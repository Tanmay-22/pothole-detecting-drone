"""
merge_datasets.py
Merge several single-class YOLO datasets (outputs of the data/*_prep converters) into one.

    python ml/scripts/merge_datasets.py \
        --inputs uavpdd=data/prepared/uav_pdd2023_yolo uapd=data/prepared/uapd_yolo highrpd=data/prepared/highrpd_yolo \
        --out data/merged/pothole_v1

What it does:
  * drops flipped / rotated duplicates within a source (UAV-PDD2023 ships lr_/tb_/r180_ copies
    of every image);
  * ignores the input splits and re-splits train/val/test PER SOURCE, GROUPED BY SOURCE PHOTO
    (tile suffixes like `_640_02_05`, `_top_left`, `_4_3` and flip prefixes are stripped to find
    the photo), so tiles of one photo never end up in both train and test;
  * sources named in --train-only (e.g. street-level photos) go entirely to train, never val/test;
  * drops boxes longer than --max-aspect x their width (crack-shaped "potholes"), except in
    --train-only sources, where oblique views make real potholes look elongated;
  * cuts images larger than --tile-above px into --tile px tiles (--overlap), clipping boxes and
    dropping boxes that are less than half visible (tiles holding such fragments are skipped);
  * keeps pothole-free images/tiles up to --neg-ratio x positives (train) or --eval-neg-ratio x
    positives (val/test) per source and split;
  * prefixes every file with its source name and writes data.yaml (no `path`: resolve the yaml
    to an absolute path when training) plus merge_report.json.
"""
import argparse, json, random, re, shutil
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# _fNNN: frame number inside a road segment (luis_drone: `seg03_f501` -> group `seg03`)
TILE_SUFFIX = re.compile(r"(_640_\d+_\d+|_(top|bottom)_(left|right)|_\d+_\d+|_f\d+)$", re.I)
AUG_PREFIX = re.compile(r"(^|_)(lr|tb|r90|r180|r270)_(?=\d)", re.I)  # UAV-PDD2023 flip/rotate copies


def photo_group(stem: str) -> str:
    return AUG_PREFIX.sub(r"\1", TILE_SUFFIX.sub("", stem))


def dihedral_key(img: Path) -> bytes:
    """Coarse fingerprint that is identical for flipped / rotated copies of the same image."""
    t = cv2.resize(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE), (16, 16), interpolation=cv2.INTER_AREA)
    r = cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE)
    variants = [t, cv2.flip(t, 1), cv2.flip(t, 0), cv2.flip(t, -1),
                r, cv2.flip(r, 1), cv2.flip(r, 0), cv2.flip(r, -1)]
    return min((v // 16).tobytes() for v in variants)


def drop_duplicates(items):
    """Keep one image per flip/rotation-duplicate set (training already applies flips)."""
    seen, kept = set(), []
    for img, boxes in items:
        k = dihedral_key(img)
        if k not in seen:
            seen.add(k)
            kept.append((img, boxes))
    return kept


def read_boxes(lbl: Path):
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append(tuple(float(v) for v in p[1:]))
    return boxes


def collect(src: Path):
    """All (image, boxes) pairs of a prepared dataset, whatever split they were in."""
    items = []
    for img in sorted((src / "images").rglob("*")):
        if img.suffix.lower() in IMG_EXT:
            lbl = src / "labels" / img.parent.name / f"{img.stem}.txt"
            items.append((img, read_boxes(lbl)))
    return items


def split_groups(groups, test_frac, val_frac, rng):
    """Assign whole photo groups to test/val/train until each reaches its share of images."""
    names = sorted(groups)
    rng.shuffle(names)
    total = sum(len(groups[g]) for g in names)
    out, n_test, n_val = {}, 0, 0
    t_test, t_val = test_frac * total, val_frac * total
    for g in names:
        size = len(groups[g])
        # skip groups that would overshoot a split's share by >50% (long video segments)
        if n_test < t_test and n_test + size <= 1.5 * t_test:
            out[g], n_test = "test", n_test + size
        elif n_val < t_val and n_val + size <= 1.5 * t_val:
            out[g], n_val = "val", n_val + size
        else:
            out[g] = "train"
    for split in ("test", "val"):  # never leave val/test empty: take the smallest train group
        if names and split not in out.values():
            g = min((g for g in names if out[g] == "train"), key=lambda g: len(groups[g]), default=None)
            if g is not None:
                out[g] = split
    return out


def tiles_of(w, h, tile, overlap):
    stride = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


def cut_tiles(w, h, boxes, tile, overlap):
    """Yield (suffix, (tx, ty), tile_boxes); skip tiles that would hold an unlabeled fragment."""
    px = [((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
          for x, y, bw, bh in boxes]
    for tx, ty in tiles_of(w, h, tile, overlap):
        kept, ambiguous = [], False
        for x0, y0, x1, y1 in px:
            cx0, cy0 = max(x0, tx), max(y0, ty)
            cx1, cy1 = min(x1, tx + tile), min(y1, ty + tile)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            if (cx1 - cx0) * (cy1 - cy0) >= 0.5 * (x1 - x0) * (y1 - y0):
                kept.append((((cx0 + cx1) / 2 - tx) / tile, ((cy0 + cy1) / 2 - ty) / tile,
                             (cx1 - cx0) / tile, (cy1 - cy0) / tile))
            else:
                ambiguous = True
        if not ambiguous:
            yield f"_t{tx}_{ty}", (tx, ty), kept


@lru_cache(maxsize=4)
def load_image(path: str):
    return cv2.imread(path)


def write_labels(path: Path, boxes):
    with open(path, "w") as f:
        for x, y, w, h in boxes:
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description="Merge pothole YOLO datasets with grouped re-split")
    ap.add_argument("--inputs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--tile-above", type=int, default=1280, help="tile images whose long side exceeds this")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--neg-ratio", type=float, default=0.3,
                    help="pothole-free images per positive in train")
    ap.add_argument("--eval-neg-ratio", type=float, default=1.0,
                    help="pothole-free images per positive in val/test (measures false alarms)")
    ap.add_argument("--train-only", nargs="*", default=[], help="source names used only for training")
    ap.add_argument("--max-aspect", type=float, default=0, help="drop boxes with long/short side above this (0 = keep all)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    report = {}
    for spec in a.inputs:
        name, src = spec.split("=", 1)
        rng = random.Random(a.seed)
        all_items = collect(Path(src))
        items = drop_duplicates(all_items)
        groups = defaultdict(list)
        for img, boxes in items:
            groups[photo_group(img.stem)].append((img, boxes))
        # split photos with potholes and pothole-free photos separately, so each split gets its
        # share of positives no matter how many negatives a source has
        pos_groups = {g: m for g, m in groups.items() if any(b for _, b in m)}
        neg_groups = {g: m for g, m in groups.items() if g not in pos_groups}
        if name in a.train_only:
            assign = {g: "train" for g in groups}
        else:
            assign = {**split_groups(pos_groups, a.test_frac, a.val_frac, rng),
                      **split_groups(neg_groups, a.test_frac, a.val_frac, rng)}

        # expand to output samples (tiling large images), grouped by split
        pos, neg = defaultdict(list), defaultdict(list)
        n_long = 0
        for g, members in groups.items():
            split = assign[g]
            for img, boxes in members:
                with Image.open(img) as im:
                    w, h = im.size
                # not for train-only sources: in oblique street photos real potholes look elongated
                if a.max_aspect and name not in a.train_only:
                    ok = [b for b in boxes
                          if max(b[2] * w, b[3] * h) <= a.max_aspect * max(min(b[2] * w, b[3] * h), 1e-6)]
                    n_long += len(boxes) - len(ok)
                    boxes = ok
                if max(w, h) > a.tile_above:
                    for suffix, origin, tboxes in cut_tiles(w, h, boxes, a.tile, a.overlap):
                        (pos if tboxes else neg)[split].append((f"{img.stem}{suffix}", img, origin, tboxes))
                else:
                    (pos if boxes else neg)[split].append((img.stem, img, None, boxes))

        rep = {"input_images": len(all_items), "duplicates_dropped": len(all_items) - len(items),
               "photo_groups": len(groups), "long_boxes_dropped": n_long}
        for split in ("train", "val", "test"):
            n = neg[split]
            rng.shuffle(n)
            ratio = a.neg_ratio if split == "train" else a.eval_neg_ratio
            n = sorted(n[: int(len(pos[split]) * ratio)], key=lambda t: str(t[1]))  # sorted: fewer re-decodes
            for stem, image, origin, boxes in pos[split] + n:
                fname = f"{name}_{stem}"
                if origin is None:
                    dst = out / "images" / split / f"{fname}{image.suffix.lower()}"
                    shutil.copy2(image, dst)
                else:
                    tx, ty = origin
                    crop = load_image(str(image))[ty:ty + a.tile, tx:tx + a.tile]
                    assert crop.shape[:2] == (a.tile, a.tile), f"{image}: size mismatch (EXIF rotation?)"
                    dst = out / "images" / split / f"{fname}.jpg"
                    cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_labels(out / "labels" / split / f"{fname}.txt", boxes)
            rep[split] = {"positives": len(pos[split]), "negatives": len(n),
                          "boxes": sum(len(b) for *_, b in pos[split]),
                          "groups": sum(1 for g in assign.values() if g == split)}
        report[name] = rep
        print(f"{name}: {json.dumps(rep)}")

    # leakage check: no photo group may appear in more than one split
    seen = {}
    for split in ("train", "val", "test"):
        for p in (out / "images" / split).iterdir():
            src_name, stem = p.stem.split("_", 1)
            g = (src_name, photo_group(re.sub(r"_t\d+_\d+$", "", stem)))
            assert seen.setdefault(g, split) == split, f"group {g} in {seen[g]} and {split}"

    (out / "data.yaml").write_text("# Resolve this file to an absolute path when training.\n"
                                   "train: images/train\nval: images/val\ntest: images/test\n"
                                   "nc: 1\nnames: ['pothole']\n")
    (out / "merge_report.json").write_text(json.dumps(report, indent=2))
    totals = {s: len(list((out / "images" / s).iterdir())) for s in ("train", "val", "test")}
    print(f"Wrote {out}  images per split: {totals}  (no photo group crosses splits)")


if __name__ == "__main__":
    main()
