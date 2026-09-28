"""
check_dataset.py
Sanity-check a YOLO detection dataset before training.

    python ml/scripts/check_dataset.py data/prepared/highrpd_yolo/data.yaml [--out DIR] [--previews 20]

Writes to <dataset>/inspection/ (or --out):
    stats.json, stats.txt     per-split images / positives / negatives / boxes, box-size summary
    box_sizes.png             histogram of box short side in pixels
    previews/*.jpg            sample images with boxes drawn (mostly positives)
"""
import argparse, json, random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
TINY_PX = 12  # boxes with a short side below this are hard to detect at native resolution


def load_splits(data_yaml: Path):
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    root = Path(d["path"]) if d.get("path") else data_yaml.resolve().parent
    if not root.is_absolute():
        root = (data_yaml.resolve().parent / root).resolve()
    return root, {s: root / d[s] for s in ("train", "val", "test") if d.get(s)}


def label_path(img: Path):
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")  # last "images" folder
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(lbl: Path):
    if not lbl.exists():
        return None
    boxes = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            boxes.append(tuple(float(v) for v in p[1:]))
    return boxes


def main():
    ap = argparse.ArgumentParser(description="Inspect a YOLO dataset")
    ap.add_argument("data_yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--previews", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data_yaml = Path(a.data_yaml)
    root, splits = load_splits(data_yaml)
    out = Path(a.out) if a.out else root / "inspection"
    (out / "previews").mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)

    stats, short_sides, positives = {}, [], []
    for split, img_dir in splits.items():
        imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT)
        s = {"images": len(imgs), "positives": 0, "negatives": 0, "boxes": 0,
             "missing_labels": 0, "tiny_boxes": 0, "image_sizes": {}}
        for img in imgs:
            boxes = read_boxes(label_path(img))
            if boxes is None:
                s["missing_labels"] += 1
                continue
            h, w = cv2.imread(str(img)).shape[:2]
            s["image_sizes"][f"{w}x{h}"] = s["image_sizes"].get(f"{w}x{h}", 0) + 1
            if boxes:
                s["positives"] += 1
                positives.append((img, boxes))
            else:
                s["negatives"] += 1
            for _, _, bw, bh in boxes:
                side = min(bw * w, bh * h)
                short_sides.append(side)
                s["tiny_boxes"] += side < TINY_PX
            s["boxes"] += len(boxes)
        stats[split] = s

    if short_sides:
        ss = sorted(short_sides)
        pct = lambda q: round(ss[min(int(q * len(ss)), len(ss) - 1)], 1)
        stats["box_short_side_px"] = {"min": round(ss[0], 1), "p10": pct(0.1), "median": pct(0.5),
                                      "p90": pct(0.9), "max": round(ss[-1], 1)}
        plt.figure(figsize=(6, 3.5))
        plt.hist(ss, bins=50)
        plt.axvline(TINY_PX, color="red", ls="--", label=f"tiny (<{TINY_PX}px)")
        plt.xlabel("box short side (px, native resolution)"); plt.ylabel("boxes"); plt.legend()
        plt.tight_layout(); plt.savefig(out / "box_sizes.png", dpi=120); plt.close()

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    lines = [f"Dataset: {data_yaml}"]
    for split in splits:
        s = stats[split]
        lines.append(f"{split:5s}: {s['images']} images | {s['positives']} with potholes | "
                     f"{s['negatives']} negatives | {s['boxes']} boxes | {s['tiny_boxes']} tiny | "
                     f"{s['missing_labels']} missing labels | sizes {s['image_sizes']}")
    if "box_short_side_px" in stats:
        lines.append(f"box short side px: {stats['box_short_side_px']}")
    (out / "stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for img, boxes in random.sample(positives, min(a.previews, len(positives))):
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        for x, y, bw, bh in boxes:
            x0, y0 = int((x - bw / 2) * w), int((y - bh / 2) * h)
            x1, y1 = int((x + bw / 2) * w), int((y + bh / 2) * h)
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), max(2, w // 300))
        cv2.imwrite(str(out / "previews" / img.name), im)
    print(f"Saved stats + {min(a.previews, len(positives))} previews to {out}")


if __name__ == "__main__":
    main()
