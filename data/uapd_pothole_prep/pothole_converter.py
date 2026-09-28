"""
pothole_converter.py
Converts a downloaded pavement-distress dataset (YOLO .txt or Pascal VOC .xml labels)
into a single-class, pothole-only YOLO dataset:

    out/
      images/train  images/val  (images/test if the source has one)
      labels/train  labels/val
      data.yaml     (nc: 1, names: ['pothole'])
      report.txt

Class detection: reads class names from data.yaml / classes.txt / *.names in the
source folder and picks the one matching --pothole-name (regex). ALWAYS check the
printed class list. Override with --pothole-id if detection is wrong.
"""
import argparse, random, re, shutil, sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_class_names(src: Path):
    for p in list(src.rglob("*.yaml")) + list(src.rglob("*.yml")):
        try:
            import yaml
            d = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore"))
            names = d.get("names") if isinstance(d, dict) else None
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)], p
            if isinstance(names, list):
                return names, p
        except Exception:
            pass
    for pat in ("classes.txt", "*.names", "obj.names"):
        for p in src.rglob(pat):
            names = [l.strip() for l in p.read_text(errors="ignore").splitlines() if l.strip()]
            if names:
                return names, p
    return None, None


def split_of(path: Path):
    parts = [x.lower() for x in path.parts]
    for s, keys in (("train", ("train", "training")), ("val", ("val", "valid", "validation")),
                    ("test", ("test", "testing"))):
        if any(k in parts for k in keys):
            return s
    return None


def find_label(img: Path, src: Path):
    """Return (kind, path) for the label of an image, or (None, None)."""
    cands = [img.with_suffix(".txt"), img.with_suffix(".xml")]
    # mirror images/ -> labels/ or Annotations/
    s = str(img)
    for a, b in (("images", "labels"), ("JPEGImages", "Annotations"), ("images", "Annotations"),
                 ("imgs", "labels"), ("Images", "Labels")):
        if a in s:
            m = Path(s.replace(a, b, 1))
            cands += [m.with_suffix(".txt"), m.with_suffix(".xml")]
    for c in cands:
        if c.exists() and c.name.lower() != "classes.txt":
            return ("yolo" if c.suffix == ".txt" else "voc"), c
    return None, None


def parse_yolo(lbl: Path, pid: int):
    out = []
    for line in lbl.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cid = int(float(p[0]))
        except ValueError:
            continue
        if cid != pid:
            continue
        vals = [float(v) for v in p[1:]]
        if len(vals) == 4:                       # bbox
            x, y, w, h = vals
        else:                                    # polygon (segmentation) -> bbox
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        out.append((x, y, w, h))
    return out


def parse_voc(lbl: Path, name_re):
    out = []
    root = ET.parse(lbl).getroot()
    W = float(root.findtext("size/width", "0")); H = float(root.findtext("size/height", "0"))
    if W <= 0 or H <= 0:
        return out
    for o in root.iter("object"):
        if not name_re.fullmatch((o.findtext("name") or "").strip()):
            continue
        b = o.find("bndbox")
        x0, y0 = float(b.findtext("xmin")), float(b.findtext("ymin"))
        x1, y1 = float(b.findtext("xmax")), float(b.findtext("ymax"))
        out.append(((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    return out


def main(default_name_regex, dataset_label):
    ap = argparse.ArgumentParser(description=f"Convert {dataset_label} to pothole-only YOLO")
    ap.add_argument("--src", required=True, help="extracted dataset folder")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--pothole-name", default=default_name_regex,
                    help="regex (case-insensitive, full match) for the pothole class name")
    ap.add_argument("--pothole-id", type=int, default=None, help="force the pothole class id")
    ap.add_argument("--neg-ratio", type=float, default=0.15,
                    help="keep pothole-free images up to this fraction of positives (0 = none)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="val split if source has none")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if not src.exists():
        sys.exit(f"Source folder not found: {src}")
    name_re = re.compile(a.pothole_name, re.I)
    random.seed(a.seed)

    names, names_file = find_class_names(src)
    pid = a.pothole_id
    if names:
        print(f"Class names from {names_file}:")
        for i, n in enumerate(names):
            print(f"  {i}: {n}")
        if pid is None:
            hits = [i for i, n in enumerate(names) if name_re.fullmatch(str(n).strip())]
            if len(hits) != 1:
                sys.exit(f"Could not pick a unique pothole class (matches: {hits}). "
                         f"Re-run with --pothole-id N.")
            pid = hits[0]
    print(f"Using pothole class id = {pid} (VOC files matched by name regex '{a.pothole_name}')")

    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
    pos, neg, missing = [], [], 0
    for img in imgs:
        kind, lbl = find_label(img, src)
        if kind is None:
            missing += 1
            continue
        if kind == "yolo":
            if pid is None:
                sys.exit("YOLO labels found but no class list. Re-run with --pothole-id N.")
            boxes = parse_yolo(lbl, pid)
        else:
            boxes = parse_voc(lbl, name_re)
        (pos if boxes else neg).append((img, boxes))

    random.shuffle(neg)
    neg = neg[: int(len(pos) * a.neg_ratio)]
    items = pos + neg
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    counts, nbox = {}, 0
    for img, boxes in items:
        s = split_of(img.relative_to(src)) or ("val" if random.random() < a.val_frac else "train")
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        stem = "_".join(img.relative_to(src).with_suffix("").parts)  # avoid name clashes
        shutil.copy2(img, out / "images" / s / f"{stem}{img.suffix.lower()}")
        with open(out / "labels" / s / f"{stem}.txt", "w") as f:
            for x, y, w, h in boxes:
                clip = lambda v: min(max(v, 0.0), 1.0)
                f.write(f"0 {clip(x):.6f} {clip(y):.6f} {clip(w):.6f} {clip(h):.6f}\n")
        counts[s] = counts.get(s, 0) + 1
        nbox += len(boxes)

    splits = {s: f"images/{s}" for s in ("train", "val", "test") if (out / "images" / s).exists()}
    yaml_txt = f"path: {out.resolve()}\n" + "".join(f"{k}: {v}\n" for k, v in splits.items())
    yaml_txt += "nc: 1\nnames: ['pothole']\n"
    (out / "data.yaml").write_text(yaml_txt)

    rep = (f"{dataset_label} -> pothole-only YOLO\n"
           f"images scanned: {len(imgs)}  (no label found: {missing})\n"
           f"with potholes: {len(pos)}  negatives kept: {len(neg)}  pothole boxes: {nbox}\n"
           f"split counts: {counts}\n")
    (out / "report.txt").write_text(rep)
    print(rep + f"Done. Train with:  yolo detect train data={out / 'data.yaml'} model=yolov8n.pt imgsz=640")
