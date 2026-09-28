"""
make_samples.py
Build the demo portal's sample images from the TEST split only (never seen in training):
    python app/make_samples.py
Writes app/samples/<id>.jpg and app/samples/samples.json (title, source, note, size, ground-truth
pothole boxes in pixels). Ground truth uses the same rules as evaluation: pothole class only,
boxes more than 3x longer than wide dropped.
"""
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "samples"
TEST = ROOT / "data" / "merged" / "pothole_v2" / "images" / "test"


def yolo_boxes(label_file: Path, w: int, h: int, keep_class="0"):
    boxes = []
    for line in label_file.read_text().splitlines():
        p = line.split()
        if len(p) < 5 or p[0] != keep_class:
            continue
        x, y, bw, bh = (float(v) for v in p[1:5])
        if max(bw * w, bh * h) > 3 * min(bw * w, bh * h):
            continue  # crack-shaped, excluded from evaluation too
        boxes.append([round((x - bw / 2) * w, 1), round((y - bh / 2) * h, 1),
                      round((x + bw / 2) * w, 1), round((y + bh / 2) * h, 1)])
    return boxes


def prepared(src: str, stem: str):
    """Full-frame image + label of a converted dataset image (before tiling)."""
    for split in ("train", "val", "test"):
        img = ROOT / "data" / "prepared" / src / "images" / split / f"{stem}.jpg"
        if img.exists():
            return img, ROOT / "data" / "prepared" / src / "labels" / split / f"{stem}.txt"
    raise FileNotFoundError(stem)


def test_stems(prefix: str):
    """Original (untiled) stems of a source in the test split, with/without potholes."""
    lab = TEST.parent.parent / "labels" / "test"
    pos, neg = [], []
    for p in sorted(TEST.glob(f"{prefix}_*")):
        stem = p.stem[len(prefix) + 1:]
        (pos if (lab / f"{p.stem}.txt").stat().st_size else neg).append(stem)
    return pos, neg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []

    def add(sid, title, source, note, img_path, boxes_fn):
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        cv2.imwrite(str(OUT / f"{sid}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        gt = boxes_fn(w, h)
        samples.append({"id": sid, "title": title, "source": source, "note": note,
                        "width": w, "height": h, "ground_truth": gt})
        print(f"{sid:14s} {w}x{h}  {len(gt)} labelled potholes")

    drone = ROOT / "data" / "raw" / "luis_drone"
    for f, title, note in (
            ("490", "Drone frame 490", "4K top-down frame; 2 labelled potholes"),
            ("494", "Drone frame 494", "4K top-down frame with many raveled patches"),
            ("537", "Drone frame 537", "4K top-down frame; 2 labelled potholes"),
            ("536", "Drone frame 536 (no labelled potholes)", "4K top-down frame; shows false-pin behaviour")):
        add(f"drone_{f}", title, "GitHub drone set (test road stretch)", note, drone / "images" / f"{f}.png",
            lambda w, h, f=f: yolo_boxes(drone / "labels" / f"{f}.txt", w, h))

    pos, neg = test_stems("uapd")
    for i, stem in enumerate(pos[:2] + neg[:1]):
        img, lbl = prepared("uapd_yolo", stem)
        free = stem in neg
        add(f"uapd_{i + 1}", f"UAPD {'pothole-free ' if free else ''}image {i + 1}", "UAPD (low-altitude UAV)",
            "512x512 image" + ("; no potholes" if free else ""), img, lambda w, h, lbl=lbl: yolo_boxes(lbl, w, h))

    pos, _ = test_stems("uavpdd")
    stem = sorted({s.rsplit("_t", 1)[0] for s in pos})[0]
    img, lbl = prepared("uav_pdd2023_yolo", stem)
    add("uavpdd_1", "UAV-PDD2023 frame", "UAV-PDD2023", "2592x1944 frame; labels are tiny specks the model misses",
        img, lambda w, h: yolo_boxes(lbl, w, h))

    syn = ROOT / "data" / "synthetic_geotagged_potholes" / "synthetic_geotagged_potholes"
    add("synthetic_1", "Synthetic frame", "Synthetic test set (never trained on)", "Generated road, 640x640",
        syn / "images" / "val" / "frame_00160.jpg", lambda w, h: yolo_boxes(syn / "labels" / "val" / "frame_00160.txt", w, h))

    (OUT / "samples.json").write_text(json.dumps(samples, indent=1))
    print(f"Wrote {len(samples)} samples to {OUT}")


if __name__ == "__main__":
    main()
