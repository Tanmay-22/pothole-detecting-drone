"""
predict.py
Run the pothole detector on an image or a folder of images.

    python ml/predict.py --model models/pothole_v1/best.pt --source path/to/frames --out runs/predict/flight_01

--conf defaults to the value in threshold.txt next to the model (written by evaluate.py), else 0.25.
Large frames (long side > --tile-above) are cut into overlapping --tile tiles, like the training
data, and the boxes are mapped back to full-frame pixels, de-duplicated with NMS, and boxes that
mostly lie inside another (a pothole split across tiles) are merged into one.

Writes to --out:
    detections.json   [{image, width, height, has_pothole, detections: [{bbox_xyxy_px, bbox_norm, confidence}]}]
    detections.csv    one row per detection (images without potholes get one row with has_pothole=0)
    annotated/        copies of the images with boxes drawn
This output is the input contract for Sprint 2 (geolocation).
"""
import argparse, csv, json
from pathlib import Path

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def tile_origins(w, h, tile, overlap):
    stride = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


def detect(model, im, a):
    """Return [(x0, y0, x1, y1, conf)] in full-image pixels."""
    return merge_fragments(detect_raw(model, im, a))


def detect_raw(model, im, a):
    """Detections with confidence >= a.conf after cross-tile NMS, before fragment merging.
    Filtering these by a higher threshold and then merging equals running detect() at that threshold."""
    h, w = im.shape[:2]
    if max(w, h) <= a.tile_above:
        crops = [(0, 0, im)]
    else:
        crops = [(x, y, im[y:y + a.tile, x:x + a.tile]) for x, y in tile_origins(w, h, a.tile, a.overlap)]
    boxes, confs = [], []
    for i in range(0, len(crops), 16):
        batch = crops[i:i + 16]
        for (ox, oy, _), r in zip(batch, model.predict([c for _, _, c in batch], imgsz=a.imgsz,
                                                         conf=a.conf, device=a.device, verbose=False)):
            for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                boxes.append([x0 + ox, y0 + oy, x1 + ox, y1 + oy])
                confs.append(c)
    if not boxes:
        return []
    b, c = torch.tensor(boxes), torch.tensor(confs)
    keep = nms(b, c, 0.5).tolist()
    return [(*b[k].tolist(), float(c[k])) for k in keep]


def merge_fragments(dets, min_overlap=0.5):
    """Merge boxes where most of the smaller box lies inside the other (one pothole split across
    overlapping tiles, or boxed twice). Keeps the union box and the highest confidence."""
    dets = sorted(dets, key=lambda d: -d[4])
    merged = True
    while merged:
        merged = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, o = dets[i], dets[j]
                iw = min(a[2], o[2]) - max(a[0], o[0])
                ih = min(a[3], o[3]) - max(a[1], o[1])
                if iw <= 0 or ih <= 0:
                    continue
                smaller = min((a[2] - a[0]) * (a[3] - a[1]), (o[2] - o[0]) * (o[3] - o[1]))
                if iw * ih >= min_overlap * smaller:
                    dets[i] = (min(a[0], o[0]), min(a[1], o[1]), max(a[2], o[2]), max(a[3], o[3]), max(a[4], o[4]))
                    del dets[j]
                    merged = True
                    break
            if merged:
                break
    return dets


def main():
    ap = argparse.ArgumentParser(description="Detect potholes in images")
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=None, help="default: threshold.txt next to model, else 0.25")
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--tile-above", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-annotated", action="store_true")
    a = ap.parse_args()

    if a.conf is None:
        for cand in (Path(a.model).parent / "threshold.txt", Path(a.model).parent.parent / "threshold.txt"):
            if cand.exists():
                a.conf = float(cand.read_text().strip())
                break
        else:
            a.conf = 0.25
    print(f"Using confidence threshold {a.conf}")

    src = Path(a.source)
    imgs = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    out = Path(a.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)
    a.imgsz = a.imgsz or (model.ckpt or {}).get("train_args", {}).get("imgsz", 640)

    results = []
    for img in imgs:
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        dets = []
        for x0, y0, x1, y1, c in sorted(detect(model, im, a), key=lambda d: -d[4]):
            dets.append({"bbox_xyxy_px": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                         "bbox_norm": [round((x0 + x1) / 2 / w, 6), round((y0 + y1) / 2 / h, 6),
                                       round((x1 - x0) / w, 6), round((y1 - y0) / h, 6)],
                         "confidence": round(c, 4)})
            if not a.no_annotated:
                cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), max(2, w // 400))
                cv2.putText(im, f"{c:.2f}", (int(x0), max(int(y0) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, w / 1500), (0, 0, 255), max(1, w // 800))
        results.append({"image": str(img.resolve()), "width": w, "height": h,
                        "has_pothole": bool(dets), "detections": dets})
        if not a.no_annotated:
            cv2.imwrite(str(out / "annotated" / img.name), im)

    (out / "detections.json").write_text(json.dumps(results, indent=2))
    with open(out / "detections.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "has_pothole", "confidence", "x0", "y0", "x1", "y1", "cx_norm", "cy_norm", "w_norm", "h_norm"])
        for r in results:
            if not r["detections"]:
                wr.writerow([r["image"], 0] + [""] * 9)
            for d in r["detections"]:
                wr.writerow([r["image"], 1, d["confidence"], *d["bbox_xyxy_px"], *d["bbox_norm"]])

    n_pos = sum(r["has_pothole"] for r in results)
    n_det = sum(len(r["detections"]) for r in results)
    print(f"{len(results)} images | {n_pos} with potholes | {n_det} detections -> {out}")


if __name__ == "__main__":
    main()
