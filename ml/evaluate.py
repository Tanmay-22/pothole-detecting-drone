"""
evaluate.py
Evaluate a trained pothole detector on one split of a dataset.

    python ml/evaluate.py --model models/pothole_v1/best.pt --data data/merged/pothole_v1/data.yaml --split test --out models/pothole_v1/eval

Writes to --out:
    metrics.json     box metrics (mAP50, mAP50-95, precision, recall) + image-level "has pothole"
                     metrics at the chosen threshold + the full threshold sweep
    threshold.txt    chosen confidence threshold (highest F1 among thresholds with recall >= --min-recall)
    pr_curve.png     image-level precision / recall / F1 vs threshold
    val/             Ultralytics validation plots (PR curve, confusion matrix, sample predictions)
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from check_dataset import IMG_EXT, label_path, load_splits, read_boxes  # noqa: E402


def image_level_sweep(max_conf, truth, thresholds):
    rows = []
    for t in thresholds:
        tp = sum(1 for c, y in zip(max_conf, truth) if c >= t and y)
        fp = sum(1 for c, y in zip(max_conf, truth) if c >= t and not y)
        fn = sum(1 for c, y in zip(max_conf, truth) if c < t and y)
        tn = sum(1 for c, y in zip(max_conf, truth) if c < t and not y)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"threshold": round(t, 2), "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate the pothole detector")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-recall", type=float, default=0.9,
                    help="image-level recall the chosen threshold must reach")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)
    a.imgsz = a.imgsz or (model.ckpt or {}).get("train_args", {}).get("imgsz", 640)
    print(f"Evaluating at imgsz {a.imgsz}")

    # 1) standard box metrics
    v = model.val(data=str(Path(a.data).resolve()), split=a.split, imgsz=a.imgsz, device=a.device,
                  project=str(out), name="val", exist_ok=True, plots=True, verbose=False)
    box = {"mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4),
           "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4)}

    # 2) image-level "has pothole?" using the highest box confidence per image
    _, splits = load_splits(Path(a.data))
    imgs = sorted(p for p in splits[a.split].glob("*") if p.suffix.lower() in IMG_EXT)
    truth = [bool(read_boxes(label_path(p))) for p in imgs]
    max_conf = []
    for i in range(0, len(imgs), 32):
        for r in model.predict([str(p) for p in imgs[i:i + 32]], imgsz=a.imgsz, device=a.device,
                               conf=0.01, verbose=False):
            max_conf.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)

    sweep = image_level_sweep(max_conf, truth, [t / 100 for t in range(1, 96)])
    ok = [r for r in sweep if r["recall"] >= a.min_recall]
    chosen = max(ok or sweep, key=lambda r: (r["f1"], r["threshold"]))
    met_target = bool(ok)

    metrics = {"model": str(Path(a.model).resolve()), "data": str(Path(a.data).resolve()),
               "split": a.split, "images": len(imgs), "positives": sum(truth),
               "negatives": len(truth) - sum(truth), "box": box,
               "image_level": {**chosen, "min_recall_target": a.min_recall,
                               "met_recall_target": met_target},
               "sweep": sweep}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "threshold.txt").write_text(f"{chosen['threshold']}\n")

    ts = [r["threshold"] for r in sweep]
    plt.figure(figsize=(6, 4))
    for k in ("precision", "recall", "f1"):
        plt.plot(ts, [r[k] for r in sweep], label=k)
    plt.axvline(chosen["threshold"], color="k", ls="--", label=f"chosen {chosen['threshold']}")
    plt.axhline(a.min_recall, color="grey", ls=":", lw=1)
    plt.xlabel("confidence threshold"); plt.ylabel("image-level score"); plt.ylim(0, 1.02)
    plt.title(f"Has pothole? ({a.split}, {len(imgs)} images)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=120); plt.close()

    print(f"Box metrics ({a.split}): {box}")
    print(f"Image-level @ threshold {chosen['threshold']}: precision {chosen['precision']} "
          f"recall {chosen['recall']} F1 {chosen['f1']} (TP {chosen['tp']} FP {chosen['fp']} "
          f"FN {chosen['fn']} TN {chosen['tn']})"
          + ("" if met_target else f"  [recall target {a.min_recall} NOT reached]"))
    print(f"Saved metrics.json, threshold.txt, pr_curve.png to {out}")


if __name__ == "__main__":
    main()
