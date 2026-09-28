"""
train.py
Train the pothole detector with Ultralytics YOLO using a YAML config.

    python ml/train.py --data data/merged/pothole_v1/data.yaml
    python ml/train.py --data data/merged/synthetic_smoke/data.yaml --set model=yolo11n.pt epochs=3 name=smoke
    python ml/train.py --data ... --device 0            # GPU (Colab)

Prints the path of best.pt at the end.
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_value(v: str):
    return yaml.safe_load(v)  # "3" -> 3, "true" -> True, "yolo11n.pt" -> str


def main():
    ap = argparse.ArgumentParser(description="Train the pothole detector")
    ap.add_argument("--config", default=str(ROOT / "ml" / "configs" / "train_v1.yaml"))
    ap.add_argument("--data", required=True, help="dataset data.yaml")
    ap.add_argument("--device", default="cpu", help="'cpu', '0' for first GPU, ...")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config keys, e.g. model=yolo11m.pt epochs=150")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8-sig"))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)

    project = Path(cfg.pop("project", "runs"))
    cfg["project"] = str(project if project.is_absolute() else ROOT / project)
    cfg["data"] = str(Path(a.data).resolve())
    cfg["device"] = a.device
    model = YOLO(cfg.pop("model"))
    print(f"Training with: {cfg}")
    model.train(**cfg)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"BEST_WEIGHTS {best}")


if __name__ == "__main__":
    main()
