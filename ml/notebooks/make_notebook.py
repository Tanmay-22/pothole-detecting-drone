"""Generate train_colab.ipynb:  python ml/notebooks/make_notebook.py ml/notebooks/train_colab.ipynb"""
import json, sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Pothole detector — training on Colab (Sprint 1, step S1-21j: dataset v2)

**Before running**, put these two files in Google Drive at `MyDrive/pothole_drone/`
(both are created by step S1-19 in `E:\\ProjectsE\\Potholes detecting drone\\dist\\`):

| File | Contents |
|---|---|
| `pothole_v2.zip` | merged dataset (`pothole_v2/images`, `labels`, `data.yaml`) |
| `ml_code.zip` | the `ml/` folder (train / evaluate / predict scripts + config) |

Then: **Runtime → Change runtime type → GPU (T4 is fine)** and run the cells top to bottom.

Runs are written **directly to Drive** (`MyDrive/pothole_drone/runs/<RUN_NAME>`), so nothing is lost
if Colab disconnects — use the *Resume* cell at the bottom.
""")

md("## 1. Settings (run 3: dataset v2 = cleaner top-down data, 1024 px). Runs 1-2 used pothole_v1.")
code("""
MODEL = "yolo11s.pt"        # if still below target later: "yolo11m.pt"
DATASET = "pothole_v2"      # name of the dataset zip / folder
RUN_NAME = "pothole_v2_s1024"
IMGSZ = 1024                # tiles are upscaled so tiny potholes get more pixels
EPOCHS = 150
BATCH = 16                  # lower to 8 if you get CUDA out-of-memory

DRIVE = "/content/drive/MyDrive/pothole_drone"
""")

md("## 2. Check the GPU and mount Google Drive")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount("/content/drive")
import os
for f in (f"{DATASET}.zip", "ml_code.zip"):
    assert os.path.exists(f"{DRIVE}/{f}"), f"Missing {DRIVE}/{f} - upload it first (step S1-19)"
print("Found both zips in", DRIVE)
""")

md("## 3. Install Ultralytics and unpack dataset + code to the fast local disk")
code("""
!pip install -q ultralytics
!rm -rf /content/data /content/ml
!mkdir -p /content/data
!unzip -q {DRIVE}/{DATASET}.zip -d /content/data
!unzip -q {DRIVE}/ml_code.zip -d /content
!ls /content/data/{DATASET} /content/ml
!for s in train val test; do echo "$s: $(ls /content/data/{DATASET}/images/$s | wc -l) images"; done
""")

md("## 4. Train (≈2–3 h on a T4; early-stops if val mAP stalls for 40 epochs; if Colab disconnects use the Resume cell)")
code("""
!cd /content && python ml/train.py --data /content/data/{DATASET}/data.yaml --device 0 --set model={MODEL} name={RUN_NAME} project={DRIVE}/runs epochs={EPOCHS} batch={BATCH} imgsz={IMGSZ} exist_ok=true
""")

md("## 5. Evaluate on the held-out **test** split and pick the confidence threshold")
code("""
RUN = f"{DRIVE}/runs/{RUN_NAME}"
!cd /content && python ml/evaluate.py --model {RUN}/weights/best.pt --data /content/data/{DATASET}/data.yaml --split test --out {RUN}/eval --device 0
""")

md("## 6. Look at the results")
code("""
import json
from IPython.display import Image, display
m = json.load(open(f"{RUN}/eval/metrics.json"))
print("Box metrics :", m["box"])
il = m["image_level"]
print(f"Has-pothole : threshold {il['threshold']}  precision {il['precision']}  recall {il['recall']}  "
      f"F1 {il['f1']}  (TP {il['tp']} FP {il['fp']} FN {il['fn']} TN {il['tn']})  "
      f"recall target met: {il['met_recall_target']}")
display(Image(f"{RUN}/results.png", width=900))
display(Image(f"{RUN}/eval/pr_curve.png", width=500))
""")

md("""
## 7. Download to the PC

Download the whole `MyDrive/pothole_drone/runs/<RUN_NAME>` folder from Google Drive and place it at
`E:\\ProjectsE\\Potholes detecting drone\\models\\<RUN_NAME>\\` (it contains `weights/best.pt`, `results.csv`,
`eval/metrics.json`, `eval/threshold.txt`). Then continue with step S1-22 on the PC.
""")

md("## (Only if Colab disconnected during training) Resume from the last saved epoch")
code("""
# Re-run cells 1-3 first, then:
from ultralytics import YOLO
YOLO(f"{DRIVE}/runs/{RUN_NAME}/weights/last.pt").train(resume=True)
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", sys.argv[1], len(cells), "cells")
