# Pothole-detecting drone

A drone flies a predefined road path, a Raspberry Pi camera captures top-down frames, a detector
finds potholes, and a web dashboard shows each one as a pin on a map (details on click).
Step-by-step plan and progress: [PLAN.md](PLAN.md). New session? Start with [HANDOFF.md](HANDOFF.md).

**Sprint 1 (done): the pothole detector** — YOLO11s, single class `pothole`, runs on the ground
after the flight. Final model: [`models/pothole_v1/`](models/pothole_v1/README.txt).

## Setup (Windows, CPU)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r ml/requirements.txt
```

Datasets, weights and runs are not in git (see `.gitignore`); rebuild them with the steps below.

## Detect potholes

```bash
.venv\Scripts\python.exe ml/predict.py --model models/pothole_v1/weights/best.pt --source <image-or-folder> --out runs/predict/<name>
```

- Frames larger than 1280 px (e.g. 4K drone frames) are cut into overlapping 640 px tiles, and the
  boxes are mapped back to full-frame pixels.
- The confidence threshold defaults to `models/pothole_v1/threshold.txt` (0.30); override with `--conf`.
- Boxes of one pothole split across overlapping tiles are merged into a single detection.
- Output: `detections.json` / `detections.csv` (per image: `has_pothole` and boxes in pixels and
  normalised coordinates, with confidence) and `annotated/` images. Sprint 2 (geolocation) builds
  on this format.
- CPU speed: about 18 s per 4K frame.

## Demo portal (Sprint 1 demo)

A local web app (FastAPI + React) to try the detector: pick a test sample or upload a road photo,
see the pothole boxes, move the confidence threshold slider (instant, same result as `predict.py`
at that threshold) and compare with the labelled potholes. Includes a model-info panel.

```bash
# one-time setup
.venv\Scripts\python.exe app/make_samples.py                  # test-split sample images -> app/samples/
cd app/frontend && npm install && npm run build && cd ../..    # React UI -> app/frontend/dist/
# run, then open http://127.0.0.1:8000
.venv\Scripts\python.exe app/backend/main.py
```

- Samples are processed in the background at startup and cached in `app/.cache/`; uploads take
  ~1 s (small images) to ~20 s (4K frames) on CPU the first time.
- Red box = detection, orange = detection that matches no labelled pothole, green dashed = label.
- UI development with hot reload: `cd app/frontend && npm run dev` (port 5173, API proxied to 8000).

## Data pipeline

| Step | Command |
|---|---|
| Download | `ml/scripts/download_file.py` (parallel, resumable; Zenodo / Mendeley) — sources in PLAN.md |
| Convert to pothole-only YOLO | `data/<dataset>_pothole_prep/prepare.py --src data/raw/<dataset> --out data/prepared/<dataset>_yolo` |
| Merge + split | `ml/scripts/merge_datasets.py` (exact v2 command in PLAN.md, step S1-21i) |
| Inspect | `ml/scripts/check_dataset.py <data.yaml>` (stats + preview images) |

The merge step removes flipped/rotated duplicate images, keeps every tile and every overlapping video
frame of one road stretch in the same split, tiles large images, drops crack-shaped boxes and keeps
street-level photos in training only.

Dataset `pothole_v2` (used by the final model): UAPD, UAV-PDD2023 and a top-down drone set
(luisaugustos/Pothole-Recognition), plus a street-level Roboflow set for training only.
HighRPD was dropped because its "pit" labels include tree shadows and stains.

## Train and evaluate

Training runs on Google Colab (GPU): upload `dist/pothole_v2.zip` and `dist/ml_code.zip` to
`MyDrive/pothole_drone/`, then run [`ml/notebooks/train_colab.ipynb`](ml/notebooks/train_colab.ipynb)
(regenerate it with `ml/notebooks/make_notebook.py`). Settings: [`ml/configs/train_v1.yaml`](ml/configs/train_v1.yaml).

```bash
# local smoke test (CPU, minutes)
.venv\Scripts\python.exe ml/train.py --data data/merged/synthetic_smoke/data.yaml --set model=models/pretrained/yolo11n.pt epochs=3 imgsz=640 name=smoke exist_ok=true
# evaluate: box mAP + image-level "has pothole?" precision/recall over thresholds
.venv\Scripts\python.exe ml/evaluate.py --model models/pothole_v1/weights/best.pt --data data/merged/pothole_v2/data.yaml --split test --out runs/eval
```

## Data sources and licenses

Datasets are not stored in this repo (except the synthetic set); the scripts download/convert them.
Training-run preview images under `models/` contain samples from these datasets.

| Dataset | Source | License |
|---|---|---|
| UAV-PDD2023 | [Zenodo 8429208](https://zenodo.org/records/8429208) | CC BY 4.0 |
| UAPD | [GitHub: tantantetetao/UAPD-Pavement-Distress-Dataset](https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset) | released for public use |
| HighRPD (used in runs 1-2 only) | [Mendeley Data sywswj7djj](https://data.mendeley.com/datasets/sywswj7djj/1) | CC BY 4.0 |
| Top-down drone set | [GitHub: luisaugustos/Pothole-Recognition](https://github.com/luisaugustos/Pothole-Recognition) (Silva et al., *Sensors* 2020) | no license stated |
| Street-level potholes (train only) | [Roboflow Universe: pothole-detection-zdizt](https://universe.roboflow.com/drone-zh0ho/pothole-detection-zdizt/dataset/1) | MIT |
| Synthetic geotagged set | generated for this project (`data/synthetic_geotagged_potholes`) | — |

## Sprint 1 results

| Test set (top-down, half pothole-free) | mAP50 |
|---|---|
| All (172 images) | 0.39 |
| Top-down drone frames | 0.53 |
| UAPD | 0.76 |
| UAV-PDD2023 (labels are tiny specks) | 0.03 |

On full 4K drone frames (22 test frames, 78 potholes):

| Threshold | Potholes found per frame | False pins per frame |
|---|---|---|
| 0.15 | 82% | 6.5 |
| **0.30 (default)** | 77% | 3.0 |
| 0.40 | 69% | 1.5 |

A drone sees each pothole in several consecutive frames, so Sprint 2 will confirm a pothole only
when it is detected in 2+ frames at the same location, which should remove most false pins.

The Sprint 1 target (mAP50 ≥ 0.6) is not met overall; the image-level recall target (≥ 0.9) is met
on tiles. Known weaknesses: lane markings, roadside grass and wide dark cracks are sometimes flagged.
The next real gain is training on this project's own drone photos.
