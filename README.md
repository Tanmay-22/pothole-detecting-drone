# Pothole-detecting drone

A drone flies a predefined road path, a Raspberry Pi camera captures top-down frames, a detector
finds potholes, and a web dashboard shows each one as a pin on a map (details on click).
Step-by-step plan and progress: [PLAN.md](PLAN.md). New session? Start with [HANDOFF.md](HANDOFF.md).

**Sprint 1 (done): the pothole detector** — YOLO11s, single class `pothole`, runs on the ground
after the flight. Final model: [`models/pothole_v1/`](models/pothole_v1/README.txt).

**Dashboard demo (done):** route planner, results map and flight upload in `web/`, built on a first
geolocation module in `geo/` and tested on a synthetic flight. See [Dashboard demo](#dashboard-demo-how-the-final-product-will-look).

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

## Dashboard demo (how the final product will look)

A separate web app (`web/`, FastAPI + React + Leaflet) with three pages:

- **Plan route**: click on the map to draw the road to fly (drag or click a waypoint to move or delete it).
  The page shows route length, flight time, number of photos, ground strip width, photo spacing and
  cm per pixel for the chosen altitude, speed, overlap and camera. Camera settings default to the
  synthetic test camera because the real camera isn't chosen yet. Mission export comes later.
- **Results map**: the flight track and one pin per pothole, coloured by severity (low / medium / high by
  diameter; dashed = unverified, low confidence). Click a pin for the photo crop, size, area, confidence,
  number of photos, time, flight and location. Filters, GeoJSON / CSV download, and streets / satellite
  basemaps.
- **Flights**: upload a flight zip. The server runs the detector on every photo, geolocates each detection
  and merges repeats into one pothole, showing a progress bar. The page also lists every flight.

```bash
# one-time setup
.venv\Scripts\python.exe web/tools/make_synthetic_flight.py          # demo flight -> web/data/flights/synthetic_001 + dist/synthetic_flight_001.zip
.venv\Scripts\python.exe geo/process_flight.py web/data/flights/synthetic_001   # ~2 min on CPU
cd web/frontend && npm install && npm run build && cd ../..
# run, then open http://127.0.0.1:8001
.venv\Scripts\python.exe web/backend/main.py
```

To demo an upload, upload `dist/synthetic_flight_001.zip` on the Flights page.
Basemap tiles come from OpenStreetMap and Esri World Imagery, so the pages need internet access.

**Flight folder / upload zip format** (see `geo/flight.py`):
- `images/`: one photo per frame.
- `frames.csv`: frame, image, timestamp.
- `telemetry.csv`: timestamp, lat, lon, alt_m (height above the road), heading_deg.
- `flight.json` (optional): name, `camera.hfov_deg`.

Settings (camera field of view, telemetry smoothing, detector threshold, merge radius, severity
thresholds) are in `geo/config.yaml`.

**How a pin is made:**
1. The detector runs at threshold 0.30.
2. Each box is geolocated from the telemetry at the photo's timestamp (smoothed over 1 s), the altitude, the
   camera field of view and the heading. The model assumes the camera points straight down, with no lens
   distortion.
3. Detections from different photos within 1 m of each other are merged. A pothole is kept only if it is
   seen in at least 2 photos.

Results on the synthetic flight (`python geo/eval_synthetic.py flight`; 45 true potholes):

| Threshold | Found | False | Location error (median / max) |
|---|---|---|---|
| 0.15 | 42 | 0 | 0.10 / 0.25 m |
| **0.30** | 37 | 0 | 0.10 / 0.23 m |
| 0.50 | 22 | 0 | 0.12 / 0.26 m |

Caveats:
- The synthetic road is easy, so these numbers say nothing about false alarms on real roads.
- Detector boxes are about 1.33× the pothole, so sizes are multiplied by `box_scale: 0.75`. That factor
  was measured on this same synthetic flight and must be re-measured on real drone photos.
- The synthetic road is simulated and does not line up with a real road on the basemap.

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
