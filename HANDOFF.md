# HANDOFF — Pothole-detecting drone (read this first in a new session)

Last updated: **2026-09-28**. Status: **Sprint 1 complete + demo portal complete. Sprint 2 not started.**
Detailed step-by-step history with every number: [PLAN.md](PLAN.md) (Progress section at the top).
How to run things: [README.md](README.md).

---

## 1. The project

A drone flies a predefined road path (the path is drawn on a web dashboard). A Raspberry Pi camera
captures top-down road frames. After landing, a detector finds potholes on the ground station, their
GPS location is computed, and they appear as pins on a web map. Clicking a pin shows: size,
severity, photo crop, time / flight / confidence (details only on click).

Sprints (Sprint 1 done; the rest are the roadmap):
| Sprint | Scope | Status |
|---|---|---|
| 1 | Pothole detector (YOLO boxes, single class `pothole`) + demo portal | **done** |
| 2 | Capture & geolocation: Pi capture (picamera2) + telemetry sync; pixel → lat/lon from altitude, FOV, heading; size in metres; merge duplicates within ~2 m; confirm a pothole only if seen in ≥ 2 consecutive frames; validate on the synthetic geotagged set | next |
| 3 | Backend: FastAPI + SQLite; upload a flight → detect → geolocate → store potholes | later |
| 4 | Dashboard: React + Leaflet map, pins, details on click; draw route → waypoints → export GeoJSON + QGC `.plan` | later |
| 5 | Field test: own drone photos, labelling, fine-tune model v2 | later |

## 2. Working with this user (important)

- **Plans must be broken into the smallest discrete, serial steps**, each with needs / does /
  produces / check, done one after another (numbered like S1-01…, D-01…). Record every step's result
  in PLAN.md's Progress section as you go.
- **Ask before any download** (state file, source, size) — datasets, weights, pip/npm packages.
- Ask questions before planning a new sprint or feature; the user likes to choose between options.
- Report results honestly (targets missed, caveats); the user makes the accept / retrain calls.
- Don't commit to git unless asked (repo initialised, **no commits yet**).

## 3. Decisions already made (don't re-ask)

- Detection = YOLO bounding boxes, class `pothole`; "has pothole" = any box ≥ threshold.
- Inference runs **on the ground after the flight**, not on the Pi → YOLO11s at 1024 px is fine.
- Training on **Google Colab** (this PC has no NVIDIA GPU). User uploads zips to
  `MyDrive/pothole_drone/`, runs `ml/notebooks/train_colab.ipynb`, downloads the run folder into `models/`.
- Dashboard / portal stack: **FastAPI + React (+ Leaflet later)**.
- Final Sprint 1 model: **`models/pothole_v1/`** (Colab run 3), shipped threshold **0.30** (user's choice).
- **Still open — ask at the start of Sprint 2:** Pi camera model, flight altitude, flight controller
  (Pixhawk + ArduPilot / PX4 / DJI / undecided). Mission export is planned as generic GeoJSON + QGC `.plan`.

## 4. Where things are

Project root: **`E:\ProjectsE\Potholes detecting drone`** (moved from D:\Projects on 2026-09-24 — D: was full).

| Path | What |
|---|---|
| `PLAN.md` | Full plan + progress checklist with all results (source of truth for history) |
| `README.md` | Setup, predict, demo portal, data pipeline, train/eval commands, results |
| `ml/predict.py` | Detector CLI: image/folder → `detections.json`/`.csv` + annotated images. Tiles frames > 1280 px into 640 px tiles (20 % overlap), NMS, merges fragments. `detect_raw()` + `merge_fragments()` reused by the portal. **Output format = input contract for Sprint 2.** |
| `ml/evaluate.py` | Box mAP + image-level "has pothole?" sweep; picks a recall-first threshold (a suggestion only) |
| `ml/train.py`, `ml/configs/train_v1.yaml` | Training wrapper + config (yolo11s, imgsz 1024, 150 epochs, patience 40, flips + rotation aug) |
| `ml/notebooks/train_colab.ipynb` | Colab notebook (regenerate with `ml/notebooks/make_notebook.py`; `DATASET` / `RUN_NAME` settings) |
| `ml/scripts/download_file.py` | Parallel resumable range downloader (+ `--md5` / `--sha256`) |
| `ml/scripts/merge_datasets.py` | Merge + dedupe (flip/rotation copies) + grouped split (tiles / video stretches never cross splits) + tiling + `--train-only` + `--max-aspect` |
| `ml/scripts/check_dataset.py` | Dataset stats + preview images |
| `data/*_pothole_prep/` | Original converters (user's) → pothole-only YOLO |
| `data/raw/` | uav_pdd2023, uapd, highrpd (zips + extracted), luis_drone (111 4K PNGs + `segments.json`), roboflow_drone1 |
| `data/prepared/*_yolo` | Converted sources (neg-ratio 5) |
| `data/merged/pothole_v2` | **Dataset used by the final model** (rebuild command in PLAN.md S1-21i); `pothole_v1` = old dataset; `synthetic_smoke` = CPU smoke-test config |
| `data/synthetic_geotagged_potholes/` | Synthetic 240 frames + telemetry.csv, frames.csv, potholes_ground_truth.csv, detections_ground_truth.csv → **use for Sprint 2 geolocation tests** (8.6 m altitude, 82° HFOV, 2.34 cm/px, top of image = north) |
| `models/pothole_v1/` | **Final model** (read-only): weights/best.pt, threshold.txt (0.30), eval/metrics*.json, README.txt, SHA256SUMS.txt |
| `models/pothole_v2_s1024/` | Colab run 3 (source of the final model) |
| `models/backup/pothole_v1_s1024_2026-09-25/` | Backup of run 2 (read-only) |
| `models/pothole_v1_s`, `models/pothole_v1_s1024` | Runs 1 and 2 (superseded) |
| `app/backend/main.py` | Demo portal API (+ serves built UI) on :8000; `model_card.json` |
| `app/frontend/` | React 19 + Vite 8 UI (`src/App.jsx`, `src/detections.js` = JS port of threshold/merge/match) |
| `app/make_samples.py` | Builds `app/samples/` (9 test-split samples + ground truth) |
| `dist/` | Colab upload zips (pothole_v2.zip, ml_code.zip; old pothole_v1.zip) |
| `.claude/launch.json` | Preview server config "demo-portal" |

Gitignored (regenerable, large): data/raw, data/prepared, data/merged, runs/, dist/, *.pt, *.zip,
app/samples, app/.cache, app/frontend/node_modules, app/frontend/dist.

## 5. How to run

```bash
# Python: always use the venv's python (Python 3.14, ultralytics 8.4.160, torch 2.14 CPU)
.venv\Scripts\python.exe ml/predict.py --model models/pothole_v1/weights/best.pt --source <img|folder> --out runs/predict/<name>
# Demo portal → http://127.0.0.1:8000 (or preview_start "demo-portal")
.venv\Scripts\python.exe app/backend/main.py
# Rebuild portal UI after editing app/frontend/src
cd app/frontend && npm run build
```

## 6. Sprint 1 results (final model `models/pothole_v1`)

- Test set (172 imgs, half pothole-free, all top-down): **mAP50 0.39** overall — target 0.6 **not met**.
  Per source: **top-down drone 0.53**, **UAPD 0.76**, UAV-PDD2023 0.03 (its labels are 10-30 px specks).
  Without UAV-PDD: 0.55. Previous model (run 2) scored 0.04 on the drone frames.
- Full 4K frames (22 test frames, 78 potholes): @0.15 82 % found / 6.5 false pins per frame ·
  **@0.30 77 % / 3.0** · @0.40 69 % / 1.5.
- Speed: ~18-20 s per 4K frame on this CPU; ~1 s for 512 px.
- Weaknesses: lane markings, roadside grass, wide dark cracks flagged; label definitions differ across datasets.

## 7. Lessons learned (don't repeat these mistakes)

- **Public pothole labels are inconsistent**: HighRPD "pits" include tree shadows / stains (dropped);
  UAV-PDD2023 ships 3 flipped copies of every image (lr_/tb_/r180_ → dedupe) and labels specks;
  the GitHub drone set's class 0 = raveled patches (kept), class 1 = streaks (dropped); the
  Roboflow "Drone" set is actually street-level photos (train-only).
- Split by **source photo / video stretch**, never randomly (tiles and overlapping frames leak).
- Judge thresholds on **full frames**, not tiles: false pins add up across ~28 tiles per 4K frame.
- Test sets need many pothole-free images, or false alarms go unmeasured.
- Next real accuracy gain = **the project's own drone photos**, labelled with a consistent definition.

## 8. Environment quirks

- The venv was moved from D: → `.venv\Scripts\pip.exe` / `gdown.exe` wrappers are broken; use
  `.venv\Scripts\python.exe -m pip ...` / `-m gdown`.
- Git Bash and PowerShell are both available; paths contain spaces (quote them).
- Zenodo/Mendeley throttle per connection → use `ml/scripts/download_file.py --connections 16`.
- GitHub API is often rate-limited here; `raw.githubusercontent.com` downloads work.
- The PC may sleep during long jobs (a 4K detection once "took" hours); rerun if timings look absurd.
- Browser-pane screenshots can time out when the app window is hidden; use page text / JS checks.

## 9. Next step

Start **Sprint 2 (capture & geolocation)**:
1. Ask the user: Pi camera model, flight altitude, flight controller (and GPS / telemetry source).
2. Propose a plan as small serial steps (S2-01 …) and add it to PLAN.md; get approval.
3. Likely first steps: a geolocation module that turns `predict.py` detections + frame telemetry into
   lat/lon and size in metres, validated on `data/synthetic_geotagged_potholes` (ground-truth CSVs),
   then multi-frame confirmation / 2 m de-duplication (the synthetic set should end with 45 potholes).

---

## Prompt to start a new session

Copy this into a new Claude Code session opened in `E:\ProjectsE\Potholes detecting drone`:

```
I'm continuing my pothole-detecting drone project. Before doing anything, read HANDOFF.md
(project context, decisions, where everything is, lessons, next step), then PLAN.md (full step-by-step
history and results) and README.md (how to run). Also check models/pothole_v1/README.txt.

Summarise back to me in a few lines: what the project is, what is finished (Sprint 1 detector +
demo portal), the final model and its results, and what comes next. Then continue with the next
step from HANDOFF.md section 9 (Sprint 2: capture & geolocation). Ask me the open questions first
(Pi camera model, flight altitude, flight controller) and propose the Sprint 2 plan as small serial
steps with checks, like Sprint 1. Ask before any download, and keep PLAN.md's progress updated
after every step.
```
