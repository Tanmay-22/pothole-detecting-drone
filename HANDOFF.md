# HANDOFF — Pothole-detecting drone (read this first in a new session)

Last updated: **2026-09-29**. Status: **Sprint 1 complete + demo portal complete + dashboard demo website complete
(W-01…W-10). Sprint 2 (real capture + geolocation hardening) not started.**
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
- Don't commit to git unless asked (Sprint 1 + demo portal pushed as commit 676e25e to github.com/Tanmay-22/pothole-detecting-drone; the dashboard work is not committed yet).

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
| `data/synthetic_geotagged_potholes/synthetic_geotagged_potholes/` | Synthetic 240 frames + telemetry.csv, frames.csv, potholes_ground_truth.csv, detections_ground_truth.csv → **use for Sprint 2 geolocation tests** (8.6 m altitude, 82° HFOV, 2.34 cm/px, top of image = north) |
| `models/pothole_v1/` | **Final model** (read-only): weights/best.pt, threshold.txt (0.30), eval/metrics*.json, README.txt, SHA256SUMS.txt |
| `models/pothole_v2_s1024/` | Colab run 3 (source of the final model) |
| `models/backup/pothole_v1_s1024_2026-09-25/` | Backup of run 2 (read-only) |
| `models/pothole_v1_s`, `models/pothole_v1_s1024` | Runs 1 and 2 (superseded) |
| `app/backend/main.py` | Demo portal API (+ serves built UI) on :8000; `model_card.json` |
| `app/frontend/` | React 19 + Vite 8 UI (`src/App.jsx`, `src/detections.js` = JS port of threshold/merge/match) |
| `app/make_samples.py` | Builds `app/samples/` (9 test-split samples + ground truth) |
| `dist/` | Colab upload zips (pothole_v2.zip, ml_code.zip; old pothole_v1.zip) |
| `.claude/launch.json` | Preview server configs "demo-portal" (:8000) and "dashboard" (:8001) |
| `geo/` | **Geolocation (dashboard demo, first cut of Sprint 2):** `flight.py` (flight folder format + loader), `geolocate.py` (telemetry smoothing/interpolation, box → lat/lon + metres), `cluster.py` (merge across frames, severity), `process_flight.py` (flight → detector → potholes.json/.geojson/.csv + crops), `eval_synthetic.py` (accuracy vs synthetic ground truth), `config.yaml` (camera HFOV, smoothing, conf 0.30, box_scale 0.75, merge 1.0 m / ≥ 2 frames, severity 0.6 / 0.9 m) |
| `web/backend/main.py` | Dashboard API on :8001 (flights, potholes, crops, zip upload → background job, progress) + serves built UI |
| `web/frontend/` | Dashboard UI (React 19 + Vite 8 + Leaflet 1.9.4 / react-leaflet 5): `Planner.jsx`, `ResultsMap.jsx`, `Flights.jsx`, `BaseMap.jsx` |
| `web/tools/make_synthetic_flight.py` | Builds `web/data/flights/synthetic_001` + `dist/synthetic_flight_001.zip` (upload demo) |
| `web/data/flights/<id>/` | Flight folders (gitignored): images/, frames.csv, telemetry.csv, flight.json, result/ |

Gitignored (regenerable, large): data/raw, data/prepared, data/merged, runs/, dist/, *.pt, *.zip,
app/samples, app/.cache, app/frontend/node_modules, app/frontend/dist, web/data, web/frontend/node_modules,
web/frontend/dist.

## 5. How to run

```bash
# Python: always use the venv's python (Python 3.14, ultralytics 8.4.160, torch 2.14 CPU)
.venv\Scripts\python.exe ml/predict.py --model models/pothole_v1/weights/best.pt --source <img|folder> --out runs/predict/<name>
# Demo portal → http://127.0.0.1:8000 (or preview_start "demo-portal")
.venv\Scripts\python.exe app/backend/main.py
# Rebuild portal UI after editing app/frontend/src
cd app/frontend && npm run build
# Dashboard demo → http://127.0.0.1:8001 (or preview_start "dashboard"); rebuild UI: cd web/frontend && npm run build
.venv\Scripts\python.exe web/backend/main.py
# Process a flight folder / evaluate on the synthetic ground truth
.venv\Scripts\python.exe geo/process_flight.py web/data/flights/synthetic_001
.venv\Scripts\python.exe geo/eval_synthetic.py flight      # also: boxes, clusters
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

**Dashboard demo website done (2026-09-29)** — plan and results in PLAN.md ("Demo website", W-01…W-10).
Synthetic flight: 37 of 45 potholes found at 0.30, 0 false, location error median 0.10 m.
Nothing is committed yet (ask the user before committing / pushing).

Open points to raise with the user:
- `detector.box_scale 0.75` in geo/config.yaml was calibrated on the same synthetic flight (circular);
  re-measure on real photos. Without it sizes are +45 % and nearly all potholes are "high".
- Threshold 0.30 finds 37/45 on synthetic, 0.15 finds 42/45 (no false ones there, but real roads differ).
- Still undecided: camera model, flight controller (→ mission export GeoJSON / QGC .plan in the planner).

Possible next work (ask the user): Sprint 2 proper (much of S2-01…S2-08 now exists in `geo/`; remaining:
Pi capture skeleton with a simulate mode, tilt/lens handling, real telemetry reader once the flight
controller is chosen), Sprint 3 backend (SQLite instead of JSON files), mission export, or Sprint 5 field
photos.

---

## Prompt to start a new session

Copy this into a new Claude Code session opened in `E:\ProjectsE\Potholes detecting drone`:

```
I'm continuing my pothole-detecting drone project. Before doing anything, read HANDOFF.md
(project context, decisions, where everything is, lessons, next step), then PLAN.md (full step-by-step
history and results) and README.md (how to run). Also check models/pothole_v1/README.txt.

Summarise back to me in a few lines: what the project is, what is finished (Sprint 1 detector +
demo portal, dashboard demo website), the final model and its results, and the open points in
HANDOFF section 9. Then ask me what I want to work on next. Plan any new work as small serial steps
with checks, like Sprint 1. Ask before any download, and keep PLAN.md's progress updated after
every step.
```
