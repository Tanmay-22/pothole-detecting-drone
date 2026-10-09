# Pothole-Detecting Drone — Project Plan (Sprint 1 in detail)

> New session? Read [HANDOFF.md](HANDOFF.md) first (current state, decisions, next step).

## Context
Full project: a drone flies a predefined path (drawn on a web dashboard), a Raspberry Pi camera
captures top-down road frames, potholes are detected + geolocated, and shown as pins on a dashboard
map; clicking a pin shows size, severity, photo crop, time/flight and confidence.

**Sprint 1 goal:** a model that, given a top-view road image, predicts whether it contains potholes
and where (bounding boxes → later become GPS pins and sizes).

Decisions made with the user:
- Output: YOLO bounding boxes, single class `pothole`; "has pothole" = any box ≥ threshold.
- Inference on the ground after flight (Pi only captures) → mid-size model (YOLO11s/m), high res.
- Training on Google Colab / Kaggle GPU (this PC: no NVIDIA GPU, Python 3.14).
- Flight controller undecided; camera/altitude asked again at Sprint 2.
- Dashboard later: FastAPI + React + Leaflet; all pin details shown only on click.

Existing state of `data/` (checked): three converter folders (`uav_pdd2023_pothole_prep/`,
`uapd_pothole_prep/`, `highrpd_pothole_prep/`, each `prepare.py` + shared `pothole_converter.py`)
but **no raw datasets downloaded**; `synthetic_geotagged_potholes/` has 240 generated frames usable
only for smoke tests (its `data.yaml` has a stale `/home/claude/...` path).

## How the steps work
Each step is small, done one after another, and self-contained: it lists what it **needs** (files
produced by earlier steps), what it **does**, what it **produces**, and a **check** that proves it
is done. A step is not started until the previous step's check passes. Steps that require the user
(downloads approval, uploading to Drive, running Colab) are marked **[USER]**.

## Progress (update this after every step)
To resume in a new session: open this file, find the first unchecked step, and continue from it.

- [x] S1-01 Project folders + .gitignore
- [x] S1-02 Python environment (`.venv`, Python 3.14, ultralytics 8.4.160, torch 2.14.0+cpu)
- [x] S1-03 Fix synthetic dataset config (`data/merged/synthetic_smoke/data.yaml`, 200 train / 40 val)
- [x] S1-04 Download UAV-PDD2023 (MD5 OK; 2,440 jpg + 2,440 VOC xml; only **195 `Pothole` boxes**;
      official splits in `ImageSets/Main` are ignored by the converter → consider honouring them in S1-12.
      Zenodo throttles ~0.1 MB/s per connection → use `ml/scripts/download_file.py --connections 16`)
- [x] S1-05 Download UAPD (`UAPD_final.zip` 203 MB via gdown; 3,151 jpg 512×512 + VOC xml;
      only **94 `Pothole` boxes**; no official split lists)
- [x] S1-06 Download HighRPD (Mendeley, SHA-256 OK; `data/raw/highrpd/HighRPD/{images,labels}`,
      11,696 jpg 640×640 + YOLO txt; class 2 = pit: **1,412 boxes in 997 images**; no class file →
      needs `--pothole-id 2`)
- [x] S1-07 Convert UAV-PDD2023 → `data/prepared/uav_pdd2023_yolo` (132 pos + 19 neg imgs, 195 boxes)
- [x] S1-08 Convert UAPD → `data/prepared/uapd_yolo` (68 pos + 10 neg imgs, 94 boxes; 1 source
      image/xml name mismatch `1177_4_3`/`1177_4_8`, crack-only, ignored)
- [x] S1-09 Convert HighRPD (`--pothole-id 2`) → `data/prepared/highrpd_yolo` (997 pos + 149 neg
      imgs, 1,412 boxes)
      **Total: 1,197 positive images, 1,701 pothole boxes, 178 negatives.**
      ⚠ Leakage: HighRPD (`DJI_<photo>_640_<r>_<c>`) and UAV-PDD2023 (`lr_<photo>_<quadrant>`) are
      tiles of larger photos; the converter's random split puts sibling tiles in train AND val →
      S1-12 must re-split **grouped by source photo**.
- [x] S1-10 Dataset inspection script (`ml/scripts/check_dataset.py`; verified on synthetic set)
- [x] S1-11 Inspect each converted dataset (`data/prepared/*/inspection/`). Boxes on real distress in
      all sets; 0 missing labels. Notes: "pothole" definition varies (UAPD/HighRPD include spalling /
      peeled patches; a few UAV-PDD boxes look like crack patches) → acceptable noise for v1.
      UAV-PDD2023 frames are 2592×1944 with median box 56 px → would shrink to ~22 px at imgsz 1024
      → S1-12 tiles large images into 640×640. Box short side medians: UAV-PDD 56, UAPD 41, HighRPD 42 px.
- [x] S1-12 Merge script (`ml/scripts/merge_datasets.py`; synthetic-only test: 240 in → 240 out,
      544 boxes preserved, leakage assert passes)
- [x] S1-13 Build merged dataset `data/merged/pothole_v1`: **train 970 (846 pos) / val 204 (180 pos) /
      test 133 (117 pos) images; 1,599 boxes**; 0 missing labels; 0 cross-split duplicates.
      ⚠ Found UAV-PDD2023 ships 3 exact flip copies of every image (`lr_`/`tb_`/`r180_` prefixes) →
      merge now dedupes flip/rotation copies (151 → 64 unique UAV-PDD images) and groups by photo
      number. HighRPD dominates (~85% of images). Some UAV-PDD "Pothole" labels are long cracks
      (source noise). Rebuild: `python ml/scripts/merge_datasets.py --inputs
      uavpdd=data/prepared/uav_pdd2023_yolo uapd=data/prepared/uapd_yolo
      highrpd=data/prepared/highrpd_yolo --out data/merged/pothole_v1`
- [x] S1-14 Training config + train script (`ml/configs/train_v1.yaml`, `ml/train.py --data ... --set k=v`).
      imgsz changed 1024 → **640**: after tiling every image is ≤640 px, so native resolution.
- [x] S1-15 Local smoke training (yolo11n, 3 epochs, CPU, 4 min → `runs/smoke/weights/best.pt`;
      pretrained weights in `models/pretrained/`)
- [x] S1-16 Evaluate script (`ml/evaluate.py`; on smoke model: mAP50 0.576 matches training log;
      writes metrics.json / threshold.txt / pr_curve.png / val plots)
- [x] S1-17 Predict script (`ml/predict.py`; tiles frames >1280 px and maps boxes back + NMS; conf
      defaults to threshold.txt; verified on synthetic val + a 2592×1944 frame)
- [x] S1-18 Colab notebook (`ml/notebooks/train_colab.ipynb`; runs write straight to Drive, resume cell)
- [x] S1-19 Package dataset for Colab (`dist/pothole_v1.zip` + `dist/ml_code.zip` → Drive `MyDrive/pothole_drone/`)
- [x] S1-20 Baseline training on Colab → `models/pothole_v1_s/`. yolo11s, 640 px. **Test mAP50 0.408,
      mAP50-95 0.151, P 0.57, R 0.39 → misses mAP50 ≥ 0.6 target.** Run stopped at epoch 79/100 while
      still improving (best = last epoch). Image-level result (recall 0.98 @ thr 0.01, TN 1/16) is
      meaningless: test set had only 16 negatives. HighRPD pits are 10–20 px specks (50 m altitude);
      faint ones missed.
- [x] S1-21 Improved retrain (user chose this over the yolo11m comparison), as sub-steps:
  - [x] S1-21a Re-convert all 3 datasets with `--neg-ratio 5` (UAV-PDD 792, UAPD 408, HighRPD 5,982 imgs;
        pothole boxes unchanged). **Project moved to `E:\ProjectsE\Potholes detecting drone`** (D: full);
        venv pip.exe wrapper still points at D: → use `.venv\Scripts\python.exe -m pip`.
  - [x] S1-21b Merge (pos/neg photos split separately; tiles cropped lazily): **train 1,074 (827 pos /
        247 neg), val 342 (171/171), test 284 (142/142); 1,594 boxes**; 0 cross-split duplicates;
        negatives include cracked road (hard negatives).
  - [x] S1-21c Config → imgsz 1024, 150 epochs, patience 40, batch 16; evaluate/predict default imgsz =
        the model's training imgsz; notebook updated (run name `pothole_v1_s1024`, IMGSZ cell)
  - [x] S1-21d Re-package zips (`dist/pothole_v1.zip` 248 MB, 1,074/342/284 imgs) — user uploaded
  - [x] S1-21e Colab run → `models/pothole_v1_s1024/` (150/150 epochs, best ep 148, still slowly rising).
        **Test (1:1 neg): mAP50 0.32, mAP50-95 0.14; has-pothole @0.03: P 0.54 R 0.92 (FP 110/142).**
        Per source (test): **UAPD mAP50 0.71** (only 11 boxes), HighRPD 0.32, UAV-PDD 0.06 (13 boxes).
        Root cause = labels: HighRPD "pits" (85% of data) include tree-shadow blobs, peeled strips and
        faint stains; UAV-PDD "potholes" include cracks. Also HighRPD consecutive photos (1x00003 /
        1x00004) overlap the same road → possible train/test leak not caught by photo grouping.
        Candidate extra data: github.com/luisaugustos/Pothole-Recognition (568 drone imgs, no license),
        Roboflow Universe drone pothole sets (free account needed → user downloads).
  - Backup of run 2 (read-only, SHA256SUMS.txt): `models/backup/pothole_v1_s1024_2026-09-25/`
  - User chose **cleaner data + retrain** (over accepting v1 or hand-cleaning HighRPD):
  - [x] S1-21f Download GitHub drone set (luisaugustos/Pothole-Recognition): 111/111 PNGs (1.8 GB),
        95 with potholes, 453 class-0 boxes kept, 108 class-1 dropped.
        Only **111 labelled 4K (3840×2160) PNGs** exist in the repo (382 labels have no image); truly
        top-down low-altitude. 2 classes, unnamed: class 0 = raveled/shallow pothole patches (KEEP as
        pothole), class 1 = faint streaks/cracks (DROP). Files via raw.githubusercontent.com (git clone
        failed: connection drop). → `data/raw/luis_drone/{images,labels}`
  - [x] S1-21g Roboflow set downloaded by user → `data/raw/roboflow_drone1/` (MIT, 465 imgs 640×640,
        1,256 boxes, no augmentation). **Not drone imagery** — street-level phone photos, but clean
        pothole labels → use in **train only**, never val/test. User found no other top-down sets.
  - [x] S1-21h Inspect new sets. GitHub frames are a video-like sequence (median consecutive overlap
        0.77) → grouped into 18 road segments (9 with potholes), `data/raw/luis_drone/segments.json`;
        prepared as `data/prepared/luis_drone_yolo` (stems `segNN_fNNN`). Roboflow flattened to
        `data/prepared/roboflow_drone1_yolo`. 0 duplicate images across the 4 sources.
  - [x] S1-21i Merged `data/merged/pothole_v2` (no HighRPD; Roboflow train-only; boxes with aspect > 3
        dropped except in Roboflow; split_groups no longer overshoots with long video segments;
        seed 8 chosen for a balanced luis split by split sizes only). **train 830 (747 pos, 1,659 boxes),
        val 82 (41 pos, 47 boxes), test 172 (86 pos / 86 neg, 109 boxes); val/test top-down only;
        median box 82 px; 0 cross-split duplicates.** Rebuild:
        `python ml/scripts/merge_datasets.py --inputs uavpdd=data/prepared/uav_pdd2023_yolo
        uapd=data/prepared/uapd_yolo luis=data/prepared/luis_drone_yolo
        roboflow=data/prepared/roboflow_drone1_yolo --train-only roboflow --max-aspect 3
        --test-frac 0.2 --val-frac 0.15 --seed 8 --out data/merged/pothole_v2`
  - [x] S1-21j Colab run 3 → `models/pothole_v2_s1024/` (yolo11s, 1024 px; converged: best val mAP50
        0.505 @ epoch 78, run ended at epoch 103). **Full v2 test: mAP50 0.39, mAP50-95 0.14; has-pothole
        @0.05 P 0.64 R 0.94.** Per source: **GitHub drone 0.53** (run 2: 0.04), **UAPD 0.76**,
        UAV-PDD 0.03 (its labels are 10-30 px specks, some on curbs; model instead flags wide dark
        cracks = a real weakness). Test without UAV-PDD (post-hoc exclusion, reported alongside):
        **mAP50 0.55; has-pothole @0.18 P 0.86 R 0.90.** Val (41 pos) too small to pick a threshold
        (collapses to 0.01). Full-test sweep: @0.10 P 0.69 R 0.86 · @0.15 P 0.75 R 0.83 · @0.18 P 0.80 R 0.80.
- [x] S1-22 Evaluate + choose final model. **User accepted run 3** (over a yolo11m run or dropping
      UAV-PDD) → installed read-only as **`models/pothole_v1/`** (weights/best.pt, threshold.txt,
      eval/metrics*.json per source, README.txt, SHA256SUMS.txt). Run 2 on the unseen luis test subset
      scored mAP50 0.04 vs run 3's 0.53 → v1 data did not transfer to low-altitude drone views.
      Target mAP50 ≥ 0.6 not met overall (0.39; 0.55 without UAV-PDD); tile-level recall target met.
- [x] S1-23 Final inference check + README. README commands verified as written (predict uses 0.30
      from threshold.txt; smoke train OK; local evaluate reproduces Colab exactly: mAP50 0.3917).
      Note: `evaluate.py` still *suggests* a recall-first threshold (0.05); the shipped one is 0.30.
      Full 4K frames (22 test-only luis frames, 78 potholes) showed many more false pins than tile
      stats: 0.15 → 11.6 false pins/frame, about half being duplicate fragments across overlapping
      tiles → **`predict.py` now merges boxes that mostly overlap** (intersection ≥ 50% of the smaller
      box). After merging: @0.15 82% found / 6.5 false pins per frame · **@0.30 77% / 3.0** · @0.40
      69% / 1.5. **User set threshold 0.30** (was 0.15). Model also finds synthetic potholes (124 of
      139 at 0.15). CPU speed ≈ 18 s per 4K frame. `README.md` written (setup, predict, pipeline,
      train/eval, results, weaknesses: lane markings, grass, wide dark cracks).
      → Sprint 2 idea: confirm a pothole only if detected in ≥ 2 consecutive frames at the same spot.

**Sprint 1 complete (2026-09-28).** Since then: demo portal (below), dashboard demo website with a
first-cut geolocation pipeline (below, W-01…W-10). Sprint 2 status and remaining work: see the
Sprint 2 section and HANDOFF.md §9.

### Sprint 1 demo portal (requested 2026-09-28)
Decisions: live local app (FastAPI + React, the future dashboard stack); main page = "Try it: upload &
detect" + a small model-info panel; audience = team / mentor (semi-technical).
Design: backend runs the model once per image at conf 0.05 and returns boxes before fragment merging;
the React page applies the threshold slider and the merge client-side, so the slider is instant and
gives exactly what `predict.py` would at that threshold. Samples come from the **test split only**,
with their ground-truth boxes (toggle) to compare. One command serves the built UI + API on :8000.
- [x] D-01 Installed fastapi 0.141.1, uvicorn 0.54.0, python-multipart (added to ml/requirements.txt)
- [x] D-02 `predict.py`: detect() = merge_fragments(detect_raw()); raw@0.05 → filter → merge is
      identical to detect() at 0.15 / 0.30 / 0.40 on frame 490
- [x] D-03 `app/make_samples.py` → `app/samples/` (9 test-only samples: drone 490/494/537/536,
      UAPD ×3, UAV-PDD ×1, synthetic ×1; 14 MB; gitignored) + `samples.json` with ground truth
- [x] D-04 FastAPI backend `app/backend/main.py` (+ `model_card.json`): GET /api/model, GET
      /api/samples, POST /api/detect (upload or sample_id; disk cache `app/.cache/`, samples warmed in
      background at startup; 400 for non-images, 404 unknown sample). Tested with curl: 512 px ≈ 1 s,
      4K ≈ 17-21 s uncached, 0.2 s cached. (A 2.5 h wait seen once = PC asleep overnight.)
      Run: `.claude/launch.json` → "demo-portal" (port 8000).
- [x] D-05 Vite + React 19 scaffold in `app/frontend/` (npm: 0 vulnerabilities); dev proxy to :8000
- [x] D-06 UI (`src/App.jsx`, `src/detections.js` = JS port of threshold + merge + IoU matching,
      `src/index.css`): sample picker + upload, box overlay (red = detection, orange = no matching
      label, green dashed = label), threshold slider + reset, verdict ("found X of Y labelled"),
      detections table, model-info panel (test mAP per source, full-frame trade-off, weaknesses)
- [x] D-07 FastAPI serves the built UI at :8000. Verified in the browser: all 9 samples, slider
      (frame 490: 0.15 → 10 boxes = predict.py, 0.30 → 4, 0.50 → 2), upload path, 0 errors.
      Samples are JPEG, so confidences differ slightly from the PNG originals (490: 4 vs 3 boxes @0.30).
- [x] D-08 README "Demo portal" section. **Demo portal done (2026-09-28).**
- Pushed to https://github.com/Tanmay-22/pothole-detecting-drone (public), commit 676e25e.

---

## Demo website — "how the final product will look" (requested 2026-09-28, done 2026-09-29)

Goal: a separate demo web app showing the final product: **plan a route → fly (synthetic flight) →
upload the flight → potholes appear as pins on a map**, details on click.

Decisions (user, 2026-09-28):
- Pages: **Route planner**, **Results map**, **Flights list + upload**. The Sprint 1 portal (`app/`) stays
  as it is; this is a **separate new app** in `web/` (FastAPI + React + Leaflet).
- Pins = **the real detector on the synthetic flight** + a simple geolocation (a small preview of
  Sprint 2), not ground truth or mock data.
- Basemap: **OpenStreetMap streets + Esri World Imagery satellite toggle** (online tiles, attribution shown).
- Upload does **real processing**: zip (images/ + frames.csv + telemetry.csv) → detector + geolocation
  in the background with progress (~4 min for the 240-frame synthetic flight on CPU).
- Storage: **JSON files**, one folder per flight (`web/data/flights/<id>/`).
- Route planner: **waypoints + stats only** (no GeoJSON / QGC export: flight controller undecided).
  Camera settings are editable, with synthetic defaults (82° HFOV, 9 m altitude).
- Still undecided (from Sprint 2 questions): camera model, flight controller; altitude **8-10 m**.
  Severity = **diameter + confidence** (size sets low / medium / high; low confidence = "unverified").

Assumptions for the geolocation preview: camera points straight down, no lens distortion; image top =
drone heading (synthetic set: north-up, heading ≈ 0°); position/heading interpolated from telemetry at
each frame's timestamp. Synthetic set is at `data/synthetic_geotagged_potholes/synthetic_geotagged_potholes/`
(Chennai, ~12.905 N 80.227 E; 45 potholes; 5 pairs closer than 2 m → merge radius < 1.34 m).

- [x] W-01 Flight folder format (`geo/flight.py` docstring + loader; `geo/config.yaml` for camera,
      smoothing, detector, merge, severity): `images/`, `frames.csv` (frame, image, timestamp),
      `telemetry.csv` (timestamp, lat, lon, alt_m, heading_deg), optional `flight.json`.
      `web/tools/make_synthetic_flight.py` → `web/data/flights/synthetic_001` (only what a drone would
      record; true positions stay in the source for evaluation) + `dist/synthetic_flight_001.zip` (38 MB).
      **Check passed:** 240 frames, 600 telemetry rows (59.9 s), zip = 240 images + 3 files.
      `web/data/` gitignored.
- [x] W-02 Geolocation `geo/geolocate.py` (Telemetry: 1 s moving-average smoothing, heading averaged
      as a unit vector, interpolation at frame time; `box_to_ground`: GSD from altitude + HFOV, offset
      rotated by heading). Eval `geo/eval_synthetic.py boxes` on the 544 ground-truth boxes:
      true frame positions → median **0.04 m** (maths correct); raw telemetry → median 0.35 m, p90 0.66,
      max 1.49; **smoothed telemetry → median 0.20 m, p90 0.42, max 0.81**. Diameter (longest box side,
      non-edge boxes) **+9-10 %** vs the nominal diameter (box includes the irregular rim). **Check passed.**
- [x] W-03 Merge + severity `geo/cluster.py`: detections in time order join the nearest pothole within
      the radius that has **no detection from the same frame yet** (keeps close pairs apart); ≥ 2 frames;
      mean position, median size of non-edge boxes, max confidence, best frame; severity by diameter
      (low < 0.6 m ≤ medium < 0.9 m ≤ high), "unverified" if best confidence < 0.45 (config).
      Eval `geo/eval_synthetic.py clusters` (ground-truth boxes, smoothed telemetry): **radius 0.8-1.5 m
      all give exactly 45 potholes, 45/45 matched, 0 mixed, 0 false; location error median 0.09 m,
      max 0.22 m** (radius 0.6 splits 7). Chosen radius **1.0 m**. Diameter +9 % median (box includes
      the rim) → severity skews up (found low/med/high 6/20/19 vs nominal 10/21/14). **Check passed.**
- [x] W-04 Pipeline `geo/process_flight.py` → `<flight>/result/`: detections.json (boxes ≥ 0.10 cached,
      so re-thresholding skips the detector), potholes.json/.csv/.geojson (ids P001…), track.json,
      crops/; flight.json gets status + summary. Synthetic_001: **240 frames in 110 s on CPU**.
      Eval `geo/eval_synthetic.py flight` (+ `result/eval_map.png`):
      | conf | found / 45 | false | missed | location median / max |
      | 0.15 | 42 | 0 | 3 | 0.10 / 0.25 m |
      | 0.20 | 41 | 0 | 4 | |
      | **0.30** | **37** | **0** | **8** | 0.10 / 0.23 m |
      | 0.40 | 30 | 0 | 15 | |
      | 0.50 | 22 | 0 | 23 | |
      Synthetic scenes are easy (≤ 8 stray boxes in 240 frames, none survive the ≥ 2-frames rule), so this
      does not measure false alarms on real roads. **Size:** detector boxes are 1.33× the label boxes →
      diameters +45 % → 35/37 "high". Added `detector.box_scale: 0.75` in config (**calibrated on this
      same synthetic flight — circular, re-measure on real photos**): diameter error +9 %, severity
      medium 18 / high 19. Kept threshold 0.30 (user's choice; 0.15 would find 42/45 here).
- [x] W-05 Backend `web/backend/main.py` (FastAPI :8001; `.claude/launch.json` "dashboard"): GET /api/config,
      /api/flights, /api/flights/{id} (meta + potholes + track), /crops/{name}, /images/{name},
      /download/{geojson|csv|json}; POST /api/flights (zip → safe extract (zip-slip check, 2 GB cap,
      files at top level or in one folder) → format validated → one background worker); GET /api/jobs/{id}.
      Interrupted jobs are re-queued at startup. **Check passed (curl):** all GETs 200, path tricks 404,
      non-zip / broken zip 400; uploading `dist/synthetic_flight_001.zip` → progress 0 → 100 % in ~100 s
      → 37 potholes (same as W-04).
- [x] W-06 Frontend scaffold `web/frontend/` (download approved with the plan): leaflet 1.9.4,
      react-leaflet 5.0.0 + React 19 / Vite 8 (npm: 0 vulnerabilities). App shell (header + 3 tabs, hash
      routes `#/plan`, `#/map?flight=…`, `#/flights`), `BaseMap` (OSM streets / Esri satellite toggle,
      scale bar, zoom 21 with native 19), `api.js`. Built UI served by FastAPI at :8001.
      **Check passed:** page loads, OSM + Esri tiles load (20/20), 0 console errors.
      Note: the synthetic road is simulated — its track does not line up with a real road on the basemap.
- [x] W-07 Results map `src/ResultsMap.jsx`: flight picker, track line, pins coloured by severity
      (dashed = unverified), hover tooltip, summary tiles, severity chips + "hide unverified", list sorted by
      severity/size, GeoJSON/CSV download; click pin or row → details (crop, diameter, W×L, area, best/mean
      confidence, photos seen, time, flight, lat/lon, spread, full photo, Google Maps link, prev/next);
      selection kept in the URL. **Check passed in the browser:** 37 pins, pin click → P018, next → P022,
      crop loads, filters 37 → 18 → 9, CSV 200, 0 console errors.
- [x] W-08 Flights page `src/Flights.jsx`: upload card (file picker or drag & drop, optional name, upload
      progress bar, format help), flights table (name, date, duration, distance, photos, potholes by
      severity, status badge, live job progress bar with ETA, "Open map"); list auto-refreshes while a
      flight is processing; newest upload first (fixed: processed uploads used to sort by flight time).
      **Check passed in the browser:** uploaded the synthetic zip through the page → "processing" with
      progress (22 %, "frame 54 of 240, ~76 s left") → processed, 37 potholes → Open map shows its 37 pins;
      an empty zip shows "not a valid flight: missing images". Test flights deleted afterwards.
- [x] W-09 Route planner `src/Planner.jsx`: click map → waypoint (H = take-off), drag to move, click to
      delete, undo / clear / "Load demo route", waypoint list with leg lengths; settings: altitude (9 m),
      speed (5 m/s), overlap (70 %), HFOV (82°), image width (640 px), shape (1:1 / 4:3 / 16:9) — defaults =
      synthetic camera; stats: length, survey time, photos, strip width, photo spacing + interval, cm/px,
      photos per spot; warnings for strip < 7 m and < 2 photos per spot; route saved in the browser.
      **Check passed:** demo route (300 m north) → hand calc 299.7 m, 59.9 s, strip 2·9·tan 41° = 15.65 m,
      spacing 4.69 m / 0.94 s, ⌈299.7/4.69⌉+1 = 65 photos, 2.44 cm/px, 3.3 photos per spot = page values;
      altitude 4 m → strip warning; click-delete 2 → 1, drag moves H (300 → 351 m), reload keeps route.
- [x] W-10 Final browser check: 1366 px desktop (380 px side panel, no horizontal scroll) and 375 px
      phone (map on top, whole page scrolls — fixed: the panel used to scroll inside ~260 px); all 3 pages
      0 horizontal overflow, no console errors except the deliberate bad-upload test. README "Dashboard
      demo" section, HANDOFF.md (status, file table, run commands, next step) and this checklist updated.
      Nothing committed — asking the user.

**Demo website complete (2026-09-29).**

---

## Sprint 2 — Capture & geolocation (proposed 2026-09-28; sprint not started as such)

History: on 2026-09-28 the user declined to start Sprint 2 as proposed. The **dashboard demo website
(W-01…W-10, 2026-09-29) then built a first cut of most of it in `geo/`**: S2-01…S2-08 below are covered
by W-01…W-04 (marked "via W-xx"). Severity = **diameter + confidence** (user's choice), as implemented
in `geo/cluster.py`. What remains is listed under S2-10; ask the user before doing any of it.

Goal: turn detections in drone frames into **one record per real pothole**: lat/lon, size in metres,
severity, confidence, frames seen, time, best photo crop — the data the dashboard will show.

Decisions (user, 2026-09-28): camera **not decided** → camera settings in a config file (synthetic:
82° HFOV); altitude **8-10 m**; flight controller **not decided** → generic telemetry CSV
(timestamp, lat, lon, alt_m, heading_deg), real MAVLink / GPS reader later; **no hardware yet** →
build and validate on `data/synthetic_geotagged_potholes` (45 potholes with true lat/lon).

Facts from the synthetic set: 240 frames at 4 FPS, 640×640, north-up, 8.6 m, GSD 2.34 cm/px;
telemetry 10 Hz with ~0.3 m GPS noise and ~1.5° heading noise; each pothole visible in 12-13 frames;
diameters 0.42-1.19 m; **5 pothole pairs are < 2 m apart (closest 1.34 m) → a flat 2 m merge
radius would wrongly join them; tune the radius on measured error instead.**
Assumptions: camera points straight down (no tilt correction yet), no lens distortion (config
placeholder for later).

- [x] S2-01 *(via W-01: `geo/flight.py`, `geo/config.yaml`, `web/data/flights/synthetic_001`)* Flight-folder format + config: define `flight/` = images/ + frames.csv (frame, timestamp)
      + telemetry.csv; `geo/config.yaml` (camera HFOV, image orientation, merge radius, min frames,
      severity thresholds). Convert the synthetic set into this format (`flights/synthetic_001/`).
      Check: loader reads 240 frames + 600 telemetry rows.
- [x] S2-02 *(via W-02: in `geo/geolocate.py`, smoothed)* Telemetry interpolation (`geo/telemetry.py`): position / altitude / heading at any
      timestamp (heading interpolated on the circle). Check: vs the true frame centres in
      frames.csv, median error ≈ GPS noise (< 0.5 m).
- [x] S2-03 *(via W-02: `box_to_ground` in `geo/geolocate.py`; true positions → median 0.04 m)* Pixel → ground projection (`geo/project.py`): box centre → lat/lon using altitude, HFOV,
      heading; box size → metres. Check with **ground-truth boxes + true frame positions**:
      median location error < 0.2 m, size error small (proves the maths).
- [x] S2-04 *(via W-02: smoothed telemetry → median 0.20 m, max 0.81 m)* Same with **noisy telemetry**: measure per-detection location error (expected ~0.3-0.5 m).
- [x] S2-05 *(via W-03: radius 1.0 m, ≥ 2 frames → 45/45 on ground-truth boxes)* Merge detections into potholes (`geo/cluster.py`): group across frames within a radius
      (chosen from S2-04 errors, < 1.34 m), average position, median size, max confidence; keep
      only potholes seen in ≥ 2 frames. Check with ground-truth boxes: **45 records**, each matching
      one true pothole.
- [x] S2-06 *(via W-03/W-04: diameter + confidence severity, crops)* Severity + record fields: diameter & area (m²), severity (low / medium / high by size,
      thresholds in config), confidence, frames seen, first/last time, best frame; save a photo crop
      per pothole. Check: fields filled for all 45.
- [x] S2-07 *(via W-04)* End-to-end CLI `geo/process_flight.py`: flight folder → run detector (predict.py
      detect_raw + merge at the model threshold) → geolocate → merge → `potholes.json`, `.csv`,
      `.geojson` + crops/. Check: runs on synthetic_001.
- [x] S2-08 *(via W-04: @0.30 37/45 found, 0 false, median 0.10 m; `result/eval_map.png`)* Evaluate on synthetic with the **real detector**: match to true potholes (≤ 1 m):
      found / missed / false potholes, location error, size error; compare thresholds; check the
      ≥ 2-frames rule removes most false pins. Plot map of true vs found (PNG).
- [ ] S2-09 Pi capture design (no hardware): `pi/capture.py` skeleton (picamera2 photos + timestamps
      + telemetry logging) with a `--simulate` mode that writes a flight folder from the synthetic
      set; document what's needed when hardware arrives. Check: simulate mode output passes S2-07.
- [ ] S2-10 Docs: README section, HANDOFF.md, this checklist; ask user about commit/push.

Remaining Sprint 2 work (not started — ask the user first): S2-09 Pi capture skeleton with a simulate
mode; camera tilt / lens-distortion handling; a real telemetry reader once the flight controller is
chosen; re-measure `detector.box_scale` (0.75, calibrated on the synthetic flight itself) on real photos.

---

## Sprint 1 steps

### S1-01 · Project folders + .gitignore
- Needs: nothing.
- Does: create `ml/`, `ml/scripts/`, `ml/configs/`, `ml/notebooks/`, `models/`, `data/raw/`,
  `data/prepared/`, `data/merged/`; write `.gitignore` (data/raw, data/prepared, data/merged,
  `*.pt`, `runs/`, `.venv/`); optional `git init`.
- Produces: folder skeleton.
- Check: folders exist; `.gitignore` present.

### S1-02 · Python environment
- Needs: S1-01.
- Does: create `.venv` (Python 3.14; fall back to 3.12 via `py` launcher if torch/ultralytics
  wheels fail); write `ml/requirements.txt` (ultralytics, torch CPU, pyyaml, pandas,
  opencv-python, matplotlib, gdown); install.
- Produces: `.venv/`, `ml/requirements.txt`.
- Check: `python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__)"`.

### S1-03 · Fix synthetic dataset config
- Needs: S1-02.
- Does: write `data/merged/synthetic_smoke/data.yaml` pointing at the existing synthetic
  images/labels (don't modify the original folder).
- Produces: a valid YAML for smoke tests.
- Check: `yolo checks` + loading the YAML with ultralytics lists 200 train / 40 val images.

### S1-04 · Download UAV-PDD2023 **[USER approves download]**
- Needs: S1-01.
- Does: download `UAV-PDD2023.zip` (2.1 GB, Zenodo record 8429208, CC-BY 4.0) and extract
  into `data/raw/uav_pdd2023/`.
- Produces: raw images + VOC XML annotations.
- Check: ~2,440 images and matching `.xml` files found.

### S1-05 · Download UAPD **[USER approves download]**
- Needs: S1-01.
- Does: download from Google Drive
  (https://drive.google.com/file/d/1yQ0GMXFwwM5qdYY_5HzJBQqqjNtWJxEc/view, link from the UAPD GitHub
  README) via `gdown` (if Drive blocks it, user downloads manually) into `data/raw/uapd/`.
- Produces: raw images + labels.
- Check: ~3,151 images with label files found.

### S1-06 · Download HighRPD **[USER approves download]**
- Needs: S1-01.
- Does: download `HighRPD.zip` (1.6 GB, CC BY 4.0) from Mendeley Data
  (https://data.mendeley.com/datasets/sywswj7djj/1) with `ml/scripts/download_file.py` into
  `data/raw/highrpd/` (the old Arthasyue mirror is 404). Classes: 0 line crack, 1 block crack, 2 pit.
- Produces: raw YOLO-format images + labels (pothole class = `pit`).
- Check: images + `.txt` labels + class list found.

### S1-07 · Convert UAV-PDD2023 → pothole-only YOLO
- Needs: S1-02, S1-04.
- Does: run existing `data/uav_pdd2023_pothole_prep/prepare.py --src data/raw/uav_pdd2023
  --out data/prepared/uav_pdd2023_yolo`; confirm the printed pothole class (force
  `--pothole-id` if wrong).
- Produces: `data/prepared/uav_pdd2023_yolo/` with `data.yaml`, `report.txt`.
- Check: `report.txt` shows >0 pothole boxes and train/val counts.

### S1-08 · Convert UAPD → pothole-only YOLO
- Needs: S1-02, S1-05. Same as S1-07 using `data/uapd_pothole_prep/prepare.py` →
  `data/prepared/uapd_yolo/`.
- Check: `report.txt` shows >0 pothole boxes.

### S1-09 · Convert HighRPD → pothole-only YOLO
- Needs: S1-02, S1-06. Same as S1-07 using `data/highrpd_pothole_prep/prepare.py` →
  `data/prepared/highrpd_yolo/`. The paper gives class 2 = pit; if the zip has no class-name file
  the converter stops with "no class list" → re-run with `--pothole-id 2`.
- Check: `report.txt` shows >0 pothole boxes (paper: 1,412 pit annotations).

### S1-10 · Dataset inspection script
- Needs: S1-02.
- Does: write `ml/scripts/check_dataset.py <data.yaml>` → per-split image/box counts, negatives
  ratio, box-size histogram (flags tiny boxes), saves ~20 preview images with boxes drawn.
- Produces: the script.
- Check: runs on `data/merged/synthetic_smoke/data.yaml` and previews show correct boxes.

### S1-11 · Inspect each converted dataset
- Needs: S1-07, S1-08, S1-09, S1-10.
- Does: run `check_dataset.py` on each prepared set; eyeball previews; note any bad labels.
- Produces: `data/prepared/<name>_yolo/inspection/` (stats + previews).
- Check: previews show boxes on real potholes for all three sets.

### S1-12 · Merge script
- Needs: S1-02.
- Does: write `ml/scripts/merge_datasets.py --inputs <dirs...> --out <dir> --test-frac 0.1`:
  copy images/labels with source prefix (`uavpdd_`, `uapd_`, `highrpd_`), **ignore the
  converter's random splits and re-split train/val/test (~75/15/10) grouped by source photo**
  (strip tile suffixes like `_640_02_05` / `_top_left`) so sibling tiles never cross splits;
  stratify so each split has positives from every source; **tile images larger than 1280 px into
  640×640 tiles (20% overlap), clipping boxes and dropping boxes <50% visible; cap pothole-free
  tiles at ~15% of positives**; write `data.yaml` with relative path.
- Produces: the script.
- Check: merging the synthetic set alone gives the same image/label counts.

### S1-13 · Build merged dataset `pothole_v1`
- Needs: S1-11, S1-12.
- Does: run merge on the three prepared sets (synthetic excluded) → `data/merged/pothole_v1/`;
  run `check_dataset.py` on it.
- Produces: `data/merged/pothole_v1/{images,labels}/{train,val,test}`, `data.yaml`, stats.
- Check: no filename clashes; test split non-empty; stats saved.

### S1-14 · Training config + train script
- Needs: S1-02.
- Does: write `ml/configs/train_v1.yaml` (model, imgsz, epochs, batch, patience, flipud=0.5,
  fliplr=0.5) and `ml/train.py --config --data --device` (thin wrapper over ultralytics train).
- Produces: config + script.
- Check: `python ml/train.py --help` works.

### S1-15 · Local smoke training (CPU)
- Needs: S1-03, S1-14.
- Does: train `yolo11n.pt`, 3 epochs, imgsz 640 on `synthetic_smoke`.
- Produces: `runs/smoke/weights/best.pt`.
- Check: training finishes and `best.pt` exists (accuracy irrelevant here).

### S1-16 · Evaluate script
- Needs: S1-02.
- Does: write `ml/evaluate.py --model --data --split test --out`: Ultralytics val metrics (mAP50,
  mAP50-95, P, R) + image-level "has pothole" precision/recall/F1/confusion matrix across
  thresholds; picks the threshold with recall ≥ 0.9 and best F1; saves `metrics.json`,
  `threshold.txt`, `pr_curve.png`.
- Produces: the script.
- Check: runs on the S1-15 smoke model (use `--split val` for synthetic) and writes all three files.

### S1-17 · Predict script (contract for later sprints)
- Needs: S1-02.
- Does: write `ml/predict.py --model --source <img|folder> --out --conf` → per image
  `{image, has_pothole, detections:[{bbox_xyxy_px, bbox_norm, confidence}]}` as
  `detections.json` + `detections.csv`, plus annotated images.
- Produces: the script.
- Check: runs on the synthetic val folder with the smoke model and writes JSON/CSV/images.

### S1-18 · Colab notebook
- Needs: S1-14, S1-16.
- Does: write `ml/notebooks/train_colab.ipynb`: mount Drive → install ultralytics → unzip
  dataset to local disk → train (config values inline) → run evaluation → copy `runs/` + `best.pt`
  back to Drive after training.
- Produces: the notebook.
- Check: notebook JSON is valid; cells reference only Drive paths the user will create.

### S1-19 · Package dataset for Colab **[USER uploads]**
- Needs: S1-13.
- Does: zip `data/merged/pothole_v1` → `pothole_v1.zip`; user uploads it and the notebook to
  Google Drive (`MyDrive/pothole_drone/`).
- Check: user confirms upload.

### S1-20 · Baseline training on Colab **[USER runs]**
- Needs: S1-18, S1-19.
- Does: run notebook with YOLO11s, imgsz 1024, ~100 epochs, patience 20.
- Produces: `best.pt`, `results.csv`, metrics on Drive.
- Check: user downloads `best.pt` + run folder into `models/pothole_v1_s/`.

### S1-21 · Comparison training on Colab **[USER runs]**
- Needs: S1-20.
- Does: same notebook with YOLO11m (or newest Ultralytics model available).
- Produces: `models/pothole_v1_m/`.
- Check: `best.pt` downloaded.

### S1-22 · Evaluate + choose final model
- Needs: S1-16, S1-20, S1-21.
- Does: run `evaluate.py` on the `test` split for both models; compare; copy the winner to
  `models/pothole_v1/` with `metrics.json` + `threshold.txt`.
- Produces: final `models/pothole_v1/`.
- Check: targets mAP50 ≥ 0.6 and image-level recall ≥ 0.9, or a written explanation of the gap
  and next action (more data / tiling / fine-tune).

### S1-23 · Final inference check + README
- Needs: S1-17, S1-22.
- Does: run `predict.py` with the final model on held-out test images and on synthetic frames;
  write `README.md` (setup, data pipeline, training, evaluation, prediction commands).
- Produces: sample outputs + README.
- Check: `has_pothole` flags and annotated images look right; README commands run as written.

---

## Later sprints (roadmap; each will get the same step breakdown when started)
- **Sprint 2 – Capture & geolocation:** Pi capture (picamera2) + telemetry sync; pixel → lat/lon
  from altitude/HFOV/heading; size in metres; merge duplicates within ~2 m; validate on synthetic
  ground-truth CSVs. (Ask: camera model, altitude, flight controller.)
- **Sprint 3 – Backend:** FastAPI + SQLite; upload a flight → detect → geolocate → store potholes.
- **Sprint 4 – Dashboard:** React + Leaflet; pins; details only on click (size, severity, photo crop,
  time/flight/confidence); draw route → waypoints → export GeoJSON + QGC `.plan`.
- **Sprint 5 – Field test:** own drone images, labelling, fine-tune model v2.

## Verification (Sprint 1 done)
All step checks S1-01 … S1-23 pass; `models/pothole_v1/` holds `best.pt`, `metrics.json`,
`threshold.txt`; `predict.py` produces correct JSON/CSV + annotated images on unseen images.
