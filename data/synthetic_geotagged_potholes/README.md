# Synthetic geotagged drone pothole dataset

**Purpose: testing your pipeline end to end, not training the final model.** The images are
procedurally generated. A model trained only on them will not work on real roads.

Simulated flight: downward camera, 8.6 m altitude, 82° HFOV (≈15 m footprint, 2.34 cm/px),
5 m/s northward, frames sampled at 4 FPS, starting at 12.904826, 80.227419.
240 frames (640×640), 45 potholes, each visible in several consecutive frames.

| File | Use it to test |
|---|---|
| `images/`, `labels/`, `data.yaml` | YOLO training/inference smoke test (class 0 = pothole) |
| `telemetry.csv` (10 Hz, noisy GPS) | frame ↔ GPS timestamp sync with your rolling buffer |
| `frames.csv` | true position of each frame centre |
| `potholes_ground_truth.csv` | true pothole lat/lon: measure geolocation error |
| `detections_ground_truth.csv` | which pothole each box is: test the 2 m duplicate filter (should end with 45 records) |

Image orientation: top of image = north (heading 0°). Pixel → metres: 0.0234 m/px from frame centre.

Regenerate with different settings: `python generate.py --length-m 500 --potholes 80 --seed 1`
