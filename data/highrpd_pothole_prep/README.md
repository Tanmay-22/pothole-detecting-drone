# HighRPD → pothole-only YOLO

## 1. Download
- Official (Mendeley Data, CC BY 4.0, HighRPD.zip ≈1.6 GB): https://data.mendeley.com/datasets/sywswj7djj/1
- Google Drive copy: https://drive.google.com/file/d/1oD2doLlF59ArYidE-WRVASKrEdhQ0rCz/view
- (Old mirror github.com/Arthasyue/Crack-detection-public-dataset-collection returned 404 in Sept 2026.)
- Paper: https://doi.org/10.1016/j.dib.2025.111377

Extract the archive into a folder, e.g. `raw_highrpd/`.

## 2. Convert
```bash
pip install pyyaml
python prepare.py --src raw_highrpd --out highrpd_pothole_yolo
```
The script prints the dataset's class list and which id it picked as "pothole".
**Check it.** If it is wrong or ambiguous, force it: `--pothole-id N`.

Options: `--neg-ratio 0.15` (pothole-free images kept as negatives),
`--val-frac 0.2` (used only if the source has no train/val folders).
Segmentation polygons are converted to bounding boxes automatically.

## 3. Train
```bash
yolo detect train data=highrpd_pothole_yolo/data.yaml model=yolov8n.pt imgsz=640 epochs=100
```

## Notes
Already YOLO format, 640x640. Potholes are labelled **pit**. Shot from ~50 m, so potholes are small; use as supplementary data. Only road-segment GPS is published (≈37.768°N, 112.772°E, Shanxi), not per-pothole coordinates.
