# UAV-PDD2023 → pothole-only YOLO

## 1. Download
- Zenodo (official): https://zenodo.org/record/8429208
- Mirror: https://github.com/Arthasyue/Crack-detection-public-dataset-collection (Google Drive link #9)

Extract the archive into a folder, e.g. `raw_uav_pdd2023/`.

## 2. Convert
```bash
pip install pyyaml
python prepare.py --src raw_uav_pdd2023 --out uav_pdd2023_pothole_yolo
```
The script prints the dataset's class list and which id it picked as "pothole".
**Check it.** If it is wrong or ambiguous, force it: `--pothole-id N`.

Options: `--neg-ratio 0.15` (pothole-free images kept as negatives),
`--val-frac 0.2` (used only if the source has no train/val folders).
Segmentation polygons are converted to bounding boxes automatically.

## 3. Train
```bash
yolo detect train data=uav_pdd2023_pothole_yolo/data.yaml model=yolov8n.pt imgsz=640 epochs=100
```

## Notes
Low-altitude UAV images of roads in Tianjin, China, 6 distress classes. Closest to an ~8–9 m downward drone view; use as your main training set.
