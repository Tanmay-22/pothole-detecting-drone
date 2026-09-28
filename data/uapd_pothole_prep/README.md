# UAPD → pothole-only YOLO

## 1. Download
- GitHub: https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset (follow the download link in its README)

Extract the archive into a folder, e.g. `raw_uapd/`.

## 2. Convert
```bash
pip install pyyaml
python prepare.py --src raw_uapd --out uapd_pothole_yolo
```
The script prints the dataset's class list and which id it picked as "pothole".
**Check it.** If it is wrong or ambiguous, force it: `--pothole-id N`.

Options: `--neg-ratio 0.15` (pothole-free images kept as negatives),
`--val-frac 0.2` (used only if the source has no train/val folders).
Segmentation polygons are converted to bounding boxes automatically.

## 3. Train
```bash
yolo detect train data=uapd_pothole_yolo/data.yaml model=yolov8n.pt imgsz=640 epochs=100
```

## Notes
3,151 UAV asphalt images, 6 classes. Check the label format after download; the script handles YOLO boxes, YOLO polygons and VOC XML.
