Backup of pothole detector run 2 (Sprint 1, step S1-21e) - saved 2026-09-25
Model: YOLO11s, imgsz 1024, 150 epochs, dataset pothole_v1 (UAV-PDD2023 + UAPD + HighRPD)
Test (284 imgs, half pothole-free): mAP50 0.32, mAP50-95 0.14
Has-pothole @ conf 0.03: precision 0.54, recall 0.92
Per source: UAPD mAP50 0.71, HighRPD 0.32, UAV-PDD2023 0.06
Use: python ml/predict.py --model models/backup/pothole_v1_s1024_2026-09-25/weights/best.pt --source <images> --out <dir>
