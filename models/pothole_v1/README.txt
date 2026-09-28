Pothole detector - Sprint 1 final model (model v1), chosen 2026-09-27 (step S1-22)
Source run: models/pothole_v2_s1024 (Colab run 3). Backup of the previous model: models/backup/pothole_v1_s1024_2026-09-25

Model   : YOLO11s, trained at imgsz 1024 on 640x640 tiles, single class "pothole"
Dataset : data/merged/pothole_v2 = UAPD + UAV-PDD2023 + GitHub top-down drone set (luisaugustos) ;
          Roboflow street-level set used for training only ; HighRPD excluded (noisy labels)
Threshold (threshold.txt): 0.30  - user's choice after full-frame test (was 0.15 from tile-level stats)

Test results (172 images, half pothole-free, all top-down):
  full test          mAP50 0.39  mAP50-95 0.14 | has-pothole @0.15: precision 0.75 recall 0.83
  GitHub drone       mAP50 0.53   (previous model: 0.04)
  UAPD               mAP50 0.76
  UAV-PDD2023        mAP50 0.03   (labels are tiny 10-30 px specks; model misses them)
  without UAV-PDD    mAP50 0.55 | has-pothole @0.18: precision 0.86 recall 0.90 (post-hoc exclusion)
Full 4K frames (22 test frames, 78 potholes, with fragment merging in predict.py):
  @0.15 found 82% per frame, 6.5 false pins/frame | @0.30 77%, 3.0/frame | @0.40 69%, 1.5/frame
Known weaknesses: lane markings, roadside grass and wide dark cracks are sometimes flagged.
Sprint 1 target mAP50 >= 0.6 not met overall; recall target met. Next gain: own drone photos.

Use:  python ml/predict.py --model models/pothole_v1/weights/best.pt --source <image|folder> --out <dir>
