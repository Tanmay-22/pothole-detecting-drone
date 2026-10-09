"""
make_synthetic_flight.py
Turn the synthetic geotagged set into a flight folder (geo/flight.py format) and an upload zip.
Only what a real drone would record is copied: images, frame timestamps, noisy telemetry.
The true positions (frames.csv lat/lon, potholes_ground_truth.csv) stay in the source folder for evaluation.

    python web/tools/make_synthetic_flight.py            # -> web/data/flights/synthetic_001 + dist/synthetic_flight_001.zip
"""
import argparse, csv, json, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/synthetic_geotagged_potholes/synthetic_geotagged_potholes"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(SRC))
ap.add_argument("--id", default="synthetic_001")
ap.add_argument("--out", default=str(ROOT / "web/data/flights"))
ap.add_argument("--zip", default=str(ROOT / "dist/synthetic_flight_001.zip"))
a = ap.parse_args()

src, out = Path(a.src), Path(a.out) / a.id
if out.exists():
    shutil.rmtree(out)
(out / "images").mkdir(parents=True)

with open(src / "frames.csv", newline="") as fh:
    rows = list(csv.DictReader(fh))
with open(out / "frames.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "image", "timestamp"])
    for r in rows:
        name = f"frame_{int(r['frame']):05d}.jpg"
        shutil.copy2(src / "images" / r["split"] / name, out / "images" / name)
        w.writerow([r["frame"], name, r["timestamp"]])
shutil.copy2(src / "telemetry.csv", out / "telemetry.csv")
meta = {"name": "Synthetic road survey (Chennai demo)", "camera": {"hfov_deg": 82.0},
        "source": "synthetic", "notes": "Generated test flight: 300 m road, 8.6 m altitude, 5 m/s north."}
(out / "flight.json").write_text(json.dumps(meta, indent=2))

Path(a.zip).parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_STORED) as z:   # jpgs don't compress
    for p in sorted(out.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(out).as_posix())
print(f"{len(rows)} frames -> {out}\nzip -> {a.zip} ({Path(a.zip).stat().st_size / 1e6:.1f} MB)")
