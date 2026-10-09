"""
flight.py
Flight folder format (what the Pi writes after a flight, and what the dashboard accepts as an upload):

    <flight>/
        images/          one photo per frame (jpg / png)
        frames.csv       frame, image, timestamp          (timestamp = unix seconds, same clock as telemetry)
        telemetry.csv    timestamp, lat, lon, alt_m, heading_deg   (any rate; extra columns are ignored)
        flight.json      optional: {"name", "camera": {"hfov_deg"}, ...}; missing values come from geo/config.yaml

alt_m is height above the road (not above sea level). heading_deg: 0 = north, clockwise.
The camera is assumed to point straight down with the top of the image towards the drone's heading.
"""
import csv, json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "geo" / "config.yaml"
IMG_EXT = {".jpg", ".jpeg", ".png"}
TELEMETRY_COLS = ["timestamp", "lat", "lon", "alt_m", "heading_deg"]


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_flight(folder, config=None):
    """Return {folder, meta, camera, frames: [{frame, image, path, timestamp}], telemetry: [{...floats}]}.
    Raises ValueError with a readable message if the folder does not follow the format."""
    folder = Path(folder)
    config = config or load_config()
    for part in ("images", "frames.csv", "telemetry.csv"):
        if not (folder / part).exists():
            raise ValueError(f"missing {part}")

    frames = []
    for r in _read_csv(folder / "frames.csv"):
        missing = {"frame", "image", "timestamp"} - r.keys()
        if missing:
            raise ValueError(f"frames.csv lacks columns {sorted(missing)}")
        p = folder / "images" / r["image"]
        if not p.exists():
            raise ValueError(f"frames.csv lists {r['image']} but it is not in images/")
        frames.append({"frame": int(r["frame"]), "image": r["image"], "path": p, "timestamp": float(r["timestamp"])})
    frames.sort(key=lambda f: f["timestamp"])

    rows = _read_csv(folder / "telemetry.csv")
    if not rows or set(TELEMETRY_COLS) - rows[0].keys():
        raise ValueError(f"telemetry.csv needs columns {TELEMETRY_COLS}")
    telemetry = sorted(({k: float(r[k]) for k in TELEMETRY_COLS} for r in rows), key=lambda t: t["timestamp"])
    if len(telemetry) < 2:
        raise ValueError("telemetry.csv needs at least 2 rows")

    meta = json.loads((folder / "flight.json").read_text()) if (folder / "flight.json").exists() else {}
    camera = {**config["camera"], **meta.get("camera", {})}
    return {"folder": folder, "meta": meta, "camera": camera, "frames": frames, "telemetry": telemetry}


if __name__ == "__main__":
    import sys
    f = load_flight(sys.argv[1])
    t = f["telemetry"]
    print(f"{len(f['frames'])} frames, {len(t)} telemetry rows, "
          f"{t[-1]['timestamp'] - t[0]['timestamp']:.1f} s, camera {f['camera']}")
