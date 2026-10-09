"""
process_flight.py
Flight folder (geo/flight.py format) -> detector -> geolocation -> merged potholes.

    python geo/process_flight.py web/data/flights/synthetic_001 [--conf 0.30] [--redetect]

Writes <flight>/result/:
    detections.json   every box >= RAW_CONF per frame, before fragment merging (cached: re-running with another
                      --conf or merge setting skips the detector unless --redetect)
    potholes.json / .csv / .geojson   one record per confirmed pothole
    track.json        smoothed flight path [[lat, lon], ...]
    crops/<id>.jpg    photo crop of each pothole from its best frame
and updates flight.json with status + summary.
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flight import ROOT, load_config, load_flight  # noqa: E402
from geolocate import locate_flight  # noqa: E402
from cluster import cluster, summarise  # noqa: E402

sys.path.insert(0, str(ROOT / "ml"))
RAW_CONF = 0.10
_model = {}


def get_model(path):
    if path not in _model:
        from ultralytics import YOLO
        m = YOLO(str(ROOT / path))
        _model.clear()
        _model[path] = (m, (m.ckpt or {}).get("train_args", {}).get("imgsz", 640))
    return _model[path]


def run_detector(flight, config, progress):
    """{frame: [(x0, y0, x1, y1, conf)]} at RAW_CONF (after cross-tile NMS, before fragment merging)."""
    from predict import detect_raw
    model, imgsz = get_model(config["detector"]["model"])
    a = SimpleNamespace(conf=RAW_CONF, imgsz=imgsz, tile=640, tile_above=1280, overlap=0.2, device="cpu")
    out, sizes = {}, {}
    frames = flight["frames"]
    t0 = time.time()
    for i, f in enumerate(frames):
        im = cv2.imread(str(f["path"]))
        if im is None:
            raise ValueError(f"cannot read image {f['image']}")
        sizes[f["frame"]] = (im.shape[1], im.shape[0])
        out[f["frame"]] = [tuple(round(v, 2) for v in d[:4]) + (round(d[4], 4),) for d in detect_raw(model, im, a)]
        left = (time.time() - t0) / (i + 1) * (len(frames) - i - 1)
        progress("detecting", (i + 1) / len(frames), f"frame {i + 1} of {len(frames)}, ~{left:.0f} s left")
    return out, sizes


def save_crop(flight, rec, path, pad=0.6, min_px=120):
    """Crop around the pothole's best box (with context) and draw the box."""
    im = cv2.imread(str(flight["folder"] / "images" / rec["best_image"]))
    x0, y0, x1, y1 = rec["best_box_px"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max((x1 - x0), (y1 - y0), min_px / (1 + 2 * pad)) * (0.5 + pad)
    H, W = im.shape[:2]
    cx0, cy0 = int(max(0, cx - half)), int(max(0, cy - half))
    cx1, cy1 = int(min(W, cx + half)), int(min(H, cy + half))
    crop = im[cy0:cy1, cx0:cx1].copy()
    cv2.rectangle(crop, (int(x0 - cx0), int(y0 - cy0)), (int(x1 - cx0), int(y1 - cy0)), (40, 40, 230), 2)
    if crop.shape[1] < 320:
        s = 320 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])


def process(folder, conf=None, redetect=False, progress=lambda stage, frac, msg: None):
    from predict import merge_fragments
    folder = Path(folder)
    config = load_config()
    conf = config["detector"]["conf"] if conf is None else conf
    progress("loading", 0.0, "reading flight folder")
    flight = load_flight(folder, config)
    res = folder / "result"
    (res / "crops").mkdir(parents=True, exist_ok=True)

    cache = res / "detections.json"
    if cache.exists() and not redetect:
        c = json.loads(cache.read_text())
        raw = {int(k): [tuple(b) for b in v] for k, v in c["boxes"].items()}
        sizes = {int(k): tuple(v) for k, v in c["sizes"].items()}
    else:
        raw, sizes = run_detector(flight, config, progress)
        cache.write_text(json.dumps({"raw_conf": RAW_CONF, "model": config["detector"]["model"],
                                     "boxes": raw, "sizes": sizes}))

    progress("locating", 0.0, "geolocating detections")
    boxes = {fr: merge_fragments([b for b in bs if b[4] >= conf]) for fr, bs in raw.items()}
    dets, tel = locate_flight(flight, boxes, sizes, config)
    dets = [d for d in dets if d["gps_ok"]]
    scale = config["detector"].get("box_scale", 1.0)
    for d in dets:
        for k in ("w_m", "h_m", "diameter_m"):
            d[k] *= scale
    groups = cluster(dets, config["merge"]["radius_m"], config["merge"]["min_frames"])
    recs = sorted((summarise(g, config, tel.local.to_ll) for g in groups), key=lambda r: r["first_seen"])
    for old in (res / "crops").glob("*.jpg"):
        old.unlink()
    for i, r in enumerate(recs, 1):
        r["id"] = f"P{i:03d}"
        save_crop(flight, r, res / "crops" / f"{r['id']}.jpg")
        r["crop"] = f"crops/{r['id']}.jpg"
        progress("crops", i / len(recs), f"pothole {i} of {len(recs)}")
    recs = [{"id": r.pop("id"), **r} for r in recs]

    (res / "potholes.json").write_text(json.dumps(recs, indent=1))
    with open(res / "potholes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()) if recs else ["id"])
        w.writeheader()
        w.writerows(recs)
    (res / "potholes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
         "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")}} for r in recs]}, indent=1))

    t = flight["telemetry"]
    t0, t1 = t[0]["timestamp"], t[-1]["timestamp"]
    n = max(2, int((t1 - t0) / 0.5))
    track = [tel.at(t0 + (t1 - t0) * k / (n - 1)) for k in range(n)]
    dist = sum(math.hypot(b["east"] - a["east"], b["north"] - a["north"]) for a, b in zip(track, track[1:]))
    (res / "track.json").write_text(json.dumps([[round(p["lat"], 7), round(p["lon"], 7)] for p in track]))

    meta_path = folder / "flight.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sev = {s: sum(r["severity"] == s for r in recs) for s in ("low", "medium", "high")}
    meta.update({
        "status": "processed", "processed_at": time.time(),
        "summary": {"frames": len(flight["frames"]), "start": t0, "end": t1, "duration_s": round(t1 - t0, 1),
                    "distance_m": round(dist, 1), "mean_alt_m": round(sum(p["alt_m"] for p in track) / len(track), 1),
                    "potholes": len(recs), "by_severity": sev, "unverified": sum(r["unverified"] for r in recs),
                    "detections": len(dets), "detector_conf": conf, "box_scale": scale,
                    "merge_radius_m": config["merge"]["radius_m"], "min_frames": config["merge"]["min_frames"],
                    "camera_hfov_deg": flight["camera"]["hfov_deg"]},
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    progress("done", 1.0, f"{len(recs)} potholes")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--conf", type=float, default=None, help="default: geo/config.yaml detector.conf")
    ap.add_argument("--redetect", action="store_true", help="run the detector even if result/detections.json exists")
    a = ap.parse_args()
    last = [0.0]

    def show(stage, frac, msg):
        if stage != "detecting" or frac >= 1 or time.time() - last[0] > 10:
            last[0] = time.time()
            print(f"[{stage}] {frac:.0%} {msg}", flush=True)

    m = process(a.folder, a.conf, a.redetect, show)
    print(json.dumps(m["summary"], indent=1))
