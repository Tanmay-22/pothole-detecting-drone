"""
cluster.py
Merge per-frame detections of the same pothole into one record, and grade it.

A drone sees each pothole in several consecutive frames. Detections are taken in time order; each joins
the nearest pothole whose average position is within `radius_m` and that has no detection from the same
frame yet (two potholes seen in one frame stay apart even when close); otherwise it starts a new pothole.
Potholes seen in fewer than `min_frames` frames are dropped (most false alarms appear in one frame only).
"""
import math, statistics as st


def cluster(dets, radius_m, min_frames):
    """dets: [{frame, timestamp, east, north, confidence, diameter_m, w_m, h_m, edge, ...}].
    Returns clusters (lists of dets) seen in >= min_frames distinct frames."""
    groups = []   # {"e", "n", "dets", "frames"}
    for d in sorted(dets, key=lambda d: (d["timestamp"], -d["confidence"])):
        best, best_dist = None, radius_m
        for g in groups:
            if d["frame"] in g["frames"]:
                continue
            dist = math.hypot(d["east"] - g["e"], d["north"] - g["n"])
            if dist <= best_dist:
                best, best_dist = g, dist
        if best is None:
            best = {"e": 0.0, "n": 0.0, "dets": [], "frames": set()}
            groups.append(best)
        best["dets"].append(d)
        best["frames"].add(d["frame"])
        k = len(best["dets"])
        best["e"] += (d["east"] - best["e"]) / k
        best["n"] += (d["north"] - best["n"]) / k
    return [g["dets"] for g in groups if len(g["frames"]) >= min_frames]


def severity(diameter_m, cfg):
    s = cfg["severity"]
    return "high" if diameter_m >= s["high_m"] else "medium" if diameter_m >= s["medium_m"] else "low"


def summarise(dets, cfg, to_ll):
    """One pothole record from its detections. Size uses boxes not cut by the frame edge when there are any."""
    e = st.fmean(d["east"] for d in dets)
    n = st.fmean(d["north"] for d in dets)
    whole = [d for d in dets if not d["edge"]] or dets
    w, h = st.median(d["w_m"] for d in whole), st.median(d["h_m"] for d in whole)
    diameter = st.median(d["diameter_m"] for d in whole)
    best = max(whole, key=lambda d: d["confidence"])
    conf = max(d["confidence"] for d in dets)
    lat, lon = to_ll(e, n)
    spread = max(math.hypot(d["east"] - e, d["north"] - n) for d in dets)
    return {
        "lat": round(lat, 7), "lon": round(lon, 7),
        "diameter_m": round(diameter, 2), "width_m": round(w, 2), "length_m": round(h, 2),
        "area_m2": round(math.pi / 4 * w * h, 2),                      # ellipse inside the box
        "severity": severity(diameter, cfg),
        "unverified": conf < cfg["severity"]["unverified_below_conf"],
        "confidence": round(conf, 3),
        "mean_confidence": round(st.fmean(d["confidence"] for d in dets), 3),
        "frames_seen": len({d["frame"] for d in dets}),
        "first_seen": min(d["timestamp"] for d in dets), "last_seen": max(d["timestamp"] for d in dets),
        "position_spread_m": round(spread, 2),
        "best_frame": best["frame"], "best_image": best.get("image"),
        "best_box_px": [round(v, 1) for v in best["box"][:4]],
    }
