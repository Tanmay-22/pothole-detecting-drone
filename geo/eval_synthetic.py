"""
eval_synthetic.py
Measure geolocation accuracy on the synthetic flight, whose true pothole positions are known.

    python geo/eval_synthetic.py boxes      # ground-truth boxes -> per-detection location / size error
    python geo/eval_synthetic.py clusters   # ground-truth boxes -> merged potholes (radius sweep)
    python geo/eval_synthetic.py flight     # real detector (cached result/detections.json) -> threshold sweep,
                                            # + map of true vs found potholes (result/eval_map.png)
"""
import argparse, csv, math, statistics as st, sys
from pathlib import Path

from flight import ROOT, load_config, load_flight
from cluster import cluster, summarise
from geolocate import LocalFrame, Telemetry, box_to_ground, locate_flight, touches_edge

SRC = ROOT / "data/synthetic_geotagged_potholes/synthetic_geotagged_potholes"
FLIGHT = ROOT / "web/data/flights/synthetic_001"
IMG = 640


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def truth():
    """True potholes {id: {lat, lon, diameter_m}}, true frame poses {frame: pose}, ground-truth boxes."""
    holes = {int(r["id"]): {"lat": float(r["lat"]), "lon": float(r["lon"]), "diameter_m": float(r["diameter_m"])}
             for r in read_csv(SRC / "potholes_ground_truth.csv")}
    frames = {int(r["frame"]): r for r in read_csv(SRC / "frames.csv")}
    boxes = []
    for r in read_csv(SRC / "detections_ground_truth.csv"):
        x, y, w, h = (float(r[k]) * IMG for k in ("x", "y", "w", "h"))
        boxes.append({"frame": int(r["frame"]), "pothole_id": int(r["pothole_id"]),
                      "box": (x - w / 2, y - h / 2, x + w / 2, y + h / 2)})
    return holes, frames, boxes


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summary(name, errs, derr):
    print(f"{name:34s} n={len(errs):3d}  location error m: median {st.median(errs):.2f}  "
          f"p90 {pct(errs, 0.9):.2f}  max {max(errs):.2f} | diameter error: median {st.median(derr):+.0%}  "
          f"median abs {st.median(abs(d) for d in derr):.0%}")


def cmd_boxes(_):
    cfg = load_config()
    fl = load_flight(FLIGHT, cfg)
    hfov = fl["camera"]["hfov_deg"]
    holes, frames, boxes = truth()
    ts = {f["frame"]: f["timestamp"] for f in fl["frames"]}
    local = LocalFrame(*(lambda r: (float(r["lat"]), float(r["lon"])))(frames[0]))
    hole_en = {i: local.to_en(h["lat"], h["lon"]) for i, h in holes.items()}

    def true_pose(fr):
        e, n = local.to_en(float(frames[fr]["lat"]), float(frames[fr]["lon"]))
        return {"east": e, "north": n, "alt_m": float(frames[fr]["alt_m"]), "heading_deg": 0.0}

    tel_raw = Telemetry(fl["telemetry"], 0.0)
    tel_smooth = Telemetry(fl["telemetry"], cfg["telemetry"]["smooth_s"])

    def telem_pose(tel):
        def pose(fr):
            p = tel.at(ts[fr])
            e, n = local.to_en(p["lat"], p["lon"])
            return {**p, "east": e, "north": n}
        return pose

    for name, pose_fn in [("true frame positions", true_pose),
                          ("telemetry, raw", telem_pose(tel_raw)),
                          (f"telemetry, smoothed {cfg['telemetry']['smooth_s']} s", telem_pose(tel_smooth))]:
        for inner in (False, True):
            errs, derr = [], []
            for b in boxes:
                edge = touches_edge(b["box"], IMG, IMG, cfg["merge"]["edge_px"])
                if inner and edge:
                    continue
                g = box_to_ground(b["box"], IMG, IMG, pose_fn(b["frame"]), hfov)
                he, hn = hole_en[b["pothole_id"]]
                errs.append(math.hypot(g["east"] - he, g["north"] - hn))
                if not edge:
                    d = holes[b["pothole_id"]]["diameter_m"]
                    derr.append((g["diameter_m"] - d) / d)
            summary(f"{name}{' (no edge boxes)' if inner else ''}", errs, derr)


def match(records, holes, max_m=1.0):
    """Greedy nearest matching of found potholes to true ones (each true pothole used once).
    Returns (pairs [(record, hole_id, dist)], unmatched records, missed hole ids)."""
    local = LocalFrame(holes[0]["lat"], holes[0]["lon"])
    hole_en = {hi: local.to_en(h["lat"], h["lon"]) for hi, h in holes.items()}
    cand = []
    for ri, r in enumerate(records):
        e, n = local.to_en(r["lat"], r["lon"])
        cand += [(math.hypot(e - he, n - hn), ri, hi) for hi, (he, hn) in hole_en.items()]
    cand.sort()
    used_r, used_h, pairs = set(), set(), []
    for dist, ri, hi in cand:
        if dist > max_m:
            break
        if ri in used_r or hi in used_h:
            continue
        used_r.add(ri); used_h.add(hi); pairs.append((records[ri], hi, dist))
    return pairs, [r for i, r in enumerate(records) if i not in used_r], sorted(set(holes) - used_h)


def report(records, holes):
    pairs, extra, missed = match(records, holes)
    loc = [d for _, _, d in pairs]
    size = [(r["diameter_m"] - holes[h]["diameter_m"]) / holes[h]["diameter_m"] for r, h, _ in pairs]
    print(f"  {len(records)} potholes found: {len(pairs)} match a true one (<= 1 m), {len(extra)} false, "
          f"{len(missed)} of {len(holes)} true missed")
    if pairs:
        print(f"  location error m: median {st.median(loc):.2f}  p90 {pct(loc, 0.9):.2f}  max {max(loc):.2f} | "
              f"diameter error median {st.median(size):+.0%}, median abs {st.median(abs(x) for x in size):.0%}")
    return pairs, extra, missed


def cmd_clusters(_):
    cfg = load_config()
    fl = load_flight(FLIGHT, cfg)
    holes, _, boxes = truth()
    by_frame, ids = {}, {}
    for b in boxes:
        by_frame.setdefault(b["frame"], []).append((*b["box"], 1.0))
        ids.setdefault(b["frame"], []).append(b["pothole_id"])
    dets, tel = locate_flight(fl, by_frame, {f["frame"]: (IMG, IMG) for f in fl["frames"]}, cfg)
    order = [i for f in fl["frames"] for i in ids.get(f["frame"], [])]   # locate_flight keeps this order
    for d, i in zip(dets, order, strict=True):
        d["pothole_id"] = i
    for radius in (0.6, 0.8, 1.0, 1.2, 1.5):
        groups = cluster(dets, radius, cfg["merge"]["min_frames"])
        mixed = sum(len({d["pothole_id"] for d in g}) > 1 for g in groups)
        split = len(groups) - len({min(d["pothole_id"] for d in g) for g in groups})
        print(f"radius {radius} m: {len(groups)} potholes, {mixed} mix two true potholes, "
              f"{split} extra pieces of an already-found pothole")
        report([summarise(g, cfg, tel.local.to_ll) for g in groups], holes)


def box_iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def cmd_flight(a):
    import json
    sys.path.insert(0, str(ROOT / "ml"))
    from predict import merge_fragments
    cfg = load_config()
    fl = load_flight(FLIGHT, cfg)
    holes, _, boxes = truth()
    cache = json.loads((FLIGHT / "result/detections.json").read_text())
    raw = {int(k): [tuple(b) for b in v] for k, v in cache["boxes"].items()}
    sizes = {int(k): tuple(v) for k, v in cache["sizes"].items()}
    gt_by_frame = {}
    for b in boxes:
        gt_by_frame.setdefault(b["frame"], []).append(b["box"])
    results = {}
    for conf in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        per_frame = {fr: merge_fragments([b for b in bs if b[4] >= conf]) for fr, bs in raw.items()}
        n_box = sum(map(len, per_frame.values()))
        hit = sum(any(box_iou(d, g) >= 0.3 for g in gt_by_frame.get(fr, [])) for fr, ds in per_frame.items() for d in ds)
        ratio = [((d[2] - d[0]) * (d[3] - d[1]) / ((g[2] - g[0]) * (g[3] - g[1]))) ** 0.5
                 for fr, ds in per_frame.items() for d in ds for g in gt_by_frame.get(fr, []) if box_iou(d, g) >= 0.3]
        dets, tel = locate_flight(fl, per_frame, sizes, cfg)
        for d in dets:
            for k in ("w_m", "h_m", "diameter_m"):
                d[k] *= a.box_scale
        one = cluster(dets, cfg["merge"]["radius_m"], 1)
        recs = [summarise(g, cfg, tel.local.to_ll) for g in cluster(dets, cfg["merge"]["radius_m"], cfg["merge"]["min_frames"])]
        print(f"conf {conf}: {n_box} boxes in frames ({n_box - hit} not on a labelled pothole); "
              f"detector box / label box side ratio median {st.median(ratio):.2f}; "
              f"{len(one)} groups before the >= {cfg['merge']['min_frames']}-frames rule")
        results[conf] = (recs, report(recs, holes))
    if a.plot:
        plot(results[cfg["detector"]["conf"]], holes, fl, cfg["detector"]["conf"])


def plot(res, holes, fl, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs, (pairs, extra, missed) = res
    local = LocalFrame(holes[0]["lat"], holes[0]["lon"])
    fig, ax = plt.subplots(figsize=(4, 10))
    tr = [local.to_en(t["lat"], t["lon"]) for t in fl["telemetry"]]
    ax.plot([p[0] for p in tr], [p[1] for p in tr], color="0.8", lw=0.8, label="GPS track")
    he = {i: local.to_en(h["lat"], h["lon"]) for i, h in holes.items()}
    ax.scatter(*zip(*he.values()), s=60, facecolors="none", edgecolors="tab:green", label="true pothole")
    if missed:
        ax.scatter(*zip(*(he[i] for i in missed)), s=60, marker="x", color="tab:orange", label="missed")
    found = [local.to_en(r["lat"], r["lon"]) for r, _, _ in pairs]
    if found:
        ax.scatter(*zip(*found), s=12, color="tab:blue", label="found (matched)")
    if extra:
        ax.scatter(*zip(*(local.to_en(r["lat"], r["lon"]) for r in extra)), s=20, marker="s", color="tab:red",
                   label="false pothole")
    ax.set_xlim(-12, 12); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.set_title(f"Synthetic flight, detector @ {conf}: {len(pairs)} found, {len(missed)} missed, {len(extra)} false",
                 fontsize=7.5)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    fig.tight_layout()
    out = FLIGHT / "result/eval_map.png"
    fig.savefig(out, dpi=110)
    print("map ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("boxes").set_defaults(fn=cmd_boxes)
    sub.add_parser("clusters").set_defaults(fn=cmd_clusters)
    f = sub.add_parser("flight")
    f.add_argument("--no-plot", dest="plot", action="store_false")
    f.add_argument("--box-scale", type=float, default=load_config()["detector"].get("box_scale", 1.0))
    f.set_defaults(fn=cmd_flight)
    a = ap.parse_args()
    a.fn(a)
