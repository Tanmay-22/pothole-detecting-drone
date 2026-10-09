"""
geolocate.py
Turn a box in a frame into a ground position (lat/lon) and a size in metres.

Model: camera points straight down, no lens distortion, image top = drone heading.
    ground width of a frame = 2 * altitude * tan(HFOV / 2)  ->  metres per pixel (GSD) = that / image width
    offset from the frame centre (right, forward) in metres, rotated by the heading into (east, north).
Position / altitude / heading at the frame's timestamp are interpolated from telemetry, after an optional
moving-average smoothing (GPS and compass samples are noisy).
"""
import math

import numpy as np

EARTH_M_PER_DEG = 111_320.0


class LocalFrame:
    """Flat east/north metres around a reference point (fine for a few km)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = EARTH_M_PER_DEG * math.cos(math.radians(lat0))

    def to_en(self, lat, lon):
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * EARTH_M_PER_DEG

    def to_ll(self, east, north):
        return self.lat0 + north / EARTH_M_PER_DEG, self.lon0 + east / self.kx


class Telemetry:
    """Interpolates position (local metres), altitude and heading at any timestamp."""

    def __init__(self, rows, smooth_s=0.0):
        self.t = np.array([r["timestamp"] for r in rows])
        self.local = LocalFrame(rows[0]["lat"], rows[0]["lon"])
        en = np.array([self.local.to_en(r["lat"], r["lon"]) for r in rows])
        hd = np.radians([r["heading_deg"] for r in rows])
        cols = {"e": en[:, 0], "n": en[:, 1], "alt": np.array([r["alt_m"] for r in rows]),
                "hs": np.sin(hd), "hc": np.cos(hd)}   # heading as a unit vector: averages across 359/1 deg
        if smooth_s > 0:
            dt = np.median(np.diff(self.t))
            k = max(1, int(round(smooth_s / dt)))
            if k > 1:
                cols = {name: _moving_average(v, k) for name, v in cols.items()}
        self.cols = cols

    def at(self, ts):
        """{east, north, lat, lon, alt_m, heading_deg} at time ts (clamped to the telemetry range)."""
        v = {name: float(np.interp(ts, self.t, col)) for name, col in self.cols.items()}
        lat, lon = self.local.to_ll(v["e"], v["n"])
        return {"east": v["e"], "north": v["n"], "lat": lat, "lon": lon, "alt_m": v["alt"],
                "heading_deg": math.degrees(math.atan2(v["hs"], v["hc"])) % 360}

    def covers(self, ts, slack=1.0):
        return self.t[0] - slack <= ts <= self.t[-1] + slack


def _moving_average(x, k):
    """Centred moving average that keeps the ends (shrinking window)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - k // 2), np.minimum(len(x), i + k // 2 + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def gsd_m(alt_m, hfov_deg, img_w):
    return 2 * alt_m * math.tan(math.radians(hfov_deg) / 2) / img_w


def box_to_ground(box, img_w, img_h, pose, hfov_deg):
    """box = (x0, y0, x1, y1) pixels; pose from Telemetry.at(). Returns east/north (local m), lat/lon,
    width/height/diameter in metres and the GSD used."""
    x0, y0, x1, y1 = box[:4]
    g = gsd_m(pose["alt_m"], hfov_deg, img_w)
    right = ((x0 + x1) / 2 - img_w / 2) * g
    fwd = (img_h / 2 - (y0 + y1) / 2) * g
    th = math.radians(pose["heading_deg"])
    east = pose["east"] + right * math.cos(th) + fwd * math.sin(th)
    north = pose["north"] - right * math.sin(th) + fwd * math.cos(th)
    w_m, h_m = (x1 - x0) * g, (y1 - y0) * g
    return {"east": east, "north": north, "w_m": w_m, "h_m": h_m, "diameter_m": max(w_m, h_m), "gsd_m": g}


def touches_edge(box, img_w, img_h, margin):
    x0, y0, x1, y1 = box[:4]
    return x0 <= margin or y0 <= margin or x1 >= img_w - margin or y1 >= img_h - margin



def locate_flight(flight, boxes_by_frame, image_size, config):
    """Geolocate every box of a flight. boxes_by_frame: {frame: [(x0, y0, x1, y1, conf), ...]};
    image_size: {frame: (w, h)}. Returns (detections with east/north/lat/lon/size, Telemetry)."""
    tel = Telemetry(flight["telemetry"], config["telemetry"]["smooth_s"])
    hfov, edge = flight["camera"]["hfov_deg"], config["merge"]["edge_px"]
    dets = []
    for f in flight["frames"]:
        boxes = boxes_by_frame.get(f["frame"], [])
        if not boxes:
            continue
        pose = tel.at(f["timestamp"])
        w, h = image_size[f["frame"]]
        for b in boxes:
            g = box_to_ground(b, w, h, pose, hfov)
            lat, lon = tel.local.to_ll(g["east"], g["north"])
            dets.append({**g, "lat": lat, "lon": lon, "frame": f["frame"], "image": f["image"],
                         "timestamp": f["timestamp"], "box": tuple(b[:4]), "confidence": float(b[4]),
                         "edge": touches_edge(b, w, h, edge), "gps_ok": tel.covers(f["timestamp"])})
    return dets, tel
