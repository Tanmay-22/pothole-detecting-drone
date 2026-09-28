"""
Synthetic geotagged drone pothole dataset (for PIPELINE TESTING, not for final training).

Simulates a downward-facing drone flying north along a road at 8.6 m altitude,
sampled at 4 FPS, starting at 12.904826, 80.227419 (the example from the project deck).
Every pothole has a known ground-truth lat/lon, so you can measure your geolocation
error and test the 2 m duplicate filter (each pothole appears in many frames).

Outputs:
  images/{train,val}/*.jpg, labels/{train,val}/*.txt   YOLO format, class 0 = pothole
  data.yaml
  telemetry.csv            timestamp, lat, lon, alt_m, heading_deg, ground_speed_mps (10 Hz)
  frames.csv               frame, split, timestamp, lat, lon, alt_m (frame centre)
  potholes_ground_truth.csv id, lat, lon, diameter_m
  detections_ground_truth.csv frame, pothole_id, x, y, w, h (normalised)
"""
import argparse, csv, math, random
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="synthetic_geotagged_potholes")
ap.add_argument("--length-m", type=float, default=300.0)
ap.add_argument("--potholes", type=int, default=45)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

rng = np.random.default_rng(a.seed); random.seed(a.seed)
LAT0, LON0, ALT = 12.904826, 80.227419, 8.6
HFOV = math.radians(82)              # typical small-drone camera
IMG = 640
FOOT_M = 2 * ALT * math.tan(HFOV / 2)  # ground width covered by one frame
GSD = FOOT_M / IMG                     # metres per pixel
SPEED, FPS = 5.0, 4.0
ROAD_W = 7.0

def m2ll(x_east, y_north):
    return (LAT0 + y_north / 111320.0,
            LON0 + x_east / (111320.0 * math.cos(math.radians(LAT0))))

# ---------- build world canvas (x: east, rows: north goes UP) ----------
H_WORLD = int(a.length_m / GSD) + IMG
W = IMG
base = rng.normal(95, 14, (H_WORLD, W)).astype(np.float32)
base = cv2.GaussianBlur(base, (0, 0), 1.2) + rng.normal(0, 6, (H_WORLD, W))
world = np.dstack([base, base, base * 1.03])
cx = W // 2; half = int(ROAD_W / 2 / GSD)
# verges: dirt/grass
grass = np.dstack([rng.normal(60, 12, (H_WORLD, W)), rng.normal(105, 15, (H_WORLD, W)),
                   rng.normal(70, 12, (H_WORLD, W))])
mask = np.zeros((H_WORLD, W), np.float32)
mask[:, cx - half:cx + half] = 1
mask = cv2.GaussianBlur(mask, (0, 0), 6)[..., None]
world = world * mask + grass * (1 - mask)
# lane markings
dash, gap = int(3 / GSD), int(6 / GSD)
for y in range(0, H_WORLD, dash + gap):
    cv2.rectangle(world, (cx - 5, y), (cx + 5, y + dash), (215, 215, 210), -1)
for e in (cx - half + 12, cx + half - 12):
    cv2.line(world, (e, 0), (e, H_WORLD), (200, 200, 195), 7)
# random patches / stains (distractors)
for _ in range(int(a.length_m / 6)):
    y, x = rng.integers(0, H_WORLD), rng.integers(cx - half, cx + half)
    ax_, ay_ = rng.integers(20, 70, 2)
    col = float(rng.choice([60, 130]))
    ov = world.copy()
    cv2.ellipse(ov, (int(x), int(y)), (int(ax_), int(ay_)), float(rng.uniform(0, 180)), 0, 360,
                (col, col, col), -1)
    world = cv2.addWeighted(ov, 0.35, world, 0.65, 0)

# ---------- potholes ----------
potholes = []
ys = np.sort(rng.uniform(8, a.length_m - 8, a.potholes))
for i, yn in enumerate(ys):
    xe = rng.uniform(-ROAD_W / 2 + 0.6, ROAD_W / 2 - 0.6)
    d = rng.uniform(0.4, 1.2)
    r = d / 2 / GSD
    px, py = cx + xe / GSD, H_WORLD - IMG / 2 - yn / GSD
    n = 18
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = r * (1 + rng.normal(0, 0.18, n))
    pts = np.stack([px + rad * np.cos(ang), py + rad * 0.8 * np.sin(ang)], 1).astype(np.int32)
    m = np.zeros(world.shape[:2], np.uint8); cv2.fillPoly(m, [pts], 255)
    inner = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=max(1, int(r / 12)))
    m_f = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None] / 255.0
    in_f = cv2.GaussianBlur(inner, (0, 0), 3)[..., None] / 255.0
    rough = rng.normal(0, 10, world.shape[:2])[..., None]
    dark = 40 + rough + (1 - in_f) * 25
    world = world * (1 - m_f) + (dark + (rng.random() < 0.3) * np.array([0, 5, 15])) * m_f
    # rim highlight (sun from top-left)
    rim = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2) - m
    rim = np.roll(rim, (-2, -2), (0, 1))
    world[rim > 0] = np.clip(world[rim > 0] + 35, 0, 255)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min(); x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    lat, lon = m2ll(xe, yn)
    potholes.append(dict(id=i, lat=lat, lon=lon, d=d, box=(x0, y0, x1, y1)))
world = np.clip(world, 0, 255).astype(np.uint8)

# ---------- fly & render ----------
out = Path(a.out)
for s in ("train", "val"):
    (out / "images" / s).mkdir(parents=True, exist_ok=True)
    (out / "labels" / s).mkdir(parents=True, exist_ok=True)

t0 = 1_758_000_000.0
n_frames = int(a.length_m / SPEED * FPS)
frames, dets = [], []
for f in range(n_frames):
    t = f / FPS
    yn = SPEED * t
    jx = rng.normal(0, 0.25)                      # small lateral drift (m)
    top = int(H_WORLD - IMG - yn / GSD)
    left = int(np.clip(jx / GSD, -40, 40))
    crop = np.roll(world, -left, 1)[top:top + IMG, :]
    img = crop.astype(np.float32)
    img = img * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)          # exposure
    img += rng.normal(0, 4, img.shape)                                   # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        k = int(rng.integers(3, 7)); img = cv2.blur(img, (k, 1))         # motion blur
    split = "val" if (f // 40) % 5 == 4 else "train"  # block split limits leakage
    name = f"frame_{f:05d}"
    lines = []
    for p in potholes:
        x0, y0, x1, y1 = p["box"]
        x0, x1 = x0 - left, x1 - left
        y0, y1 = y0 - top, y1 - top
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, IMG), min(y1, IMG)
        if cx1 - cx0 < 6 or cy1 - cy0 < 6:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.4 * (x1 - x0) * (y1 - y0):
            continue  # mostly out of frame
        bx, by = (cx0 + cx1) / 2 / IMG, (cy0 + cy1) / 2 / IMG
        bw, bh = (cx1 - cx0) / IMG, (cy1 - cy0) / IMG
        lines.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
        dets.append([f, p["id"], round(bx, 6), round(by, 6), round(bw, 6), round(bh, 6)])
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    lat, lon = m2ll(jx, yn)
    frames.append([f, split, f"{t0 + t:.3f}", f"{lat:.7f}", f"{lon:.7f}", ALT])

with open(out / "frames.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "split", "timestamp", "lat", "lon", "alt_m"]); w.writerows(frames)
with open(out / "telemetry.csv", "w", newline="") as fh:   # 10 Hz GPS, slightly noisy
    w = csv.writer(fh); w.writerow(["timestamp", "lat", "lon", "alt_m", "heading_deg", "ground_speed_mps"])
    for k in range(int(a.length_m / SPEED * 10)):
        t = k / 10; lat, lon = m2ll(rng.normal(0, 0.3), SPEED * t + rng.normal(0, 0.3))
        w.writerow([f"{t0 + t:.3f}", f"{lat:.7f}", f"{lon:.7f}", round(ALT + rng.normal(0, 0.1), 2),
                    round(rng.normal(0, 1.5) % 360, 1), round(SPEED + rng.normal(0, 0.2), 2)])
with open(out / "potholes_ground_truth.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["id", "lat", "lon", "diameter_m"])
    for p in potholes:
        w.writerow([p["id"], f"{p['lat']:.7f}", f"{p['lon']:.7f}", round(p["d"], 2)])
with open(out / "detections_ground_truth.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "pothole_id", "x", "y", "w", "h"]); w.writerows(dets)
(out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                               "nc: 1\nnames: ['pothole']\n")
print(f"{n_frames} frames, {len(potholes)} potholes, {len(dets)} boxes, GSD={GSD*100:.2f} cm/px, "
      f"footprint={FOOT_M:.1f} m -> {out}")
