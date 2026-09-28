"""
Demo portal backend (Sprint 1): runs the pothole detector on uploaded or sample images.

    .venv\\Scripts\\python.exe app/backend/main.py          # http://127.0.0.1:8000

GET  /api/model     model card: threshold, test scores, full-frame trade-off, weaknesses
GET  /api/samples   test-split sample images (built by app/make_samples.py) with ground truth
POST /api/detect    form field `file` (image upload) or `sample_id`
                    -> all detections with confidence >= 0.05, BEFORE fragment merging.
                    The page filters by its threshold slider and merges, which gives exactly what
                    ml/predict.py returns at that threshold (see detect_raw in ml/predict.py).
The built React UI (app/frontend/dist) is served at /.
"""
import hashlib, json, sys, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from predict import detect_raw, tile_origins  # noqa: E402

MODEL_DIR = ROOT / "models" / "pothole_v1"
SAMPLES = ROOT / "app" / "samples"
CACHE = ROOT / "app" / ".cache"
FRONTEND = ROOT / "app" / "frontend" / "dist"
RAW_CONF = 0.05          # lowest threshold the slider can reach
MAX_UPLOAD = 60 * 1024 * 1024

state = {"model": None, "weights_sha": None}
model_lock = threading.Lock()


def load_model():
    from ultralytics import YOLO
    weights = MODEL_DIR / "weights" / "best.pt"
    state["weights_sha"] = hashlib.sha1(weights.read_bytes()).hexdigest()[:12]
    state["model"] = YOLO(str(weights))
    state["imgsz"] = (state["model"].ckpt or {}).get("train_args", {}).get("imgsz", 640)


def run_detection(data: bytes):
    """Detections for an encoded image, cached on disk by image + model hash."""
    key = f"{hashlib.sha1(data).hexdigest()}_{state['weights_sha']}_{RAW_CONF}"
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return {**json.loads(cached.read_text()), "cached": True}
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise HTTPException(400, "Not a readable image (use JPG or PNG)")
    h, w = im.shape[:2]
    a = SimpleNamespace(conf=RAW_CONF, imgsz=state["imgsz"], tile=640, tile_above=1280, overlap=0.2, device="cpu")
    tiles = len(tile_origins(w, h, a.tile, a.overlap)) if max(w, h) > a.tile_above else 1
    with model_lock:
        t0 = time.time()
        dets = detect_raw(state["model"], im, a)
        seconds = round(time.time() - t0, 1)
    result = {"width": w, "height": h, "tiles": tiles, "seconds": seconds,
              "detections": [{"box": [round(v, 1) for v in d[:4]], "confidence": round(d[4], 4)}
                             for d in sorted(dets, key=lambda d: -d[4])]}
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result))
    return {**result, "cached": False}


def samples():
    f = SAMPLES / "samples.json"
    return json.loads(f.read_text()) if f.exists() else []


def warm_cache():
    """Pre-compute the samples in the background so the demo is instant."""
    for s in samples():
        try:
            run_detection((SAMPLES / f"{s['id']}.jpg").read_bytes())
        except Exception as e:  # never block the server on a bad sample
            print(f"warm-up failed for {s['id']}: {e}")
    print("Sample detections cached.")


@asynccontextmanager
async def lifespan(_app):
    load_model()
    threading.Thread(target=warm_cache, daemon=True).start()
    yield


app = FastAPI(title="Pothole detector demo", lifespan=lifespan)


@app.get("/api/model")
def model_info():
    card = json.loads((Path(__file__).with_name("model_card.json")).read_text())
    ev = MODEL_DIR / "eval"
    per_source = {}
    for name, label in (("luis", "Top-down drone frames"), ("uapd", "UAPD"),
                        ("uavpdd", "UAV-PDD2023 (speck labels)"), ("no_uavpdd", "All except UAV-PDD2023")):
        m = json.loads((ev / f"metrics_test_{name}.json").read_text())
        per_source[label] = m["box"]["mAP50"]
    full = json.loads((ev / "metrics.json").read_text())
    return {**card, "threshold": float((MODEL_DIR / "threshold.txt").read_text().strip()),
            "raw_conf": RAW_CONF, "imgsz": state["imgsz"], "weights_sha": state["weights_sha"],
            "test": {"images": full["images"], "positives": full["positives"], "negatives": full["negatives"],
                     "mAP50": full["box"]["mAP50"], "mAP50_95": full["box"]["mAP50_95"], "per_source_mAP50": per_source}}


@app.get("/api/samples")
def list_samples():
    return [{**s, "url": f"/samples/{s['id']}.jpg"} for s in samples()]


@app.post("/api/detect")
def detect(file: UploadFile | None = File(None), sample_id: str | None = Form(None)):
    if file is not None:
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Image larger than 60 MB")
    elif sample_id:
        s = next((s for s in samples() if s["id"] == sample_id), None)
        if s is None:
            raise HTTPException(404, f"Unknown sample {sample_id}")
        data = (SAMPLES / f"{sample_id}.jpg").read_bytes()
    else:
        raise HTTPException(400, "Send an image file or a sample_id")
    return run_detection(data)


if SAMPLES.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
