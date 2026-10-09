"""
Demo website backend (FastAPI): flights, potholes and flight uploads. Serves the built React UI.

    .venv\\Scripts\\python.exe web/backend/main.py        # http://127.0.0.1:8001

Flights live in web/data/flights/<id>/ (geo/flight.py format + result/ from geo/process_flight.py).
An uploaded zip is extracted into a new flight folder and processed by one background worker
(detector -> geolocation -> merge); GET /api/jobs/{id} reports progress.
"""
import json, queue, re, shutil, sys, threading, time, traceback, uuid, zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "geo"))
from flight import load_config, load_flight  # noqa: E402
from process_flight import process  # noqa: E402

FLIGHTS = ROOT / "web/data/flights"
UI = ROOT / "web/frontend/dist"
MAX_UPLOAD = 2 * 1024 ** 3                     # 2 GB unpacked
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# share of the progress bar per stage (detection dominates the time)
STAGES = {"queued": (0, 0), "loading": (0, 0.02), "detecting": (0.02, 0.92), "locating": (0.92, 0.95),
          "crops": (0.95, 1.0), "done": (1, 1)}

jobs, work = {}, queue.Queue()


def read_meta(fid):
    p = FLIGHTS / fid / "flight.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_meta(fid, **kv):
    p = FLIGHTS / fid / "flight.json"
    meta = {**read_meta(fid), **kv}
    p.write_text(json.dumps(meta, indent=2))
    return meta


def flight_dir(fid):
    if not ID_RE.match(fid) or not (FLIGHTS / fid).is_dir():
        raise HTTPException(404, "unknown flight")
    return FLIGHTS / fid


def worker():
    while True:
        jid = work.get()
        job = jobs[jid]
        fid = job["flight_id"]

        def progress(stage, frac, msg):
            lo, hi = STAGES.get(stage, (0, 1))
            job.update(status="running", stage=stage, message=msg, progress=round(lo + (hi - lo) * frac, 3))

        try:
            write_meta(fid, status="processing")
            process(FLIGHTS / fid, progress=progress)
            job.update(status="done", stage="done", progress=1.0, finished=time.time())
        except Exception as e:  # report any failure to the UI and keep the worker alive
            traceback.print_exc()
            write_meta(fid, status="failed", error=str(e))
            job.update(status="failed", error=str(e), finished=time.time())


def enqueue(fid):
    jid = uuid.uuid4().hex[:12]
    jobs[jid] = {"id": jid, "flight_id": fid, "status": "queued", "stage": "queued", "progress": 0.0,
                 "message": "waiting for the worker", "created": time.time()}
    write_meta(fid, status="queued", job_id=jid)
    work.put(jid)
    return jobs[jid]


@asynccontextmanager
async def lifespan(_):
    FLIGHTS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    for d in sorted(FLIGHTS.iterdir()):     # resume flights interrupted by a restart
        if d.is_dir() and read_meta(d.name).get("status") in ("queued", "processing"):
            enqueue(d.name)
    yield


app = FastAPI(title="Pothole drone dashboard (demo)", lifespan=lifespan)


@app.get("/api/config")
def config():
    c = load_config()
    return {"camera": c["camera"], "severity": c["severity"], "merge": c["merge"],
            "detector": {"conf": c["detector"]["conf"], "box_scale": c["detector"].get("box_scale", 1.0)}}


@app.get("/api/flights")
def flights():
    out = []
    for d in FLIGHTS.iterdir():
        if d.is_dir() and (d / "flight.json").exists():
            m = read_meta(d.name)
            out.append({"id": d.name, "name": m.get("name", d.name), "status": m.get("status", "new"),
                        "uploaded_at": m.get("uploaded_at"), "summary": m.get("summary"),
                        "error": m.get("error"), "job_id": m.get("job_id")})
    # newest upload first; flights added without an upload (the demo flight) by flight time
    return sorted(out, key=lambda f: -(f["uploaded_at"] or (f["summary"] or {}).get("start") or 0))


@app.get("/api/flights/{fid}")
def flight(fid: str):
    d = flight_dir(fid)
    meta = read_meta(fid)
    res = d / "result"
    potholes = json.loads((res / "potholes.json").read_text()) if (res / "potholes.json").exists() else []
    track = json.loads((res / "track.json").read_text()) if (res / "track.json").exists() else []
    return {"id": fid, **meta, "potholes": potholes, "track": track}


@app.get("/api/flights/{fid}/crops/{name}")
def crop(fid: str, name: str):
    p = flight_dir(fid) / "result" / "crops" / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "no such crop")
    return FileResponse(p)


@app.get("/api/flights/{fid}/images/{name}")
def image(fid: str, name: str):
    p = flight_dir(fid) / "images" / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "no such image")
    return FileResponse(p)


@app.get("/api/flights/{fid}/download/{kind}")
def download(fid: str, kind: str):
    files = {"geojson": "potholes.geojson", "csv": "potholes.csv", "json": "potholes.json"}
    if kind not in files or not (flight_dir(fid) / "result" / files[kind]).exists():
        raise HTTPException(404, "not available")
    return FileResponse(FLIGHTS / fid / "result" / files[kind], filename=f"{fid}_{files[kind]}")


def extract(upload_path, dest):
    """Extract a flight zip safely. Accepts the flight files at the top level or inside one folder."""
    with zipfile.ZipFile(upload_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and "__MACOSX" not in i.filename]
        if sum(i.file_size for i in infos) > MAX_UPLOAD:
            raise ValueError("zip is larger than 2 GB unpacked")
        names = [i.filename for i in infos]
        prefix = ""
        if "frames.csv" not in names:
            tops = {n.split("/")[0] for n in names}
            if len(tops) == 1 and f"{next(iter(tops))}/frames.csv" in names:
                prefix = next(iter(tops)) + "/"
        dest_r = dest.resolve()
        for i in infos:
            if not i.filename.startswith(prefix):
                continue
            target = (dest / i.filename[len(prefix):]).resolve()
            if dest_r not in target.parents:
                raise ValueError(f"unsafe path in zip: {i.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


@app.post("/api/flights")
def upload(file: UploadFile = File(...), name: str = Form("")):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "upload a .zip with images/, frames.csv and telemetry.csv")
    fid = time.strftime("flight_%Y%m%d_%H%M%S")
    while (FLIGHTS / fid).exists():
        fid += "_b"
    d = FLIGHTS / fid
    d.mkdir(parents=True)
    tmp = d / "_upload.zip"
    try:
        with open(tmp, "wb") as out:
            shutil.copyfileobj(file.file, out)
        extract(tmp, d)
        tmp.unlink()
        fl = load_flight(d)                       # validates the format before queueing
    except (ValueError, zipfile.BadZipFile) as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"not a valid flight: {e}")
    shutil.rmtree(d / "result", ignore_errors=True)          # never trust results inside an upload
    meta = {k: v for k, v in fl["meta"].items() if k not in ("status", "summary", "processed_at", "job_id", "error")}
    meta.update(name=name.strip() or meta.get("name") or fid, uploaded_at=time.time(), original_file=file.filename)
    (d / "flight.json").write_text(json.dumps(meta, indent=2))
    job = enqueue(fid)
    return {"flight_id": fid, "job": job, "frames": len(fl["frames"])}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[jid]


if UI.exists():
    app.mount("/", StaticFiles(directory=UI, html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
