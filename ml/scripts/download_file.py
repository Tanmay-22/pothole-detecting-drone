"""
download_file.py
Parallel, resumable HTTP download using byte-range requests. Useful for servers that
throttle each connection (e.g. Zenodo) but allow many connections.

    python ml/scripts/download_file.py --url URL --out data/raw/x/file.zip --connections 16 [--md5 HASH | --sha256 HASH]

Progress is kept in <out>.progress, so re-running the same command resumes.
"""
import argparse, hashlib, json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 16 * 1024 * 1024
UA = {"User-Agent": "pothole-drone-downloader/1.0"}


def remote_size(url):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            sys.exit("Server does not support range requests; use a normal download instead.")
        return int(r.headers["Content-Range"].split("/")[-1])


def fetch_chunk(url, out, start, end, retries=30):
    """Download bytes [start, end]; on a dropped connection, resume from the last byte written."""
    pos = start
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "r+b") as f:
                f.seek(pos)
                while pos <= end:
                    block = r.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
                    pos += len(block)
            if pos > end:
                return start
            raise IOError(f"connection closed at {pos - start}/{end - start + 1} bytes")
        except Exception as e:
            wait = min(60, 2 ** min(attempt, 6))
            print(f"  chunk {start}: {e} (retry in {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"chunk at {start} failed after {retries} retries")


def hash_of(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Parallel resumable range download")
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--connections", type=int, default=16)
    ap.add_argument("--md5", default=None, help="expected MD5 to verify at the end")
    ap.add_argument("--sha256", default=None, help="expected SHA-256 to verify at the end")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prog_file = out.with_name(out.name + ".progress")
    size = remote_size(a.url)
    print(f"Remote size: {size / 1e6:.1f} MB", flush=True)

    done = set(json.loads(prog_file.read_text())) if prog_file.exists() and out.exists() else set()
    if not out.exists() or out.stat().st_size != size:
        with open(out, "wb") as f:
            f.truncate(size)
        done = set()

    starts = [s for s in range(0, size, CHUNK) if s not in done]
    lock, t0, got = threading.Lock(), time.time(), 0
    print(f"Chunks: {len(done)} done, {len(starts)} to fetch, {a.connections} connections", flush=True)

    with ThreadPoolExecutor(a.connections) as ex:
        futs = [ex.submit(fetch_chunk, a.url, out, s, min(s + CHUNK, size) - 1) for s in starts]
        for fut in as_completed(futs):
            s = fut.result()
            with lock:
                done.add(s)
                got += min(CHUNK, size - s)
                prog_file.write_text(json.dumps(sorted(done)))
                pct = 100 * len(done) * CHUNK / size
                rate = got / 1e6 / max(time.time() - t0, 1e-6)
                print(f"progress {min(pct, 100):.0f}%  {rate:.2f} MB/s", flush=True)

    for algo, expected in (("md5", a.md5), ("sha256", a.sha256)):
        if expected:
            actual = hash_of(out, algo)
            if actual != expected.lower():
                sys.exit(f"{algo.upper()} MISMATCH: expected {expected}, got {actual}")
            print(f"{algo.upper()} OK", flush=True)
    prog_file.unlink(missing_ok=True)
    print(f"DONE {out} ({size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
