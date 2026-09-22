"""Fetch the pinned public encoder by verified byte ranges and LFS digest."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import threading
import time

import requests

from evotac.config import ROOT


REVISION = "e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd"
SHA256 = "28097c91a1a1d051f65266fc7fdd84420dc184da9dcfc2006aa40af144cf15bf"
SIZE = 165482707
URL = f"https://huggingface.co/datasets/byml/UniVTAC/resolve/{REVISION}/checkpoints/encoder.pth"


def main():
    root = ROOT/"checkpoints/univtac_release"
    root.mkdir(parents=True, exist_ok=True)
    destination = root/"encoder.pth"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Existing encoder digest differs from pinned release")
        return
    listing = requests.get(f"https://huggingface.co/api/datasets/byml/UniVTAC/tree/{REVISION}/checkpoints", timeout=30)
    listing.raise_for_status()
    source = next(r for r in listing.json() if r["path"] == "checkpoints/encoder.pth")
    if source["size"] != SIZE or source["lfs"]["oid"] != SHA256:
        raise ValueError("Pinned metadata differs from official LFS record")
    partial = root/"encoder.pth.ranged.partial"
    progress = root/"encoder.ranges.json"
    chunk = 4*1024*1024
    count = (SIZE+chunk-1)//chunk
    saved = json.loads(progress.read_text()) if progress.exists() else {}
    identity = {"sha256": SHA256, "size": SIZE, "chunk_size": chunk}
    done = set(saved.get("done", [])) if partial.exists() and all(saved.get(k) == v for k, v in identity.items()) else set()
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, SIZE)
    lock = threading.Lock()
    def fetch(index):
        start, end = index*chunk, min(SIZE, (index+1)*chunk)-1
        for attempt in range(3):
            try:
                with requests.get(URL+f"?download=true&part={index}", headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end}/{SIZE}":
                        raise RuntimeError("Server did not honor requested byte range")
                    offset = start
                    for data in response.iter_content(1024*1024):
                        if offset+len(data) > end+1 or os.pwrite(fd, data, offset) != len(data):
                            raise RuntimeError("Invalid range write")
                        offset += len(data)
                    if offset != end+1:
                        raise RuntimeError("Incomplete range")
                with lock:
                    os.fsync(fd)
                    done.add(index)
                    progress.write_text(json.dumps({**identity, "done": sorted(done)}))
                    print(f"encoder: {len(done)}/{count} verified ranges downloaded", flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, [i for i in range(count) if i not in done]))
    finally:
        os.close(fd)
    actual = hashlib.sha256(partial.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError("Downloaded encoder LFS SHA256 mismatch")
    partial.replace(destination)
    (root/"encoder_source.json").write_text(json.dumps({"repository": "byml/UniVTAC", "revision": REVISION,
        "url": URL, "sha256": actual, "bytes": SIZE, "downloaded_unix": time.time()}, indent=2))
    print(f"Verified encoder SHA256: {actual}", flush=True)


if __name__ == "__main__":
    main()
