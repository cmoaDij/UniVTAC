"""Download the official UniVTAC FTP-1 checkpoint with resumable SHA256 checks."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests

from evotac.config import ROOT

REPO = "MJJJJ1064/ftp1_univtac_finetune"
REVISION = "620ac69b4fffd2341300cfef1b1d224d56710ed3"
PREFIX = "FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999/"
MIRROR = "https://www.modelscope.cn/models/michaelyuancb/ftp1_univtac_finetune/resolve/master/ftp1_univtac_finetune/"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def download_ranges(url, destination, size, expected_sha, workers=4):
    if destination.exists() and destination.stat().st_size == size and sha256(destination) == expected_sha:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix+".partial")
    progress = destination.with_suffix(destination.suffix+".chunks.json")
    chunk_size = 64*1024*1024
    identity = {"size": size, "sha256": expected_sha, "chunk_size": chunk_size}
    saved = json.loads(progress.read_text()) if progress.exists() else {}
    done = set(saved.get("done", [])) if all(saved.get(k) == v for k, v in identity.items()) and partial.exists() else set()
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    count = (size+chunk_size-1)//chunk_size
    def fetch(index):
        start, end = index*chunk_size, min(size, (index+1)*chunk_size)-1
        for attempt in range(4):
            try:
                session = requests.Session()
                session.trust_env = False  # Direct official mirror avoids slow proxy route on this host.
                with session.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 90)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Server did not honor the exact byte range")
                    offset = start
                    for block in r.iter_content(1024*1024):
                        if offset+len(block) > end+1 or os.pwrite(fd, block, offset) != len(block):
                            raise RuntimeError("Invalid or incomplete range write")
                        offset += len(block)
                    if offset != end+1:
                        raise RuntimeError("Incomplete download range")
                with lock:
                    done.add(index)
                    progress.write_text(json.dumps({**identity, "done": sorted(done)}))
                    print(f"{destination.name}: {len(done)}/{count} chunks", flush=True)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, [i for i in range(count) if i not in done]))
        os.fsync(fd)
    finally:
        os.close(fd)
    if sha256(partial) != expected_sha:
        raise RuntimeError("Checkpoint SHA256 mismatch; refusing to load")
    partial.replace(destination)
    progress.unlink(missing_ok=True)


def main():
    root = ROOT/"checkpoints/ftp1_univtac"
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}", params={"blobs": "true"}, timeout=30)
    response.raise_for_status()
    files = [f for f in response.json()["siblings"] if f["rfilename"].startswith(PREFIX)]
    evidence = {"repo_id": REPO, "revision": REVISION, "mirror": MIRROR, "files": {}}
    for record in files:
        name = record["rfilename"]
        dest = root/name
        if record.get("lfs"):
            digest = record["lfs"]["sha256"]
            download_ranges(MIRROR+name, dest, record["size"], digest)
        else:
            r = requests.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=30)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            digest = sha256(dest)
        evidence["files"][name] = {"sha256": digest, "size": dest.stat().st_size}
        (root/"verified_source.json").write_text(json.dumps(evidence, indent=2))
    print("Verified checkpoint:", root/PREFIX, flush=True)


if __name__ == "__main__":
    main()
