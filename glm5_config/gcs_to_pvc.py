"""Robust parallel GCS->PVC copy (pod has no gsutil). Thread-local storage
clients (avoid shared-pool contention/hangs), per-download timeout + retry,
atomic .tmp rename, resumable (skip files already present with matching size).
Loops until TARGET_SHARDS local so it drains GCS as the download fills it.

Env: GCS_PREFIX (glm5_full), LOCAL_DIR (/mnt/ckpt/glm5), WORKERS (16),
TARGET_SHARDS (260; 0 = one pass).
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage

BUCKET = "vtc-pathways-scratch-infinipod-shared-dev"
PREFIX = os.environ.get("GCS_PREFIX", "glm5_full")
LOCAL = os.environ.get("LOCAL_DIR", "/mnt/ckpt/glm5")
WORKERS = int(os.environ.get("WORKERS", "16"))
TARGET = int(os.environ.get("TARGET_SHARDS", "260"))
os.makedirs(LOCAL, exist_ok=True)
_tl = threading.local()
_lock = threading.Lock()
_n = [0]


def client():
    if not hasattr(_tl, "c"):
        _tl.c = storage.Client()
    return _tl.c


def dl(name, size, total):
    fn = name.rsplit("/", 1)[-1]
    if not fn:
        return
    dst = os.path.join(LOCAL, fn)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return
    tmp = dst + ".tmp"
    for attempt in range(4):
        try:
            b = client().bucket(BUCKET).blob(name)
            b.download_to_filename(tmp, timeout=300)
            if os.path.getsize(tmp) == size:
                os.rename(tmp, dst)
                with _lock:
                    _n[0] += 1
                    print(f"[{_n[0]}/{total}] dl {fn} {size/1e9:.1f}GB", flush=True)
                return
        except Exception as e:
            with _lock:
                print(f"retry {fn} ({attempt}): {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    with _lock:
        print(f"GIVEUP {fn}", flush=True)


while True:
    c = storage.Client()
    blobs = [(b.name, b.size) for b in c.list_blobs(BUCKET, prefix=PREFIX + "/")
             if not b.name.endswith("/")]
    todo = [(n, s) for n, s in blobs
            if not (os.path.exists(os.path.join(LOCAL, n.rsplit("/", 1)[-1]))
                    and os.path.getsize(os.path.join(LOCAL, n.rsplit("/", 1)[-1])) == s)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda t: dl(t[0], t[1], len(blobs)), todo))
    nshard = len([f for f in os.listdir(LOCAL) if f.endswith(".safetensors")])
    print(f"pass: {len(blobs)} in gcs, {nshard} shards local, {len(todo)} were todo",
          flush=True)
    if TARGET <= 0 or nshard >= TARGET:
        break
    time.sleep(20)
print("GCS_TO_PVC_DONE", flush=True)
