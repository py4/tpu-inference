"""Parallel HF->GCS streamer for GLM-5.2. N worker threads each download a shard
and upload to GCS, to beat the ~15MB/s single-stream limit. Subset-first.
Resumable (skips shards already in GCS). Per-shard temp file deleted after upload.
Env: WORKERS (default 16), SUBSET_LAYERS (16), PREFIX (glm5_full).
"""
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download

REPO = "zai-org/GLM-5.2"
BUCKET = "vtc-pathways-scratch-infinipod-shared-dev"
PREFIX = os.environ.get("PREFIX", "glm5_full")
SUBSET_LAYERS = int(os.environ.get("SUBSET_LAYERS", "16"))
WORKERS = int(os.environ.get("WORKERS", "16"))
TMP = os.environ.get("TMP_DIR", "/home/pooyam_google_com/hf_dl_tmp")
os.makedirs(TMP, exist_ok=True)
_lock = threading.Lock()
_done = [0]


def gcs_existing():
    r = subprocess.run(["gsutil", "ls", f"gs://{BUCKET}/{PREFIX}/"],
                       capture_output=True, text=True)
    return {ln.strip().rsplit("/", 1)[-1]
            for ln in r.stdout.splitlines()
            if ln.strip() and not ln.strip().endswith("/")}


def needs(t, max_layer):
    if ".indexer." in t:
        return False
    if t.startswith("model.embed_tokens.") or t == "model.norm.weight" \
            or t == "lm_head.weight":
        return True
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", t)
    return m is not None and int(m.group(1)) < max_layer


def fetch(fname, total):
    try:
        p = hf_hub_download(REPO, fname, local_dir=TMP)
        rc = subprocess.run(["gsutil", "-q", "cp", p,
                             f"gs://{BUCKET}/{PREFIX}/{fname}"]).returncode
        sz = os.path.getsize(p) / 1e9
        os.remove(p)
        with _lock:
            _done[0] += 1
            print(f"[{_done[0]}/{total}] {'OK' if rc == 0 else 'FAIL'} {fname} "
                  f"{sz:.1f}GB", flush=True)
    except Exception as e:
        with _lock:
            print(f"ERR {fname}: {e}", flush=True)


def main():
    idx = hf_hub_download(REPO, "model.safetensors.index.json", local_dir=TMP)
    wm = json.load(open(idx))["weight_map"]
    subset, full = set(), set()
    for t, sh in wm.items():
        if needs(t, SUBSET_LAYERS):
            subset.add(sh)
        if needs(t, 10_000):
            full.add(sh)
    order = sorted(subset) + sorted(full - subset)
    aux = ["model.safetensors.index.json", "config.json", "tokenizer.json",
           "tokenizer_config.json", "generation_config.json",
           "special_tokens_map.json", "tokenizer.model"]
    existing = gcs_existing()
    todo = [f for f in (aux + order) if f not in existing]
    total = len(todo)
    print(f"subset shards={len(subset)} full shards={len(full)} "
          f"in_gcs={len(existing)} todo={total} workers={WORKERS}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, f, total) for f in todo]
        for _ in as_completed(futs):
            pass
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
