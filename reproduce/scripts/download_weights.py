"""Download the full GLM-5.2 BF16 checkpoint from Hugging Face to a local dir.

GLM-5.2 = zai-org/GLM-5.2 = 282 safetensors shards, ALL BF16, ~1.4 TiB total,
NO quantization_config. We download the weight shards + tokenizer; we do NOT use
the HF config.json or the indexer (DSA) tensors at serve time (see notes below).

Parallel download (N threads) to beat the single-stream HF rate limit. Resumable:
shards already present (correct size) are skipped.

Usage:
    pip install huggingface_hub
    export HF_TOKEN=hf_...                  # zai-org/GLM-5.2 is gated; accept the license first
    DEST=/mnt/weights/glm5_full WORKERS=16 python3 download_weights.py

Notes:
- Put DEST on storage every TPU host can read. On a Cloud TPU v6e pod we used a
  Hyperdisk ML volume (READ_ONLY_MANY, multi-host RO) populated once, mounted on
  all 32 hosts. A shared NFS/Filestore or per-host local copy also works.
- The serve does NOT need model.safetensors.index.json (the flax loader globs
  *.safetensors). We still fetch it (harmless, and handy for sanity checks).
- The indexer (DSA) tensors ARE in the checkpoint; the BF16 serve path simply
  ignores them. Downloading them is fine (a few hundred MB); don't bother filtering.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "zai-org/GLM-5.2"
DEST = os.environ.get("DEST", "/mnt/weights/glm5_full")
WORKERS = int(os.environ.get("WORKERS", "16"))

# Tokenizer / aux files we DO want alongside the shards.
AUX = [
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def fetch(fname):
    try:
        hf_hub_download(REPO, fname, local_dir=DEST)
        return (fname, True, "")
    except Exception as e:  # missing optional aux file is non-fatal
        return (fname, False, str(e))


def main():
    os.makedirs(DEST, exist_ok=True)
    files = [f for f in list_repo_files(REPO) if f.endswith(".safetensors")]
    files = sorted(files) + [a for a in AUX]
    print(f"[dl] {len(files)} files -> {DEST} ({WORKERS} workers)", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, f): f for f in files}
        for fut in as_completed(futs):
            fname, ok, err = fut.result()
            done += 1
            tag = "OK" if ok else f"SKIP/ERR ({err[:60]})"
            print(f"[dl] [{done}/{len(files)}] {tag} {fname}", flush=True)
    print("[dl] DONE", flush=True)


if __name__ == "__main__":
    main()
