"""Stream the full GLM-5.2 (~1.5TB) from HuggingFace to GCS, shard by shard, so
the v6e pod can read it via GCS-fuse. Resumable: skips files already in GCS.
Low local disk: downloads one file, uploads, deletes.

gs://<BUCKET>/<PREFIX>/  <- all .safetensors + *.json + tokenizer files.
"""
import os
import subprocess
import sys

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "zai-org/GLM-5.2"
BUCKET = "vtc-pathways-scratch-infinipod-shared-dev"
PREFIX = "glm5_full"
TMP = os.environ.get("TMP_DIR", "/home/pooyam_google_com/hf_dl_tmp")
os.makedirs(TMP, exist_ok=True)


def gcs_existing():
    r = subprocess.run(["gsutil", "ls", f"gs://{BUCKET}/{PREFIX}/"],
                       capture_output=True, text=True)
    names = set()
    for line in r.stdout.splitlines():
        line = line.strip()
        if line and not line.endswith("/"):
            names.add(line.rsplit("/", 1)[-1])
    return names


def main():
    files = list_repo_files(REPO)
    want = [
        f for f in files
        if (f.endswith(".safetensors") or f.endswith(".json")
            or "tokenizer" in f or f.endswith(".model")) and "/" not in f
    ]
    want.sort()
    existing = gcs_existing()
    print(f"repo files wanted: {len(want)}; already in GCS: {len(existing)}",
          flush=True)
    done = 0
    for i, f in enumerate(want):
        if f in existing:
            done += 1
            continue
        try:
            p = hf_hub_download(REPO, f, local_dir=TMP,
                                local_dir_use_symlinks=False)
        except Exception as e:
            print(f"[{i+1}/{len(want)}] DL FAIL {f}: {e}", flush=True)
            continue
        rc = subprocess.run(
            ["gsutil", "-q", "cp", p, f"gs://{BUCKET}/{PREFIX}/{f}"]).returncode
        sz = os.path.getsize(p) / 1e9
        try:
            os.remove(p)
        except OSError:
            pass
        done += 1
        print(f"[{i+1}/{len(want)}] {'OK' if rc == 0 else 'UPLOAD_FAIL'} {f} "
              f"{sz:.1f}GB ({done}/{len(want)})", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
