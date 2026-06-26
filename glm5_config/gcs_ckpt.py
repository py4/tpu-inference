"""Stage/fetch the 1-layer GLM-5.2 checkpoint to/from GCS so it survives pod
restarts (avoids re-downloading from HF). Uses google.cloud.storage (the
vtc-pathways-sa service account has access). gcloud CLI is not in the image.

Usage (on the pod):
  python3 gcs_ckpt.py put   # upload /workspace/glm5_real -> gs://.../glm5_1layer/
  python3 gcs_ckpt.py get   # download gs://.../glm5_1layer/ -> /workspace/glm5_real
"""
import os
import sys

from google.cloud import storage

BUCKET = "vtc-pathways-scratch-infinipod-shared-dev"
PREFIX = os.environ.get("GCS_PREFIX", "glm5_1layer")
LOCAL = os.environ.get("LOCAL_DIR", "/workspace/glm5_real")


def put():
    c = storage.Client()
    b = c.bucket(BUCKET)
    for fn in sorted(os.listdir(LOCAL)):
        p = os.path.join(LOCAL, fn)
        if not os.path.isfile(p):
            continue
        b.blob(f"{PREFIX}/{fn}").upload_from_filename(p)
        print("uploaded", fn, f"{os.path.getsize(p)/1e9:.2f}GB")
    print("PUT_DONE")


def get():
    c = storage.Client()
    b = c.bucket(BUCKET)
    os.makedirs(LOCAL, exist_ok=True)
    blobs = list(c.list_blobs(BUCKET, prefix=PREFIX + "/"))
    if not blobs:
        print("GCS_EMPTY")
        return 1
    for bl in blobs:
        fn = bl.name.split("/")[-1]
        if not fn:
            continue
        bl.download_to_filename(os.path.join(LOCAL, fn))
        print("downloaded", fn)
    print("GET_DONE")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "put"
    sys.exit(get() if mode == "get" else (put() or 0))
