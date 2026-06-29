# RUNBOOK — GLM-5.2 on v6e-128, start to finish

Exact commands. Assumes a **v6e-128** pod, `gcloud` SSH access, a shared FS at the
same path on every host, and an HF token for `zai-org/GLM-5.2`. Adjust the names
in the `## 0. Variables` block to your environment; the rest follows.

Throughout: drive all hosts at once with
```
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ \
  --worker=all --internal-ip --command="<cmd>"
```
`--internal-ip` is required. You need ssh-agent + key loaded.

---

## 0. Variables (edit me)

```bash
export VM=glm5-ray-v6e128                 # your TPU pod name
export ZONE=us-central1-b
export PROJ=infinipod-shared-dev
export SHARED_FS=/mnt/glm5fs              # shared writable FS, SAME path on all hosts
export WEIGHTS_FS=/mnt/glm5               # RO weights volume, SAME path on all hosts
export MODELDIR=$SHARED_FS/glm5_model     # the model dir we will build
export HF_TOKEN=hf_xxx                    # access to gated zai-org/GLM-5.2
# Image + head IP get set in steps 1 and 3.
```

---

## 1. Get the container images on every host

You need a tpu-inference image (vllm + jax + ray + libtpu + editable tpu_inference)
and the public vbar control agent. We used:

- `IMG = us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference/vllm-tpu:latest`
  (vllm 0.23.x, jax 0.10.2, libtpu 0.0.42.1, ray 2.55.1). Needs registry access.
- `VBAR = gcr.io/cloud-tpu-v2-images/vbar_control_agent:latest` (public).

If you can pull our image, `ray_node.sh pull <oauth_token>` logs in + pulls both on
each host. Otherwise build/obtain an equivalent tpu-inference image (any recent
`vllm/vllm-tpu` nightly with a compatible tpu_inference works — the fork overlay in
step 4 replaces the model code anyway; only the deps + kernels must match jax/libtpu).

```bash
# put this kit + scripts on the shared FS first (so all hosts see them):
#   copy reproduce/ to $SHARED_FS/reproduce on the pod (gcloud scp, or via the
#   weights volume). Then on each host:
TOKEN=$(gcloud auth print-access-token)
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ --worker=all --internal-ip \
  --command="IMG=<your_img> bash $SHARED_FS/reproduce/scripts/ray_node.sh pull $TOKEN"
# expect: PULL_IMG_OK / PULL_VBAR_OK on every worker.
```

---

## 2. Download the weights (once) to the RO volume

`zai-org/GLM-5.2` = 282 BF16 safetensors shards (~1.4 TiB), all BF16, no quant.
Download once to a volume every host can read at `$WEIGHTS_FS`.

```bash
pip install huggingface_hub
DEST=$WEIGHTS_FS/glm5_full WORKERS=16 HF_TOKEN=$HF_TOKEN \
  python3 $SHARED_FS/reproduce/scripts/download_weights.py
```
On v6e, the durable multi-host pattern is: populate the disk once (e.g. from a
temporary GCE VM), snapshot it, create a **Hyperdisk ML** (`--access-mode=
READ_ONLY_MANY`) from the snapshot, then `gcloud alpha compute tpus tpu-vm
attach-disk $VM --disk=<hml> --mode=read-only`. (Plain PD types fail to attach on
v6e — see `GOTCHAS.md`.) Mount RO on each host and point `$WEIGHTS_FS` at it.

---

## 3. Bring up the Ray cluster (every host)

First find the **internal IP of worker 0** (the Ray head):
```bash
gcloud compute tpus tpu-vm describe $VM --zone=$ZONE --project=$PROJ \
  --format='value(networkEndpoints[0].ipAddress)'
export HEAD_IP=<that internal ip>
```
Then start the per-host stack on all hosts. `ray_node.sh start` does: enable
`allow_unsafe_interrupts`, bind the 4 TPU chips to vfio-pci, (re)start the vbar
control agent (:8353), and start the Ray/vLLM `node` container (head on worker 0,
workers elsewhere).
```bash
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ --worker=all --internal-ip \
  --command="HEAD_IP=$HEAD_IP SHARED_FS=$SHARED_FS WEIGHTS_FS=$WEIGHTS_FS IMG=<your_img> \
             bash $SHARED_FS/reproduce/scripts/ray_node.sh start"
# expect on each host: NODE_STARTED w=<n> role=head|worker vbar8353=1
```
Sanity-check the cluster sees 128 chips (run inside the head container):
```bash
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ --worker=0 --internal-ip \
  --command="sudo docker exec node ray status"   # expect 128 TPU
```

---

## 4. Stage the fork code (overlay) on the shared FS

The container runs the GLM-5.2 fork by bind-mounting the fork's `tpu_inference`
package read-only over the image's copy (`ray_node.sh` mounts
`$OVERLAY -> /workspace/tpu_inference/tpu_inference:ro`). Build `$OVERLAY` once:

```bash
# clone the fork (branch has all GLM-5.2 code; see CODE_CHANGES.md for the file list)
git clone -b glm-5.2-native-flax https://github.com/vllm-project/tpu-inference.git \
  $SHARED_FS/tpu-inference-glm5
# overlay = the image's package + the fork's files. SAFEST: start from the image's
# own package, then copy the fork's changed files over it (so deps match the image
# exactly and only the GLM-5.2 code differs). See CODE_CHANGES.md "Building the overlay".
export OVERLAY=$SHARED_FS/overlay/tpu_inference
```
Re-run step 3 if you built the overlay after the containers started (or just restart
the `node` container). The default `$OVERLAY` in `ray_node.sh` is
`$SHARED_FS/overlay/tpu_inference`.

---

## 5. Build the model dir

```bash
WEIGHTS=$WEIGHTS_FS/glm5_full MODELDIR=$MODELDIR REPRO=$SHARED_FS/reproduce CTX=8192 \
  bash $SHARED_FS/reproduce/scripts/build_model_dir.sh
# -> $MODELDIR with config.json (8192) + tokenizer + chat_template + 282 symlinks
```
(`CTX=1m` swaps in the 1M config; only do that for the experimental 1M serve, and
also flip the one-line RoPE global in glm5.py — see `GOTCHAS.md` "1M context".)

---

## 6a. Offline generation smoke (fastest way to prove the model works)

Run inside the head container. This is the canonical "does it produce English" test.
```bash
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ --worker=0 --internal-ip \
  --command="sudo docker exec node env \
      GEN_MODEL=$MODELDIR GLM_NUM_EXPERTS=256 \
      SMOKE_TP=64 SMOKE_EP=128 GLM_ATTN_EXPERT_SHARD=32 \
      GEN_KV_BLOCKS=128 GEN_MAXTOK=48 GEN_GPU_UTIL=0.85 \
      LOG=$SHARED_FS/glm5_full_bf16.log \
      bash $SHARED_FS/reproduce/scripts/run_glm5.sh"
# then poll the log:
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ --worker=0 --internal-ip \
  --command="tail -40 $SHARED_FS/glm5_full_bf16.log"
```
Expect (~110 min cold: ~50 load + ~60 compile) `GEN_OK` and coherent completions,
e.g. "The capital of France is Paris...". `run_glm5.sh` detaches with `setsid` so it
survives the SSH/exec disconnect; always poll the log, never hold a foreground exec
through the compile (see `GOTCHAS.md`).

## 6b. OpenAI server (for opencode / API use)

```bash
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ --worker=0 --internal-ip \
  --command="sudo docker exec node env \
      MAXLEN=8192 MAXBATCH=2048 KVBLOCKS=256 \
      LOG=$SHARED_FS/glm5_serve.log \
      bash $SHARED_FS/reproduce/scripts/run_serve.sh"
```
`run_serve.sh` already pins `GLM_ATTN_EXPERT_SHARD=32`, EP=128, TP=64, the kernel
env, the chat template, and the glm47 tool/reasoning parsers. **Do not** set
`SKIP_JAX_PRECOMPILE` for the server (it is a driver-only env → driver/worker
mismatch hangs warmup; let the full bucket sweep run). Up after ~85 min
(load + warmup). Wait for HTTP 200 on `/v1/models`:
```bash
gcloud compute tpus tpu-vm ssh $VM --zone=$ZONE --project=$PROJ --worker=0 --internal-ip \
  --command="bash $SHARED_FS/reproduce/scripts/test_api.sh"   # curl smoke
```

---

## 7. Verify

- **API smoke:** `scripts/test_api.sh` (a `/v1/completions` curl).
- **>2K context:** `python3 scripts/test_long_ctx.py` inside the container — plants a
  needle ~3K tokens before the question; expect exact recall and `prompt_tokens>2048`.
- **1M context** (only if you started the 1M serve): `python3 scripts/test_1m.py`.
- **opencode:** install on host 0 (`curl -fsSL https://opencode.ai/install | bash`),
  point an `openai-compatible` provider at `http://localhost:8000/v1`, model
  `glm-5.2`. To keep small requests lean we used an isolated `HOME`, a `tiny`
  agent (bash-only tools), and `limit.context`/`output` caps — see `GOTCHAS.md`.

---

## 8. Restart / teardown

- **Restart the whole fleet:** re-run step 3 (`ray_node.sh start` is idempotent;
  it `docker rm -f` the old containers first).
- **Stop the server cleanly:** SIGTERM the api-server pid in the head container
  (graceful → all Ray placement groups removed). **Do not** `pkill -f "vllm serve"`
  on host 0 only — it orphans Ray worker actors on the other 31 hosts holding a
  stale placement group, and the next serve hangs "Waiting for creating a placement
  group". If that happens, see `GOTCHAS.md` for the placement-group cleanup.
