# Serving GLM-5.2 on TPU v6e-128 (pure BF16) — reproduce kit

This directory is a self-contained handoff for running **GLM-5.2** (744B-A40B, the
DeepSeek-V3.2 topology: MLA + sigmoid MoE + DSA) on a **Cloud TPU v6e-128** pod in
**pure BF16**, served through an OpenAI-compatible vLLM endpoint. It is implemented
natively in **flax_nnx** inside the `tpu-inference` project (no torchax wrapper).

It was built and validated on our pod; this kit is everything needed to do it on a
*different* v6e-128. Read this file, then `RUNBOOK.md` for the exact commands.

---

## What actually works (validated on v6e-128)

- **Full 78-layer, 256-expert GLM-5.2, pure BF16 (no FP8), loads and runs on
  v6e-128 and produces coherent English.** Per-chip weights ~13 GB; load ~50 min,
  first-shape compile ~60 min, decode ~24 s/iter for the offline smoke.
- **OpenAI-compatible server** (`vllm serve`) on port 8000: `/v1/completions`,
  `/v1/chat/completions`, tool-calling (`glm47` parser), reasoning parser.
- **Agentic coding (opencode) end-to-end** against the TPU backend, verified with
  request contexts **>2K tokens** (measured 5.8K–7.2K tokens/step live).
- **Full-dense attention past the 2048 window**: needle-in-haystack recalled at
  ~3K tokens of context; coherent and accurate.
- **Layer-0 and MoE-layer numerical parity vs the vLLM GPU / HF reference** (BF16
  cross-hardware noise only).
- **Optional, experimental:** 1M-context serve (one-line RoPE + config bump) and a
  DSA "lightning indexer" path (numerically correct; decode-efficient up to ~4.3×
  at 128K; NOT in the default serve). See `GOTCHAS.md` and `CODE_CHANGES.md`.
- **Fast dev iteration with random weights:** `LOADFMT=dummy ALLOWDUMMYMOE=1`
  generates weights on-device and skips the ~61 min disk read (weight load ~42 s).
  Output is gibberish — for "does the pipeline run / shapes / OOM / scheduling"
  checks only, not correctness. See `RANDOM_WEIGHTS.md`.

> Honesty note: the proven, default configuration is **8192-context pure BF16
> without DSA / without spec-decoding**. The 1M and DSA paths are extensions with
> caveats documented inline — don't assume they are production-ready.

---

## Hardware / environment assumptions

- **TPU v6e-128** = 32 hosts × 4 chips, 31.25 GB usable HBM/chip. The whole
  sharding recipe (TP=64, EP=128, attention sharded 32-way on the expert axis) is
  tuned so pure BF16 *just* fits in HBM. A smaller pod will **not** fit pure BF16;
  a different topology needs the sharding re-derived. See `GOTCHAS.md`.
- A **shared writable filesystem** mounted at the *same path on every host*
  (we used `/mnt/glm5fs`). Ray + the run scripts coordinate through it.
- The **weights** (~1.4 TiB BF16) readable from every host at the same path. On
  v6e we used a **Hyperdisk ML** volume (READ_ONLY_MANY) — note v6e does **not**
  support plain pd-ssd/pd-balanced; only Hyperdisk Balanced / Hyperdisk ML.
- A **tpu-inference container image** (vllm + jax + ray + libtpu + tpu_inference)
  and the public **vbar control agent** image. See `RUNBOOK.md` step 1.
- `gcloud` with SSH access to the pod (`--worker=all --internal-ip`), and a
  Hugging Face token with access to the gated `zai-org/GLM-5.2` repo.

---

## Directory map

```
reproduce/
├── README.md            <- you are here (overview)
├── RUNBOOK.md           <- exact step-by-step commands, start to finish
├── GOTCHAS.md           <- the hard-won traps; READ before debugging
├── CODE_CHANGES.md      <- the code: which branch, which 8 files, why
├── RANDOM_WEIGHTS.md    <- fast dev: random on-device weights, skip the ~60min disk read
├── BUILD_LOG.md         <- raw append-only build log (deep reference, unfiltered;
│                           includes our internal IPs/projects + dead-ends)
├── config/
│   ├── config.8192.json     <- model dir config.json (scaffold), 8192 ctx (PROVEN)
│   ├── config.1m.json       <- same, max_position_embeddings=1048576 (experimental)
│   ├── generation_config.json
│   └── chat_template.jinja  <- GLM-5.2 chat template, edited to default no-think
└── scripts/
    ├── download_weights.py  <- HF zai-org/GLM-5.2 -> local dir (parallel, resumable)
    ├── build_model_dir.sh   <- assemble the model dir (config+tokenizer+282 symlinks)
    ├── ray_node.sh          <- per-host bring-up (vfio bind + vbar + ray/vllm node)
    ├── run_serve.sh         <- start the OpenAI server (head container)
    ├── run_glm5.sh          <- offline generation smoke (head container)
    ├── glm5_generate.py     <- the offline-gen entrypoint run by run_glm5.sh
    ├── test_api.sh          <- curl smoke against the server
    ├── test_long_ctx.py     <- >2K needle-in-haystack test
    └── test_1m.py           <- ~1.04M-token needle test (only for the 1M serve)
```

---

## The 30-second mental model

1. **Code** lives in a fork of `tpu-inference` (branch `glm-5.2-native-flax`): a new
   native flax model `glm5.py` + 7 supporting edits (loader registration, MoE
   sharded loader, attention-head sharding knob, two MLA-kernel VMEM fixes). See
   `CODE_CHANGES.md`. You bind-mount this package read-only over the image's copy.
2. **Bring-up** (`ray_node.sh start` on every host): bind TPU chips to vfio-pci,
   run the vbar control agent (:8353), start a Ray head (worker 0) / workers in a
   docker container. Native libtpu auto-coordinates — no `jax.distributed.initialize`.
3. **Model dir** (`build_model_dir.sh`): a folder with a scaffold `config.json`,
   tokenizer, chat template, and 282 BF16 shards (symlinks to the RO weights vol).
4. **Run** inside the head container: `run_serve.sh` (server) or `run_glm5.sh`
   (offline). The magic is the env: `GLM_ATTN_EXPERT_SHARD=32` + EP=128 + the
   kernel env flags. That's what makes pure BF16 fit and the kernels not OOM.

Next: open **`RUNBOOK.md`**.
