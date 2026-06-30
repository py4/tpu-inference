# Random (dummy) weights — fast dev iteration without the disk read

Loading the real 1.4 TiB BF16 checkpoint from disk takes **~61 min** (measured:
`Loading weights took 3650 s`, ~384 MB/s off Hyperdisk ML). For any change where you
only need to know **"does the pipeline run?"** — shapes, sharding, HBM/OOM,
scheduling, a new kernel, an env flag — you do not need the real weights. This path
generates **random weights directly on-device** and skips the disk read entirely.

| | Weight load | Total to ready |
|---|---|---|
| Real weights (disk) | **~61 min** | ~76 min |
| Random weights (on-device) | **~42 s** | ~17 min |

The weight step is **~87× faster**. End-to-end it is ~4.5× faster; the remaining
~15 min is **warmup compile**, which is identical for real and random weights (it
compiles the unrolled 78-layer bucket graph, it does not touch the disk).

> **Output is gibberish.** Random weights produce incoherent tokens. Use this ONLY
> to validate that the serve *runs without error*. For any correctness / needle /
> opencode check, load the real weights.

---

## How to use

Add two env vars to the normal serve launch (`scripts/run_serve.sh`):

```bash
# inside the head 'node' container (see RUNBOOK.md step 6b for the docker exec wrapper)
LOADFMT=dummy ALLOWDUMMYMOE=1 \
  MODELDIR=$SHARED_FS/glm5_model \
  MAXLEN=8192 \
  LOG=$SHARED_FS/glm5_serve.log \
  bash $SHARED_FS/reproduce/scripts/run_serve.sh
```

- `LOADFMT=dummy`   → vLLM `--load-format dummy` (no checkpoint read; the model dir
  is still needed for `config.json` / tokenizer / chat template).
- `ALLOWDUMMYMOE=1` → `GLM_ALLOW_DUMMY_MOE=1`, which lifts an (overly broad) guard
  that otherwise refuses dummy weights for the fused-MoE backends.

Everything else (sharding, EP/TP, kernel envs) is unchanged, so the run exercises
the **same** code path as a real serve. Wait for `Application startup complete` /
`GET /v1/models -> 200`, then hit the API as usual (`scripts/test_api.sh`); expect
gibberish completions.

### Make warmup fast too (optional)

The ~15 min warmup compiles every `num_tokens` bucket up to
`max_num_batched_tokens × dp_size`. For a quick "does-it-run" check, shrink it:

```bash
LOADFMT=dummy ALLOWDUMMYMOE=1 MAXLEN=8192 MAXBATCH=512 ... bash run_serve.sh
```

`MAXBATCH=512` drops the largest compiled bucket from 16384 → 4096, cutting warmup
substantially. (This narrows the prefill chunk size; fine for a smoke test, not for
a throughput run.)

---

## What changed in the code (committed with this doc)

Two files, both strictly about dummy-weight loading:

1. **`models/jax/glm5.py`** — the dummy-MoE guard in `DeepSeekV3.__init__` is now
   opt-out via `GLM_ALLOW_DUMMY_MOE=1` instead of an unconditional `raise`. The
   loader below does support MoE; the guard predated that.

2. **`models/jax/utils/weight_utils.py`** — `JaxDummyModelLoader.load_weights`
   rewritten to generate **on-device, sharded** instead of on the host CPU. The old
   path generated each full param on the host (`cpu_mesh_context`) and then copied
   to TPU; for a 744B-param MoE that holds ~1 TB in host RAM and **OOM-kills the Ray
   node**. The rewrite, and the traps it works around (each was a real failure):

   - **Generate per-shard on-device** via `jax.jit(out_shardings=...)` — no host RAM,
     no host→device copy.
   - **Sequential** generation (no `ThreadPoolExecutor`): on-device sharded
     generation is a multi-host SPMD dispatch and must launch in the same order on
     every worker, or libtpu aborts (`Terminating the libtpu controller proc`).
   - **Interleaved per-module fusion** (`_gen_and_fuse`): generate a layer's expert
     kernels, fuse them, free the raw, then move on — mirrors the real per-layer
     load. Generating *all* raw experts first holds ~181 G/chip → HBM OOM.
   - **Full-mesh expert sharding** for the raw MoE kernels: their `nnx.Param`
     sharding metadata reads back empty, so the generic path would generate them
     **replicated** (full 6.4 G/chip per param) → OOM. Pass the layer's expert
     sharding explicitly and generate ~50 M/chip per param.
   - **Initialize derived caches** (`initialize_cache`, e.g. the RoPE sin/cos cache)
     after loading — the real `model.load_weights` does this, the dummy loader
     bypasses it (else warmup asserts `RoPE cache not initialized`).
   - **Shape-keyed jit cache** for the generators — without it each param recompiles
     its own program (428 s); with it, params that share a shape reuse one (42 s).

No change to the real-weight load path.

---

## Gotchas

- **Not for correctness.** Output is random. Don't read anything into the tokens.
- **The model dir is still required** (`config.json`, tokenizer, chat template) —
  only the `*.safetensors` reads are skipped.
- **Warmup still runs** and dominates the ~17 min; it is the same compile a real
  serve does. Use `MAXBATCH=512` (above) if you want the loop to be minutes.
- **Same sharding as the real serve.** That is the point — an OOM / shape / kernel
  bug reproduced here reproduces with real weights. Memory is *slightly* lower than
  real (random init range is tiny), so treat a near-miss on HBM as suspect.
