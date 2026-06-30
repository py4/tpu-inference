# CODE_CHANGES — what was added to tpu-inference, and how to build the overlay

All GLM-5.2 code lives in a fork of `tpu-inference`. The default working serve is
the branch **`glm-5.2-native-flax`** (pushed to `origin`). DSA is a separate branch
(local-only; see bottom).

```
git clone -b glm-5.2-native-flax https://github.com/vllm-project/tpu-inference.git
```

> Note: the branch was cut from a specific upstream base (commit `3a72d49f`). The
> container **image** ships a (newer) `tpu_inference`. The overlay reconciles them —
> see "Building the overlay" below — so deps match the image and only the GLM-5.2
> code differs.

---

## The changed files (committed, `glm-5.2-native-flax` vs `origin/main`)

```
 tpu_inference/models/jax/glm5.py                       1570 ++++  (NEW model)
 tpu_inference/models/common/model_loader.py               8      (register arch)
 tpu_inference/layers/jax/moe/moe.py                      34      (sharded MoE loader)
 tpu_inference/layers/jax/quantization/unquantized.py    15      (MoE reshard safety)
 tpu_inference/layers/common/sharding.py                 13      (GLM_ATTN_EXPERT_SHARD)
 tpu_inference/kernels/mla/v2/tuned_params.py             7      (MLA decode VMEM fix)
 tpu_inference/models/jax/utils/weight_utils.py          17      (o_proj 2D fix)
 tpu_inference/models/common/pathways_dummy_loader.py    17      (dummy-load fix)
```

### `models/jax/glm5.py` (new, ~1570 lines) — the model
Native flax_nnx GLM-5.2. Started as a copy of `deepseek_v3.py` (GLM-5.2 *is* the
DeepSeek-V3.2 topology) with:
- **Module-global config block set to GLM-5.2** (the deepseek_v3 globals are
  DeepSeek-V3 values and must be overridden): hidden 6144, 64 Q/KV heads, q_lora
  2048, kv_lora 512, qk_nope 192, qk_rope 64, v_head 256, vocab 154880, ffw 12288,
  rms_eps 1e-5, rope_theta 8e6, n_group 1, interleaved RoPE (factor 1.0).
- `k_up_proj` / `v_up_proj` created in `__post_init__` (stock only creates them in
  `MLAEinsum.load_weights`, which assumes an FP8 ckpt → AttributeError on BF16).
- Class renamed → `GlmMoeDsaForCausalLM`.
- `load_weights` / `MLAEinsum.load_weights`: BF16 `kv_b_proj` split path (see
  `GOTCHAS.md` "Loader correctness"); `_filter_weights` skips `indexer.*` and any
  layer ≥ built count.
- `GLM_NUM_EXPERTS` env (default 256) for reduced-expert parity runs.

### `models/common/model_loader.py` — registration
Inline import + `_MODEL_REGISTRY["GlmMoeDsaForCausalLM"]`, added to
`_PP_DISABLED_MODELS`. Deliberately **not** in `_VLLM_PREFERRED_ARCHITECTURES`, so
it stays on the native flax_nnx path (not the torchax/vLLM wrapper).

### `layers/jax/moe/moe.py` — sharded MoE loader
Pass the intended sharding spec to `shard_put` (the `nnx.Param.sharding` attr is
`None` there → would replicate all 256 experts/chip → OOM). Loads experts directly
sharded; ~13× faster, no OOM. Critical for the full model.

### `layers/jax/quantization/unquantized.py` — reshard safety
Re-shards loaded experts to the intended layout before the fused-weight concat (now
a no-op given the moe.py fix; kept as a safety net).

### `layers/common/sharding.py` — `GLM_ATTN_EXPERT_SHARD`
Adds the env knob that splits `attn_dp_expert //= N; expert *= N` after the
`is_kv_fully_sharded` branch — shards attention heads on the shared `expert` axis
while keeping experts 128-way. **This is the trick that makes pure BF16 fit v6e-128.**

### `kernels/mla/v2/tuned_params.py` — MLA decode VMEM fix
Default fallback → `decode_batch_size=1, num_kv_pages_per_block=1` (stock 4/3 →
`CompileTimeScopedVmemOom` for 64-head MLA). We serve `max_num_seqs=1` anyway.

### `models/jax/utils/weight_utils.py` — o_proj 2D fix + on-device dummy loader
MLA `o_proj.weight` is 2D `(N*v_head_dim, D)`; the generic branch assumed 3D
per-head and crashed. Made ndim-aware.

Also rewrote `JaxDummyModelLoader.load_weights` to generate random weights
**on-device and sharded** (was host-CPU generation, which holds ~1 TB on the host
for a 744B-param MoE → Ray node OOM). Paired with the `glm5.py` guard below, this
enables a `--load-format dummy` serve that skips the ~61 min disk read (weight load
~42 s) for fast dev iteration. Full how-to + the chain of fixes in
**`RANDOM_WEIGHTS.md`**.

### `models/jax/glm5.py` — dummy-MoE guard is opt-out
`DeepSeekV3.__init__` used to unconditionally `raise` for `--load-format dummy` on
the fused-MoE backends. The dummy loader supports MoE now, so the guard is gated by
`GLM_ALLOW_DUMMY_MOE=1` (default still raises). See `RANDOM_WEIGHTS.md`.

### `models/common/pathways_dummy_loader.py` — dummy-load fix
(Only relevant to dummy-weight smokes / the old Pathways path.) Assigns the
already-sharded dummy array directly instead of a redundant slow `device_put`.
Not on the critical path for real-weight Ray serving.

---

## Building the overlay (what `ray_node.sh` bind-mounts)

`ray_node.sh` mounts `$OVERLAY -> /workspace/tpu_inference/tpu_inference:ro`,
replacing the image's package. **Build it so deps match the image exactly** and only
GLM-5.2 code differs:

1. Extract the image's own `tpu_inference` package to `$SHARED_FS/overlay/tpu_inference`:
   ```bash
   cid=$(sudo docker create $IMG)
   sudo docker cp $cid:/workspace/tpu_inference/tpu_inference $SHARED_FS/overlay/
   sudo docker rm $cid
   ```
2. Copy the fork's GLM-5.2 files over it. `glm5.py` is new (just drop it in). For the
   7 supporting files, copy them if the image's base matches the fork's base on those
   files; if a supporting file differs in the image (we hit this with
   `weight_utils.py`, where the image had a newer `cpu_mesh_context` fix), apply the
   GLM-5.2 edit **surgically** to the image's version instead of overwriting, so you
   keep the image's fix. Diff each before copying:
   ```bash
   for f in models/jax/glm5.py models/common/model_loader.py layers/jax/moe/moe.py \
            layers/jax/quantization/unquantized.py layers/common/sharding.py \
            kernels/mla/v2/tuned_params.py models/jax/utils/weight_utils.py \
            models/common/pathways_dummy_loader.py; do
     diff $SHARED_FS/overlay/tpu_inference/$f $SHARED_FS/tpu-inference-glm5/tpu_inference/$f
   done
   ```
3. Quick import check inside the container:
   `docker exec node python3 -c "import tpu_inference, tpu_inference.models.jax.glm5"`.

Re-create the `node` container (step 3 of the RUNBOOK) after changing the overlay.

---

## DSA branch (optional, experimental — not pushed)

`glm-5.2-dsa` (local-only at time of handoff) = `glm-5.2-native-flax` + ~250 lines in
`glm5.py`: the V3.2 `Indexer` module (`wq_b`, `wk`, `k_norm` LayerNorm, `weights_proj`),
the 21-full-layer schedule, mask + gather attention paths, and the `GLM_DSA` /
`GLM_DSA_GATHER` / `GLM_DSA_BENCH` env flags. Correct, decode-efficient, prefill not
TPU-efficient (see `GOTCHAS.md` "DSA"). **Push it before relying on it**, or request
the branch from us. The default serve does not use it.
