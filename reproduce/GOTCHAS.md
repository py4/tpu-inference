# GOTCHAS — hard-won traps (read before debugging)

Distilled from the build log. Each item cost real time. Grouped by area.

---

## Multi-host TPU bring-up (v6e)

- **All four of these are required** for multi-host JAX on a v6e GCE pod; missing
  any one gives a confusing failure:
  1. `allow_unsafe_interrupts=1` on `vfio_iommu_type1` — GCE VMs lack interrupt
     remapping; without it: *"Couldn't set iommu to type 3: Operation not permitted"*.
  2. **Bind the 4 TPU chips to vfio-pci BEFORE `docker run`** — else the container's
     `/dev` snapshot lacks `/dev/vfio/<group>` → libtpu sees *"0 chips / jellyfish"*.
     Bound chips also let Ray auto-detect `TPU=128`.
  3. **Run the vbar control agent on every host** (`:8353`). This VM's metadata
     `VBARCONTROL_AGENT_DOCKER_URL` is empty so the systemd unit never starts it —
     run it yourself. Without it: *"Failed to connect to vbar control service at
     [::]:8353"*.
  4. node container needs `--privileged --network host` and `-v /var/run/tpu-plugin`.
  `ray_node.sh start` does all four.
- **No `jax.distributed.initialize` needed.** Once vbar + bound chips are present,
  native libtpu auto-coordinates. Verified `jax.device_count()` → global 128.
- **v6e does NOT support plain Persistent Disks** (pd-ssd / pd-balanced). Every
  `attach-disk` returns a server-side *"internal error"*. Use **Hyperdisk Balanced**
  (single-host RW) or **Hyperdisk ML** (`READ_ONLY_MANY`, multi-host RO). Build an
  HML from a snapshot of a populated disk; HML create may need explicit
  `--provisioned-throughput=20000` (default exceeds the per-project quota).

## Don't kill a running TPU job the wrong way

- The model run is launched **detached** (`setsid ... </dev/null >log &`). A
  **foreground** `gcloud ... ssh --command="docker exec ..."` that hits the local
  120 s timeout sends SIGTERM → kills the remote python → cluster/IFRT drop. Always
  launch detached and **poll the log** with short calls; never hold a foreground
  exec through the (~60 min) compile.
- A frozen log + low CPU is **not** proof of compiling — a real compile pegs a host
  CPU. Check host loadavg + advancing log line-count to tell "compiling/loading"
  from "hung". (The slow MoE reshard once looked hung for 6 min but was working.)

## Stale Ray placement groups (the #1 "serve won't start" cause)

- `pkill -f "vllm serve"` on host 0 only leaves Ray worker actors alive on the
  other 31 hosts, holding a placement group. Next serve hangs *"Waiting for
  creating a placement group"*, TPU shows 128/128 used.
- **Prefer graceful SIGTERM** of the api-server pid (removes all PGs, frees TPU).
- If already stuck, remove the PG via the Ray API from the head container:
  ```python
  import ray; ray.init(address="auto")
  from ray.util.placement_group import remove_placement_group, PlacementGroup
  from ray._raylet import PlacementGroupID
  for e in ray.util.placement_group_table().values():
      remove_placement_group(PlacementGroup(PlacementGroupID(bytes.fromhex(e["placement_group_id"]))))
  ```
  (`docker exec -i node python3 - <<'PY' ... PY` — the `-i` is needed for heredocs.)
  Or just kill `RayWorkerProc` on **all** hosts (`--worker=all`) and re-run step 3.

## Pure-BF16 HBM fit — the central sharding trick

- Naive sharding OOMs at load (~73–99%): experts shard 128-way fine (~10.7 GB/chip),
  but **DP-attention replicates the non-expert weights** (attention q/k/v/o sit on
  `ATTN_HEAD=(model,expert,dcp)=1` → replicated ~20 GB/chip) → no room → OOM.
- **Fix:** `ATTN_HEAD` and `ATTN_DATA_EXPERT` both contain the `expert` axis, so you
  can split EP into `attn_dp_expert * expert` — this shards attention heads on the
  shared `expert` axis **while keeping experts 128-way**. The overlay adds env
  `GLM_ATTN_EXPERT_SHARD=N` (in `layers/common/sharding.py`).
- **Winning config:** `SMOKE_TP=64 SMOKE_EP=128 GLM_ATTN_EXPERT_SHARD=32` →
  attention 32-way (2 heads/chip, ~0.8 GB), experts 128-way (~10.7 GB), ~13 GB
  weights/chip → pure BF16 fits with headroom. **No FP8 needed.**
- **Constraint:** `GLM_ATTN_EXPERT_SHARD` must be ≤ 32 (expert=64 fails: the KV
  cache axis of size 32 can't shard 64-way — *"should evenly divide"*).
- **vLLM head check:** `num_attention_heads(64) % tensor_parallel_size == 0`, so set
  `tensor_parallel_size=64`. The V2 executor places workers by
  `sharding_config.total_devices` (=128 via EP), **not** by TP.
- MLA **requires** `NEW_MODEL_DESIGN=1` + `enable_dp_attention:true` (tpu_platform
  validation). You can't drop DP-attention; the `tensor_parallelism:1 / attn_dp_size:1`
  trick satisfies both it and EP.

## KV cache auto-sizing is broken with attention-sharding

- With `GLM_ATTN_EXPERT_SHARD>1`, the KV cache shards only `attn_dp_expert`-way, but
  vLLM's auto block-sizing assumes wide sharding and **massively over-allocates**
  (saw 146–210 GB → OOM). **Cap it** with `num_gpu_blocks_override`
  (`GEN_KV_BLOCKS` for offline / `KVBLOCKS` for serve). Short prompts need very few
  blocks; MLA page_size is 1024, so e.g. `KVBLOCKS=256` = 262k token-slots.

## GLM kernels have no tuned params → default VMEM/SparseCore OOMs

All baked into `ray_node.sh` env / the overlay; listed so you know *why*:
- **MoE SparseCore gather:** `RAGGED_GATHER_VERSION=v1` + `RAGGED_GATHER_REDUCE_VERSION=v1`
  (v2 → `CompileTimeSparseCoreAllocationFailure`, SPMEM over by 1 word).
- **MLA decode kernel:** overlay `kernels/mla/v2/tuned_params.py` default fallback →
  `decode_batch_size=1, num_kv_pages_per_block=1` (stock 4/3 → `CompileTimeScopedVmemOom`;
  we serve `max_num_seqs=1` anyway).
- **MLA v-absorb transpose:** `MLA_XPOSE_N_TILE_SIZE=64` (stock 160 → scoped VMEM
  33.56M > 32M).

## Loader correctness (BF16 real weights)

- The MoE weight_loader must pass the **intended** sharding spec to `shard_put`; the
  `nnx.Param.sharding` attribute returns **None** there → `P()` → fully replicated
  256 experts/chip → 36 GB transient OOM + a slow post-hoc reshard. The overlay
  builds explicit `param_shardings` (gating/up = edf, down = edf/efd) → experts
  load **directly** sharded (~9 s/param vs ~6 min). This is in `moe.py`.
- MLA `kv_b_proj` is **BF16, not FP8** in this checkpoint → gate the BF16 split path
  on `not hasattr(self,"weight_scale_inv")` (NOT `quant_config is None` — vLLM hands
  you a non-None quant_config even when unquantized).
- `o_proj.weight` for MLA is **2D** `(N*v_head_dim, D)`; the generic loader branch
  assumed 3D per-head → crash. The overlay makes that branch ndim-aware
  (`weight_utils.py`).
- The model **drops** the indexer (`self_attn.indexer.*`) and any layer ≥ built
  count in `_filter_weights`. That's why the default BF16 serve ignores DSA tensors.

## Config / model dir

- Use our **scaffold `config.json`** (`model_type=deepseek_v3`,
  `architectures=["GlmMoeDsaForCausalLM"]`), NOT the HF one. GLM's real
  `model_type=glm_moe_dsa` won't load in stock transformers; `deepseek_v3` loads a
  Config class that has all the MLA fields, and `architectures` routes to our native
  flax `glm5.py`. All GLM dims are pinned in both the config and `glm5.py` globals.
- The flax loader **globs `*.safetensors`** — no `index.json` needed; shards can be
  symlinks. `lm_head` is in the **last** shard (00282), so the model can't load
  until all shards are present.
- Don't `cd /workspace` in the container: `/workspace/vllm` shadows the editable
  vllm as a namespace package (`vllm.__file__` becomes None →
  *"cannot import name SamplingParams"*). Run from `/tmp` with
  `PYTHONPATH=/workspace/vllm:/workspace/tpu_inference`.

## opencode driving the TPU backend (keeping requests small)

- opencode's default request was ~3500 tokens. To stay lean: (1) run in a clean dir
  with an **isolated `HOME`** so it doesn't inject `~/.claude/CLAUDE.md` + project
  CLAUDE.md; (2) a custom `tiny` agent with only `bash` tools (drops the big tool
  schemas); (3) short agent prompt; (4) `limit.context` / `output` caps.
- Tool/reasoning parser names in the vllm runtime registry are **glm45 / glm47**
  (NOT `glm5`, despite `--help`). Use `glm47` for both `--tool-call-parser` and
  `--reasoning-parser`. Tool-calling verified (model emits a valid `bash` tool_call).

## 1M context (experimental)

- Two gates, both in this kit: (1) `glm5.py` RoPE global
  `original_max_position_embeddings` was 8192 → bump to 1048576 (factor stays 1.0,
  so only the cache *size* grows; positions ≥ old max otherwise silently clamp).
  (2) `config.json` `max_position_embeddings` 8192 → 1048576 (vLLM caps
  `max_model_len` at the config). Use `config/config.1m.json` + serve with
  `MAXLEN=1048576 KVBLOCKS=1100`. KV at 1M is ~440 MB/chip (fits); prefill is ~512
  chunked steps (slow, silent until first token — may exceed an HTTP client timeout).
  Model is trained well below 1M, so treat quality as unverified.

## DSA (lightning indexer) — status, not in default serve

- Implemented and **numerically correct** (CPU exact: gather==mask, selection==brute
  force; real-model coherent ≤2048; fires with correct per-layer schedule at >2K, no
  OOM). On a separate branch (`glm-5.2-dsa`); toggled by `GLM_DSA` / `GLM_DSA_GATHER`.
- **Efficiency reality on TPU:** the FLOP reduction is real but **not realizable for
  prefill** with naive gather — TPU gather/scatter is slow and duplicates keys
  across query blocks (moved ~275 GB at 16K vs dense's 134 MB single read) → 100–800×
  *slower*. **DECODE** (batch=1) gather *is* faster: attention is ~constant ~0.11 ms
  at any context vs dense's linear growth → up to ~4.3× at 128K. Prefill speedup
  would need KV compression (DeepSeek-V4 style) or a custom block-sparse Pallas
  kernel. The default serve runs **full-dense** (DSA dropped) and is coherent >2K.
