#!/bin/bash
# Runs INSIDE the head 'node' container. Starts an OpenAI-compatible vLLM server
# for GLM-5.2 (pure BF16, attention-head-sharded config) on port 8000.
# Env: MODELDIR (model dir), MAXLEN (default 2048; use 8192 for >2K), MAXBATCH,
# KVBLOCKS, LOG.
# Fast dev iteration with RANDOM weights (skip the ~60 min disk read; output is
# gibberish, use only to validate the pipeline runs / shapes / OOM / scheduling):
#   LOADFMT=dummy ALLOWDUMMYMOE=1 ... bash run_serve.sh    # see reproduce/RANDOM_WEIGHTS.md
# NOTE: do NOT set SKIP_JAX_PRECOMPILE for the server — it is a driver-only env, so
# the driver/worker mismatch hangs warmup. Let the full bucket sweep run (~40 min).
set -uo pipefail
MODELDIR="${MODELDIR:-/mnt/glm5fs/glm5_model}"
LOG="${LOG:-/mnt/glm5fs/glm5_serve.log}"
: > "$LOG"
cd /tmp
setsid env \
  PYTHONPATH=/workspace/vllm:/workspace/tpu_inference \
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=flax_nnx TPU_MULTIHOST_BACKEND=ray \
  GLM_NUM_EXPERTS=256 GLM_ATTN_EXPERT_SHARD=32 \
  GLM_ALLOW_DUMMY_MOE="${ALLOWDUMMYMOE:-0}" \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 MLA_XPOSE_N_TILE_SIZE=64 \
  JAX_PLATFORMS= PYTHONUNBUFFERED=1 \
  vllm serve "$MODELDIR" \
    --served-model-name glm-5.2 \
    --dtype bfloat16 --max-model-len "${MAXLEN:-2048}" --max-num-seqs 1 --max-num-batched-tokens "${MAXBATCH:-2048}" \
    --load-format "${LOADFMT:-auto}" \
    --tensor-parallel-size 64 \
    --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":128,"tensor_parallelism":1,"attn_dp_size":1}}}' \
    --num-gpu-blocks-override "${KVBLOCKS:-256}" \
    --gpu-memory-utilization 0.85 \
    --chat-template "$MODELDIR/chat_template.jinja" \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
    --trust-remote-code --no-enable-prefix-caching \
    --host 0.0.0.0 --port 8000 >"$LOG" 2>&1 &
echo "serve launched pid=$! log=$LOG"
