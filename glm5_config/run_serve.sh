#!/bin/bash
# Runs INSIDE the head 'node' container. Starts an OpenAI-compatible vLLM server
# for GLM-5.2 (pure BF16, attention-head-sharded config) on port 8000.
# max_model_len=2048 (DSA stays a no-op at <=2048). SKIP_JAX_PRECOMPILE=1 so the
# server is up right after weight load; shapes compile lazily on first request.
set -uo pipefail
LOG="${LOG:-/mnt/glm5fs/glm5_serve.log}"
: > "$LOG"
cd /tmp
setsid env \
  PYTHONPATH=/workspace/vllm:/workspace/tpu_inference \
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=flax_nnx TPU_MULTIHOST_BACKEND=ray \
  GLM_NUM_EXPERTS=256 GLM_ATTN_EXPERT_SHARD=32 \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 MLA_XPOSE_N_TILE_SIZE=64 \
  JAX_PLATFORMS= PYTHONUNBUFFERED=1 \
  vllm serve /mnt/glm5fs/glm5_model \
    --served-model-name glm-5.2 \
    --dtype bfloat16 --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --tensor-parallel-size 64 \
    --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":128,"tensor_parallelism":1,"attn_dp_size":1}}}' \
    --num-gpu-blocks-override 256 \
    --gpu-memory-utilization 0.85 \
    --chat-template /mnt/glm5fs/glm5_model/chat_template.jinja \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
    --trust-remote-code --no-enable-prefix-caching \
    --host 0.0.0.0 --port 8000 >"$LOG" 2>&1 &
echo "serve launched pid=$! log=$LOG"
