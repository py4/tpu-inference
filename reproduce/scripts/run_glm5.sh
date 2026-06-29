#!/bin/bash
# Runs INSIDE the head 'node' container. Launches glm5 generation across the Ray
# cluster (detached via setsid so it survives the docker-exec disconnect).
set -uo pipefail
MODEL="${GEN_MODEL:-/mnt/glm5fs/glm5_model}"
EXPERTS="${GLM_NUM_EXPERTS:-256}"
TP="${SMOKE_TP:-128}"
EP="${SMOKE_EP:-128}"
MAXTOK="${GEN_MAXTOK:-48}"
UTIL="${GEN_GPU_UTIL:-0.92}"
LOG="${LOG:-/mnt/glm5fs/glm5_run.log}"
EXTRA_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-}"
# Path to glm5_generate.py (this kit ships it next to this script). Override with
# GEN_SCRIPT if you keep it elsewhere.
GEN_SCRIPT="${GEN_SCRIPT:-$(cd "$(dirname "$0")" && pwd)/glm5_generate.py}"

: > "$LOG"
# IMPORTANT: do NOT cd /workspace -- /workspace/vllm shadows the editable vllm
# package as a namespace package (vllm.__file__ becomes None). Run from /tmp and
# rely on PYTHONPATH (set on the container) for editable package resolution.
cd /tmp
setsid env \
  PYTHONPATH=/workspace/vllm:/workspace/tpu_inference \
  NEW_MODEL_DESIGN=1 \
  MODEL_IMPL_TYPE=flax_nnx \
  TPU_MULTIHOST_BACKEND=ray \
  GLM_NUM_EXPERTS="$EXPERTS" \
  GLM_ATTN_EXPERT_SHARD="${GLM_ATTN_EXPERT_SHARD:-1}" \
  SMOKE_TP="$TP" SMOKE_EP="$EP" \
  GEN_MODEL="$MODEL" GEN_MAXTOK="$MAXTOK" GEN_GPU_UTIL="$UTIL" \
  ${EXTRA_DTYPE:+MOE_REQUANTIZE_WEIGHT_DTYPE=$EXTRA_DTYPE} \
  PYTHONUNBUFFERED=1 \
  python3 "$GEN_SCRIPT" >"$LOG" 2>&1 </dev/null &
echo "launched pid=$! log=$LOG model=$MODEL TP=$TP EP=$EP experts=$EXPERTS"
