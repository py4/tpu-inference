#!/bin/bash
# Build the local "model dir" that the flax_nnx loader points at.
#
# The model dir is just: a config.json (our deepseek_v3-SCAFFOLD config, NOT the
# HF one), tokenizer files, chat template, generation config, and the 282 BF16
# safetensors shards. The flax loader globs *.safetensors, so the shards can be
# SYMLINKS to a read-only weights volume (what we did) or real files.
#
# Why a scaffold config.json (model_type=deepseek_v3) instead of the HF one?
#   GLM-5.2's real model_type is `glm_moe_dsa`, which stock transformers cannot
#   load. We point `architectures=["GlmMoeDsaForCausalLM"]` (routes to our native
#   flax glm5.py) and `model_type=deepseek_v3` (so a stock transformers Config
#   class loads with all the MLA fields). All GLM dims are pinned in the config
#   AND in glm5.py module globals. See config/config.8192.json.
#
# Usage:
#   WEIGHTS=/mnt/weights/glm5_full \    # where download_weights.py put the shards
#   MODELDIR=/mnt/glm5fs/glm5_model \   # output dir (readable by all hosts)
#   REPRO=/mnt/glm5fs/reproduce \       # this reproduce/ dir
#   CTX=8192 \                          # 8192 (proven) or 1m (experimental)
#   bash build_model_dir.sh
set -euo pipefail
WEIGHTS="${WEIGHTS:-/mnt/weights/glm5_full}"
MODELDIR="${MODELDIR:-/mnt/glm5fs/glm5_model}"
REPRO="${REPRO:-$(cd "$(dirname "$0")/.." && pwd)}"
CTX="${CTX:-8192}"

mkdir -p "$MODELDIR"

# 1) config.json — our scaffold (NOT HF's). Pick 8192 (proven) or 1m.
if [ "$CTX" = "1m" ]; then
  cp "$REPRO/config/config.1m.json" "$MODELDIR/config.json"
else
  cp "$REPRO/config/config.8192.json" "$MODELDIR/config.json"
fi

# 2) tokenizer / generation / chat template
cp "$REPRO/config/generation_config.json" "$MODELDIR/"
cp "$REPRO/config/chat_template.jinja"     "$MODELDIR/"
# tokenizer.json + tokenizer_config.json come from the HF download:
cp "$WEIGHTS/tokenizer.json"        "$MODELDIR/" 2>/dev/null || \
  echo "WARN: tokenizer.json not in $WEIGHTS — copy it from the HF repo"
cp "$WEIGHTS/tokenizer_config.json" "$MODELDIR/" 2>/dev/null || true

# 3) the 282 shards — SYMLINK them in (no copy; weights volume is read-only)
n=0
for f in "$WEIGHTS"/model-*-of-*.safetensors; do
  [ -e "$f" ] || { echo "ERROR: no shards found in $WEIGHTS"; exit 1; }
  ln -sf "$f" "$MODELDIR/$(basename "$f")"
  n=$((n+1))
done

echo "built $MODELDIR : config(ctx=$CTX) + tokenizer + chat_template + $n shards"
echo "sanity: ls $MODELDIR | head ; expect 282 safetensors symlinks"
