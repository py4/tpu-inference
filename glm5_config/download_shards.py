"""Download only the GLM-5.2 shards needed for a 1-layer checkpoint
(layer 0 + embed + lm_head are in shard 00001; model.norm in 00282).
Set HF_HOME=/workspace/hf. HF_TOKEN is read from env."""
from huggingface_hub import hf_hub_download

for f in [
    "model.safetensors.index.json",
    "config.json",
    "model-00001-of-00282.safetensors",
    "model-00282-of-00282.safetensors",
]:
    p = hf_hub_download("zai-org/GLM-5.2", f)
    print("OK", f)
print("DL_DONE")
