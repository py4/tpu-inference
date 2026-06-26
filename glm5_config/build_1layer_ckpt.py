"""Build a self-consistent 1-layer GLM-5.2 checkpoint from the downloaded shards.

Extracts embed_tokens, lm_head, all model.layers.0.* EXCEPT the DSA indexer
params (we don't implement DSA yet), and model.norm, into a single
model.safetensors + index.json, alongside our num_hidden_layers=1 config.
vLLM needs a complete checkpoint matching the config, so we materialize exactly
the tensors a 1-layer model needs (kv_b_proj kept fused; the model splits it to
k_up/v_up at load).
"""
import glob
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

CACHE = "/workspace/hf"
OUT = "/workspace/glm5_real"
os.makedirs(OUT, exist_ok=True)


def find(name):
    hits = glob.glob(f"{CACHE}/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


s1 = find("model-00001-of-00282.safetensors")
s_last = find("model-00282-of-00282.safetensors")

tensors = {}
with safe_open(s1, framework="pt") as f:
    keys = list(f.keys())
    want = ["model.embed_tokens.weight", "lm_head.weight"]
    want += [
        k for k in keys
        if k.startswith("model.layers.0.") and ".indexer." not in k
    ]
    for k in want:
        tensors[k] = f.get_tensor(k)

with safe_open(s_last, framework="pt") as f:
    tensors["model.norm.weight"] = f.get_tensor("model.norm.weight")

save_file(tensors, f"{OUT}/model.safetensors", metadata={"format": "pt"})

total = sum(t.numel() * t.element_size() for t in tensors.values())
idx = {
    "metadata": {"total_size": total},
    "weight_map": {k: "model.safetensors" for k in tensors},
}
json.dump(idx, open(f"{OUT}/model.safetensors.index.json", "w"))

print(f"wrote {len(tensors)} tensors, {total/1e9:.2f} GB -> {OUT}/model.safetensors")
for k in sorted(tensors):
    print("  ", k, tuple(tensors[k].shape), tensors[k].dtype)
