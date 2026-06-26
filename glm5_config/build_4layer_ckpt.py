"""Build a 4-layer GLM-5.2 checkpoint (layers 0-2 dense + layer 3 = first MoE)
to verify the MoE path. Incremental: download one shard at a time into a temp
dir, extract only the needed tensors, delete the shard (keeps peak disk low).
Excludes DSA indexer params. kv_b_proj kept fused (model splits to k_up/v_up).
"""
import gc
import json
import os

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "zai-org/GLM-5.2"
OUT = os.environ.get("OUT_DIR", "/dev/shm/glm5_real4")  # RAM-backed, avoids overlay disk eviction
TMP = os.environ.get("TMP_DIR", "/dev/shm/dl_tmp")
LAYERS = (0, 1, 2, 3)
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)


def need(k):
    if ".indexer." in k:
        return False
    return (k.startswith("model.embed_tokens.") or k == "model.norm.weight"
            or k == "lm_head.weight"
            or any(k.startswith(f"model.layers.{i}.") for i in LAYERS))


ip = hf_hub_download(REPO, "model.safetensors.index.json")
wm = json.load(open(ip))["weight_map"]
wanted = [k for k in wm if need(k)]
byshard = {}
for k in wanted:
    byshard.setdefault(wm[k], []).append(k)
print(f"need {len(wanted)} tensors across {len(byshard)} shards", flush=True)

tensors = {}
for i, (sh, keys) in enumerate(sorted(byshard.items())):
    p = hf_hub_download(REPO, sh, local_dir=TMP, local_dir_use_symlinks=False)
    with safe_open(p, framework="pt") as f:
        for k in keys:
            tensors[k] = f.get_tensor(k)
    os.remove(p)
    print(f"[{i+1}/{len(byshard)}] {sh}: extracted {len(keys)}, total {len(tensors)}",
          flush=True)
    gc.collect()

save_file(tensors, f"{OUT}/model.safetensors", metadata={"format": "pt"})
total = sum(t.numel() * t.element_size() for t in tensors.values())
json.dump({"metadata": {"total_size": total},
           "weight_map": {k: "model.safetensors" for k in tensors}},
          open(f"{OUT}/model.safetensors.index.json", "w"))
print(f"DONE wrote {len(tensors)} tensors, {total/1e9:.2f} GB -> {OUT}", flush=True)
