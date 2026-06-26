"""Truncate the 4-layer GLM-5.2 ckpt from 256 routed experts -> N (default 64),
so the MoE layer (layer 3) fits v6e HBM for a TPU-vs-GPU numerical parity test.

The MoE math (sigmoid router, noaux_tc top-k, GMM experts, shared expert,
routed_scaling) is identical for any expert count, so a reduced-expert match
verifies the implementation. Keeps experts 0..N-1, truncates the router gate
[256,H]->[N,H] and e_score_correction_bias [256]->[N]. Drops nothing else.
Both TPU and GPU must use this SAME ckpt for the comparison to be valid.

Env: SRC (in dir), DST (out dir), NEXP (kept experts, default 64), MOE_LAYER (3).
"""
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ.get("SRC", "/dev/shm/glm5_real4")
DST = os.environ.get("DST", "/dev/shm/glm5_real4_e64")
NEXP = int(os.environ.get("NEXP", "64"))
MOE_LAYER = int(os.environ.get("MOE_LAYER", "3"))
os.makedirs(DST, exist_ok=True)

# expert weight names look like: model.layers.3.mlp.experts.<idx>.<proj>.weight
exp_re = re.compile(rf"^model\.layers\.{MOE_LAYER}\.mlp\.experts\.(\d+)\.")
gate_name = f"model.layers.{MOE_LAYER}.mlp.gate.weight"
bias_name = f"model.layers.{MOE_LAYER}.mlp.gate.e_score_correction_bias"

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
shard = sorted(set(idx.values()))[0]  # build_4layer wrote a single shard
out = {}
dropped = 0
with safe_open(f"{SRC}/{shard}", framework="pt") as f:
    for k in f.keys():
        m = exp_re.match(k)
        if m:
            if int(m.group(1)) >= NEXP:
                dropped += 1
                continue
            out[k] = f.get_tensor(k)
        elif k == gate_name:
            t = f.get_tensor(k)
            out[k] = t[:NEXP].contiguous()
            print(f"gate {tuple(t.shape)} -> {tuple(out[k].shape)}", flush=True)
        elif k == bias_name:
            t = f.get_tensor(k)
            out[k] = t[:NEXP].contiguous()
            print(f"bias {tuple(t.shape)} -> {tuple(out[k].shape)}", flush=True)
        else:
            out[k] = f.get_tensor(k)

save_file(out, f"{DST}/model.safetensors", metadata={"format": "pt"})
total = sum(t.numel() * t.element_size() for t in out.values())
json.dump({"metadata": {"total_size": total},
           "weight_map": {k: "model.safetensors" for k in out}},
          open(f"{DST}/model.safetensors.index.json", "w"))

# config: set n_routed_experts = NEXP
cfg = json.load(open(f"{SRC}/config.json"))
for key in ("n_routed_experts", "num_experts", "num_local_experts"):
    if key in cfg:
        cfg[key] = NEXP
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
print(f"DONE kept {NEXP} experts, dropped {dropped} expert tensors, "
      f"{total/1e9:.2f} GB -> {DST}; n_routed_experts={cfg.get('n_routed_experts')}",
      flush=True)
