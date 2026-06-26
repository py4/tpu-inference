"""GPU reference (HF transformers DeepseekV3) for the MoE parity check.

GLM-5.2 IS the DeepSeek-V3 topology (MLA + sigmoid/noaux_tc MoE); transformers
DeepseekV3 is the authoritative reference our flax glm5 + vLLM GlmMoeDsa both
target. Config fully specifies the MoE routing (scoring_func, topk_method,
n_group, norm_topk_prob, routed_scaling_factor), so HF reproduces it exactly.

Dumps the SAME data as the TPU run (glm5_compare.py) for a fixed input:
  gen_token, last_pos_top20 (log_softmax), prompt_argmax (per-position next-token).
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("REF_MODEL", "/mnt/lustre/glm5_real4_e64")
OUT = os.environ.get("REF_OUT", "/mnt/lustre/cmp_gpu_moe.json")
IDS = list(range(1, 17))

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

ids = torch.tensor([IDS], device="cuda")
with torch.no_grad():
    all_logits = model(ids).logits[0].float()  # [seq, vocab]

last = all_logits[-1]
lp = torch.log_softmax(last, dim=-1)
vals, idx = torch.topk(lp, 20)

# per-position next-token argmax (position i predicts token i+1), with its logprob
prompt_argmax = []
for i in range(all_logits.shape[0]):
    plp = torch.log_softmax(all_logits[i], dim=-1)
    t = int(plp.argmax().item())
    prompt_argmax.append([t, float(plp[t].item())])

res = {
    "platform": "gpu_hf",
    "input_ids": IDS,
    "gen_token": int(last.argmax().item()),
    "last_pos_top20": [[int(t), float(v)] for t, v in zip(idx.tolist(), vals.tolist())],
    "prompt_argmax": prompt_argmax,
}
json.dump(res, open(OUT, "w"), indent=2)
print("gen_token", res["gen_token"], "top5", res["last_pos_top20"][:5], flush=True)
print("REF_OK wrote", OUT, flush=True)
