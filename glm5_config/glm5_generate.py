"""Generate text from the FULL GLM-5.2 on TPU with the REAL tokenizer, to check
the output is coherent English. Uses the same GMM_EP sharding config as the
parity runs. Model dir is a local path (PVC) with config.json (architectures
GlmMoeDsa, model_type deepseek_v3, num_hidden_layers 78) + tokenizer files.

Env: GEN_MODEL (/mnt/ckpt/glm5), SMOKE_TP (64), SMOKE_EP (64),
GLM_NUM_EXPERTS (256), GEN_MAXTOK (48).
"""
import os
import sys
import traceback

from vllm import LLM, SamplingParams

MODEL = os.environ.get("GEN_MODEL", "/mnt/ckpt/glm5")
TP = int(os.environ.get("SMOKE_TP", "64"))
MAXTOK = int(os.environ.get("GEN_MAXTOK", "48"))

# Conservative memory: full model is ~23G/chip of weights, so keep KV+activations
# small (short seq, single sequence) to avoid HBM OOM during the forward/profiling.
kw = dict(model=MODEL, dtype="bfloat16", max_model_len=128, max_num_seqs=1,
          max_num_batched_tokens=256,
          gpu_memory_utilization=float(os.environ.get("GEN_GPU_UTIL", "0.92")),
          tensor_parallel_size=TP, trust_remote_code=True)
# When attention heads are sharded on the 'expert' axis (GLM_ATTN_EXPERT_SHARD>1),
# the KV cache shards only attn_dp_expert-way, and vllm's auto block-sizing (which
# assumes wide sharding) massively over-allocates -> 146G OOM. Cap the KV block
# count for short-prompt generation. GEN_KV_BLOCKS pages is plenty for max_len=128.
_kvb = int(os.environ.get("GEN_KV_BLOCKS", "0"))
if _kvb > 0:
    kw["num_gpu_blocks_override"] = _kvb
# MLA + GMM_EP sharding (same as parity runs).
ss = {"enable_dp_attention": True}
ep = int(os.environ.get("SMOKE_EP", "1"))
if ep > 1:
    ss.update({"expert_parallelism": ep, "tensor_parallelism": 1, "attn_dp_size": 1})
kw["additional_config"] = {"sharding": {"sharding_strategy": ss}}

PROMPTS = [
    "The capital of France is",
    "Here is a short poem about the ocean:\n",
    "Question: What is the largest planet in the solar system?\nAnswer:",
    "Once upon a time, in a small village,",
]

print(f"[gen] MODEL={MODEL} TP={TP} EP={ep} maxtok={MAXTOK}", flush=True)
try:
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=MAXTOK, temperature=0.0)
    outs = llm.generate(PROMPTS, sp)
    print("\n========== GENERATIONS ==========", flush=True)
    for o in outs:
        print("PROMPT:", repr(o.prompt), flush=True)
        print("OUTPUT:", repr(o.outputs[0].text), flush=True)
        print("-" * 60, flush=True)
    print("[gen] GEN_OK", flush=True)
except Exception:
    print("[gen] GEN_FAILED", flush=True)
    traceback.print_exc()
    sys.exit(1)
