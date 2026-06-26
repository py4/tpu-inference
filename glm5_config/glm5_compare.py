"""Dump layer-0 logits parity data for a FIXED input, so TPU (our flax glm5) and
the GPU vLLM reference (GlmMoeDsa) can be compared numerically.

Same 1-layer dense-MLA checkpoint + same input ids on both sides. Writes JSON:
  { platform, input_ids, gen_token, last_pos_top20: [[tok, logprob]...],
    prompt_argmax: [[tok, logprob] per position] }

Env:
  PLATFORM   tpu|gpu   (tpu adds DP-attention additional_config; default tpu)
  SMOKE_MODEL  checkpoint dir (default /workspace/glm5_real)
  SMOKE_TP     tensor_parallel_size (default 64 tpu / 1 gpu)
  CMP_OUT      output json path
"""
import json
import os
import sys
import traceback

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

PLATFORM = os.environ.get("PLATFORM", "tpu")
MODEL = os.environ.get("SMOKE_MODEL", "/workspace/glm5_real")
TP = int(os.environ.get("SMOKE_TP", "64" if PLATFORM == "tpu" else "1"))
OUT = os.environ.get("CMP_OUT", f"/workspace/cmp_{PLATFORM}.json")
IDS = list(range(1, 17))

kw = dict(model=MODEL, skip_tokenizer_init=True, dtype="bfloat16",
          max_model_len=128, max_num_seqs=4, tensor_parallel_size=TP,
          enforce_eager=(PLATFORM == "gpu"))
# Stream weights from object storage (gs://...) instead of a local dir.
LOAD_FORMAT = os.environ.get("SMOKE_LOAD_FORMAT", "")
if LOAD_FORMAT:
    kw["load_format"] = LOAD_FORMAT
if PLATFORM == "tpu":
    # MLA requires enable_dp_attention (tpu_platform.py validation). For MoE we
    # also need GMM_EP (experts sharded on the expert axis) to avoid the GMM_TP
    # 51.5GB unsharded reorder transient -> HBM OOM. GMM_EP needs total_tp==1, so
    # force tensor_parallelism=1 + attn_dp_size=1 and put all 64 chips on experts:
    # -> mesh (1,1,attn_dp_expert=64,1,1,1), use_ep=True.
    ss = {"enable_dp_attention": True}
    ep = int(os.environ.get("SMOKE_EP", "1"))
    if ep > 1:
        ss["expert_parallelism"] = ep
        ss["tensor_parallelism"] = 1
        ss["attn_dp_size"] = 1
    kw["additional_config"] = {"sharding": {"sharding_strategy": ss}}

print(f"[cmp] PLATFORM={PLATFORM} MODEL={MODEL} TP={TP} OUT={OUT}", flush=True)
try:
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20,
                        prompt_logprobs=20)
    outs = llm.generate([TokensPrompt(prompt_token_ids=IDS)], sp)
    o = outs[0]
    comp = o.outputs[0]
    gen_tok = int(comp.token_ids[0])

    top = []
    if comp.logprobs:
        lp = comp.logprobs[0]
        top = sorted(((int(t), float(l.logprob)) for t, l in lp.items()),
                     key=lambda x: -x[1])

    prompt_argmax = []
    if o.prompt_logprobs:
        for pos in o.prompt_logprobs:
            if not pos:
                prompt_argmax.append(None)
                continue
            best = max(pos.items(), key=lambda kv: kv[1].logprob)
            prompt_argmax.append([int(best[0]), float(best[1].logprob)])

    res = {"platform": PLATFORM, "input_ids": IDS, "gen_token": gen_tok,
           "last_pos_top20": top, "prompt_argmax": prompt_argmax}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[cmp] gen_token={gen_tok} top5={top[:5]}", flush=True)
    print(f"[cmp] CMP_OK wrote {OUT}", flush=True)
except Exception:
    print("[cmp] CMP_FAILED", flush=True)
    traceback.print_exc()
    sys.exit(1)
