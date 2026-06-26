"""GLM-5.2 Step-1 smoke: build the native-Flax model with dummy BF16 weights and run
one forward/decode on TPU v6e-64 (pathways). No tokenizer, no checkpoint.

Env knobs:
  SMOKE_TP        tensor_parallel_size (default 64 = full v6e-64)
  SMOKE_NLAYERS   override num_hidden_layers via hf_overrides (M1=unset->1, M2=4)
  USE_DENSE_MOE=1 REQUIRED (selects DENSE_MAT so dummy-weights check passes)
"""
import os
import sys
import traceback

# Diagnostics: dump all thread stacks to stderr (the log) on SIGUSR1, and
# automatically every 90s, so we can locate any hang without py-spy/gdb (no
# ptrace cap on the pod).
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1, all_threads=True)
faulthandler.dump_traceback_later(30, repeat=True)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

TP = int(os.environ.get("SMOKE_TP", "64"))
NL = os.environ.get("SMOKE_NLAYERS")
hf_overrides = {"num_hidden_layers": int(NL)} if NL else {}
MODEL = os.environ.get("SMOKE_MODEL", "/workspace/glm5_config")
DUMMY = os.environ.get("SMOKE_DUMMY", "1") == "1"
LOAD_FORMAT = "dummy" if DUMMY else "auto"

print(f"[smoke] MODEL={MODEL} DUMMY={DUMMY} load_format={LOAD_FORMAT} "
      f"USE_DENSE_MOE={os.environ.get('USE_DENSE_MOE')} TP={TP} "
      f"hf_overrides={hf_overrides}", flush=True)

try:
    llm = LLM(
        model=MODEL,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        load_format=LOAD_FORMAT,
        max_model_len=128,
        max_num_seqs=4,
        tensor_parallel_size=TP,
        hf_overrides=hf_overrides or None,
        # MLA on tpu-inference requires NEW_MODEL_DESIGN=1 (env) + DP attention.
        additional_config={
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": True
                }
            }
        },
    )
    sp = SamplingParams(max_tokens=4, temperature=0.0)
    prompts = [TokensPrompt(prompt_token_ids=list(range(1, 17)))]
    outs = llm.generate(prompts, sp)
    toks = list(outs[0].outputs[0].token_ids)
    print(f"[smoke] generated token ids: {toks}", flush=True)
    assert len(toks) > 0, "no tokens generated"
    print("[smoke] SMOKE_OK", flush=True)
except Exception:
    print("[smoke] SMOKE_FAILED", flush=True)
    traceback.print_exc()
    sys.exit(1)
