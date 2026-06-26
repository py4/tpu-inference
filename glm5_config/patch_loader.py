"""Idempotent in-place registration of GlmMoeDsaForCausalLM into the image's
tpu_inference model_loader.py (which differs from our local clone, so we patch
rather than overwrite). Adds the inline import + _MODEL_REGISTRY entry only.
The _PP_DISABLED_MODELS edit is intentionally skipped (only matters for PP>1)."""
import sys

F = "/usr/local/lib/python3.12/site-packages/tpu_inference/models/common/model_loader.py"
s = open(F).read()

imp_anchor = "    from tpu_inference.models.jax.deepseek_v3 import DeepseekV3ForCausalLM\n"
imp_new = "    from tpu_inference.models.jax.glm5 import GlmMoeDsaForCausalLM\n"
reg_anchor = '    _MODEL_REGISTRY["DeepseekV3ForCausalLM"] = DeepseekV3ForCausalLM\n'
reg_new = '    _MODEL_REGISTRY["GlmMoeDsaForCausalLM"] = GlmMoeDsaForCausalLM\n'

changed = False
if "glm5 import GlmMoeDsaForCausalLM" not in s:
    if imp_anchor not in s:
        sys.exit("ERROR: import anchor not found; image model_loader.py changed")
    s = s.replace(imp_anchor, imp_anchor + imp_new, 1)
    changed = True
if 'GlmMoeDsaForCausalLM"] = GlmMoeDsaForCausalLM' not in s:
    if reg_anchor not in s:
        sys.exit("ERROR: registry anchor not found; image model_loader.py changed")
    s = s.replace(reg_anchor, reg_anchor + reg_new, 1)
    changed = True

open(F, "w").write(s)
print("patched" if changed else "already patched")
