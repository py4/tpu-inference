"""Patch the image's pathways_dummy_loader.py. Two idempotent changes:

(1) PATCH-GLM5-DUMMY-1: skip the redundant slow device_put. create_dummy_weights_on_tpu()
    already makes each array on-device with the target sharding via
    jax.jit(out_shardings=sharding); assign_and_shard_param() then does ANOTHER
    jax.device_put -> slow host round-trip that broadcasts huge replicated params
    (model=1 mesh) -> Pathways timeout. Assign the already-sharded array directly.

(2) PATCH-GLM5-DUMMY-2: initialize model caches (e.g. RoPE sin/cos) that the real
    load_weights path sets up via model.initialize_cache(). The dummy loader bypasses
    the model's load_weights, so without this the forward hits
    "AssertionError: RoPE cache not initialized." Call initialize_cache() like the real path.
"""
import sys

F = "/usr/local/lib/python3.12/site-packages/tpu_inference/models/common/pathways_dummy_loader.py"
s = open(F).read()
changed = False

# (1) skip redundant device_put
anchor1 = "        assign_and_shard_param(param, dummy, param_name)\n"
repl1 = (
    "        # PATCH-GLM5-DUMMY-1: dummy already has target sharding; skip the slow\n"
    "        # device_put inside assign_and_shard_param (host broadcast -> timeout).\n"
    "        param.set_value(dummy)\n"
    "        param.set_metadata(\"_is_loaded\", True)\n"
)
if "PATCH-GLM5-DUMMY-1" not in s:
    if anchor1 not in s:
        sys.exit("ERROR: anchor1 (assign_and_shard_param) not found")
    s = s.replace(anchor1, repl1, 1)
    changed = True

# (2) initialize RoPE/other caches the dummy path otherwise skips
anchor2 = "    _process_weights_after_loading_jax(model)\n"
repl2 = (
    anchor2 +
    "    # PATCH-GLM5-DUMMY-2: real load_weights calls model.initialize_cache()\n"
    "    # (builds RoPE sin/cos); the dummy path skips it. Do it here.\n"
    "    _ic = getattr(model, \"initialize_cache\", None) or getattr(\n"
    "        getattr(model, \"model\", None), \"initialize_cache\", None)\n"
    "    if callable(_ic):\n"
    "        _ic()\n"
)
if "PATCH-GLM5-DUMMY-2" not in s:
    if anchor2 not in s:
        sys.exit("ERROR: anchor2 (_process_weights_after_loading_jax) not found")
    s = s.replace(anchor2, repl2, 1)
    changed = True

open(F, "w").write(s)
print("patched dummy loader" if changed else "already patched (both changes)")
