"""Direct 1M-context test against the serve (verify the model runs + is coherent
at ~1M tokens before opencode). Builds a needle-in-haystack of ~1.04M tokens via
the real tokenizer (truncated exactly, so no 'exceeds max' overshoot), sends it
to /v1/completions as prompt_token_ids, times the prefill, checks needle recall.

Run inside the container: python3 /mnt/glm5fs/test_1m.py
"""
import json
import time
import urllib.request

from transformers import AutoTokenizer

MODEL_DIR = "/mnt/glm5fs/glm5_model"
URL = "http://localhost:8000/v1/completions"
TARGET_TOKENS = 1_040_000          # < max_model_len (1048576), leave room to gen

NEEDLE = "IMPORTANT FACT: the vault passcode is CRIMSON-3382.\n"
FILLER = ("The facility logbook records routine status checks. Each line notes "
          "the time, the operator on duty, and the measured result for the shift. ")


def main():
    print("[1m] loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # Build ~4MB text: needle near the START, then filler, then the question.
    text = ("Read this logbook, then answer the final question.\n" + NEEDLE
            + FILLER * 60000
            + "\nQUESTION: What is the vault passcode? Answer:")
    ids = tok(text).input_ids
    print(f"[1m] raw prompt tokens={len(ids)}", flush=True)
    if len(ids) < TARGET_TOKENS:
        # pad by repeating filler region if needed (rare)
        extra = tok(FILLER * 20000).input_ids
        while len(ids) < TARGET_TOKENS:
            ids = ids[:1] + extra + ids[1:]
    # keep the needle (near front) + the tail question; truncate the middle.
    head = ids[:TARGET_TOKENS - 40]
    tail = tok("\nQUESTION: What is the vault passcode? Answer:").input_ids
    ids = head + tail
    print(f"[1m] final prompt tokens={len(ids)} (max_model_len=1048576)", flush=True)

    body = {"model": "glm-5.2", "prompt": ids, "max_tokens": 16,
            "temperature": 0.0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    print("[1m] sending 1M-token prefill (this can take many minutes)...",
          flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=5400) as r:  # up to 90 min
        resp = json.loads(r.read())
    dt = time.time() - t0
    out = resp["choices"][0]["text"]
    usage = resp.get("usage", {})
    print(f"[1m] DONE in {dt:.1f}s  prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')}", flush=True)
    print(f"[1m] OUTPUT={out!r}", flush=True)
    print(f"[1m] needle recalled: {'YES' if 'CRIMSON-3382' in out else 'NO'}",
          flush=True)
    print("[1m] TEST_1M_OK", flush=True)


if __name__ == "__main__":
    main()
