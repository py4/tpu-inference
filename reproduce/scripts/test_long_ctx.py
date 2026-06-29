"""Test full-dense attention past 2K. Builds a needle-in-haystack prompt whose
token count exceeds 2048 (verified via usage.prompt_tokens in the response), with
a planted fact near the START and the question at the END. If full-dense
attention works past 2K, the model recalls the early needle; if attention breaks
beyond 2K, it fails. Also a plain long-continuation prompt to judge fluency.

Run on host 0 (serve binds 0.0.0.0:8000 via --net=host):  python3 test_long_ctx.py
"""
import json
import sys
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

# Filler: many distinct, mundane sentences so the model can't trivially compress.
TOPICS = [
    "The maintenance crew inspected the north corridor lighting on Tuesday.",
    "Quarterly inventory counts were reconciled against the warehouse ledger.",
    "A new batch of seedlings was moved to the eastern greenhouse for testing.",
    "The bus schedule on route 12 shifted by seven minutes during the holiday.",
    "Rainfall totals for the valley exceeded the seasonal average this spring.",
    "The library extended its weekend hours after a survey of local students.",
    "Technicians recalibrated the pressure sensors in the secondary pump room.",
    "The cafe introduced a rotating menu featuring regional autumn produce.",
    "Volunteers repainted the community center mural over two long weekends.",
    "The ferry terminal added a covered waiting area near the south dock.",
]

NEEDLE = ("IMPORTANT FACT: The access code for the riverside server cabinet is "
          "MAGENTA-7741, and it must only be used by the night-shift operator.")

def build_messages(target_tokens_hint=3000):
    parts = [
        "You are reading an operations logbook. Read it carefully; you will be "
        "asked one specific question at the end.\n\n",
        "--- LOGBOOK START ---\n",
        "Entry 1. " + NEEDLE + "\n",
    ]
    # ~ generate lots of filler entries
    n = 2
    for i in range(200):
        parts.append(f"Entry {n}. " + TOPICS[i % len(TOPICS)] + "\n")
        n += 1
    parts.append("--- LOGBOOK END ---\n\n")
    parts.append("QUESTION: What is the access code for the riverside server "
                 "cabinet, and who is allowed to use it? Answer in one sentence.")
    user = "".join(parts)
    return [{"role": "user", "content": user}]

def call(messages, max_tokens=120):
    body = {
        "model": "glm-5.2",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def main():
    msgs = build_messages()
    print("[test] sending needle-in-haystack prompt...", flush=True)
    resp = call(msgs)
    usage = resp.get("usage", {})
    pt = usage.get("prompt_tokens")
    out = resp["choices"][0]["message"].get("content", "")
    reasoning = resp["choices"][0]["message"].get("reasoning_content", "")
    print(f"[test] prompt_tokens = {pt}  (must be > 2048 to exercise >2K)",
          flush=True)
    print(f"[test] completion_tokens = {usage.get('completion_tokens')}",
          flush=True)
    if reasoning:
        print("[test] reasoning_content:", repr(reasoning[:400]), flush=True)
    print("[test] ANSWER:", repr(out), flush=True)
    ok_len = (pt or 0) > 2048
    ok_needle = ("MAGENTA-7741" in out) or ("MAGENTA-7741" in reasoning)
    print(f"[test] >2K context: {'YES' if ok_len else 'NO'} | "
          f"needle recalled: {'YES' if ok_needle else 'NO'}", flush=True)
    if not ok_len:
        print("[test] WARNING: prompt did not exceed 2048; increase filler.",
              flush=True)

if __name__ == "__main__":
    main()
