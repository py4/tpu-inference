#!/bin/bash
echo "=== /v1/models ==="; curl -s -m10 localhost:8000/v1/models | python3 -c "import sys,json;print([m['id'] for m in json.load(sys.stdin)['data']])" 2>&1
echo "=== simple chat (warms compile; may take minutes on first call) ==="
time curl -s -m 900 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"glm-5.2","max_tokens":40,"temperature":0,
  "messages":[{"role":"user","content":"Reply with exactly: hello from tpu"}]}' | python3 -c "import sys,json;d=json.load(sys.stdin);print('CONTENT:',repr(d['choices'][0]['message'].get('content')));print('usage:',d.get('usage'))" 2>&1
echo "=== tool-calling chat (does glm5 parser produce tool_calls?) ==="
curl -s -m 900 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"glm-5.2","max_tokens":120,"temperature":0,
  "messages":[{"role":"user","content":"Create a file named hi.txt containing the word hi. Use the bash tool."}],
  "tools":[{"type":"function","function":{"name":"bash","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"the command"}},"required":["command"]}}}],
  "tool_choice":"auto"}' | python3 -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('CONTENT:',repr(m.get('content')));print('TOOL_CALLS:',json.dumps(m.get('tool_calls'),indent=1));print('usage:',d.get('usage'))" 2>&1
