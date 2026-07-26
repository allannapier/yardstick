#!/usr/bin/env bash
# Drives one full ys start -> mock turns -> ys end cycle for testing compare/report.
set -euo pipefail
ARM="$1"
N_TURNS="${2:-2}"
PORT="${3:-4020}"

cd /home/allan/code/measure
source .venv/bin/activate

ys start --exp experiments/example.yaml --arm "$ARM"
MK="$LITELLM_MASTER_KEY"

MSGS='[{"role":"user","content":"turn 1"}]'
for i in $(seq 1 "$N_TURNS"); do
  curl -sf "http://localhost:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $MK" -H "Content-Type: application/json" \
    -d "{\"model\":\"probe-claude-mock\",\"max_tokens\":50,\"system\":\"You are a coding agent.\",\"messages\":$MSGS}" > /dev/null
  MSGS=$(python3 -c "import json,sys; m=json.loads(sys.argv[1]); m.append({'role':'assistant','content':'ok'}); m.append({'role':'user','content':'turn'}); print(json.dumps(m))" "$MSGS")
done

ys end
