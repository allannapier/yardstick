#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MASTER_KEY="$(cat .master_key)"

curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -H "x-ys-run: ${YS_RUN_ID:-probe-run-mock-001}" \
  -d '{
    "model": "probe-claude-mock",
    "max_tokens": 200,
    "system": "You are a coding agent. Use tools when helpful.",
    "messages": [
      {"role": "user", "content": "List the files in the current directory."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "list_files",
          "description": "List files in a directory",
          "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
          }
        }
      }
    ]
  }' | python3 -m json.tool
