#!/bin/bash
MSG="$*"
curl -s http://127.0.0.1:11437/chat \
  -H "Content-Type: application/json" \
  -H "X-Token: REMOVED-TOKEN" \
  -d "{\"message\": \"$MSG\"}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'[{d[\"model\"]}]\n{d[\"answer\"]}')"
