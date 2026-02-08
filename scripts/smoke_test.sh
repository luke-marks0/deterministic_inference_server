#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example to .env first." >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

PORT="${VLLM_PORT:-8000}"
MODEL="${SERVED_MODEL_NAME:-qwen3-235b-a22b-instruct-2507}"
URL="http://127.0.0.1:${PORT}/v1/completions"

response="$(
  curl -fsS "${URL}" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"prompt\": \"Write exactly one sentence that says the server is healthy.\",
      \"max_tokens\": 32,
      \"temperature\": 0.0,
      \"seed\": ${VLLM_SEED:-424242}
    }"
)"

python3 - <<'PY' "${response}"
import json
import sys

payload = json.loads(sys.argv[1])
text = payload["choices"][0]["text"].strip()
tokens = payload.get("usage", {}).get("completion_tokens", "unknown")
print("Smoke test response:", text)
print("Completion tokens:", tokens)
PY
