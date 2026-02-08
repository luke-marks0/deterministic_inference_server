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
URL="http://127.0.0.1:${PORT}/v1/models"
TIMEOUT_SECONDS="${1:-7200}"

start_epoch="$(date +%s)"
echo "Waiting for ${URL} (timeout: ${TIMEOUT_SECONDS}s)"

while true; do
  if curl -fsS "${URL}" >/dev/null 2>&1; then
    echo "Server is ready."
    exit 0
  fi

  now_epoch="$(date +%s)"
  elapsed="$((now_epoch - start_epoch))"
  if (( elapsed >= TIMEOUT_SECONDS )); then
    echo "Timed out after ${elapsed}s waiting for server readiness." >&2
    exit 1
  fi
  sleep 10
done
