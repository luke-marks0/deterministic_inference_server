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

if [[ -z "${MODEL_REVISION:-}" || "${MODEL_REVISION}" == "UNSET_RUN_LOCK_SCRIPT" ]]; then
  echo "MODEL_REVISION is not set. Run ./scripts/lock_model_revision.sh first." >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/state/hf" "${ROOT_DIR}/artifacts/manifests" "${ROOT_DIR}/artifacts/samples"

cd "${ROOT_DIR}"
docker compose --env-file "${ENV_FILE}" up -d
echo "vLLM server started. Use ./scripts/wait_ready.sh to wait for model load."
