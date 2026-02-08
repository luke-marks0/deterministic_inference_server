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

if [[ -z "${MODEL_ID:-}" || -z "${MODEL_REVISION:-}" ]]; then
  echo "MODEL_ID and MODEL_REVISION are required in ${ENV_FILE}" >&2
  exit 1
fi

if [[ "${MODEL_REVISION}" == "UNSET_RUN_LOCK_SCRIPT" ]]; then
  echo "MODEL_REVISION is not locked yet. Run ./scripts/lock_model_revision.sh first." >&2
  exit 1
fi

cache_model_path="models--${MODEL_ID//\//--}"
snapshot_dir="${ROOT_DIR}/state/hf/hub/${cache_model_path}/snapshots/${MODEL_REVISION}"

if [[ ! -d "${snapshot_dir}" ]]; then
  echo "Snapshot directory not found: ${snapshot_dir}" >&2
  echo "Start the server once and wait for model download to complete first." >&2
  exit 1
fi

manifest_dir="${ROOT_DIR}/artifacts/manifests"
mkdir -p "${manifest_dir}"
manifest_file="${manifest_dir}/model_snapshot_${MODEL_REVISION}.sha256"

(
  cd "${snapshot_dir}"
  LC_ALL=C find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "${manifest_file}"

echo "Wrote snapshot manifest:"
echo "  ${manifest_file}"
