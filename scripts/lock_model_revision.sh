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

if [[ -z "${MODEL_ID:-}" ]]; then
  echo "MODEL_ID is required in ${ENV_FILE}" >&2
  exit 1
fi

if ! lock_output="$(python3 - "${MODEL_ID}" <<'PY'
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

model_id = sys.argv[1]
url = f"https://huggingface.co/api/models/{model_id}"
try:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
except urllib.error.URLError as exc:
    print(f"Failed to query Hugging Face API: {exc}", file=sys.stderr)
    raise SystemExit(2)

sha = payload.get("sha")
if not sha:
    raise SystemExit("Could not read model sha from Hugging Face API response.")

locked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
print(sha)
print(locked_at)
PY
 )"; then
  echo "Failed to resolve model revision from Hugging Face API. Check network and DNS, then retry." >&2
  exit 1
fi

readarray -t lock_values <<< "${lock_output}"

if [[ "${#lock_values[@]}" -lt 2 || -z "${lock_values[0]}" || -z "${lock_values[1]}" ]]; then
  echo "Could not parse revision lock values from Hugging Face API response." >&2
  exit 1
fi

revision="${lock_values[0]}"
locked_at="${lock_values[1]}"

upsert_env_var() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${ENV_FILE}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
}

upsert_env_var "MODEL_REVISION" "${revision}"
upsert_env_var "MODEL_LOCKED_AT_UTC" "${locked_at}"

echo "Locked ${MODEL_ID} to revision ${revision}"
echo "Updated ${ENV_FILE}"
