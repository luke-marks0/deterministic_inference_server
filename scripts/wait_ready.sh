#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  timeout_seconds="$1"
  shift
  exec python3 "${ROOT_DIR}/scripts/serve.py" wait --timeout-seconds "${timeout_seconds}" "$@"
fi

exec python3 "${ROOT_DIR}/scripts/serve.py" wait "$@"
