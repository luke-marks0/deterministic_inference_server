#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/workflow.sh run [run_profile_options...]
  ./scripts/workflow.sh <serve_subcommand> [options...]

Top-level workflow entrypoint for all non-token-generation operations.

Examples:
  ./scripts/workflow.sh run --config configs/qwen3-235b-a22b-instruct-2507.json --stop
  ./scripts/workflow.sh start --config configs/qwen3-235b-a22b-instruct-2507.json
  ./scripts/workflow.sh wait --config configs/qwen3-235b-a22b-instruct-2507.json --verify-manifest
  ./scripts/workflow.sh smoke --config configs/qwen3-235b-a22b-instruct-2507.json
  ./scripts/workflow.sh verify --config configs/qwen3-235b-a22b-instruct-2507.json
EOF
}

if [[ $# -eq 0 ]]; then
  exec "${ROOT_DIR}/scripts/atomic/run_profile.sh"
fi

case "$1" in
  -h|--help|help)
    usage
    exit 0
    ;;
  run)
    shift
    exec "${ROOT_DIR}/scripts/atomic/run_profile.sh" "$@"
    ;;
  start|stop|wait|smoke|hash|verify|lock-model|lock-image|render|show)
    subcommand="$1"
    shift
    exec python3 "${ROOT_DIR}/scripts/core/serve.py" "${subcommand}" "$@"
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
