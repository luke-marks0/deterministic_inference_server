#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

config="configs/qwen3-235b-a22b-instruct-2507.json"
secrets_file=".env"
timeout_seconds=7200
expected_manifest=""
run_smoke=1
run_hash=0
hash_output=""
auto_stop=0
stop_on_error=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/workflow.sh [options]

Orchestrates the standard profile workflow:
  start -> wait (with strict manifest verify) -> smoke -> optional hash -> optional stop
Requires an existing pinned snapshot manifest; the workflow never auto-bootstraps trust roots.

Options:
  --config <path>              Profile JSON path.
  --secrets-file <path>        Secrets env file (default: .env).
  --timeout-seconds <int>      Readiness timeout (default: 7200).
  --expected-manifest <path>   Override expected manifest path/template.
  --no-smoke                   Skip smoke test.
  --hash                       Generate snapshot manifest after readiness/smoke.
  --hash-output <path>         Output path/template for hash manifest.
  --stop                       Stop server at end of successful run.
  --stop-on-error              Stop server if any step fails.
  -h, --help                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="$2"
      shift 2
      ;;
    --secrets-file)
      secrets_file="$2"
      shift 2
      ;;
    --timeout-seconds)
      timeout_seconds="$2"
      shift 2
      ;;
    --expected-manifest)
      expected_manifest="$2"
      shift 2
      ;;
    --no-smoke)
      run_smoke=0
      shift
      ;;
    --hash)
      run_hash=1
      shift
      ;;
    --hash-output)
      hash_output="$2"
      shift 2
      ;;
    --stop)
      auto_stop=1
      shift
      ;;
    --stop-on-error)
      stop_on_error=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

step=1
log_step() {
  echo "[${step}] $1"
  step=$((step + 1))
}

stop_server() {
  python3 "${ROOT_DIR}/scripts/core/serve.py" stop \
    --config "${config}" \
    --secrets-file "${secrets_file}" >/dev/null 2>&1 || true
}

if [[ "${stop_on_error}" -eq 1 ]]; then
  trap stop_server ERR
fi

resolve_manifest_path() {
  python3 - "${ROOT_DIR}" "${config}" "${expected_manifest}" <<'PY'
import sys
from pathlib import Path

root_dir = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
override = sys.argv[3].strip()

sys.path.insert(0, str(root_dir / "scripts" / "core"))
from integrity_utils import resolve_manifest_template  # type: ignore
from profile_config import load_profile  # type: ignore

profile = load_profile(config_path)

template = override or profile.integrity.expected_snapshot_manifest.strip() or "manifests/{profile_id}/{revision}.sha256"
path = resolve_manifest_template(
    template=template,
    root_dir=profile.root_dir,
    profile_id=profile.profile_id,
    revision=profile.model.revision,
    model_id=profile.model.model_id,
)
print(path)
PY
}

manifest_path="$(resolve_manifest_path)"

if [[ ! -f "${manifest_path}" ]]; then
  echo "Pinned snapshot manifest is required but missing:" >&2
  echo "  ${manifest_path}" >&2
  echo "Refusing to auto-bootstrap trust roots from local state." >&2
  echo "Create/commit the manifest out-of-band, then rerun." >&2
  exit 1
fi

log_step "Starting server"
python3 "${ROOT_DIR}/scripts/core/serve.py" start \
  --config "${config}" \
  --secrets-file "${secrets_file}"

log_step "Manifest detected at ${manifest_path}; waiting with strict verification"
python3 "${ROOT_DIR}/scripts/core/serve.py" wait \
  --config "${config}" \
  --secrets-file "${secrets_file}" \
  --timeout-seconds "${timeout_seconds}" \
  --verify-manifest \
  --expected-manifest "${manifest_path}"

if [[ "${run_smoke}" -eq 1 ]]; then
  log_step "Running smoke test"
  python3 "${ROOT_DIR}/scripts/core/serve.py" smoke \
    --config "${config}" \
    --secrets-file "${secrets_file}"
else
  log_step "Skipping smoke test (--no-smoke)"
fi

if [[ "${run_hash}" -eq 1 ]]; then
  log_step "Writing snapshot hash manifest"
  hash_cmd=(
    python3 "${ROOT_DIR}/scripts/core/serve.py" hash
    --config "${config}"
    --secrets-file "${secrets_file}"
  )
  if [[ -n "${hash_output}" ]]; then
    hash_cmd+=(--output "${hash_output}")
  fi
  "${hash_cmd[@]}"
else
  log_step "Skipping snapshot hash generation (--hash to enable)"
fi

if [[ "${auto_stop}" -eq 1 ]]; then
  log_step "Stopping server (--stop)"
  python3 "${ROOT_DIR}/scripts/core/serve.py" stop \
    --config "${config}" \
    --secrets-file "${secrets_file}"
else
  log_step "Server left running (pass --stop to shut it down)"
fi

echo "Profile run completed."
