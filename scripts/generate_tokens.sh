#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/state"
PYCACHE_DIR="${ROOT_DIR}/scripts/core/__pycache__"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python3"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

# If invoked with sudo, repair ownership and drop back to the original user
# so dependencies resolve from the non-root environment.
if [[ "${EUID}" -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
  if [[ -d "${STATE_DIR}" ]]; then
    chown -R "${SUDO_USER}:${SUDO_USER}" "${STATE_DIR}" >/dev/null 2>&1 || true
  fi
  if [[ -d "${PYCACHE_DIR}" ]]; then
    chown -R "${SUDO_USER}:${SUDO_USER}" "${PYCACHE_DIR}" >/dev/null 2>&1 || true
  fi
  exec sudo -u "${SUDO_USER}" -H "${PYTHON_BIN}" "${ROOT_DIR}/scripts/core/sample_session.py" "$@"
fi

if [[ -d "${STATE_DIR}" && ! -w "${STATE_DIR}" ]]; then
  echo "Error: ${STATE_DIR} is not writable by $(id -un)." >&2
  echo "Fix with: sudo chown -R $(id -un):$(id -gn) ${STATE_DIR}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/core/sample_session.py" "$@"
