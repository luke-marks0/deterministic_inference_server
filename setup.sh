#!/usr/bin/env bash
set -euo pipefail

NODE_MAJOR="22"
NVM_VERSION="v0.39.7"
VENV_DIR=".venv"
REQ_FILE="requirements.txt"

sudo apt-get update -y
sudo apt-get install -y curl ca-certificates git build-essential

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
  curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash
fi

source "$NVM_DIR/nvm.sh"

nvm install "${NODE_MAJOR}"
nvm use "${NODE_MAJOR}"
nvm alias default "${NODE_MAJOR}"

if ! command -v uv >/dev/null 2>&1; then
  # uv via pip; you could also use the official installer, but this is fine.
  python3 -m pip install --user -U uv
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [[ -f "$REQ_FILE" ]]; then
  uv pip install -r "$REQ_FILE"
fi

export TERM="${TERM:-xterm-256color}"

echo "Done."
