# this is a temp script that provisions a lambda labs instance for what i need

sudo apt update
sudo apt install npm
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22
nvm use 22
nvm alias default 22
npm install -g @openai/codex@latest
pip install uv
uv pip install -r requirements.txt
codex login --device-auth
