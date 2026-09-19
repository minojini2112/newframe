#!/usr/bin/env bash
# ISRO PS12 — Next.js mission console (no Streamlit)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONT="$ROOT/Frontend"
PORT="${FRONTEND_PORT:-3000}"

if ! command -v node &>/dev/null; then
  echo "Node.js not found. On Ubuntu EC2:"
  echo "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
  echo "  sudo apt-get install -y nodejs"
  exit 1
fi

cd "$FRONT"

if [[ ! -d node_modules ]]; then
  echo "[frontend] npm install (first time)..."
  npm install
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -d .next ]]; then
  echo "[frontend] npm run build..."
  export NODE_ENV=production
  npm run build
fi

echo "[frontend] Dashboard: http://0.0.0.0:${PORT}"
echo "[frontend] Open in browser: http://<instance-public-ip>:${PORT}"
exec npx next start -H 0.0.0.0 -p "$PORT"
