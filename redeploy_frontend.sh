#!/usr/bin/env bash
# Pull latest code, rebuild Next.js, restart in background (judge demo redeploy).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PORT="${FRONTEND_PORT:-3000}"

log() { echo "[redeploy] $*"; }

if ! command -v node &>/dev/null; then
  log "Install Node.js first — see AWS_DEPLOY.md"
  exit 1
fi

log "Pulling latest from git..."
git pull origin main

log "Stopping old Next.js process (if any)..."
pkill -f "next start" 2>/dev/null || true
sleep 1

log "Installing deps and building..."
cd "$ROOT/Frontend"
npm install
npm run build

chmod +x "$ROOT/run_frontend.sh"
log "Starting frontend on port ${PORT} (background)..."
cd "$ROOT"
nohup env FRONTEND_PORT="$PORT" ./run_frontend.sh > "$ROOT/frontend.log" 2>&1 &

sleep 2
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  PUBLIC_IP="$(curl -fsSL --max-time 3 http://checkip.amazonaws.com 2>/dev/null || true)"
  log "Frontend is up."
  [[ -n "$PUBLIC_IP" ]] && log "Judge link: http://${PUBLIC_IP}:${PORT}"
  log "Log file: $ROOT/frontend.log"
else
  log "WARNING: port ${PORT} not listening yet — check frontend.log"
  tail -20 "$ROOT/frontend.log" 2>/dev/null || true
  exit 1
fi
