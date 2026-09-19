#!/usr/bin/env bash
# =============================================================================
# Deploy FillFrame for public access on AWS EC2 (backend + frontend).
#
# Prerequisites:
#   - Security group allows inbound TCP 3000 and 8000 from 0.0.0.0/0
#   - Elastic IP associated (recommended — link stays stable after reboot)
#   - ./setup_aws.sh already run once (venv, RIFE, checkpoint)
#
# Usage:
#   chmod +x deploy_public.sh run_backend.sh run_frontend.sh
#   ./deploy_public.sh
#
# After git push:
#   ./deploy_public.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

log() { echo "[deploy] $*"; }
die() { log "ERROR: $*"; exit 1; }

# -----------------------------------------------------------------------------
# Public IP (for browser → API calls and CORS)
# -----------------------------------------------------------------------------
PUBLIC_IP="${PUBLIC_IP:-}"
if [[ -z "$PUBLIC_IP" ]]; then
  PUBLIC_IP="$(curl -fsSL --max-time 5 http://checkip.amazonaws.com 2>/dev/null || true)"
fi
if [[ -z "$PUBLIC_IP" ]]; then
  PUBLIC_IP="$(curl -fsSL --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
fi
[[ -n "$PUBLIC_IP" ]] || die "Could not detect public IP. Set PUBLIC_IP=1.2.3.4 and re-run."

API_URL="http://${PUBLIC_IP}:${BACKEND_PORT}"
FRONT_URL="http://${PUBLIC_IP}:${FRONTEND_PORT}"
export NEXT_PUBLIC_API_URL="$API_URL"
export CORS_ORIGINS="${FRONT_URL},http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}"
export ISRO_PS12_AWS="${ISRO_PS12_AWS:-1}"

log "Public frontend: ${FRONT_URL}"
log "Public API:      ${API_URL}"

# -----------------------------------------------------------------------------
# Backend setup check
# -----------------------------------------------------------------------------
[[ -d "$ROOT/.venv" ]] || die "Run ./setup_aws.sh first (creates venv + RIFE + checkpoint check)"
[[ -f "$ROOT/checkpoints/goes_finetuned/flownet.pkl" ]] || die "Missing checkpoints/goes_finetuned/flownet.pkl"

if ! command -v node &>/dev/null; then
  log "Installing Node.js 20..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

# -----------------------------------------------------------------------------
# Pull latest (optional — skip with SKIP_GIT_PULL=1)
# -----------------------------------------------------------------------------
if [[ "${SKIP_GIT_PULL:-0}" != "1" ]] && [[ -d "$ROOT/.git" ]]; then
  log "git pull..."
  git pull origin main || git pull || true
fi

chmod +x "$ROOT/run_backend.sh" "$ROOT/run_frontend.sh" 2>/dev/null || true

stop_port() {
  local port="$1"
  if command -v fuser &>/dev/null; then
    fuser -k "${port}/tcp" 2>/dev/null || true
  fi
}

# -----------------------------------------------------------------------------
# Stop old processes
# -----------------------------------------------------------------------------
log "Stopping old backend/frontend processes..."
pkill -f "uvicorn api_server:app" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
stop_port "$BACKEND_PORT"
stop_port "$FRONTEND_PORT"
sleep 2

# -----------------------------------------------------------------------------
# Build frontend (NEXT_PUBLIC_API_URL is baked in at build time)
# -----------------------------------------------------------------------------
log "Building frontend (API → ${API_URL})..."
cd "$ROOT/Frontend"
npm install
log "Removing old .next build (prevents stale CSS/JS 400/404 errors)..."
rm -rf .next
export NODE_ENV=production
NEXT_PUBLIC_API_URL="$API_URL" npm run build
CSS_COUNT="$(find .next/static/css -name '*.css' 2>/dev/null | wc -l | tr -d ' ')"
[[ "${CSS_COUNT:-0}" -gt 0 ]] || die "Frontend build produced no CSS bundles — check frontend.log / npm output"
log "Built ${CSS_COUNT} CSS bundle(s)"

# -----------------------------------------------------------------------------
# Start backend
# -----------------------------------------------------------------------------
log "Starting backend on port ${BACKEND_PORT}..."
cd "$ROOT"
nohup env \
  BACKEND_PORT="$BACKEND_PORT" \
  CORS_ORIGINS="$CORS_ORIGINS" \
  ISRO_PS12_AWS="$ISRO_PS12_AWS" \
  ./run_backend.sh > "$ROOT/backend.log" 2>&1 &

# -----------------------------------------------------------------------------
# Start frontend
# -----------------------------------------------------------------------------
log "Starting frontend on port ${FRONTEND_PORT}..."
nohup env \
  FRONTEND_PORT="$FRONTEND_PORT" \
  NODE_ENV=production \
  REBUILD=0 \
  ./run_frontend.sh > "$ROOT/frontend.log" 2>&1 &

sleep 4

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
BACKEND_OK=0
FRONTEND_OK=0
if ss -tlnp 2>/dev/null | grep -q ":${BACKEND_PORT} "; then BACKEND_OK=1; fi
if ss -tlnp 2>/dev/null | grep -q ":${FRONTEND_PORT} "; then FRONTEND_OK=1; fi

if [[ "$BACKEND_OK" -eq 1 && "$FRONTEND_OK" -eq 1 ]]; then
  log "=============================================="
  log "Deploy complete. Share this link:"
  log ""
  log "  ${FRONT_URL}"
  log ""
  log "Backend health: ${API_URL}/api/health"
  log "Logs: backend.log, frontend.log"
  log "=============================================="
else
  log "WARNING: one or both services failed to bind."
  [[ "$BACKEND_OK" -eq 0 ]] && { log "Backend log:"; tail -30 "$ROOT/backend.log" 2>/dev/null || true; }
  [[ "$FRONTEND_OK" -eq 0 ]] && { log "Frontend log:"; tail -30 "$ROOT/frontend.log" 2>/dev/null || true; }
  exit 1
fi
