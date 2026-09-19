#!/usr/bin/env bash
# ISRO PS12 — FastAPI backend for Next.js dashboard
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PORT="${BACKEND_PORT:-8000}"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "[backend] Virtual env missing. Run: ./setup_aws.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

if [[ -f "$ROOT/.env.aws" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.aws"
fi

export ISRO_PS12_AWS="${ISRO_PS12_AWS:-1}"

echo "[backend] API: http://0.0.0.0:${PORT}"
echo "[backend] Health: http://0.0.0.0:${PORT}/api/health"
if [[ -n "${CORS_ORIGINS:-}" ]]; then
  echo "[backend] CORS: ${CORS_ORIGINS}"
fi

exec uvicorn api_server:app --host 0.0.0.0 --port "$PORT"
