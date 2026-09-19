#!/usr/bin/env bash
# =============================================================================
# Run this ONCE on a fresh AWS EC2 instance to clone the repo and set everything up.
#
# Usage:
#   chmod +x clone_on_aws.sh
#   ./clone_on_aws.sh https://github.com/YOUR_USER/YOUR_REPO.git
#
# Or set REPO_URL and run:
#   export REPO_URL=https://github.com/YOUR_USER/YOUR_REPO.git
#   ./clone_on_aws.sh
# =============================================================================
set -euo pipefail

REPO_URL="${1:-${REPO_URL:-}}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/isro}"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: ./clone_on_aws.sh <git-repo-url>"
  echo "Example: ./clone_on_aws.sh https://github.com/you/isro-ps12.git"
  exit 1
fi

if ! command -v git &>/dev/null; then
  echo "Installing git..."
  if command -v dnf &>/dev/null; then sudo dnf install -y git
  elif command -v apt-get &>/dev/null; then sudo apt-get update -qq && sudo apt-get install -y git
  else echo "Install git manually and re-run."; exit 1
  fi
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "Repo already exists at $INSTALL_DIR — pulling latest..."
  cd "$INSTALL_DIR"
  git pull
else
  echo "Cloning $REPO_URL -> $INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi

chmod +x setup_aws.sh run_app.sh deploy_public.sh run_backend.sh run_frontend.sh 2>/dev/null || true
./setup_aws.sh --start
