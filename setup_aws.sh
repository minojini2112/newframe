#!/usr/bin/env bash
# =============================================================================
# ISRO PS12 — one-shot AWS EC2 setup
#
# --- Fresh instance (clone from GitHub first) ---
#   curl -fsSL https://raw.githubusercontent.com/YOU/REPO/main/clone_on_aws.sh -o clone_on_aws.sh
#   chmod +x clone_on_aws.sh
#   ./clone_on_aws.sh https://github.com/YOU/REPO.git
#
# --- Already cloned ---
#   cd ~/isro
#   git pull
#   ./setup_aws.sh
#   ./run_app.sh
#
#   ./setup_aws.sh --start      # install + launch Streamlit
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

START_APP=0
if [[ "${1:-}" == "--start" ]]; then
  START_APP=1
fi

log() { echo "[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. System packages (Amazon Linux 2023 / Ubuntu)
# -----------------------------------------------------------------------------
log "Checking system packages..."
if command -v dnf &>/dev/null; then
  sudo dnf install -y git gcc gcc-c++ make \
    python3 python3-pip python3-devel \
    libjpeg-turbo-devel zlib-devel hdf5-devel netcdf-devel \
    2>/dev/null || sudo dnf install -y git gcc gcc-c++ make python3 python3-pip python3-devel
elif command -v apt-get &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y git build-essential \
    python3 python3-pip python3-venv python3-dev \
    libhdf5-dev libnetcdf-dev \
    libgl1 libglib2.0-0
else
  log "Unknown package manager — assuming Python/git already installed."
fi

# Prefer Python 3.12 → 3.11 → 3
PYTHON=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" &>/dev/null; then
    PYTHON="$cand"
    break
  fi
done
[[ -n "$PYTHON" ]] || die "No python3 found. Install Python 3.11+ first."
log "Using $PYTHON ($($PYTHON --version))"

# -----------------------------------------------------------------------------
# 2. Virtual environment
# -----------------------------------------------------------------------------
if [[ ! -d "$ROOT/.venv" ]]; then
  log "Creating virtual environment..."
  "$PYTHON" -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel "setuptools>=70,<82"

# -----------------------------------------------------------------------------
# 3. PyTorch (GPU if available, else CPU)
# -----------------------------------------------------------------------------
log "Installing PyTorch..."
if command -v nvidia-smi &>/dev/null; then
  log "NVIDIA GPU detected — installing CUDA PyTorch..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  log "No GPU — installing CPU PyTorch..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# -----------------------------------------------------------------------------
# 4. Python dependencies
# -----------------------------------------------------------------------------
log "Installing requirements.txt..."
pip install -r "$ROOT/requirements.txt"
pip install --only-binary=:all: scikit-image 2>/dev/null || pip install "scikit-image>=0.22"

log "Installing satpy (Himawari-9 Band 13 reader)..."
pip install "satpy>=0.43"

# Fix common boto3/s3fs version skew
pip install --upgrade "boto3>=1.34" "botocore>=1.34" "s3fs>=2023.6"

# Fix OpenCV on headless Ubuntu (libGL.so.1)
if command -v apt-get &>/dev/null; then
  sudo apt-get install -y libgl1 libglib2.0-0 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# 5. RIFE model (Practical-RIFE + v4.25 weights)
# -----------------------------------------------------------------------------
log "Setting up RIFE (Practical-RIFE + weights)..."
python "$ROOT/setup_rife.py"

# -----------------------------------------------------------------------------
# 5b. GOES fine-tuned checkpoint (required for inference)
# -----------------------------------------------------------------------------
FINETUNED_CKPT="$ROOT/checkpoints/goes_finetuned/flownet.pkl"
if [[ -f "$FINETUNED_CKPT" ]]; then
  log "Fine-tuned checkpoint OK: checkpoints/goes_finetuned/flownet.pkl"
else
  die "Missing GOES fine-tuned weights: $FINETUNED_CKPT
Copy the checkpoint before deploy, e.g.:
  scp checkpoints/goes_finetuned/flownet.pkl ec2-user@HOST:~/isro/checkpoints/goes_finetuned/
Or commit checkpoints/goes_finetuned/flownet.pkl in the repo and git pull on the instance."
fi

# -----------------------------------------------------------------------------
# 6. Streamlit server config (headless EC2)
# -----------------------------------------------------------------------------
mkdir -p "$ROOT/.streamlit"
cat > "$ROOT/.streamlit/config.toml" << 'TOML'
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
TOML

# -----------------------------------------------------------------------------
# 7. Environment file — auto-detect INSAT mount
# -----------------------------------------------------------------------------
INSAT_MOUNT=""
for candidate in \
  "${SATELLITE_DATA_MOUNT:-}" \
  "/data/insat" \
  "/data/INSAT-3DS" \
  "/data/insat3ds" \
  "/mnt/satellite-data" \
  "/mnt/satellite-data/insat" \
  "/opt/isro-data" \
  "/opt/isro-data/insat" \
  "/home/ubuntu/data/insat" \
  "/home/ec2-user/data/insat"
do
  if [[ -n "$candidate" && -d "$candidate" ]]; then
    INSAT_MOUNT="$candidate"
    log "Found INSAT data mount: $INSAT_MOUNT"
    break
  fi
done

cat > "$ROOT/.env.aws" << EOF
# Generated by setup_aws.sh — source before running the app
export ISRO_PS12_AWS=1
export SATELLITE_DATA_MODE=s3

# INSAT HDF5 on this instance (edit if ISRO gave a different path):
export SATELLITE_DATA_MOUNT="${INSAT_MOUNT}"

# If ISRO provided a private S3 bucket for INSAT, uncomment and fill in:
# export ISRO_S3_BUCKET=your-bucket-name
# export INSAT_S3_PREFIX=INSAT-3DS/L1B/{year}/{month:02d}/{day:02d}/

# Optional: bind to a different port
export STREAMLIT_PORT=8501
EOF

# -----------------------------------------------------------------------------
# 8. Launcher script
# -----------------------------------------------------------------------------
cat > "$ROOT/run_app.sh" << 'RUN'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
if [[ -f "$ROOT/.env.aws" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.aws"
fi
PORT="${STREAMLIT_PORT:-8501}"
echo "Starting ISRO PS12 app on http://0.0.0.0:${PORT}"
echo "Open in browser: http://<instance-public-ip>:${PORT}"
exec streamlit run app.py --server.port="$PORT" --server.address=0.0.0.0
RUN
chmod +x "$ROOT/run_app.sh"

# -----------------------------------------------------------------------------
# 9. Smoke tests
# -----------------------------------------------------------------------------
log "Running smoke tests..."
python - << 'PY'
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "Practical-RIFE"))

errors = []
for mod in ("streamlit", "torch", "cv2", "xarray", "boto3", "s3fs", "h5py", "skimage"):
    try:
        __import__(mod)
    except ImportError as e:
        errors.append(f"  missing {mod}: {e}")

from train_log.RIFE_HDv3 import Model
import torch
m = Model()
sd = torch.load(ROOT / "Practical-RIFE/train_log/flownet.pkl", map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.items()}
wanted = m.flownet.state_dict()
sd = {k: v for k, v in sd.items() if k in wanted}
m.flownet.load_state_dict(sd, strict=True)
print("  RIFE model OK")

from s3_catalog import list_scans_for_day, satellite_sources
from datetime import date
refs = list_scans_for_day("gk2a_ir105", date(2024, 6, 30))
assert len(refs) > 0, "S3 catalog returned 0 GK-2A files — check network/IAM"
print(f"  S3 catalog OK ({len(refs)} GK-2A scans)")

if errors:
    print("SMOKE TEST WARNINGS (non-fatal):")
    print("\n".join(errors))
    if any("cv2" in e or "libGL" in e for e in errors):
        print("\n  Fix: sudo apt-get install -y libgl1 libglib2.0-0")
else:
    print("All smoke tests passed.")
PY

log "=============================================="
log "Setup complete."
log ""
log "  Start Streamlit:   ./run_app.sh"
log "  Public deploy:     ./deploy_public.sh   (backend :8000 + frontend :3000)"
log "  Or manually:       source .venv/bin/activate && source .env.aws && streamlit run app.py"
log ""
log "  Security group:    allow inbound TCP 3000 + 8000 (public demo) or TCP ${STREAMLIT_PORT:-8501} (Streamlit only)"
if [[ -n "$INSAT_MOUNT" ]]; then
  log "  INSAT mount:       $INSAT_MOUNT"
else
  log "  INSAT mount:       not found — set SATELLITE_DATA_MOUNT in .env.aws when ISRO provides path"
fi
log "=============================================="

if [[ "$START_APP" -eq 1 ]]; then
  exec "$ROOT/run_app.sh"
fi
