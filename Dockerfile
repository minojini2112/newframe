# FillFrame FastAPI backend for Render (CPU)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ISRO_PS12_AWS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libhdf5-dev \
    libnetcdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU PyTorch first (keeps image smaller than CUDA)
RUN pip install --upgrade pip wheel "setuptools>=70,<82" \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip uninstall -y opencv-python || true \
    && pip install opencv-python-headless \
    && pip install gdown

COPY . .

# GOES fine-tuned checkpoint (must be present in build context)
RUN test -f checkpoints/goes_finetuned/flownet.pkl \
    || (echo "MISSING checkpoints/goes_finetuned/flownet.pkl — add it before deploy" && exit 1)

# RIFE repo + base weights (needed for model code path)
RUN python setup_rife.py

EXPOSE 8000

# Render injects $PORT
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
