# FillFrame

### Physics-informed deep learning for temporal super-resolution of geostationary satellite imagery

> We do not interpolate pixels. We propagate radiance through a flow field constrained by atmospheric physics — so every synthetic frame is a physically admissible state of the cloud field, not a statistical hallucination.

**Fill in the frames seamlessly** — synthesize missing thermal-infrared scans between fixed satellite cadences, with motion-aware deep learning and physics-informed training.

Live UI: [newframe-three.vercel.app](https://newframe-three.vercel.app) · Demo assets: [Drive](https://drive.google.com/drive/folders/1RjTIGuIH572-jUZwdtExbAzGAXTq56DV?usp=sharing)

---

## Why it exists

Geostationary satellites observe Earth on fixed schedules — **INSAT-3DS/3DR ~30 minutes**, **GOES-19 and Himawari ~10 minutes**. Between scans, fires spread, cyclones intensify, and convective systems reorganize. Classical optical-flow interpolation often blurs cloud edges, creates ghosting, and violates thermodynamic consistency.

FillFrame closes that gap with **RIFE** optical-flow interpolation plus a **physics-informed loss stack**, trained on high-cadence GOES and deployable on INSAT TIR.

**Target:** effective cadence **30 → 15 → 7.5 minutes** via recursive midpoint synthesis.

---

## Features

- Stream multi-satellite TIR catalogs from public NOAA S3 (GOES, Himawari, GK-2A) without copying archives to disk
- Midpoint frame prediction (t0 + t2 → t1) with GOES fine-tuned RIFE weights
- Atmospheric motion vectors (AMV), Farneback baseline, and ablation views
- Batch validation with **SSIM · MSE · PSNR · FSIM**
- PDF evaluation reports and NetCDF frame export
- Next.js mission console + FastAPI backend (+ Streamlit science UI)

---

## Satellite targets

| Satellite | TIR band | Native cadence | After FillFrame |
|-----------|----------|----------------|-----------------|
| INSAT-3DS / 3DR | TIR1 (~10.8 µm) | ~30 min | 15 min · 7.5 min |
| GOES-19 ABI | Ch.13 (~10.3 µm) | ~10 min | 5 min |
| Himawari-8/9 AHI | Band 13 (~10.4 µm) | ~10 min | 5 min |
| GK-2A AMI | IR105 (~10.5 µm) | ~10 min | 5 min |

---

## Architecture

```
┌─────────────────┐     NEXT_PUBLIC_API_URL      ┌──────────────────────┐
│  Next.js UI     │ ───────────────────────────► │  FastAPI (uvicorn)   │
│  Mission Console│                              │  RIFE + S3 catalog   │
└─────────────────┘                              └──────────┬───────────┘
                                                            │
                                              Public NOAA S3 / mounts
```

| Layer | Stack |
|-------|--------|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind, Recharts |
| API | FastAPI, Uvicorn, PyTorch, OpenCV, xarray, s3fs |
| Models | Practical-RIFE v4.25 + `checkpoints/goes_finetuned` |
| Science UI | Streamlit (`app.py`) |
| Metrics | scikit-image, piq |

---

## Quick start (local)

### Prerequisites

- Python **3.12**
- Node.js **20+**
- Git

### 1. Backend

```bash
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
python setup_rife.py

# Ensure checkpoint exists:
#   checkpoints/goes_finetuned/flownet.pkl

uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Health: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### 2. Frontend

```bash
cd Frontend
npm install
set NEXT_PUBLIC_API_URL=http://127.0.0.1:8000   # Windows PowerShell: $env:NEXT_PUBLIC_API_URL=...
npm run dev
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000)

### Typical workflow

1. **List scans** (e.g. GOES-19, date `2026-06-12`)
2. Set **Size → 256** on CPU hosts
3. **Load triplet** → **Predict t1**
4. Inspect **Interpolation**, **Motion / AMV**, **Batch Validate**

---

## Deployed demos

| Surface | Link |
|---------|------|
| Mission console (UI) | [newframe-three.vercel.app](https://newframe-three.vercel.app) |
| API health | [newframe.onrender.com/api/health](https://newframe.onrender.com/api/health) |
| Demo assets / video | [Google Drive folder](https://drive.google.com/drive/folders/1RjTIGuIH572-jUZwdtExbAzGAXTq56DV?usp=sharing) |

> **Note:** Full Load triplet / Predict pulls large NetCDF objects from S3 and runs RIFE on CPU. Free hosting tiers may time out; use a stronger host or the demo video for lightweight review.

---

## Repository layout

```
├── api_server.py          # FastAPI backend
├── app.py                 # Streamlit science UI
├── s3_catalog.py          # Multi-satellite S3 / mount catalog
├── amv.py                 # Atmospheric motion vector helpers
├── physics_losses.py      # Physics-informed training losses
├── metrics.py             # SSIM / MSE / PSNR / FSIM
├── finetune_goes.py       # GOES fine-tune
├── finetune_physics.py    # Physics-informed fine-tune
├── setup_rife.py          # Clone RIFE + base weights
├── report_pdf.py          # PDF evaluation report
├── checkpoints/           # goes_finetuned/flownet.pkl
├── Frontend/              # Next.js mission console
├── Dockerfile             # Render / container API deploy
└── docs                   # AWS_DEPLOY.md · DATA_ACCESS.md · RENDER_DEPLOY.md
```

---

## Environment

| Variable | Where | Purpose |
|----------|--------|---------|
| `NEXT_PUBLIC_API_URL` | Frontend / Vercel | Backend base URL |
| `NEXT_PUBLIC_DEMO_VIDEO_URL` | Frontend / Vercel | Landing “demo video” link |
| `CORS_ORIGINS` | Backend | Allowed UI origins (comma-separated) |
| Catalog / S3 mode | Backend | Enable public NOAA catalog defaults via deploy scripts |

---

## License & acknowledgment

Built for the *Fill in the Frames Seamlessly* temporal super-resolution challenge. Satellite open data courtesy of **NOAA** public AWS registries (GOES, Himawari, GK-2A).
