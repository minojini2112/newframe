# Deploy FillFrame backend on Render

Satellite **data** still comes from public NOAA S3 (free).
Render only hosts the **FastAPI + RIFE** API.

## Before you start

1. Commit & push including the checkpoint:
   ```bash
   git add checkpoints/goes_finetuned/flownet.pkl checkpoints/goes_finetuned/training_log.json
   git add Dockerfile .dockerignore render.yaml
   git commit -m "Add Render Docker deploy for FastAPI backend"
   git push origin main
   ```
2. Free Render is **512MB RAM** — PyTorch may crash (OOM). If so, upgrade to **Starter** (~$7/mo).

## Render dashboard steps

1. [https://render.com](https://render.com) → **New** → **Web Service**
2. Connect GitHub → `minojini2112/newframe`
3. Settings:
   - **Runtime:** Docker
   - **Dockerfile path:** `Dockerfile` (repo root)
   - **Branch:** `main`
4. Environment:
   | Key | Value |
   |-----|--------|
   | `ISRO_PS12_AWS` | `1` |
   | `CORS_ORIGINS` | `https://YOUR-APP.vercel.app,http://localhost:3000` |
5. Create Web Service → wait for build (can take **15–30+ min** first time)

## After deploy

Health check:
```text
https://YOUR-SERVICE.onrender.com/api/health
```

Expect:
```json
{ "status": "ok", "rife_ready": true, "checkpoint_exists": true, ... }
```

Then on **Vercel** set:
```text
NEXT_PUBLIC_API_URL=https://YOUR-SERVICE.onrender.com
```
Redeploy the frontend.

## Notes

- Free tier **spins down** after idle ~15 min — first request is slow.
- Predict on CPU can take **1–3+ minutes** — may hit timeouts; try size 256 in UI.
- If build fails on memory/disk, switch plan to Starter and retry.
