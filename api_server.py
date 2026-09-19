"""
FastAPI backend for the ISRO PS12 Next.js dashboard.
Wraps inference, catalog, batch validation, and AMV helpers from app.py.

Run (from repo root, with venv active):
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
import tempfile
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

matplotlib.use("Agg")

from report_pdf import generate_report_pdf
import app as ps12  # noqa: E402
from s3_catalog import (  # noqa: E402
    NcRef,
    _mounted_path,
    _s3_anonymous,
    _s3_client,
    aws_deploy_mode,
    parse_goes_scan_time,
    satellite_sources,
)

app = FastAPI(title="ISRO PS12 API", version="1.0.0")

_default_cors = "http://localhost:3000,http://127.0.0.1:3000"
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _default_cors).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_to_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_img(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def read_ref_file_bytes(ref: NcRef) -> tuple[bytes, str]:
    """Read original satellite file bytes (local path, mount, or S3)."""
    uri = ref.uri.replace("\\", "/")
    fname = ref.name
    if uri.startswith("s3://"):
        rest = uri[5:]
        bucket, key = rest.split("/", 1)
        local = _mounted_path(bucket, key)
        if local:
            return ps12.Path(local).read_bytes(), fname
        client = _s3_client(_s3_anonymous(uri, ref.source_id))
        return client.get_object(Bucket=bucket, Key=key)["Body"].read(), fname
    path = ps12.Path(uri)
    if path.is_file():
        return path.read_bytes(), fname
    raise HTTPException(status_code=404, detail=f"File not found: {uri}")


def write_predicted_nc_bytes(
    pred_rgb: np.ndarray,
    rad_var: str,
    vmin: float,
    vmax: float,
) -> bytes:
    """Write RIFE predicted midpoint as a minimal NetCDF radiance grid."""
    import xarray as xr

    g = ps12.gray01(pred_rgb)
    rad = g.astype(np.float32) * (vmax - vmin) + vmin
    var = rad_var or "Rad"
    buf = io.BytesIO()
    ds = xr.Dataset(
        {var: (["y", "x"], rad)},
        attrs={
            "title": "RIFE predicted midpoint",
            "source": "ISRO PS12 fillframe",
            "note": "Radiance rescaled from RGB inference output at display resolution",
        },
    )
    ds.to_netcdf(buf, engine="h5netcdf")
    return buf.getvalue()


def predicted_nc_filename(base_name: str) -> str:
    stem = base_name.rsplit(".", 1)[0] if "." in base_name else base_name
    return f"{stem}_predicted_t1.nc"


# Pre-built NetCDF files keyed by triplet — warmed during predict-and-motion for instant download.
_frame_nc_cache: dict[str, dict[str, tuple[bytes, str, str]]] = {}
_upload_nc_cache: dict[str, dict[str, tuple[bytes, str, str]]] = {}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_NC_EXTENSIONS = {".nc", ".hdf", ".hdf5", ".h5"}


def _triplet_cache_key(req: TripletLoadRequest) -> str:
    return "|".join(
        [
            str(req.source_id),
            str(req.day),
            str(req.hour),
            str(req.local_folder),
            str(req.gap_min),
            str(req.triplet_index),
            str(req.size),
            str(getattr(req, "scale", 1.0)),
        ]
    )


def write_radiance_nc_bytes(rad: np.ndarray, rad_var: str, title: str) -> bytes:
    import xarray as xr

    var = rad_var or "Rad"
    buf = io.BytesIO()
    ds = xr.Dataset(
        {var: (["y", "x"], np.asarray(rad, dtype=np.float32))},
        attrs={"title": title, "source": "ISRO PS12 fillframe"},
    )
    ds.to_netcdf(buf, engine="h5netcdf")
    return buf.getvalue()


def warm_frame_nc_cache(
    req: TripletLoadRequest,
    names: tuple[str, str, str],
    raw_rads: list[np.ndarray],
    pred_rgb: np.ndarray,
    rad_var: str,
) -> None:
    """Build all four NetCDF exports once after predict — downloads are then instant."""
    vmin, vmax = ps12.global_stretch(raw_rads)
    n0, n1, n2 = names
    entries: dict[str, tuple[bytes, str, str]] = {}
    for frame, rad, name in (
        ("t0", raw_rads[0], n0),
        ("t1_gt", raw_rads[1], n1),
        ("t2", raw_rads[2], n2),
    ):
        nc = write_radiance_nc_bytes(rad, rad_var, f"Scan {frame}")
        fname = name if name.lower().endswith((".nc", ".h5")) else f"{name}.nc"
        entries[frame] = (nc, fname, "application/x-netcdf")
    nc_pred = write_predicted_nc_bytes(pred_rgb, rad_var, vmin, vmax)
    entries["t1_pred"] = (nc_pred, predicted_nc_filename(n0), "application/x-netcdf")
    _frame_nc_cache[_triplet_cache_key(req)] = entries


def get_cached_frame_nc(req: FrameDownloadRequest) -> tuple[bytes, str, str] | None:
    entries = _frame_nc_cache.get(_triplet_cache_key(req))
    if not entries:
        return None
    return entries.get(req.frame)


def get_upload_frame_nc(upload_id: str, frame: str) -> tuple[bytes, str, str] | None:
    entries = _upload_nc_cache.get(upload_id)
    if not entries:
        return None
    return entries.get(frame)


async def _read_upload_bytes(upload: UploadFile) -> tuple[str, bytes]:
    name = upload.filename or "upload"
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Empty file: {name}")
    return name, content


def _load_radiance_bytes(name: str, content: bytes) -> np.ndarray:
    ext = Path(name).suffix.lower() or ".nc"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        path = Path(tmp.name)
    try:
        return ps12.load_nc_radiance(path)
    finally:
        path.unlink(missing_ok=True)


def _load_image_rgb(name: str, content: bytes, size: int) -> np.ndarray:
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(
        img.resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _is_nc_upload(name: str) -> bool:
    return Path(name).suffix.lower() in _NC_EXTENSIONS


async def _decode_upload_frame(
    upload: UploadFile,
    size: int,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str, bool]:
    """Return RGB frame, optional radiance grid, filename, is_nc."""
    name, content = await _read_upload_bytes(upload)
    ext = Path(name).suffix.lower()
    if ext in _NC_EXTENSIONS:
        rad = _load_radiance_bytes(name, content)
        if vmin is None or vmax is None:
            vmin, vmax = ps12.global_stretch([rad])
        rgb = ps12.rad_to_rgb(rad, vmin, vmax, size)
        return rgb, rad, name, True
    if ext in _IMAGE_EXTENSIONS or ext == "":
        return _load_image_rgb(name, content, size), None, name, False
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type for {name}. Use PNG/JPG/WebP or NetCDF/HDF5.",
    )


async def _load_upload_triplet_frames(
    t0_file: UploadFile,
    t2_file: UploadFile,
    t1_file: UploadFile | None,
    size: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, str, str, str, list[np.ndarray | None], list[np.ndarray | None]]:
    t0_name, t0_bytes = await _read_upload_bytes(t0_file)
    t2_name, t2_bytes = await _read_upload_bytes(t2_file)
    t1_bytes: bytes | None = None
    t1_name = ""
    if t1_file is not None and t1_file.filename:
        t1_name, t1_bytes = await _read_upload_bytes(t1_file)

    nc_inputs = [_is_nc_upload(t0_name), _is_nc_upload(t2_name)]
    if t1_bytes is not None:
        nc_inputs.append(_is_nc_upload(t1_name))
    if any(nc_inputs) and not all(nc_inputs):
        raise HTTPException(
            status_code=400,
            detail="Use the same format for all frames: either all images (PNG/JPG) or all NetCDF/HDF5.",
        )

    rads: list[np.ndarray | None] = [None, None, None]
    vmin = vmax = None
    if all(nc_inputs):
        r0 = _load_radiance_bytes(t0_name, t0_bytes)
        r2 = _load_radiance_bytes(t2_name, t2_bytes)
        r1 = _load_radiance_bytes(t1_name, t1_bytes) if t1_bytes else None
        stretch = [r0, r2] if r1 is None else [r0, r1, r2]
        vmin, vmax = ps12.global_stretch(stretch)
        rads = [r0, r1, r2]
        t0 = ps12.rad_to_rgb(r0, vmin, vmax, size)
        t2 = ps12.rad_to_rgb(r2, vmin, vmax, size)
        t1_gt = ps12.rad_to_rgb(r1, vmin, vmax, size) if r1 is not None else None
    else:
        t0 = _load_image_rgb(t0_name, t0_bytes, size)
        t2 = _load_image_rgb(t2_name, t2_bytes, size)
        t1_gt = _load_image_rgb(t1_name, t1_bytes, size) if t1_bytes else None

    disk_masks: list[np.ndarray | None] = []
    for rad in rads:
        disk_masks.append(ps12.rad_to_diskmask(rad, size) if rad is not None else None)

    n1 = t1_name or "(not provided)"
    return t0, t1_gt, t2, t0_name, n1, t2_name, rads, disk_masks


def warm_upload_nc_cache(
    upload_id: str,
    names: tuple[str, str, str],
    raw_rads: list[np.ndarray | None],
    pred_rgb: np.ndarray,
    rad_var: str,
) -> None:
    """Cache NetCDF exports for uploaded radiance triplets."""
    finite = [r for r in raw_rads if r is not None]
    if not finite:
        return
    vmin, vmax = ps12.global_stretch(finite)
    n0, n1, n2 = names
    entries: dict[str, tuple[bytes, str, str]] = {}
    for frame, rad, name in (
        ("t0", raw_rads[0], n0),
        ("t1_gt", raw_rads[1], n1),
        ("t2", raw_rads[2], n2),
    ):
        if rad is None:
            continue
        nc = write_radiance_nc_bytes(rad, rad_var, f"Scan {frame}")
        fname = name if name.lower().endswith((".nc", ".h5")) else f"{name}.nc"
        entries[frame] = (nc, fname, "application/x-netcdf")
    nc_pred = write_predicted_nc_bytes(pred_rgb, rad_var, vmin, vmax)
    entries["t1_pred"] = (nc_pred, predicted_nc_filename(n0), "application/x-netcdf")
    _upload_nc_cache[upload_id] = entries


@lru_cache(maxsize=1)
def cached_model():
    return ps12.get_model()


def require_model():
    if not ps12.FINETUNED_CKPT.exists():
        raise HTTPException(
            status_code=503,
            detail=f"GOES fine-tuned checkpoint missing: {ps12.FINETUNED_CKPT}",
        )
    if not ps12.rife_ready():
        raise HTTPException(status_code=503, detail=f"RIFE not installed. Run: {ps12.setup_rife_hint()}")
    return cached_model()


def resolve_files(
    source_id: str | None,
    day: str | None,
    hour: int | None,
    local_folder: str | None,
) -> list[NcRef]:
    if local_folder:
        folder = ps12.Path(local_folder)
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail=f"Folder not found: {local_folder}")
        return ps12.list_nc_files(folder)
    if not source_id or not day:
        raise HTTPException(status_code=400, detail="source_id and day required for catalog mode")
    dicts = ps12._cached_list_scans(source_id, day, hour)
    if not dicts:
        raise HTTPException(status_code=404, detail="No scans found for this date/source")
    return [NcRef.from_dict(d) for d in dicts]


def build_timing_rows(
    name_t0: str,
    name_t1: str | None,
    name_t2: str,
    target_gap_min: float | None = None,
) -> list[dict[str, str]]:
    t0_scan = parse_goes_scan_time(name_t0)
    t2_scan = parse_goes_scan_time(name_t2)
    t1_gt_scan = parse_goes_scan_time(name_t1) if name_t1 else None

    if not (t0_scan and t2_scan):
        rows = []
        if target_gap_min is not None:
            rows.append({
                "frame": "Target step",
                "time_utc": f"{target_gap_min:.0f} min each · {target_gap_min * 2:.0f} min total",
            })
        return rows

    gap_total = t2_scan - t0_scan
    t_pred = t0_scan + gap_total / 2
    rows: list[dict[str, str]] = []
    if target_gap_min is not None:
        rows.append({
            "frame": "Target step (t0→t1, t1→t2)",
            "time_utc": f"{target_gap_min:.0f} min each · {target_gap_min * 2:.0f} min total",
        })
    rows.append({"frame": "t0 — input to model", "time_utc": ps12.format_utc(t0_scan)})
    if t1_gt_scan:
        rows.append({"frame": "t1 — ground truth (real scan)", "time_utc": ps12.format_utc(t1_gt_scan)})
        rows.append({"frame": "Gap t0 → t1 (actual)", "time_utc": ps12.format_gap(t1_gt_scan - t0_scan)})
        rows.append({"frame": "Gap t1 → t2 (actual)", "time_utc": ps12.format_gap(t2_scan - t1_gt_scan)})
    rows.extend([
        {"frame": "t2 — input to model", "time_utc": ps12.format_utc(t2_scan)},
        {"frame": "Gap t0 → t2 (total)", "time_utc": ps12.format_gap(gap_total)},
        {"frame": "Predicted t1 — RIFE midpoint", "time_utc": ps12.format_utc(t_pred)},
    ])
    if t1_gt_scan:
        offset_s = (t1_gt_scan - t_pred).total_seconds()
        rows.append({"frame": "GT t1 vs predicted offset", "time_utc": f"{offset_s:.0f} s"})
    return rows


def triplet_meta(name_t0: str, name_t1: str, name_t2: str, gap_min: float, satellite: str) -> dict:
    t0_scan = parse_goes_scan_time(name_t0)
    t2_scan = parse_goes_scan_time(name_t2)
    t1_scan = parse_goes_scan_time(name_t1)
    cadence_from = gap_min * 2 if gap_min else 30
    return {
        "t0": t0_scan.isoformat() + "Z" if t0_scan else None,
        "t1_pred": (
            (t0_scan + (t2_scan - t0_scan) / 2).isoformat() + "Z"
            if t0_scan and t2_scan
            else (t1_scan.isoformat() + "Z" if t1_scan else None)
        ),
        "t1_gt": t1_scan.isoformat() + "Z" if t1_scan else None,
        "t2": t2_scan.isoformat() + "Z" if t2_scan else None,
        "cadence_from": cadence_from,
        "cadence_to": gap_min,
        "satellite": satellite,
    }


class ProgressShim:
    def progress(self, frac: float, text: str = "") -> None:
        pass


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TripletLoadRequest(BaseModel):
    source_id: str | None = None
    day: str | None = None
    hour: int | None = None
    gap_min: float = 20.0
    triplet_index: int = 0
    size: int = Field(default=512, ge=128, le=768)
    local_folder: str | None = None


class PredictRequest(TripletLoadRequest):
    scale: float = Field(default=1.0, ge=0.25, le=1.0)


class BatchRequest(TripletLoadRequest):
    scale: float = 1.0
    max_triplets: int = Field(default=24, ge=1, le=100)


class FlowRequest(PredictRequest):
    quiver_step: int = 24
    quiver_window: int = 12
    timelapse_ms: int = 700
    # Streamlit single-tab timelapse: subsample grid only — no Phase-3 quality mask
    min_mag_px: float = 0.001


class FrameDownloadRequest(PredictRequest):
    frame: Literal["t0", "t1_gt", "t2", "t1_pred"]


class ReportRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    triplet: dict[str, Any] | None = None
    predict: dict[str, Any] | None = None
    flow: dict[str, Any] | None = None
    batch: dict[str, Any] | None = None
    health: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    scorecard: list[dict[str, Any]] = Field(default_factory=list)
    images: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "status": "ok" if ps12.FINETUNED_CKPT.exists() and ps12.rife_ready() else "degraded",
        "rife_ready": ps12.rife_ready(),
        "checkpoint_exists": ps12.FINETUNED_CKPT.exists(),
        "checkpoint_path": str(ps12.FINETUNED_CKPT),
        "device": "cuda" if cuda else "cpu",
        "cuda_available": cuda,
        "aws_mode": aws_deploy_mode(),
        "practical_rife": str(ps12.PRACTICAL_RIFE),
    }


@app.get("/api/sources")
def list_sources() -> list[dict[str, Any]]:
    sources = satellite_sources()
    return [
        {
            "id": sid,
            "label": src.label,
            "default_cadence_min": src.default_cadence_min,
            "fmt": src.fmt,
            "bucket": src.bucket,
        }
        for sid, src in sources.items()
    ]


@app.get("/api/local-folders")
def local_folders() -> list[dict[str, Any]]:
    folders = ps12.find_nc_folders()
    return [
        {"path": str(p), "label": ps12.folder_label(p), "n_scans": len(ps12.list_nc_files(p))}
        for p in folders
    ]


@app.get("/api/scans")
def list_scans(
    source_id: str = Query(...),
    day: str = Query(...),
    hour: int | None = Query(None),
) -> dict[str, Any]:
    try:
        dicts = ps12._cached_list_scans(source_id, day, hour)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    scans = [
        {
            "file": d.get("name", d["uri"].rsplit("/", 1)[-1]),
            "scan_utc": ps12.format_utc(
                datetime.fromisoformat(d["scan_time"]) if d.get("scan_time") else None
            ),
            "uri": d["uri"][:120],
        }
        for d in dicts[:50]
    ]
    return {"total": len(dicts), "scans": scans}


@app.get("/api/gap-info")
def gap_info(
    source_id: str | None = Query(None),
    day: str | None = Query(None),
    hour: int | None = Query(None),
    gap_min: float = Query(20.0),
    local_folder: str | None = Query(None),
) -> dict[str, Any]:
    if local_folder:
        folder = ps12.Path(local_folder)
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail=f"Folder not found: {local_folder}")
        files = ps12.list_nc_files(folder)
    elif source_id and day:
        dicts = ps12._cached_list_scans(source_id, day, hour)
        files = [NcRef.from_dict(d) for d in dicts]
    else:
        raise HTTPException(status_code=400, detail="source_id and day required for catalog mode")

    if not files:
        skip = max(1, int(round(gap_min / 10.0)))
        return {
            "cadence_min": 10.0,
            "skip": skip,
            "actual_step_min": float(skip * 10),
            "total_span_min": float(skip * 20),
            "n_triplets": 0,
            "min_files_needed": 2 * skip + 1,
            "n_triplets_listed": 0,
        }

    info = ps12.gap_step_info(files, gap_min)
    triplets = ps12.find_triplets_by_gap(files, gap_min)
    return {**info, "n_triplets_listed": len(triplets)}


@app.post("/api/triplet/load")
def load_triplet(req: TripletLoadRequest) -> dict[str, Any]:
    files = resolve_files(req.source_id, req.day, req.hour, req.local_folder)
    triplets = ps12.find_triplets_by_gap(files, req.gap_min)
    if not triplets:
        raise HTTPException(status_code=404, detail="No triplets for this gap setting")
    if req.triplet_index >= len(triplets):
        raise HTTPException(status_code=400, detail=f"triplet_index out of range (max {len(triplets) - 1})")

    info = ps12.gap_step_info(files, req.gap_min)
    i0, i1, i2 = triplets[req.triplet_index]
    src = satellite_sources().get(req.source_id or "")
    rad_var = src.rad_var if src else None
    label = src.label if src else (ps12.folder_label(ps12.Path(req.local_folder)) if req.local_folder else "Local")

    t0, t1_gt, t2, n0, n1, n2, _rads, _disk, _refs = ps12.load_triplet_at_indices(
        files, i0, i1, i2, req.size, rad_var=rad_var
    )

    return {
        "triplet_index": req.triplet_index,
        "n_triplets": len(triplets),
        "indices": {"i0": i0, "i1": i1, "i2": i2},
        "names": {"t0": n0, "t1": n1, "t2": n2},
        "gap_info": info,
        "timing": build_timing_rows(n0, n1, n2, info["actual_step_min"]),
        "meta": triplet_meta(n0, n1, n2, info["actual_step_min"], label),
        "frames": {
            "t0": img_to_b64(t0),
            "t1_gt": img_to_b64(t1_gt),
            "t2": img_to_b64(t2),
        },
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    model, device, _version, ckpt, fp16 = require_model()
    files = resolve_files(req.source_id, req.day, req.hour, req.local_folder)
    triplets = ps12.find_triplets_by_gap(files, req.gap_min)
    if req.triplet_index >= len(triplets):
        raise HTTPException(status_code=400, detail="triplet_index out of range")

    i0, i1, i2 = triplets[req.triplet_index]
    src = satellite_sources().get(req.source_id or "")
    rad_var = src.rad_var if src else None
    t0, t1_gt, t2, n0, n1, n2, _rads, _disk, _refs = ps12.load_triplet_at_indices(
        files, i0, i1, i2, req.size, rad_var=rad_var
    )

    pred = ps12.predict_t1(t0, t2, model, device, req.scale, fp16)
    lin = ps12.linear_t1(t0, t2)
    rife_metrics = ps12.all_metrics(pred, t1_gt)
    linear_metrics = ps12.all_metrics(lin, t1_gt)

    return {
        "checkpoint": ckpt,
        "names": {"t0": n0, "t1": n1, "t2": n2},
        "metrics": {"rife": rife_metrics, "linear": linear_metrics},
        "frames": {
            "predicted": img_to_b64(pred),
            "linear": img_to_b64(lin),
            "ground_truth": img_to_b64(t1_gt),
            "t0": img_to_b64(t0),
            "t2": img_to_b64(t2),
        },
    }


@app.post("/api/batch/validate")
def batch_validate(req: BatchRequest) -> dict[str, Any]:
    model, device, version, _ckpt, fp16 = require_model()
    files = resolve_files(req.source_id, req.day, req.hour, req.local_folder)
    src = satellite_sources().get(req.source_id or "")
    rad_var = src.rad_var if src else None

    class _Prog:
        def progress(self, frac: float, text: str = "") -> None:
            pass

    rows = ps12.run_batch_validation_gaps(
        files,
        req.size,
        req.gap_min,
        model,
        device,
        version,
        req.scale,
        fp16,
        _Prog(),
        max_triplets=req.max_triplets,
        rad_var=rad_var,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No triplets to validate")

    rife_ssim = [r["rife_ssim"] for r in rows]
    linear_ssim = [r["linear_ssim"] for r in rows]
    summary = {
        "rife_ssim_mean": float(np.mean(rife_ssim)),
        "linear_ssim_mean": float(np.mean(linear_ssim)),
        "delta_ssim": float(np.mean(rife_ssim) - np.mean(linear_ssim)),
        "rife_fsim_mean": float(np.mean([r["rife_fsim"] for r in rows])),
        "n": len(rows),
    }
    return {"rows": rows, "summary": summary}


def _build_flow_payload(
    req: FlowRequest,
    t0: np.ndarray,
    t1_gt: np.ndarray | None,
    t2: np.ndarray,
    rows: list,
    data: dict,
    disk_masks: list | None,
    refs: list,
    ckpt: str,
) -> dict[str, Any]:
    u_amv, v_amv = data["amv_uv"]
    h_amv, w_amv = u_amv.shape

    try:
        if refs and refs[0].uri:
            lat, lon = ps12._cached_geogrid(
                refs[0].uri, refs[0].source_id, h_amv, w_amv, refs[0].segment_uris,
            )
            geo_note = f"Projection from {refs[0].name}"
        else:
            lat, lon = ps12._fallback_geogrid(h_amv, w_amv)
            geo_note = "Approximate full-disk lat/lon"
    except Exception as exc:
        lat, lon = ps12._fallback_geogrid(h_amv, w_amv)
        geo_note = f"Geo fallback: {exc}"

    from amv import (  # noqa: E402
        amv_speed_ms,
        build_motion_timelapses,
        composite_sparse_arrows,
        full_disk_geometric_mask,
        render_arrow_layer,
        resize_mask_nearest,
        subsample_amv_grid,
    )

    ms = amv_speed_ms(u_amv, v_amv, lat, lon, req.gap_min)

    if disk_masks:
        on_disk = disk_masks[0] & disk_masks[2]
        if on_disk.shape != u_amv.shape:
            on_disk = resize_mask_nearest(on_disk, *u_amv.shape)
    else:
        on_disk = full_disk_geometric_mask(*u_amv.shape)

    sparse_px = subsample_amv_grid(
        u_amv,
        v_amv,
        stride=req.quiver_step,
        window=req.quiver_window,
        on_disk=on_disk,
        min_mag_px=req.min_mag_px,
    )
    sparse = sparse_px
    t1_pred = data["full"]

    arrow_kw = {
        "min_mag_px": 0.0,
        "color_by_speed": True,
        "disk_mask": on_disk,
        "thickness": 1,
        "tip_length": 0.35,
        "thin_arrows": True,
    }

    overlay_frames: dict[str, str] = {}
    gifs: dict[str, str] = {}
    arrow_stats = {
        "n_shown": int(sparse_px.get("n_shown", 0)),
        "n_grid": int(sparse_px.get("n_grid", 0)),
    }

    if sparse_px.get("n_shown", 0) > 0:
        layer = render_arrow_layer(t0.shape[:2], sparse_px, **arrow_kw)
        overlay_frames = {
            "t0": img_to_b64(composite_sparse_arrows(t0, sparse_px, arrow_layer=layer)),
            "t2": img_to_b64(composite_sparse_arrows(t2, sparse_px, arrow_layer=layer)),
            "t1_pred": img_to_b64(composite_sparse_arrows(t1_pred, sparse_px, arrow_layer=layer)),
        }
        if t1_gt is not None:
            overlay_frames["t1_gt"] = img_to_b64(composite_sparse_arrows(t1_gt, sparse_px, arrow_layer=layer))
        lapses = build_motion_timelapses(t0, t1_gt, t2, t1_pred, sparse_px, arrow_kw=arrow_kw)
        gifs["predicted"] = base64.b64encode(
            ps12.frames_to_gif_bytes(lapses["predicted"], duration_ms=req.timelapse_ms)
        ).decode("ascii")
        if lapses.get("ground_truth"):
            gifs["ground_truth"] = base64.b64encode(
                ps12.frames_to_gif_bytes(lapses["ground_truth"], duration_ms=req.timelapse_ms)
            ).decode("ascii")

    ablation_chart = [
        {"method": r["method"].replace(" ", "\n"), "ssim": r.get("ssim"), "fsim": r.get("fsim")}
        for r in rows
        if r.get("ssim") is not None and not r["method"].startswith("---")
    ]

    speed_stats = {}
    if ms and ms.get("speed_ms") is not None:
        sm = ms["speed_ms"]
        valid = np.isfinite(sm) & (sm > 0)
        if np.any(valid):
            speed_stats = {
                "mean_ms": float(np.mean(sm[valid])),
                "p99_ms": float(np.percentile(sm[valid], 99)),
                "max_ms": float(np.max(sm[valid])),
            }

    arrows = []
    if sparse.get("n_shown", 0) > 0:
        X = np.asarray(sparse["X"]).ravel()
        Y = np.asarray(sparse["Y"]).ravel()
        U = np.asarray(sparse["U"]).ravel()
        V = np.asarray(sparse["V"]).ravel()
        for x, y, u, v in zip(X, Y, U, V):
            if np.isfinite(u) and np.isfinite(v):
                arrows.append({"x": float(x), "y": float(y), "u": float(u), "v": float(v)})

    return {
        "checkpoint": ckpt,
        "geo_note": geo_note,
        "ablation_rows": rows,
        "ablation_chart": ablation_chart,
        "speed_stats": speed_stats,
        "arrows": arrows[:500],
        "n_arrows": len(arrows),
        "arrow_stats": arrow_stats,
        "flow_rgb": img_to_b64(ps12.flow_to_rgb(u_amv, v_amv)),
        "overlay_frames": overlay_frames,
        "gifs": gifs,
        "frames": {
            "linear": img_to_b64(data["linear"]),
            "farneback": img_to_b64(data["farneback"]),
            "flow_only": img_to_b64(data["flow_only"]),
            "full": img_to_b64(data["full"]),
            **({"ground_truth": img_to_b64(t1_gt)} if t1_gt is not None else {}),
            "background": img_to_b64(t0),
        },
    }


def _metrics_or_none(pred: np.ndarray, gt: np.ndarray | None, baseline: np.ndarray) -> dict[str, Any] | None:
    if gt is None:
        return None
    return {
        "rife": ps12.all_metrics(pred, gt),
        "linear": ps12.all_metrics(baseline, gt),
    }


def _run_predict_motion(
    req: FlowRequest,
    t0: np.ndarray,
    t1_gt: np.ndarray | None,
    t2: np.ndarray,
    n0: str,
    n1: str,
    n2: str,
    raw_rads: list[np.ndarray | None],
    disk_masks: list | None,
    refs: list,
    *,
    upload_id: str | None = None,
) -> dict[str, Any]:
    model, device, _version, ckpt, _fp16 = require_model()
    rows, data = ps12.compare_interpolation_methods(t0, t2, t1_gt, model, device, req.scale)
    pred = data["full"]
    lin = data["linear"]
    metrics = _metrics_or_none(pred, t1_gt, lin)

    src = satellite_sources().get(req.source_id or "")
    rad_var = (src.rad_var if src else None) or "Rad"
    finite_rads = [r for r in raw_rads if r is not None]
    if upload_id and finite_rads:
        warm_upload_nc_cache(upload_id, (n0, n1, n2), raw_rads, pred, rad_var)
    elif finite_rads and len(finite_rads) >= 3 and all(r is not None for r in raw_rads):
        warm_frame_nc_cache(req, (n0, n1, n2), raw_rads, pred, rad_var)

    predict_result: dict[str, Any] = {
        "checkpoint": ckpt,
        "names": {"t0": n0, "t1": n1, "t2": n2},
        "metrics": metrics,
        "has_ground_truth": t1_gt is not None,
        "frames": {
            "predicted": img_to_b64(pred),
            "linear": img_to_b64(lin),
            **({"ground_truth": img_to_b64(t1_gt)} if t1_gt is not None else {}),
            "t0": img_to_b64(t0),
            "t2": img_to_b64(t2),
        },
    }
    if upload_id:
        predict_result["upload_id"] = upload_id
    flow_result = _build_flow_payload(req, t0, t1_gt, t2, rows, data, disk_masks, refs, ckpt)
    return {"predict": predict_result, "flow": flow_result}


def _load_triplet_for_request(req: TripletLoadRequest):
    files = resolve_files(req.source_id, req.day, req.hour, req.local_folder)
    triplets = ps12.find_triplets_by_gap(files, req.gap_min)
    if not triplets:
        raise HTTPException(status_code=404, detail="No triplets for this gap setting")
    if req.triplet_index >= len(triplets):
        raise HTTPException(status_code=400, detail="triplet_index out of range")
    i0, i1, i2 = triplets[req.triplet_index]
    src = satellite_sources().get(req.source_id or "")
    rad_var = src.rad_var if src else None
    rads = [ps12.load_nc_radiance(files[i], rad_var=rad_var) for i in (i0, i1, i2)]
    vmin, vmax = ps12.global_stretch(rads)
    t0 = ps12.rad_to_rgb(rads[0], vmin, vmax, req.size)
    t1_gt = ps12.rad_to_rgb(rads[1], vmin, vmax, req.size)
    t2 = ps12.rad_to_rgb(rads[2], vmin, vmax, req.size)
    disk_masks = [ps12.rad_to_diskmask(r, req.size) for r in rads]
    names = (files[i0].name, files[i1].name, files[i2].name)
    refs = [files[i0], files[i1], files[i2]]
    return t0, t1_gt, t2, names[0], names[1], names[2], rads, disk_masks, refs


@app.post("/api/predict-and-motion")
def predict_and_motion(req: FlowRequest) -> dict[str, Any]:
    """Single S3 load: RIFE midpoint + AMV overlays + timelapse GIFs."""
    t0, t1_gt, t2, n0, n1, n2, raw_rads, disk_masks, refs = _load_triplet_for_request(req)
    return _run_predict_motion(req, t0, t1_gt, t2, n0, n1, n2, raw_rads, disk_masks, refs)


@app.post("/api/flow/extract")
def flow_extract(req: FlowRequest) -> dict[str, Any]:
    model, device, _version, ckpt, _fp16 = require_model()
    t0, t1_gt, t2, _n0, _n1, _n2, _rads, disk_masks, refs = _load_triplet_for_request(req)

    rows, data = ps12.compare_interpolation_methods(t0, t2, t1_gt, model, device, req.scale)
    return _build_flow_payload(req, t0, t1_gt, t2, rows, data, disk_masks, refs, ckpt)


@app.post("/api/download/frame")
def download_frame(req: FrameDownloadRequest) -> dict[str, Any]:
    """Download radiance NetCDF — instant when pre-warmed by predict-and-motion."""
    cached = get_cached_frame_nc(req)
    if cached:
        nc_bytes, fname, mime = cached
    else:
        src = satellite_sources().get(req.source_id or "")
        rad_var = (src.rad_var if src else None) or "Rad"
        if req.frame == "t1_pred":
            model, device, _version, _ckpt, _fp16 = require_model()
            t0, t1_gt, t2, n0, _n1, _n2, raw_rads, _disk, _refs = _load_triplet_for_request(req)
            _rows, data = ps12.compare_interpolation_methods(t0, t2, t1_gt, model, device, req.scale)
            vmin, vmax = ps12.global_stretch(raw_rads)
            nc_bytes = write_predicted_nc_bytes(data["full"], rad_var, vmin, vmax)
            fname = predicted_nc_filename(n0)
            mime = "application/x-netcdf"
        else:
            _t0, _t1, _t2, _n0, _n1, _n2, raw_rads, _disk, _refs = _load_triplet_for_request(req)
            idx = {"t0": 0, "t1_gt": 1, "t2": 2}[req.frame]
            name = (_n0, _n1, _n2)[idx]
            nc_bytes = write_radiance_nc_bytes(raw_rads[idx], rad_var, f"Scan {req.frame}")
            fname = name if name.lower().endswith((".nc", ".h5")) else f"{name}.nc"
            mime = "application/x-netcdf"

    return {
        "filename": fname,
        "mime": mime,
        "data_b64": base64.b64encode(nc_bytes).decode("ascii"),
        "cached": cached is not None,
    }


@app.post("/api/download/report")
def download_report(req: ReportRequest) -> dict[str, Any]:
    """Generate PS12 evaluation PDF with metrics, ablation, and frame comparisons."""
    if not req.predict:
        raise HTTPException(status_code=400, detail="Run Predict t1 before downloading the report")
    try:
        pdf_bytes = generate_report_pdf(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return {
        "filename": f"isro_ps12_report_{stamp}.pdf",
        "mime": "application/pdf",
        "data_b64": base64.b64encode(pdf_bytes).decode("ascii"),
    }


@app.post("/api/upload/triplet")
async def upload_triplet_load(
    t0: UploadFile = File(..., description="Input frame t0 (PNG/JPG or NetCDF/HDF5)"),
    t2: UploadFile = File(..., description="Input frame t2 (PNG/JPG or NetCDF/HDF5)"),
    t1: UploadFile | None = File(None, description="Optional ground-truth t1"),
    size: int = Form(512),
    gap_min: float = Form(20.0),
) -> dict[str, Any]:
    """Preview uploaded frames without running RIFE inference."""
    t0_rgb, t1_gt, t2_rgb, n0, n1, n2, _rads, _masks = await _load_upload_triplet_frames(
        t0, t2, t1, size
    )
    cadence_to = max(gap_min / 2.0, 1.0)
    return {
        "triplet_index": 0,
        "n_triplets": 1,
        "indices": {"i0": 0, "i1": 0, "i2": 0},
        "names": {"t0": n0, "t1": n1, "t2": n2},
        "gap_info": {
            "cadence_min": gap_min,
            "skip": 1,
            "actual_step_min": gap_min,
            "total_span_min": gap_min * 2,
            "n_triplets": 1,
            "min_files_needed": 2,
        },
        "timing": [
            {"frame": "t0 (upload)", "time_utc": n0},
            {"frame": "t1 GT (upload)" if t1_gt is not None else "t1 GT (optional)", "time_utc": n1},
            {"frame": "t2 (upload)", "time_utc": n2},
        ],
        "meta": {
            "t0": None,
            "t1_pred": None,
            "t1_gt": None,
            "t2": None,
            "cadence_from": gap_min,
            "cadence_to": cadence_to,
            "satellite": "User upload",
        },
        "frames": {
            "t0": img_to_b64(t0_rgb),
            "t1_gt": img_to_b64(t1_gt) if t1_gt is not None else "",
            "t2": img_to_b64(t2_rgb),
        },
        "has_ground_truth": t1_gt is not None,
    }


@app.post("/api/upload/predict-and-motion")
async def upload_predict_and_motion(
    t0: UploadFile = File(...),
    t2: UploadFile = File(...),
    t1: UploadFile | None = File(None),
    size: int = Form(512),
    scale: float = Form(1.0),
    gap_min: float = Form(20.0),
    quiver_step: int = Form(24),
    quiver_window: int = Form(12),
    timelapse_ms: int = Form(700),
    min_mag_px: float = Form(0.001),
) -> dict[str, Any]:
    """Predict midpoint frame from uploaded t0 + t2; optional ground-truth t1 for metrics."""
    t0_rgb, t1_gt, t2_rgb, n0, n1, n2, rads, disk_masks = await _load_upload_triplet_frames(
        t0, t2, t1, size
    )
    upload_id = secrets.token_hex(8)
    req = FlowRequest(
        size=size,
        scale=scale,
        gap_min=gap_min,
        quiver_step=quiver_step,
        quiver_window=quiver_window,
        timelapse_ms=timelapse_ms,
        min_mag_px=min_mag_px,
    )
    return _run_predict_motion(
        req,
        t0_rgb,
        t1_gt,
        t2_rgb,
        n0,
        n1,
        n2,
        rads,
        disk_masks,
        [],
        upload_id=upload_id,
    )


@app.get("/api/upload/download/frame")
def upload_download_frame(
    upload_id: str = Query(...),
    frame: Literal["t0", "t1_gt", "t2", "t1_pred"] = Query(...),
) -> dict[str, Any]:
    """Download NetCDF for uploaded radiance frames (after predict)."""
    cached = get_upload_frame_nc(upload_id, frame)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="NetCDF not available — upload .nc/.h5 frames and run Predict first.",
        )
    nc_bytes, fname, mime = cached
    return {
        "filename": fname,
        "mime": mime,
        "data_b64": base64.b64encode(nc_bytes).decode("ascii"),
        "cached": True,
    }


@app.get("/api/training/status")
def training_status() -> dict[str, Any]:
    log_path = ps12.FINETUNED_DIR / "training_log.json"
    meta: dict[str, Any] = {}
    if log_path.exists():
        meta = json.loads(log_path.read_text())
    stat_ckpt = ps12.ROOT / "checkpoints" / "goes_finetuned_statistical" / "flownet.pkl"
    return {
        "checkpoint_exists": ps12.FINETUNED_CKPT.exists(),
        "checkpoint_path": str(ps12.FINETUNED_CKPT),
        "stage1_backup_exists": stat_ckpt.exists(),
        "training_mode": meta.get("training_mode"),
        "stage": meta.get("stage"),
        "full_eval": meta.get("full_eval", {}),
    }
