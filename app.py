r"""
ISRO PS12 â€” ONE FILE Streamlit app (entire ISRO2026 workflow).

Run (from anywhere):
    cd D:\AI\llmScience\ISRO2026
    ..\..venv\Scripts\streamlit run app.py

Or from repo root:
    .venv\Scripts\streamlit run ISRO2026\app.py

Prerequisite (run once):
    .venv\Scripts\python.exe ISRO2026\setup_rife.py

Tabs:
  1. Single triplet  â€” t0 + t2 -> predict t1 (+ metrics vs real t1)
  2. Batch validate  â€” whole .nc folder, ISRO metrics report
  3. Video / GIF     â€” increase frame rate of an animation
  4. Motion vectors  â€” optical flow (u,v), quiver, ablation vs Farneback
  5. Fine-tune       â€” train on GOES data
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from s3_catalog import (
    NcRef,
    aws_deploy_mode,
    list_scans_for_day,
    load_nc_latlon_ref,
    load_nc_radiance_ref,
    load_nc_radiance_uri,
    mount_roots,
    parse_goes_scan_time,
    satellite_sources,
    source_help_text,
)
from amv import (
    amv_interval_caption,
    amv_speed_ms,
    bidirectional_disagreement,
    build_filtered_amv,
    build_motion_timelapses,
    composite_sparse_arrows,
    erode_mask,
    filtered_amv_caption,
    full_disk_geometric_mask,
    render_arrow_layer,
    resize_geogrid,
    resize_mask_nearest,
    rife_amv_uv,
    rife_forward_uv,
    speed_ms_caption,
    subsample_amv_grid,
)

# =============================================================================
# PATHS
# =============================================================================
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def _resolve_practical_rife() -> Path:
    for cand in (ROOT / "Practical-RIFE", REPO_ROOT / "ISRO2026" / "Practical-RIFE"):
        if (cand / "train_log" / "flownet.pkl").exists():
            return cand
    return ROOT / "Practical-RIFE"


PRACTICAL_RIFE = _resolve_practical_rife()
TRAIN_LOG = PRACTICAL_RIFE / "train_log"
FINETUNED_DIR = ROOT / "checkpoints" / "goes_finetuned"
FINETUNED_CKPT = FINETUNED_DIR / "flownet.pkl"
OUTPUT_DIR = ROOT / "output"
NC_SEARCH_DIRS = [
    ROOT / "goes19_c13",
    REPO_ROOT / "goes19_c13",
    REPO_ROOT / "ISRO2026" / "goes19_c13",
    REPO_ROOT / "goes19_c13_arthur",
]


# =============================================================================
# 1. LOAD SATELLITE DATA (.nc / images)
# =============================================================================

def load_nc_radiance(path: Path | NcRef | str, rad_var: str | None = None) -> np.ndarray:
    """Load radiance from local path, s3:// URI, or HDF5/Himawari HSD (streamed)."""
    if isinstance(path, NcRef):
        return load_nc_radiance_ref(path, rad_var)
    return load_nc_radiance_uri(str(path), rad_var=rad_var)


def global_stretch(rads: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([r[np.isfinite(r)].ravel() for r in rads])
    return float(np.percentile(flat, 2)), float(np.percentile(flat, 98))


def rad_to_rgb(rad: np.ndarray, vmin: float, vmax: float, size: int) -> np.ndarray:
    img = np.nan_to_num(rad, nan=vmin)
    if vmax <= vmin:
        vmax = vmin + 1.0
    g = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    rgb = np.stack([(g * 255).astype(np.uint8)] * 3, axis=-1)
    return np.array(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))


def rad_to_gray01(rad: np.ndarray, vmin: float, vmax: float, size: int) -> np.ndarray:
    """Stretched grayscale [0,1] at display resolution â€” aligned with rad_to_rgb."""
    img = np.nan_to_num(rad, nan=vmin)
    if vmax <= vmin:
        vmax = vmin + 1.0
    g = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    gray_u8 = (g * 255).astype(np.uint8)
    resized = np.array(Image.fromarray(gray_u8).resize((size, size), Image.Resampling.BILINEAR))
    return resized.astype(np.float64) / 255.0


def rad_to_diskmask(rad: np.ndarray, size: int) -> np.ndarray:
    """
    Boolean Earth-disk mask at display resolution: True where the raw radiance had a
    real reading, False where it was off-disk fill/space (NaN in `rad`).

    `rad_to_gray01`/`rad_to_rgb` call `np.nan_to_num(rad, nan=vmin)`, which makes space
    visually identical to a very cold pixel â€” that's exactly why downstream motion-vector
    filtering couldn't reliably tell space from clouds. This keeps the real footprint
    around. Resized with NEAREST (not bilinear) so the limb edge stays crisp instead of
    picking up a blurred halo of "maybe valid" pixels.
    """
    mask_u8 = (np.isfinite(rad).astype(np.uint8)) * 255
    resized = np.array(Image.fromarray(mask_u8).resize((size, size), Image.Resampling.NEAREST))
    return resized > 127


def load_nc_folder_frames(folder: Path, size: int, limit: int | None = None) -> tuple[list[np.ndarray], list[Path]]:
    files = sorted(folder.glob("*.nc"))
    if limit:
        files = files[:limit]
    if len(files) < 3:
        raise ValueError(f"Need >=3 .nc files in {folder}")
    rads = [load_nc_radiance(p) for p in files]
    vmin, vmax = global_stretch(rads)
    frames = [rad_to_rgb(r, vmin, vmax, size) for r in rads]
    return frames, files


def list_nc_files(folder: Path) -> list[NcRef]:
    return [
        NcRef(uri=str(f.resolve()), scan_time=parse_goes_scan_time(f.name))
        for f in sorted(folder.glob("*.nc"))
    ]


def _scan_time(ref: NcRef) -> datetime | None:
    return ref.scan_time or parse_goes_scan_time(ref.name)


def estimate_scan_cadence_min(files: list[NcRef]) -> float:
    times = [_scan_time(f) for f in files]
    gaps = []
    for i in range(len(times) - 1):
        if times[i] and times[i + 1]:
            gaps.append((times[i + 1] - times[i]).total_seconds() / 60.0)
    return float(np.median(gaps)) if gaps else 10.0


def resolve_gap_step(files: list[NcRef], gap_min: float) -> tuple[int, float]:
    """Map requested minutes to scan skip count (GOES ABI â‰ˆ10 min cadence)."""
    cadence = estimate_scan_cadence_min(files)
    skip = max(1, int(round(gap_min / cadence)))
    return skip, skip * cadence


def gap_step_info(files: list[NcRef], gap_min: float) -> dict:
    skip, actual_step = resolve_gap_step(files, gap_min)
    n_files = len(files)
    n_triplets = max(0, n_files - 2 * skip)
    return {
        "cadence_min": estimate_scan_cadence_min(files),
        "skip": skip,
        "actual_step_min": actual_step,
        "total_span_min": actual_step * 2,
        "n_triplets": n_triplets,
        "min_files_needed": 2 * skip + 1,
    }


def find_triplets_by_gap(
    files: list[NcRef],
    gap_min: float,
    tolerance_min: float | None = None,
) -> list[tuple[int, int, int]]:
    """
    Build triplets by skipping scans: t0, t0+skip, t0+2Â·skip.
    On ~10 min GOES cadence: 10â†’skip1, 20â†’skip2, 30â†’skip3.
    """
    if len(files) < 3:
        return []
    skip, _ = resolve_gap_step(files, gap_min)
    return [(i, i + skip, i + 2 * skip) for i in range(len(files) - 2 * skip)]


def load_triplet_at_indices(
    files: list[NcRef],
    i0: int,
    i1: int,
    i2: int,
    size: int,
    rad_var: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str, list[np.ndarray], list[np.ndarray], list[NcRef]]:
    rads = [load_nc_radiance(files[i], rad_var=rad_var) for i in (i0, i1, i2)]
    vmin, vmax = global_stretch(rads)
    frames = [rad_to_rgb(r, vmin, vmax, size) for r in rads]
    rads_display = [rad_to_gray01(r, vmin, vmax, size) for r in rads]
    disk_masks = [rad_to_diskmask(r, size) for r in rads]
    names = [files[i].name for i in (i0, i1, i2)]
    refs = [files[i] for i in (i0, i1, i2)]
    return frames[0], frames[1], frames[2], names[0], names[1], names[2], rads_display, disk_masks, refs


def load_triplet(folder: Path, triplet_idx: int, size: int, gap_min: float = 10.0):
    files = list_nc_files(folder)
    triplets = find_triplets_by_gap(files, gap_min)
    if not triplets:
        raise ValueError(
            f"No triplets for {gap_min:.0f} min step in {folder}. "
            "Try 10 min or add more .nc files."
        )
    if triplet_idx >= len(triplets):
        triplet_idx = len(triplets) - 1
    i0, i1, i2 = triplets[triplet_idx]
    return load_triplet_at_indices(files, i0, i1, i2, size)


def pil_to_rgb(uploaded, size: int) -> np.ndarray:
    return np.array(
        Image.open(uploaded).convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def load_gif_frames(source) -> list[np.ndarray]:
    img = Image.open(source)
    frames = []
    try:
        while True:
            frames.append(np.array(img.convert("RGB"), dtype=np.uint8))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


def load_video_frames(uploaded, max_frames: int) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(uploaded.getvalue())
        path = f.name
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
        Path(path).unlink(missing_ok=True)
    return frames


def find_nc_folders() -> list[Path]:
    found = []
    for p in NC_SEARCH_DIRS:
        if p.is_dir() and any(p.glob("*.nc")):
            found.append(p)
    # Arthur storm first â€” more scans, better for wider gaps
    return sorted(found, key=lambda p: (0 if "arthur" in p.name.lower() else 1, str(p)))


def folder_label(path: Path) -> str:
    n = len(list_nc_files(path))
    if "arthur" in path.name.lower():
        return f"Arthur storm â€” {n} scans (20â€“30 min gaps, active weather)"
    if "goes19" in path.name.lower() or "c13" in path.name.lower():
        return f"Calm day â€” {n} scans (10 min gaps only)"
    return f"{path.name} ({n} scans)"



def format_utc(dt: datetime | None) -> str:
    if dt is None:
        return "â€”"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_gap(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m} min"
    return f"{s} s"


def show_time_info(
    name_t0: str,
    name_t1: str | None,
    name_t2: str,
    target_gap_min: float | None = None,
):
    """Compact timing table (avoids oversized st.metric truncation)."""
    t0_scan = parse_goes_scan_time(name_t0)
    t2_scan = parse_goes_scan_time(name_t2)
    t1_gt_scan = parse_goes_scan_time(name_t1) if name_t1 else None

    if not (t0_scan and t2_scan):
        st.caption("Timestamps not parsed (need GOES filenames with `_sYYYYJJJHHMMSSd_`).")
        if target_gap_min is not None:
            st.caption(f"Target step: **{target_gap_min:.0f} min** (t0â†’t2 span **{target_gap_min * 2:.0f} min**).")
        return

    gap_total = t2_scan - t0_scan
    t_pred = t0_scan + gap_total / 2

    rows = [
        {"Frame": "t0 â€” input to model", "Time (UTC)": format_utc(t0_scan)},
    ]
    if t1_gt_scan:
        rows.append({"Frame": "t1 â€” ground truth (real scan)", "Time (UTC)": format_utc(t1_gt_scan)})
        rows.append({"Frame": "Gap t0 â†’ t1 (actual)", "Time (UTC)": format_gap(t1_gt_scan - t0_scan)})
        rows.append({"Frame": "Gap t1 â†’ t2 (actual)", "Time (UTC)": format_gap(t2_scan - t1_gt_scan)})
    rows.extend([
        {"Frame": "t2 â€” input to model", "Time (UTC)": format_utc(t2_scan)},
        {"Frame": "Gap t0 â†’ t2 (total)", "Time (UTC)": format_gap(gap_total)},
        {"Frame": "Predicted t1 â€” RIFE midpoint", "Time (UTC)": format_utc(t_pred)},
    ])
    if target_gap_min is not None:
        rows.insert(
            0,
            {
                "Frame": "Target step (t0â†’t1, t1â†’t2)",
                "Time (UTC)": f"{target_gap_min:.0f} min each Â· {target_gap_min * 2:.0f} min total",
            },
        )
    if t1_gt_scan:
        offset_s = (t1_gt_scan - t_pred).total_seconds()
        rows.append({"Frame": "GT t1 vs predicted offset", "Time (UTC)": f"{offset_s:.0f} s"})

    st.markdown("**Timing**")
    st.dataframe(rows, width="stretch", hide_index=True, height=min(35 * len(rows) + 38, 280))


def show_input_row(
    t0,
    t1_gt,
    t2,
    name_t0: str = "",
    name_t1: str = "",
    name_t2: str = "",
    target_gap_min: float | None = None,
):
    """Three columns: t0 input | ground-truth t1 | t2 input."""
    c0, c1, c2 = st.columns(3)
    with c0:
        st.markdown("**Input t0**")
        st.caption("Given to model")
        if name_t0:
            st.caption(f"`{name_t0}`")
        st.image(t0, width="stretch")
    with c1:
        st.markdown("**Ground-truth t1**")
        st.caption("Hidden from model Â· used only for metrics")
        if name_t1:
            st.caption(f"`{name_t1}`")
        if t1_gt is not None:
            st.image(t1_gt, width="stretch")
        else:
            st.info("No ground-truth t1 â€” metrics disabled.")
    with c2:
        st.markdown("**Input t2**")
        st.caption("Given to model")
        if name_t2:
            st.caption(f"`{name_t2}`")
        st.image(t2, width="stretch")

    if name_t0 and name_t2:
        show_time_info(name_t0, name_t1 or None, name_t2, target_gap_min=target_gap_min)


def gap_minutes_input(key: str, default: float = 10.0) -> float:
    return st.number_input(
        "Minutes between frames (t0â†’t1 and t1â†’t2)",
        min_value=5.0,
        max_value=60.0,
        value=default,
        step=5.0,
        key=key,
        help="GOES scans every ~10 min. Values snap to cadence: 10â†’10 min, 15â†’20 min, "
        "20â†’20 min, 30â†’30 min. Total t0â†’t2 = 2Ã— actual step.",
    )


def show_gap_resolution(
    files: list[NcRef],
    gap_min: float,
    folder: Path | None = None,
    label: str = "",
) -> dict:
    """Show how requested gap maps to scan skips; warn if too few files."""
    info = gap_step_info(files, gap_min)
    actual = info["actual_step_min"]
    if abs(actual - gap_min) > 0.5:
        st.info(
            f"Requested **{gap_min:.0f} min** â†’ using **{actual:.0f} min** per step "
            f"({info['skip']} scans Ã— {info['cadence_min']:.0f} min cadence). "
            f"t0â†’t2 span **â‰ˆ {info['total_span_min']:.0f} min**."
        )
    else:
        st.caption(
            f"Step **{actual:.0f} min** Â· t0â†’t2 **â‰ˆ {info['total_span_min']:.0f} min** Â· "
            f"cadence **{info['cadence_min']:.0f} min**"
        )
    if info["n_triplets"] == 0:
        name = label or (folder.name if folder else "selection")
        st.warning(
            f"**{name}** has only **{len(files)}** scans â€” need **â‰¥{info['min_files_needed']}** "
            f"for **{actual:.0f} min** steps. Try another date or a smaller gap."
        )
    return info


# =============================================================================
# 2. RIFE MODEL
# =============================================================================

def rife_ready() -> bool:
    return (TRAIN_LOG / "flownet.pkl").exists() and (PRACTICAL_RIFE / "model").exists()


def setup_rife_hint() -> str:
    return f".venv\\Scripts\\python.exe {ROOT / 'setup_rife.py'}"


def model_checkpoint_dir() -> Path:
    if not FINETUNED_CKPT.exists():
        raise FileNotFoundError(
            f"GOES fine-tuned checkpoint not found: {FINETUNED_CKPT}\n"
            "Copy flownet.pkl into checkpoints/goes_finetuned/ or run finetune_physics.py."
        )
    return FINETUNED_DIR


def load_rife_checkpoint(model, ckpt_dir: Path) -> None:
    """Load flownet.pkl â€” handles `module.*` prefix and ignores unused teacher/caltime keys."""
    sd = torch.load(ckpt_dir / "flownet.pkl", map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    wanted = model.flownet.state_dict()
    sd = {k: v for k, v in sd.items() if k in wanted}
    missing = [k for k in wanted if k not in sd]
    if missing:
        raise RuntimeError(f"Checkpoint missing {len(missing)} keys (e.g. {missing[0]})")
    model.flownet.load_state_dict(sd, strict=True)


def model_input_dtype(model) -> torch.dtype:
    return next(model.flownet.parameters()).dtype


@st.cache_resource
def get_model(_weights_ver: str = "v3"):
    if str(PRACTICAL_RIFE) not in sys.path:
        sys.path.insert(0, str(PRACTICAL_RIFE))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    from train_log.RIFE_HDv3 import Model

    ckpt = model_checkpoint_dir()
    m = Model()
    load_rife_checkpoint(m, ckpt)
    m.eval()
    m.flownet.to(device=device, dtype=torch.float32)
    version = getattr(m, "version", 4.25)
    return m, device, version, str(ckpt), False


def to_tensor(frame: np.ndarray, device, dtype: torch.dtype, scale: float):
    h, w = frame.shape[:2]
    tile = max(128, int(128 / scale))
    ph = ((h - 1) // tile + 1) * tile
    pw = ((w - 1) // tile + 1) * tile
    t = torch.from_numpy(frame.transpose(2, 0, 1)).to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
    t = F.pad(t, (0, pw - w, 0, ph - h))
    if dtype == torch.float16:
        t = t.half()
    return t, h, w


def from_tensor(t, h, w) -> np.ndarray:
    return (t[0, :3].float() * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]


def predict_t1(t0: np.ndarray, t2: np.ndarray, model, device, scale: float, fp16: bool) -> np.ndarray:
    """t0 + t2 -> middle frame. HOOK: add PINN after model.inference."""
    dtype = model_input_dtype(model)
    a, h, w = to_tensor(t0, device, dtype, scale)
    b, _, _ = to_tensor(t2, device, dtype, scale)
    with torch.no_grad():
        out = model.inference(a, b, timestep=0.5, scale=scale)
    return from_tensor(out, h, w)


def rife_between(t0, t2, model, device, n_mid: int, scale: float, fp16: bool, version: float) -> list[np.ndarray]:
    """Insert n_mid frames between t0 and t2."""
    dtype = model_input_dtype(model)
    a, h, w = to_tensor(t0, device, dtype, scale)
    b, _, _ = to_tensor(t2, device, dtype, scale)
    mids = []
    with torch.no_grad():
        if version >= 3.9:
            for i in range(n_mid):
                t = (i + 1) / (n_mid + 1)
                mids.append(from_tensor(model.inference(a, b, timestep=t, scale=scale), h, w))
        else:
            mid = model.inference(a, b, scale=scale)
            mids = [from_tensor(mid, h, w)]
    return mids


def linear_t1(t0, t2):
    return ((t0.astype(np.float32) + t2.astype(np.float32)) / 2).astype(np.uint8)


def upscale_sequence(frames: list[np.ndarray], multi: int, model, device, version, scale, fp16) -> list[np.ndarray]:
    if multi < 2 or len(frames) < 2:
        return frames
    out = [frames[0]]
    for i in range(len(frames) - 1):
        mids = rife_between(frames[i], frames[i + 1], model, device, multi - 1, scale, fp16, version)
        out.extend(mids)
        out.append(frames[i + 1])
    return out


def frames_to_gif_bytes(frames: list[np.ndarray], duration_ms: int = 100) -> bytes:
    pil = [Image.fromarray(np.ascontiguousarray(f, dtype=np.uint8)) for f in frames]
    buf = io.BytesIO()
    pil[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return buf.getvalue()


# =============================================================================
# 3. METRICS
# =============================================================================

def gray01(img):
    g = img[..., 0] if img.ndim == 3 else img
    return g.astype(np.float64) / 255.0


def mse(a, b):
    x, y = gray01(a), gray01(b)
    return float(np.mean((x - y) ** 2))


def psnr(a, b):
    e = mse(a, b)
    return float("inf") if e == 0 else float(20 * np.log10(1.0) - 10 * np.log10(e))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gray01(a), gray01(b), data_range=1.0))


def fsim(a, b):
    """FSIM via piq â€” luminance-only (thermal IR is single-channel, not FSIMc RGB)."""
    from piq import fsim as fsim_fn

    x = torch.from_numpy(gray01(a)).unsqueeze(0).unsqueeze(0).float()
    y = torch.from_numpy(gray01(b)).unsqueeze(0).unsqueeze(0).float()
    return float(fsim_fn(x, y, data_range=1.0, chromatic=False).item())


def all_metrics(pred, gt):
    return {"mse": mse(pred, gt), "psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "fsim": fsim(pred, gt)}


# =============================================================================
# 4. MOTION VECTORS (optical flow)
# =============================================================================

def flow_to_rgb(u: np.ndarray, v: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """HSV color wheel: hue=direction, value=speed."""
    mag = np.sqrt(u * u + v * v)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99)) + 1e-6
    ang = np.arctan2(v, u)
    h = (ang + np.pi) / (2 * np.pi)
    val = np.clip(mag / max_mag, 0, 1)
    hsv = np.stack([h, np.ones_like(h), val], axis=-1)
    import matplotlib.colors as mcolors
    return (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def flow_smoothness(u: np.ndarray, v: np.ndarray) -> float:
    """Mean gradient magnitude â€” lower = smoother flow."""
    du = np.diff(u, axis=1)
    dv = np.diff(v, axis=0)
    return float(np.mean(np.abs(du)) + np.mean(np.abs(dv)))


def photometric_warp_error(src: np.ndarray, u: np.ndarray, v: np.ndarray, target: np.ndarray) -> float:
    """MSE after warping src by (u,v) toward target (grayscale)."""
    g = gray01(src)
    gt = gray01(target)
    h, w = g.shape
    xs = np.arange(w, dtype=np.float32)[None, :] + u.astype(np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None] + v.astype(np.float32)
    warped = cv2.remap(g.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.mean((warped - gt) ** 2))


def extract_rife_flow(
    t0: np.ndarray,
    t2: np.ndarray,
    model,
    device,
    scale: float,
) -> dict:
    """Run IFNet and return dense flow (u,v), mask, and flow-only synthesis."""
    if str(PRACTICAL_RIFE) not in sys.path:
        sys.path.insert(0, str(PRACTICAL_RIFE))
    from model.warplayer import warp

    dtype = model_input_dtype(model)
    a, h, w = to_tensor(t0, device, dtype, scale)
    b, _, _ = to_tensor(t2, device, dtype, scale)
    imgs = torch.cat((a, b), 1)
    scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    with torch.no_grad():
        flow_list, mask, _merged = model.flownet(imgs, 0.5, scale_list)
        flow = flow_list[-1]
        mask = torch.sigmoid(mask)
        warped0 = warp(a, flow[:, :2])
        warped1 = warp(b, flow[:, 2:4])
        flow_only = warped0 * mask + warped1 * (1 - mask)
    return {
        "u0": flow[0, 0, :h, :w].float().cpu().numpy(),
        "v0": flow[0, 1, :h, :w].float().cpu().numpy(),
        "u1": flow[0, 2, :h, :w].float().cpu().numpy(),
        "v1": flow[0, 3, :h, :w].float().cpu().numpy(),
        "mask": mask[0, 0, :h, :w].float().cpu().numpy(),
        "flow_only": from_tensor(flow_only, h, w),
    }


def estimate_motion_vectors(t0: np.ndarray, t2: np.ndarray, model=None, device=None, scale: float = 1.0):
    """Dense midpoint AMV (u,v) in pixels â€” bidirectional average from RIFE IFNet."""
    if model is None or device is None:
        raise ValueError("Pass model and device from get_model()")
    out = extract_rife_flow(t0, t2, model, device, scale)
    return rife_amv_uv(out)


def farneback_mid(t0: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical optical flow baseline: Farneback warp + average."""
    g0 = gray01(t0).astype(np.float32)
    g2 = gray01(t2).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(g0, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0] * 0.5, flow[..., 1] * 0.5
    h, w = g0.shape
    xs = np.arange(w, dtype=np.float32)[None, :] + u
    ys = np.arange(h, dtype=np.float32)[:, None] + v
    w0 = cv2.remap(g0, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    xs2 = np.arange(w, dtype=np.float32)[None, :] - u
    ys2 = np.arange(h, dtype=np.float32)[:, None] - v
    w2 = cv2.remap(g2, xs2, ys2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mid = np.clip((w0 + w2) * 0.5, 0, 1)
    rgb = np.stack([(mid * 255).astype(np.uint8)] * 3, axis=-1)
    return rgb, u, v


def plot_flow_quiver(
    u: np.ndarray,
    v: np.ndarray,
    step: int = 24,
    title: str = "Motion vectors",
    background: np.ndarray | None = None,
    units: str = "px",
    sparse: dict | None = None,
):
    """Quiver with fixed arrow length for visibility; color encodes true speed."""
    h, w = u.shape
    if sparse is not None and sparse.get("n_shown", 0) > 0:
        X = np.asarray(sparse["X"], dtype=np.float64).ravel()
        Y = np.asarray(sparse["Y"], dtype=np.float64).ravel()
        U = np.asarray(sparse["U"], dtype=np.float64).ravel()
        V = np.asarray(sparse["V"], dtype=np.float64).ravel()
        spacing = float(step)
        ux = np.unique(X)
        if ux.size > 1:
            spacing = float(np.median(np.diff(ux)))
    else:
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        X, Y = np.meshgrid(xs, ys)
        U = u[Y, X]
        V = v[Y, X]
        X = X.ravel()
        Y = Y.ravel()
        U = U.ravel()
        V = V.ravel()
        spacing = float(step)

    mag = np.sqrt(U * U + V * V)
    min_mag = 0.05 if units == "ms" else 0.005
    keep = np.isfinite(mag) & (mag >= min_mag)
    X, Y, U, V, mag = X[keep], Y[keep], U[keep], V[keep], mag[keep]

    fig, ax = plt.subplots(figsize=(6, 6))
    if background is not None:
        if background.ndim == 3:
            ax.imshow(background, origin="upper", extent=[0, w, h, 0])
        else:
            ax.imshow(background, cmap="gray", vmin=0, vmax=1, origin="upper", extent=[0, w, h, 0])

    unit_lbl = "m/s" if units == "ms" else "px"
    if X.size == 0:
        ax.text(
            0.5, 0.5, "No vectors above threshold",
            transform=ax.transAxes, ha="center", va="center", color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=6),
        )
    else:
        arrow_len = max(spacing * 0.55, 6.0)
        inv = np.maximum(mag, 1e-12)
        U_draw = U / inv * arrow_len
        V_draw = V / inv * arrow_len
        vmax = float(np.percentile(mag, 98)) if mag.size else 1.0
        vmax = max(vmax, min_mag * 2)
        q = ax.quiver(
            X, Y, U_draw, V_draw, mag,
            cmap="turbo", angles="xy", scale_units="xy", scale=1,
            width=0.004, clim=(0, vmax),
        )
        fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label=unit_lbl)
        ax.text(
            0.02, 0.98,
            f"Arrow length fixed Â· color = speed ({unit_lbl})",
            transform=ax.transAxes, va="top", color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.55, pad=3),
        )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return fig


def plot_validity_mask(
    valid: np.ndarray,
    background: np.ndarray | None = None,
    title: str = "AMV quality mask (white = valid)",
):
    h, w = valid.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    if background is not None:
        if background.ndim == 3:
            ax.imshow(background, origin="upper", extent=[0, w, h, 0], alpha=0.4)
        else:
            ax.imshow(background, cmap="gray", vmin=0, vmax=1, origin="upper", extent=[0, w, h, 0], alpha=0.4)
    ax.imshow(valid.astype(np.float32), cmap="gray", vmin=0, vmax=1, origin="upper", extent=[0, w, h, 0], alpha=0.65)
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig


def plot_speed_heatmap(
    speed_ms: np.ndarray,
    background: np.ndarray | None = None,
    title: str = "AMV speed (m/s)",
):
    h, w = speed_ms.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    if background is not None:
        if background.ndim == 3:
            ax.imshow(background, origin="upper", extent=[0, w, h, 0], alpha=0.45)
        else:
            ax.imshow(background, cmap="gray", vmin=0, vmax=1, origin="upper", extent=[0, w, h, 0], alpha=0.45)
    vmax = float(np.nanpercentile(speed_ms, 99)) if np.any(np.isfinite(speed_ms)) else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(
        np.clip(speed_ms, 0, vmax), cmap="turbo", origin="upper",
        extent=[0, w, h, 0], vmin=0, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, label="m/s")
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def _cached_geogrid(
    uri: str,
    source_id: str,
    height: int,
    width: int,
    segment_uris: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = load_nc_latlon_ref(NcRef(uri=uri, source_id=source_id, segment_uris=segment_uris))
    return resize_geogrid(lat, lon, height, width)


def _fallback_geogrid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Approximate GOES-West full-disk lat/lon when no NetCDF ref is available."""
    lat = np.linspace(55.0, -55.0, height, dtype=np.float64)[:, None]
    lon = np.linspace(-135.0, -15.0, width, dtype=np.float64)[None, :]
    return np.broadcast_to(lat, (height, width)).copy(), np.broadcast_to(lon, (height, width)).copy()


def compare_interpolation_methods(
    t0: np.ndarray,
    t2: np.ndarray,
    t1_gt: np.ndarray | None,
    model,
    device,
    scale: float,
) -> list[dict]:
    """Ablation: linear, Farneback, RIFE flow-only, RIFE full."""
    lin = linear_t1(t0, t2)
    fb, fu, fv = farneback_mid(t0, t2)
    rife = extract_rife_flow(t0, t2, model, device, scale)
    flow_only = rife["flow_only"]
    full = predict_t1(t0, t2, model, device, scale, False)

    rows = []
    for name, pred in [
        ("Linear (t0+t2)/2", lin),
        ("Farneback + warp", fb),
        ("RIFE flow-only (warp+mask)", flow_only),
        ("RIFE full", full),
    ]:
        row = {"method": name}
        if t1_gt is not None:
            m = all_metrics(pred, t1_gt)
            row.update({k: round(v, 6) if k != "psnr" else round(v, 2) for k, v in m.items()})
        rows.append(row)

    if t1_gt is not None:
        rows.append({
            "method": "--- flow diagnostics ---",
            "ssim": None,
            "psnr": None,
            "mse": None,
            "fsim": None,
        })
        rows.append({
            "method": "RIFE warp photometric err (t0)",
            "mse": round(photometric_warp_error(t0, rife["u0"], rife["v0"], t1_gt), 6),
            "ssim": None,
            "psnr": None,
            "fsim": None,
        })
        rows.append({
            "method": "Flow smoothness |u,v|",
            "mse": round(flow_smoothness(rife["u0"], rife["v0"]), 4),
            "ssim": None,
            "psnr": None,
            "fsim": None,
        })
    return rows, {
        "linear": lin,
        "farneback": fb,
        "flow_only": flow_only,
        "full": full,
        "rife_flow": rife,
        "farneback_uv": (fu, fv),
        "amv_uv": rife_amv_uv(rife),
        "forward_uv": rife_forward_uv(rife),
        "flow_disagreement": bidirectional_disagreement(rife),
    }


@st.cache_data(ttl=300, show_spinner=False)
def _cached_list_scans(source_id: str, day_iso: str, hour: int | None) -> list[dict]:
    refs = list_scans_for_day(source_id, date.fromisoformat(day_iso), hour)
    return [r.to_dict() for r in refs]


def browse_catalog_ui(key_prefix: str, size: int) -> tuple | None:
    """Pick satellite + UTC date; list and load triplets from S3/mount (no local copy)."""
    sources = satellite_sources()
    st.markdown("**AWS satellite catalog** â€” data streamed from S3 or instance mount")

    source_id = st.selectbox(
        "Satellite",
        list(sources.keys()),
        format_func=lambda k: sources[k].label,
        key=f"{key_prefix}_source",
    )
    src = sources[source_id]
    with st.expander("How to access this dataset", expanded=False):
        st.markdown(source_help_text(source_id))

    c1, c2 = st.columns(2)
    with c1:
        day = st.date_input("UTC date", value=date.today(), key=f"{key_prefix}_date")
    with c2:
        hour_opts = ["All hours"] + [f"{h:02d}:00 UTC" for h in range(24)]
        hour_sel = st.selectbox("Hour filter", hour_opts, key=f"{key_prefix}_hour")

    hour = None if hour_sel == "All hours" else int(hour_sel.split(":")[0])
    if src.fmt == "mount":
        roots = mount_roots()
        st.caption(
            f"Mount paths: `{', '.join(roots)}` Â· filter: `{src.file_filter}` Â· var: `{src.rad_var}`"
        )
    else:
        st.caption(f"Bucket: `{src.bucket}` Â· filter: `{src.file_filter}` Â· var: `{src.rad_var}`")

    list_done_key = f"{key_prefix}_list_done"
    if st.button("List scans", key=f"{key_prefix}_list", type="secondary"):
        with st.spinner("Listing objects (S3 API / mount)â€¦"):
            try:
                dicts = _cached_list_scans(source_id, day.isoformat(), hour)
            except Exception as exc:
                st.error(f"Could not list data: {exc}")
                return None
            st.session_state[f"{key_prefix}_refs"] = dicts
            st.session_state[list_done_key] = True
            st.session_state[f"{key_prefix}_rad_var"] = src.rad_var
            st.session_state[f"{key_prefix}_source_label"] = src.label
            if dicts:
                st.success(f"Found **{len(dicts)}** scans for {day.isoformat()}.")
            elif src.fmt == "mount":
                st.warning(
                    f"No INSAT `.h5` files for **{day.isoformat()}** under: "
                    f"`{'`, `'.join(roots) if roots else 'no mount paths'}`. "
                    "On a local PC use **GOES-19 ABI Ch.13** (public NOAA S3). "
                    "For INSAT, run on the ISRO EC2 instance or set env "
                    "`SATELLITE_DATA_MOUNT` to a folder with `3SIMG_*.h5` / `3RIMG_*.h5` files "
                    "(or place files in `data/insat/` next to this app)."
                )
            else:
                st.warning(
                    f"No files for **{day.isoformat()}** in `{src.bucket}` "
                    f"(prefix filter `{src.file_filter}`). Try another UTC date or hour."
                )

    if not st.session_state.get(list_done_key):
        if src.fmt == "mount":
            st.info(
                "INSAT reads from the **instance mount**, not public AWS. "
                "Pick a UTC date and click **List scans**, or switch to **GOES-19** for local testing."
            )
        else:
            st.info("Choose a **UTC date** and click **List scans** to browse available files.")
        return None

    raw = st.session_state.get(f"{key_prefix}_refs", [])
    if not raw:
        return None

    refs = [NcRef.from_dict(d) for d in raw]
    if refs:
        st.dataframe(
            [
                {
                    "file": r.name,
                    "scan_utc": format_utc(r.scan_time),
                    "uri": r.uri[:80] + ("â€¦" if len(r.uri) > 80 else ""),
                }
                for r in refs[:50]
            ],
            width="stretch",
            hide_index=True,
            height=min(35 * min(len(refs), 50) + 38, 280),
        )
        if len(refs) > 50:
            st.caption(f"Showing first 50 of **{len(refs)}** scans.")

    gap_min = gap_minutes_input(f"{key_prefix}_gap", default=src.default_cadence_min)
    label = st.session_state.get(f"{key_prefix}_source_label", src.label)
    info = show_gap_resolution(refs, gap_min, label=label)
    triplets = find_triplets_by_gap(refs, gap_min)
    if not triplets:
        return None

    st.caption(f"**{len(triplets)}** triplets Â· step **{info['actual_step_min']:.0f} min**")
    idx = st.slider("Triplet index", 0, max(0, len(triplets) - 1), 0, key=f"{key_prefix}_idx")
    if not st.button("Load triplet", key=f"{key_prefix}_load"):
        return None

    i0, i1, i2 = triplets[idx]
    rad_var = st.session_state.get(f"{key_prefix}_rad_var", src.rad_var)
    with st.spinner("Streaming radianceâ€¦"):
        out = load_triplet_at_indices(refs, i0, i1, i2, size, rad_var=rad_var)
    return out + (info["actual_step_min"],)


def load_triplet_ui(key_prefix: str, size: int) -> tuple | None:
    """Shared triplet loader â€” AWS catalog or local folder."""
    if aws_deploy_mode() or not find_nc_folders():
        return browse_catalog_ui(key_prefix, size)

    folders = find_nc_folders()
    if not folders:
        return browse_catalog_ui(key_prefix, size)
    folder = st.selectbox("Folder", folders, format_func=folder_label, key=f"{key_prefix}_folder")
    gap_min = gap_minutes_input(f"{key_prefix}_gap")
    files = list_nc_files(folder)
    info = show_gap_resolution(files, gap_min, folder=folder)
    triplets = find_triplets_by_gap(files, gap_min)
    if not triplets:
        return None
    st.caption(f"**{len(triplets)}** triplets Â· step **{info['actual_step_min']:.0f} min**")
    idx = st.slider("Triplet index", 0, max(0, len(triplets) - 1), 0, key=f"{key_prefix}_idx")
    if not st.button("Load triplet", key=f"{key_prefix}_load"):
        return None
    i0, i1, i2 = triplets[idx]
    return load_triplet_at_indices(files, i0, i1, i2, size) + (info["actual_step_min"],)


# =============================================================================
# 5. BATCH VALIDATION
# =============================================================================

def run_batch_validation(frames, model, device, version, scale, fp16, progress):
    rows = []
    n = len(frames) - 2
    for i in range(n):
        t0, gt, t2 = frames[i], frames[i + 1], frames[i + 2]
        pred = predict_t1(t0, t2, model, device, scale, fp16)
        lin = linear_t1(t0, t2)
        rm, lm = all_metrics(pred, gt), all_metrics(lin, gt)
        rows.append({"i": i, "rife_ssim": rm["ssim"], "rife_fsim": rm["fsim"], "rife_psnr": rm["psnr"],
                     "linear_ssim": lm["ssim"], "linear_fsim": lm["fsim"]})
        progress.progress((i + 1) / n, text=f"Triplet {i+1}/{n}")
    return rows


def run_batch_validation_gaps(
    files: list[NcRef],
    size: int,
    gap_min: float,
    model,
    device,
    version,
    scale,
    fp16,
    progress,
    max_triplets: int | None = None,
    rad_var: str | None = None,
):
    triplets = find_triplets_by_gap(files, gap_min)
    if max_triplets:
        triplets = triplets[:max_triplets]
    rows = []
    n = len(triplets)
    for ti, (i0, i1, i2) in enumerate(triplets):
        t0, gt, t2, n0, n1, n2, _rads, _disk, _refs = load_triplet_at_indices(files, i0, i1, i2, size, rad_var=rad_var)
        pred = predict_t1(t0, t2, model, device, scale, fp16)
        lin = linear_t1(t0, t2)
        rm, lm = all_metrics(pred, gt), all_metrics(lin, gt)
        t0s, t1s, t2s = parse_goes_scan_time(n0), parse_goes_scan_time(n1), parse_goes_scan_time(n2)
        actual_gap = (t1s - t0s).total_seconds() / 60 if t0s and t1s else None
        rows.append({
            "i": ti,
            "file_i0": i0,
            "file_i1": i1,
            "file_i2": i2,
            "gap_t0_t1_min": round(actual_gap, 1) if actual_gap is not None else None,
            "rife_ssim": rm["ssim"],
            "rife_fsim": rm["fsim"],
            "rife_psnr": rm["psnr"],
            "linear_ssim": lm["ssim"],
            "linear_fsim": lm["fsim"],
        })
        progress.progress((ti + 1) / n, text=f"Triplet {ti+1}/{n}")
    return rows


# =============================================================================
# 6. STREAMLIT UI
# =============================================================================

def main():
    import streamlit as st
    st.set_page_config("ISRO PS12", layout="wide")
    st.title("ISRO PS12 â€” FillFrame")
    st.info("Core inference ready â€” UI tabs landing next.")

if __name__ == "__main__":
    main()
