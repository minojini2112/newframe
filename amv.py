"""
AMV (Atmospheric Motion Vector) helpers.

Phase 1: bidirectional RIFE flow → midpoint motion (pixels).
Phase 2: geolocation → physical velocity (m/s).
"""

from __future__ import annotations

import numpy as np

# Typical ABI nadir resolution (m) for fallback when projection metadata is missing.
GOES_NADIR_M_PER_PX = 2000.0
MAX_PLAUSIBLE_MS = 80.0


def rife_amv_uv(rife_out: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Midpoint AMV from RIFE IFNet 4-channel flow.

    u0,v0 warp t0 toward the midpoint; u1,v1 warp t2 toward the midpoint.
    Averaging both directions cancels systematic bias from occlusions.
    """
    u = (rife_out["u0"] - rife_out["u1"]) / 2.0
    v = (rife_out["v0"] - rife_out["v1"]) / 2.0
    return u, v


def rife_forward_uv(rife_out: dict) -> tuple[np.ndarray, np.ndarray]:
    """One-way flow from t0 toward midpoint (legacy / comparison)."""
    return rife_out["u0"], rife_out["v0"]


def bidirectional_disagreement(rife_out: dict) -> np.ndarray:
    """Per-pixel |forward + backward| in px — high values → uncertain motion."""
    du = rife_out["u0"] + rife_out["u1"]
    dv = rife_out["v0"] + rife_out["v1"]
    return np.sqrt(du * du + dv * dv)


def flow_step_minutes(gap_min: float) -> float:
    """Minutes represented by one step of midpoint flow (t0→t1 or t1→t2)."""
    return float(gap_min)


def flow_span_minutes(gap_min: float) -> float:
    """Full t0→t2 span for the loaded triplet."""
    return float(gap_min) * 2.0


def amv_interval_caption(gap_min: float) -> str:
    step = flow_step_minutes(gap_min)
    span = flow_span_minutes(gap_min)
    return (
        f"AMV vectors: **{step:.0f} min** motion (t0→t1 step); "
        f"triplet span t0→t2 = **{span:.0f} min**. "
        "Multiply vector magnitude by 2 for full-span displacement."
    )


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (m) between point pairs (element-wise)."""
    r = 6371000.0
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def resize_geogrid(lat: np.ndarray, lon: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize lat/lon to match the display / flow grid."""
    import cv2

    lat_r = cv2.resize(lat.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    lon_r = cv2.resize(lon.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return lat_r.astype(np.float64), lon_r.astype(np.float64)


def meters_per_pixel_grid(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local east-west and north-south meters per pixel from lat/lon grid."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    h, w = lat.shape
    mx = np.full((h, w), np.nan, dtype=np.float64)
    my = np.full((h, w), np.nan, dtype=np.float64)

    mx[:, 1:-1] = haversine_m(lat[:, :-2], lon[:, :-2], lat[:, 2:], lon[:, 2:]) / 2.0
    mx[:, 0] = haversine_m(lat[:, 0], lon[:, 0], lat[:, 1], lon[:, 1])
    mx[:, -1] = haversine_m(lat[:, -2], lon[:, -2], lat[:, -1], lon[:, -1])

    my[1:-1, :] = haversine_m(lat[:-2, :], lon[:-2, :], lat[2:, :], lon[2:, :]) / 2.0
    my[0, :] = haversine_m(lat[0, :], lon[0, :], lat[1, :], lon[1, :])
    my[-1, :] = haversine_m(lat[-2, :], lon[-2, :], lat[-1, :], lon[-1, :])
    return mx, my


def nadir_meters_per_pixel(
    lat: np.ndarray,
    native_res_m: float = GOES_NADIR_M_PER_PX,
    native_shape: tuple[int, int] | None = None,
    display_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: ~2 km at nadir, scaled by cos(lat) and resize factor."""
    scale = 1.0
    if native_shape and display_shape:
        scale = max(native_shape) / max(display_shape)
    m = (native_res_m / scale) / np.maximum(np.cos(np.radians(lat)), 0.15)
    return m, m.copy()


def flow_pixels_to_ms(
    u: np.ndarray,
    v: np.ndarray,
    m_per_px_x: np.ndarray,
    m_per_px_y: np.ndarray,
    dt_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert pixel displacement over dt_sec to east/north components and speed (m/s)."""
    dt = max(float(dt_sec), 1e-6)
    ux = u * m_per_px_x / dt
    uy = v * m_per_px_y / dt
    speed = np.sqrt(ux * ux + uy * uy)
    return ux, uy, speed


def amv_speed_ms(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    gap_min: float,
    *,
    from_geo: bool = True,
) -> dict:
    """
    Full Phase-2 conversion: pixel AMV + lat/lon → m/s field and summary stats.

    lat/lon must already match u,v shape (display resolution).
    """
    dt_sec = flow_step_minutes(gap_min) * 60.0
    if from_geo and np.any(np.isfinite(lat)):
        mx, my = meters_per_pixel_grid(lat, lon)
        finite = np.isfinite(mx) & np.isfinite(my)
        if np.mean(finite) < 0.5:
            mx, my = nadir_meters_per_pixel(lat, display_shape=lat.shape)
            geo_mode = "nadir_fallback"
        else:
            mx = np.where(finite, mx, np.nanmedian(mx))
            my = np.where(finite, my, np.nanmedian(my))
            geo_mode = "latlon_grid"
    else:
        mx, my = nadir_meters_per_pixel(lat, display_shape=lat.shape)
        geo_mode = "nadir_only"

    ux, uy, speed = flow_pixels_to_ms(u, v, mx, my, dt_sec)
    valid = np.isfinite(speed) & (speed <= MAX_PLAUSIBLE_MS)
    speed_clip = np.where(speed > MAX_PLAUSIBLE_MS, np.nan, speed)

    return {
        "ux_ms": ux,
        "uy_ms": uy,
        "speed_ms": speed,
        "speed_ms_clipped": speed_clip,
        "m_per_px_x": mx,
        "m_per_px_y": my,
        "geo_mode": geo_mode,
        "dt_sec": dt_sec,
        "mean_ms": float(np.nanmean(speed)),
        "p99_ms": float(np.nanpercentile(speed, 99)),
        "max_ms": float(np.nanmax(speed)),
        "plausible_frac": float(np.mean(valid & (speed > 0.5))) if speed.size else 0.0,
        "implausible_frac": float(np.mean(speed > MAX_PLAUSIBLE_MS)) if speed.size else 0.0,
    }


def speed_ms_caption(stats: dict) -> str:
    return (
        f"Speed (m/s): mean **{stats['mean_ms']:.2f}**, "
        f"p99 **{stats['p99_ms']:.2f}**, max **{stats['max_ms']:.2f}** · "
        f"geo **{stats['geo_mode']}** · "
        f"plausible (0.5–{MAX_PLAUSIBLE_MS:.0f} m/s): **{stats['plausible_frac']*100:.1f}%** · "
        f"implausible (>{MAX_PLAUSIBLE_MS:.0f} m/s): **{stats['implausible_frac']*100:.1f}%**"
    )


# --- Phase 3: coarse grid + quality filters ---

def texture_magnitude(radiance_gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on stretched radiance — high where clouds have texture."""
    import cv2

    g = np.asarray(radiance_gray, dtype=np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a boolean mask with nearest-neighbor so the Earth-limb edge stays crisp
    (bilinear/area resizing would blur True/False into a fuzzy gray band)."""
    import cv2

    m = np.asarray(mask, dtype=np.uint8)
    resized = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """
    Shrink a boolean mask inward by `px` pixels.

    Used to drop a thin band just inside the Earth limb from AMV validity gating.
    `rad_to_gray01`/`rad_to_rgb` set off-disk pixels to a flat fill value *before*
    bilinear-resizing to display resolution, so the resize blends real edge-of-disk
    brightness with that flat fill — producing a spurious high-gradient "halo" ring
    that traces the limb even where the real data is texture-free. That halo passes
    the texture filter and gets marked valid even though it isn't real cloud motion.
    Eroding the disk mask a few pixels inward excludes it.
    """
    if px <= 0:
        return mask
    import cv2

    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def full_disk_geometric_mask(height: int, width: int, shrink: float = 0.99) -> np.ndarray:
    """
    Fallback on-Earth mask for when no real fill-value/NaN footprint is available
    (e.g. a plain image upload with no radiance data behind it). GOES-style full-disk
    imagery frames the globe as a circle inscribed in the square scan grid, so that
    geometry is used as an approximation rather than assuming every pixel is Earth.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    r = min(height, width) / 2.0 * shrink
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)


def amv_quality_mask(
    radiance_gray: np.ndarray,
    disagreement_px: np.ndarray,
    *,
    speed_ms: np.ndarray | None = None,
    rife_mask: np.ndarray | None = None,
    on_disk: np.ndarray | None = None,
    texture_percentile: float = 15.0,
    disagree_px_max: float = 0.5,
    min_speed_ms: float = 0.5,
    max_speed_ms: float = MAX_PLAUSIBLE_MS,
    min_rife_mask: float = 0.3,
    use_rife_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel validity for AMV arrows.

    Drops off-Earth/space pixels outright (on_disk), then within the Earth disk drops
    uniform regions (low texture), high forward/backward disagreement, implausible
    speed, and optionally low RIFE blend confidence.

    `on_disk` matters not just as a final AND: space is typically 20%+ of the square
    canvas and is perfectly uniform (texture ~= 0), so if it's included when computing
    the texture percentile it drags the threshold down near 0 and the texture filter
    stops doing anything useful for real cloud pixels. Restricting the percentile to
    on-disk pixels keeps the threshold meaningful.
    """
    tex = texture_magnitude(radiance_gray)
    finite = np.isfinite(tex)
    if on_disk is not None:
        finite = finite & on_disk
    tex_thr = float(np.percentile(tex[finite], texture_percentile)) if np.any(finite) else 0.0
    valid = (tex >= tex_thr) & (disagreement_px <= disagree_px_max)
    if speed_ms is not None:
        valid &= np.isfinite(speed_ms)
        valid &= (speed_ms >= min_speed_ms) & (speed_ms <= max_speed_ms)
    if use_rife_mask and rife_mask is not None:
        valid &= rife_mask >= min_rife_mask
    if on_disk is not None:
        valid &= on_disk
    return valid, tex


def subsample_amv_grid(
    u: np.ndarray,
    v: np.ndarray,
    stride: int = 24,
    window: int = 10,
    valid_mask: np.ndarray | None = None,
    min_valid_frac: float = 0.5,
    on_disk: np.ndarray | None = None,
    min_mag_px: float = 0.0,
) -> dict:
    """Coarse AMV grid with window-averaged (u,v); skip cells with too few valid pixels.

    If `on_disk` is given, a window's valid-fraction is measured against its on-Earth
    pixels only (not the full window). Otherwise a window straddling the Earth limb
    would have its fraction diluted by off-disk pixels that were never going to be
    valid anyway, unfairly hiding an arrow over a perfectly good cloud window. Windows
    that are entirely off-disk are always skipped, and `n_grid` only counts candidate
    cells that actually touch the Earth disk, so the shown/total ratio reflects real
    cloud coverage rather than being deflated by the off-disk background.

    When ``min_mag_px > 0``, a cell is skipped if the window-averaged speed is below
    that threshold (no motion in that patch). No Phase-3 quality mask is applied.
    """
    h, w = u.shape
    half = max(window // 2, 1)
    xs_grid = np.arange(0, w, stride)
    ys_grid = np.arange(0, h, stride)
    xs_out, ys_out, u_out, v_out = [], [], [], []
    n_grid = 0

    for y in ys_grid:
        for x in xs_grid:
            y0, y1 = max(0, y - half), min(h, y + half + 1)
            x0, x1 = max(0, x - half), min(w, x + half + 1)

            if on_disk is not None:
                disk_win = on_disk[y0:y1, x0:x1]
                if not np.any(disk_win):
                    continue  # window entirely off-Earth — not a candidate cell at all
            n_grid += 1

            if valid_mask is not None:
                win = valid_mask[y0:y1, x0:x1]
                denom = disk_win if on_disk is not None else np.ones_like(win, dtype=bool)
                denom_n = int(np.count_nonzero(denom))
                frac = float(np.count_nonzero(win & denom) / denom_n) if denom_n else 0.0
                if denom_n == 0 or frac < min_valid_frac:
                    continue
            u_win = u[y0:y1, x0:x1]
            v_win = v[y0:y1, x0:x1]
            if on_disk is not None:
                disk_m = on_disk[y0:y1, x0:x1]
                u_win = np.where(disk_m, u_win, np.nan)
                v_win = np.where(disk_m, v_win, np.nan)
            if valid_mask is not None:
                m = valid_mask[y0:y1, x0:x1]
                if not np.any(m):
                    continue
                u_win = np.where(m, u_win, np.nan)
                v_win = np.where(m, v_win, np.nan)
            if not np.any(np.isfinite(u_win)):
                continue
            du = float(np.nanmean(u_win))
            dv = float(np.nanmean(v_win))
            if min_mag_px > 0:
                mag = float(np.hypot(du, dv))
                if not np.isfinite(mag) or mag < min_mag_px:
                    continue
            xs_out.append(x)
            ys_out.append(y)
            u_out.append(du)
            v_out.append(dv)

    return {
        "X": np.asarray(xs_out, dtype=np.float64),
        "Y": np.asarray(ys_out, dtype=np.float64),
        "U": np.asarray(u_out, dtype=np.float64),
        "V": np.asarray(v_out, dtype=np.float64),
        "n_shown": len(xs_out),
        "n_grid": n_grid,
    }


def motion_sparse_from_dense(
    u: np.ndarray,
    v: np.ndarray,
    *,
    on_disk: np.ndarray | None = None,
    stride: int = 1,
    min_mag_px: float = 1e-8,
) -> dict:
    """
    One arrow per site where the dense Phase-1 flow is non-zero — no quality mask.

    Reads ``u[y, x], v[y, x]`` directly (not window-averaged). Skips off-disk pixels
    and sites where speed is below ``min_mag_px`` (numerical noise only). Use stride=1
    to place an arrow wherever motion exists on the grid without stride gaps.
    """
    h, w = u.shape
    if on_disk is None:
        on_disk = np.ones((h, w), dtype=bool)
    stride = max(1, int(stride))
    xs_out, ys_out, u_out, v_out = [], [], [], []
    n_candidates = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if not on_disk[y, x]:
                continue
            n_candidates += 1
            du = float(u[y, x])
            dv = float(v[y, x])
            mag = float(np.hypot(du, dv))
            if not np.isfinite(mag) or mag < min_mag_px:
                continue
            xs_out.append(x)
            ys_out.append(y)
            u_out.append(du)
            v_out.append(dv)
    return {
        "X": np.asarray(xs_out, dtype=np.float64),
        "Y": np.asarray(ys_out, dtype=np.float64),
        "U": np.asarray(u_out, dtype=np.float64),
        "V": np.asarray(v_out, dtype=np.float64),
        "n_shown": len(xs_out),
        "n_grid": n_candidates,
    }


def build_filtered_amv(
    u_px: np.ndarray,
    v_px: np.ndarray,
    radiance_gray: np.ndarray,
    disagreement_px: np.ndarray,
    *,
    ux_ms: np.ndarray | None = None,
    uy_ms: np.ndarray | None = None,
    speed_ms: np.ndarray | None = None,
    rife_mask: np.ndarray | None = None,
    on_disk: np.ndarray | None = None,
    stride: int = 24,
    window: int = 10,
    texture_percentile: float = 15.0,
    disagree_px_max: float = 0.5,
    use_rife_mask: bool = True,
) -> dict:
    """Phase-3 AMV product: quality mask + subsampled px and m/s grids.

    `on_disk` (bool array, same shape as u_px/v_px) marks which pixels are real Earth
    data vs. off-disk space/fill. When omitted, a geometric full-disk mask is assumed
    via the caller (see app.py) — arrows are never drawn off-Earth.
    """
    valid, tex = amv_quality_mask(
        radiance_gray,
        disagreement_px,
        speed_ms=speed_ms,
        rife_mask=rife_mask,
        on_disk=on_disk,
        texture_percentile=texture_percentile,
        disagree_px_max=disagree_px_max,
        use_rife_mask=use_rife_mask,
    )
    sparse_px = subsample_amv_grid(u_px, v_px, stride, window, valid, on_disk=on_disk)
    sparse_ms = None
    if ux_ms is not None and uy_ms is not None:
        sparse_ms = subsample_amv_grid(ux_ms, uy_ms, stride, window, valid, on_disk=on_disk)
    denom_mask = on_disk if on_disk is not None else np.ones_like(valid, dtype=bool)
    denom_n = int(np.count_nonzero(denom_mask))
    return {
        "valid_mask": valid,
        "texture": tex,
        "sparse_px": sparse_px,
        "sparse_ms": sparse_ms,
        "valid_pixel_frac": float(np.count_nonzero(valid & denom_mask) / denom_n) if denom_n else 0.0,
    }


def filtered_amv_caption(filt: dict, units: str = "m/s") -> str:
    px = filt["sparse_px"]
    ms = filt.get("sparse_ms")
    parts = [
        f"Grid **{px['n_shown']}/{px['n_grid']}** arrows shown",
        f"valid pixels **{filt['valid_pixel_frac']*100:.1f}%**",
    ]
    if ms is not None and ms["n_shown"] > 0:
        spd = np.sqrt(ms["U"] ** 2 + ms["V"] ** 2)
        parts.append(f"mean arrow speed **{float(np.mean(spd)):.2f} {units}**")
    return " · ".join(parts)


# --- Phase 4: motion timelapse overlays ---

def draw_sparse_arrows(
    frame_rgb: np.ndarray,
    sparse: dict,
    color: tuple[int, int, int] = (255, 60, 60),
    thickness: int = 2,
    tip_length: float = 0.35,
    min_mag_px: float = 0.3,
    disk_mask: np.ndarray | None = None,
    color_by_speed: bool = False,
    thin_arrows: bool = False,
) -> np.ndarray:
    """
    Burn a sparse arrow field onto a copy of an RGB frame.

    `sparse` is the same X/Y/U/V dict produced by `subsample_amv_grid` (or the
    `sparse_px`/`sparse_ms` entries of `build_filtered_amv`'s output) — i.e. it is
    already stride-subsampled and, if built from a quality-filtered `valid_mask`,
    already restricted to trustworthy vectors. Arrow length is a fixed fraction of
    grid spacing (not literal pixel displacement) so small-but-valid vectors stay
    legible; only direction carries the physical meaning.

    Intended use: draw the *same* field onto several frames of a sequence (e.g.
    t0, mid, t2) so that, played back, you can visually check whether the arrows
    point the way the clouds are actually moving.
    """
    import cv2

    out = np.ascontiguousarray(np.asarray(frame_rgb, dtype=np.uint8))
    if out.ndim == 2:
        out = np.ascontiguousarray(np.stack([out] * 3, axis=-1))
    elif not out.flags["WRITEABLE"]:
        out = out.copy()

    X = np.asarray(sparse.get("X", []), dtype=np.float64)
    Y = np.asarray(sparse.get("Y", []), dtype=np.float64)
    U = np.asarray(sparse.get("U", []), dtype=np.float64)
    V = np.asarray(sparse.get("V", []), dtype=np.float64)
    if X.size == 0:
        return out

    ux = np.unique(X)
    spacing = float(np.median(np.diff(ux))) if ux.size > 1 else 24.0
    if thin_arrows:
        length = max(spacing * 0.5, 7.0)
        thickness = 1
        tip_length = max(tip_length, 0.35)
        line_type = cv2.LINE_AA
    else:
        length = max(spacing * 0.55, 8.0)
        line_type = cv2.LINE_AA
    h, w = out.shape[:2]

    cmap = None
    vmax = 1.0
    if color_by_speed:
        from matplotlib import colormaps

        cmap = colormaps["turbo"]
        mags_all = np.hypot(U, V)
        finite = mags_all[np.isfinite(mags_all)]
        vmin = float(np.percentile(finite, 2)) if finite.size else 0.0
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        vmax = max(vmax, vmin + 1e-6)
    else:
        vmin = 0.0

    for x, y, du, dv in zip(X, Y, U, V):
        xi, yi = int(round(x)), int(round(y))
        if disk_mask is not None:
            if not (0 <= yi < h and 0 <= xi < w and disk_mask[yi, xi]):
                continue
        mag = float(np.hypot(du, dv))
        if not np.isfinite(mag):
            continue
        if min_mag_px > 0 and mag < min_mag_px:
            continue
        nx, ny = du / mag, dv / mag
        x0, y0 = xi, yi
        x1, y1 = int(round(x + nx * length)), int(round(y + ny * length))
        if color_by_speed and cmap is not None:
            t = float(np.clip((mag - vmin) / (vmax - vmin), 0, 1))
            r, g, b = [int(c * 255) for c in cmap(t)[:3]]
            line_color = (r, g, b)
        else:
            line_color = color
        cv2.arrowedLine(
            out, (x0, y0), (x1, y1), line_color, thickness, line_type, tipLength=tip_length,
        )
    return out


def render_arrow_layer(shape: tuple[int, int], sparse: dict, **kwargs) -> np.ndarray:
    """Draw arrows on a black canvas — identical pixels for compositing onto any frame."""
    h, w = shape[:2]
    return draw_sparse_arrows(np.zeros((h, w, 3), dtype=np.uint8), sparse, **kwargs)


def composite_sparse_arrows(
    frame_rgb: np.ndarray,
    sparse: dict,
    arrow_layer: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Paste a pre-rendered arrow layer — vivid speed colors, identical thickness on every frame."""
    frame = np.ascontiguousarray(np.asarray(frame_rgb, dtype=np.uint8))
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    h, w = frame.shape[:2]
    if arrow_layer is None or arrow_layer.shape[:2] != (h, w):
        arrow_layer = render_arrow_layer((h, w), sparse, **kwargs)
    arrow_f = arrow_layer.astype(np.float32)
    frame_f = frame.astype(np.float32)
    strength = np.max(arrow_f, axis=-1)
    out = frame_f.copy()
    solid = strength >= 28
    out[solid] = arrow_f[solid]
    fringe = (strength > 0) & ~solid
    if np.any(fringe):
        a = (strength[fringe] / 255.0)[:, np.newaxis]
        out[fringe] = frame_f[fringe] * (1.0 - a) + arrow_f[fringe] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def build_motion_timelapses(
    t0: np.ndarray,
    t1_gt: np.ndarray | None,
    t2: np.ndarray,
    t1_pred: np.ndarray,
    sparse_px: dict,
    *,
    gt_color: tuple[int, int, int] = (255, 170, 40),
    pred_color: tuple[int, int, int] = (80, 170, 255),
    arrow_kw: dict | None = None,
) -> dict:
    """
    Apply the same filtered AMV arrow field to all four frames (t0, ground-truth t1,
    t2, and RIFE-predicted t1), grouped into two 3-frame timelapse sequences:

      - "ground_truth": t0 → real t1 → t2 (arrows over what actually happened)
      - "predicted":     t0 → RIFE t1 → t2 (arrows over the model's midpoint guess)

    Both use the *same* `sparse_px` vectors (from `build_filtered_amv`), so any
    difference between the two GIFs comes from the frame content, not the arrows —
    looping each side by side shows whether cloud displacement in each sequence
    agrees with the AMV direction.

    Returns {"predicted": [3 frames], "ground_truth": [3 frames] | None}.
    """
    ak = dict(arrow_kw or {})
    use_speed_color = bool(ak.get("color_by_speed", False))
    use_composite = bool(ak.get("thin_arrows", False))
    layer = None
    if use_composite:
        h, w = t0.shape[:2]
        layer = render_arrow_layer((h, w), sparse_px, **ak)

    def _draw(frame: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        if use_composite:
            return composite_sparse_arrows(frame, sparse_px, arrow_layer=layer)
        if use_speed_color:
            return draw_sparse_arrows(frame, sparse_px, **ak)
        kw = {k: v for k, v in ak.items() if k != "color_by_speed"}
        return draw_sparse_arrows(frame, sparse_px, color=color, **kw)

    pred_frames = [
        _draw(t0, pred_color),
        _draw(t1_pred, pred_color),
        _draw(t2, pred_color),
    ]
    gt_frames = None
    if t1_gt is not None:
        gt_frames = [
            _draw(t0, gt_color),
            _draw(t1_gt, gt_color),
            _draw(t2, gt_color),
        ]
    return {"predicted": pred_frames, "ground_truth": gt_frames}