"""ISRO PS12 validation metrics — SSIM, MSE, PSNR, FSIM on radiance frames."""

from __future__ import annotations

import numpy as np
import torch


def gray01(img: np.ndarray) -> np.ndarray:
    g = img[..., 0] if img.ndim == 3 else img
    return g.astype(np.float64) / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    x, y = gray01(a), gray01(b)
    return float(np.mean((x - y) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    e = mse(a, b)
    return float("inf") if e == 0 else float(20 * np.log10(1.0) - 10 * np.log10(e))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(gray01(a), gray01(b), data_range=1.0))


def fsim(a: np.ndarray, b: np.ndarray) -> float:
    """FSIM via piq — luminance-only (thermal IR is single-channel)."""
    from piq import fsim as fsim_fn

    x = torch.from_numpy(gray01(a)).unsqueeze(0).unsqueeze(0).float()
    y = torch.from_numpy(gray01(b)).unsqueeze(0).unsqueeze(0).float()
    return float(fsim_fn(x, y, data_range=1.0, chromatic=False).item())


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(pred, gt),
        "psnr": psnr(pred, gt),
        "ssim": ssim(pred, gt),
        "fsim": fsim(pred, gt),
    }
