"""
Physics-inspired (PINN-style) losses for satellite frame interpolation.

Constraints:
  1. Photometric consistency — radiance advects with estimated flow
  2. Flow smoothness — coherent cloud motion (total variation)
  3. Midpoint symmetry — displacements from t0 and t2 are anti-symmetric at t=0.5
  4. Weak divergence — squared div(u,v) penalty (normalized)
  5. BT envelope — prediction stays within t0/t2 radiance envelope (+ slack)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_total_variation(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Spatial smoothness on optical flow (lower = smoother)."""
    return (
        (u[:, :, :, 1:] - u[:, :, :, :-1]).abs().mean()
        + (v[:, :, 1:, :] - v[:, :, :-1, :]).abs().mean()
    )


def flow_divergence_squared(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Weak incompressibility: mean(div^2) over the field."""
    du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]
    dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
    div = du_dx[:, :, :-1, :] + dv_dy[:, :, :, :-1]
    return (div ** 2).mean()


def midpoint_symmetry_loss(u0: torch.Tensor, v0: torch.Tensor, u1: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """At temporal midpoint, forward flow from t0 mirrors backward flow from t2."""
    return F.l1_loss(u0, -u1) + F.l1_loss(v0, -v1)


def photometric_consistency_loss(warped0: torch.Tensor, warped1: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Brightness constancy: warp(t0) and warp(t2) should match ground-truth t1."""
    return F.l1_loss(warped0, target) + F.l1_loss(warped1, target)


def brightness_envelope_loss(
    pred: torch.Tensor,
    t0: torch.Tensor,
    t2: torch.Tensor,
    slack: float = 0.05,
) -> torch.Tensor:
    """
  BT / radiance envelope: t1 should lie between t0 and t2; pred must not exceed
  that envelope (plus slack). Fires unlike [0,1] clamp when pred is already in-range.
    """
    lo = torch.min(t0, t2) - slack
    hi = torch.max(t0, t2) + slack
    return (F.relu(pred - hi) + F.relu(lo - pred)).mean()


def compute_physics_losses(
    pred: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    warped0: torch.Tensor,
    warped1: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
    u1: torch.Tensor,
    v1: torch.Tensor,
    bt_slack: float = 0.05,
) -> dict[str, torch.Tensor]:
    return {
        "photo": photometric_consistency_loss(warped0, warped1, t1),
        "smooth": flow_total_variation(u0, v0) + flow_total_variation(u1, v1),
        "sym": midpoint_symmetry_loss(u0, v0, u1, v1),
        "div": flow_divergence_squared(u0, v0),
        "bound": brightness_envelope_loss(pred, t0, t2, slack=bt_slack),
    }


def curriculum_scale(epoch: int, phase1_end: int, phase2_end: int) -> tuple[float, float, str]:
    """
    Three-phase curriculum:
      1..phase1_end       — statistical only (physics_scale=0)
      phase1+1..phase2    — physics ramps 0 -> 1
      phase2+1..end       — full joint (physics_scale=1, lr_factor=0.5)
    """
    if epoch <= phase1_end:
        return 0.0, 1.0, "statistical"
    if epoch <= phase2_end:
        span = max(1, phase2_end - phase1_end)
        t = (epoch - phase1_end) / span
        return float(t), 1.0, "physics_ramp"
    return 1.0, 0.5, "joint"


@torch.no_grad()
def eval_physics_metrics(
    warped0: torch.Tensor,
    warped1: torch.Tensor,
    t1: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
) -> dict[str, float]:
    """Validation diagnostics (mirrors app motion-tab metrics)."""
    photo = float(photometric_consistency_loss(warped0, warped1, t1).item()) / 2.0
    smooth = float(flow_total_variation(u0, v0).item())
    return {"photo_err": photo, "flow_smooth": smooth}
