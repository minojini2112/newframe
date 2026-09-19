"""
Stage 2 — PINN physics refinement on top of statistical fine-tune.

Requires stage 1 complete:
    checkpoints/goes_finetuned_statistical/flownet.pkl

Keeps full statistical losses (L1 + SSIM + beat-linear) at ISRO weights
and adds light physics terms to improve flow without sacrificing SSIM.

Usage:
    python finetune_goes.py --epochs 80        # stage 1 first
    python finetune_physics.py --epochs 40     # stage 2

Output overwrites:
    checkpoints/goes_finetuned/flownet.pkl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finetune_goes import (
    CKPT_DIR,
    PRACTICAL_RIFE,
    STAT_BACKUP_DIR,
    GoesTripletDataset,
    collect_triplets,
    load_rife_checkpoint,
    pad_tensor,
    predict_middle,
    save_rife_checkpoint,
)
from physics_losses import compute_physics_losses, eval_physics_metrics

ROOT = Path(__file__).resolve().parent


def physics_forward(model, t0: torch.Tensor, t2: torch.Tensor, scale: float = 1.0) -> dict:
    from model.warplayer import warp

    t0p, h, w = pad_tensor(t0)
    t2p, _, _ = pad_tensor(t2)
    imgs = torch.cat((t0p, t2p), dim=1)
    scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    flow_list, mask, merged = model.flownet(imgs, 0.5, scale_list)
    flow = flow_list[-1]
    mask = torch.sigmoid(mask)
    warped0 = warp(t0p, flow[:, :2])
    warped1 = warp(t2p, flow[:, 2:4])
    return {
        "pred": merged[-1][:, :, :h, :w],
        "warped0": warped0[:, :, :h, :w],
        "warped1": warped1[:, :, :h, :w],
        "u0": flow[:, 0:1, :h, :w],
        "v0": flow[:, 1:2, :h, :w],
        "u1": flow[:, 2:3, :h, :w],
        "v1": flow[:, 3:4, :h, :w],
    }


@torch.no_grad()
def eval_triplets(model, triplets, device) -> dict:
    from skimage.metrics import structural_similarity

    model.eval()
    rife_ssim, lin_ssim, rife_psnr = [], [], []
    photo_errs, flow_smooths = [], []

    for t0_np, t1_np, t2_np in triplets:
        t0 = torch.from_numpy(t0_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t1 = torch.from_numpy(t1_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t2 = torch.from_numpy(t2_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        out = physics_forward(model, t0, t2)
        pred, lin = out["pred"], (t0 + t2) * 0.5

        ag, bg = pred[0, 0].cpu().numpy(), t1[0, 0].cpu().numpy()
        mse = float(np.mean((ag - bg) ** 2))
        psnr = float("inf") if mse == 0 else 20 * np.log10(1.0) - 10 * np.log10(mse)
        rs = float(structural_similarity(ag, bg, data_range=1.0))
        lg = lin[0, 0].cpu().numpy()
        ls = float(structural_similarity(lg, bg, data_range=1.0))

        pm = eval_physics_metrics(out["warped0"], out["warped1"], t1, out["u0"], out["v0"])
        rife_ssim.append(rs)
        lin_ssim.append(ls)
        rife_psnr.append(psnr)
        photo_errs.append(pm["photo_err"])
        flow_smooths.append(pm["flow_smooth"])

    return {
        "rife_ssim": float(np.mean(rife_ssim)),
        "linear_ssim": float(np.mean(lin_ssim)),
        "rife_psnr": float(np.mean(rife_psnr)),
        "delta_ssim": float(np.mean(rife_ssim) - np.mean(lin_ssim)),
        "photo_err": float(np.mean(photo_errs)),
        "flow_smooth": float(np.mean(flow_smooths)),
        "n": len(triplets),
    }


def verify_reload(ckpt_dir: Path, triplets, device) -> dict:
    from train_log.RIFE_HDv3 import Model

    m = Model()
    load_rife_checkpoint(m, ckpt_dir)
    m.eval()
    m.flownet.to(device)
    return eval_triplets(m, triplets, device)


def statistical_ckpt_path() -> Path:
    p = STAT_BACKUP_DIR / "flownet.pkl"
    if p.exists():
        return p
    p2 = CKPT_DIR / "flownet.pkl"
    if p2.exists():
        log = CKPT_DIR / "training_log.json"
        if log.exists():
            meta = json.loads(log.read_text())
            if meta.get("training_mode") == "statistical" or meta.get("stage") == 1:
                return p2
    raise SystemExit(
        "Stage 1 checkpoint not found.\n"
        "Run first: python finetune_goes.py --epochs 80"
    )


def train(args):
    if str(PRACTICAL_RIFE) not in sys.path:
        sys.path.insert(0, str(PRACTICAL_RIFE))

    from model.pytorch_msssim import ssim as ssim_torch
    from train_log.RIFE_HDv3 import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stat_ckpt = statistical_ckpt_path()
    print(f"Device: {device}")
    print(f"Stage 2: PINN physics refinement")
    print(f"Load statistical weights: {stat_ckpt}\n")

    triplets = collect_triplets(args.resize)
    if len(triplets) < 6:
        raise SystemExit(f"Need >=6 triplets, found {len(triplets)}")

    rng = random.Random(args.seed)
    idx = list(range(len(triplets)))
    rng.shuffle(idx)
    n_val = max(3, int(len(triplets) * args.val_frac))
    val_idx = set(idx[:n_val])
    train_trips = [triplets[i] for i in range(len(triplets)) if i not in val_idx]
    val_trips = [triplets[i] for i in range(len(triplets)) if i in val_idx]
    print(f"Triplets: {len(triplets)} total, {len(train_trips)} train, {len(val_trips)} val")

    train_dl = DataLoader(
        GoesTripletDataset(train_trips, args.patch, args.crops_per_triplet),
        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
    )

    model = Model()
    load_rife_checkpoint(model, stat_ckpt.parent)
    model.flownet.to(device)
    model.train()

    stat_full = eval_triplets(model, triplets, device)
    stat_floor_ssim = stat_full["rife_ssim"] - args.ssim_guard
    stat_floor_delta = stat_full["delta_ssim"] - args.delta_guard
    print(
        f"Statistical reference: SSIM={stat_full['rife_ssim']:.4f}  "
        f"delta={stat_full['delta_ssim']:+.4f}  photo={stat_full['photo_err']:.5f}  "
        f"smooth={stat_full['flow_smooth']:.4f}",
        flush=True,
    )
    print(f"Guardrails: SSIM>={stat_floor_ssim:.4f}  delta>={stat_floor_delta:+.4f}\n", flush=True)

    optim = torch.optim.AdamW(model.flownet.parameters(), lr=args.lr, weight_decay=1e-4)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stat_ckpt, CKPT_DIR / "flownet.pkl")

    best_score = (
        stat_full["rife_ssim"]
        - stat_full["photo_err"] * 5.0
        - stat_full["flow_smooth"] * 2.0
    )
    history = [{"epoch": 0, "phase": "statistical_ref", **stat_full}]
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = {k: [] for k in ("total", "l1", "ssim", "beat", "photo", "smooth", "sym", "div", "bound")}

        for t0b, t1, t2 in train_dl:
            t0b, t1, t2 = t0b.to(device), t1.to(device), t2.to(device)
            out = physics_forward(model, t0b, t2)
            pred = out["pred"]
            lin = (t0b + t2) * 0.5

            loss_l1 = F.l1_loss(pred, t1)
            loss_ssim = 1.0 - ssim_torch(pred, t1, val_range=1.0)
            mse_pred = F.mse_loss(pred, t1)
            mse_lin = F.mse_loss(lin.detach(), t1)
            loss_beat = F.relu(mse_pred - mse_lin + args.margin)

            phys = compute_physics_losses(
                pred, t0b, t1, t2,
                out["warped0"], out["warped1"],
                out["u0"], out["v0"], out["u1"], out["v1"],
                bt_slack=args.bt_slack,
            )

            loss = (
                loss_l1
                + args.w_ssim * loss_ssim
                + args.w_beat * loss_beat
                + args.w_photo * phys["photo"]
                + args.w_smooth * phys["smooth"]
                + args.w_sym * phys["sym"]
                + args.w_div * phys["div"]
                + args.w_bound * phys["bound"]
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.flownet.parameters(), 1.0)
            optim.step()

            losses["total"].append(float(loss.item()))
            losses["l1"].append(float(loss_l1.item()))
            losses["ssim"].append(float(loss_ssim.item()))
            losses["beat"].append(float(loss_beat.item()))
            losses["photo"].append(float(phys["photo"].item()))
            losses["smooth"].append(float(phys["smooth"].item()))
            losses["sym"].append(float(phys["sym"].item()))
            losses["div"].append(float(phys["div"].item()))
            losses["bound"].append(float(phys["bound"].item()))

        lr_now = args.lr * (0.5 * (1 + np.cos(np.pi * epoch / args.epochs)))
        for pg in optim.param_groups:
            pg["lr"] = lr_now

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = eval_triplets(model, val_trips, device)
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(np.mean(losses["total"]))
            for k in losses:
                metrics[f"loss_{k}"] = float(np.mean(losses[k]))
            history.append(metrics)

            score = (
                metrics["rife_ssim"]
                - metrics["photo_err"] * 5.0
                - metrics["flow_smooth"] * 2.0
            )
            ok_ssim = metrics["rife_ssim"] >= stat_floor_ssim
            ok_delta = metrics["delta_ssim"] >= stat_floor_delta
            improved = score > best_score and ok_ssim and ok_delta

            if improved:
                best_score = score
                save_rife_checkpoint(model, CKPT_DIR)
                reloaded = verify_reload(CKPT_DIR, val_trips, device)
                if abs(reloaded["rife_ssim"] - metrics["rife_ssim"]) > 0.015:
                    raise SystemExit("Checkpoint reload mismatch")

            flag = " *best*" if improved else ""
            guard = "" if (ok_ssim and ok_delta) else " [guardrail]"
            print(
                f"Epoch {epoch:3d}/{args.epochs}  loss={metrics['train_loss']:.4f}  "
                f"SSIM={metrics['rife_ssim']:.4f}  delta={metrics['delta_ssim']:+.4f}  "
                f"photo={metrics['photo_err']:.5f}  smooth={metrics['flow_smooth']:.4f}{flag}{guard}",
                flush=True,
            )
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}  loss={float(np.mean(losses['total'])):.4f}", flush=True)

    full = verify_reload(CKPT_DIR, triplets, device)
    if full["rife_ssim"] < stat_floor_ssim:
        print("WARNING: stage 2 hurt SSIM — restoring statistical checkpoint.")
        shutil.copy2(stat_ckpt, CKPT_DIR / "flownet.pkl")
        full = stat_full
        full["restored_statistical"] = True

    meta = {
        "training_mode": "physics_pinn_stage2",
        "stage": 2,
        "args": vars(args),
        "statistical_reference": stat_full,
        "best_score": best_score,
        "full_eval": full,
        "history": history,
        "elapsed_sec": round(time.time() - t0, 1),
        "statistical_ckpt": str(stat_ckpt),
    }
    (CKPT_DIR / "training_log.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {CKPT_DIR / 'flownet.pkl'}")
    print(
        f"Final: SSIM={full['rife_ssim']:.4f}  delta={full['delta_ssim']:+.4f}  "
        f"photo={full['photo_err']:.5f}  smooth={full['flow_smooth']:.4f}"
    )


def main():
    p = argparse.ArgumentParser(description="Stage 2 — PINN physics on statistical checkpoint")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-6, help="Lower LR — refine, don't relearn")
    p.add_argument("--w-ssim", type=float, default=0.3, dest="w_ssim")
    p.add_argument("--w-beat", type=float, default=0.5, dest="w_beat")
    p.add_argument("--margin", type=float, default=1e-5)
    p.add_argument("--w-photo", type=float, default=0.2, dest="w_photo")
    p.add_argument("--w-smooth", type=float, default=0.02, dest="w_smooth")
    p.add_argument("--w-sym", type=float, default=0.05, dest="w_sym")
    p.add_argument("--w-div", type=float, default=0.05, dest="w_div")
    p.add_argument("--w-bound", type=float, default=0.05, dest="w_bound")
    p.add_argument("--bt-slack", type=float, default=0.05, dest="bt_slack")
    p.add_argument("--ssim-guard", type=float, default=0.002, dest="ssim_guard")
    p.add_argument("--delta-guard", type=float, default=0.0005, dest="delta_guard")
    p.add_argument("--crops-per-triplet", type=int, default=12)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
