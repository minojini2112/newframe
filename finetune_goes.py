"""
Stage 1 — Statistical fine-tuning (identical to ISRO2026/finetune_goes.py).

Maximizes image quality: L1 + SSIM + beat-linear margin.
Run this first, then finetune_physics.py for PINN stage 2.

Usage:
    python finetune_goes.py --epochs 80 --patch 256

Output:
    checkpoints/goes_finetuned/flownet.pkl
    checkpoints/goes_finetuned_statistical/flownet.pkl  (backup for stage 2)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def resolve_practical_rife() -> Path:
    for cand in (ROOT / "Practical-RIFE", REPO / "ISRO2026" / "Practical-RIFE"):
        if (cand / "train_log" / "flownet.pkl").exists():
            return cand
    return ROOT / "Practical-RIFE"


PRACTICAL_RIFE = resolve_practical_rife()
TRAIN_LOG = PRACTICAL_RIFE / "train_log"
CKPT_DIR = ROOT / "checkpoints" / "goes_finetuned"
STAT_BACKUP_DIR = ROOT / "checkpoints" / "goes_finetuned_statistical"

NC_DIRS = [
    REPO / "ISRO2026" / "goes19_c13",
    REPO / "goes19_c13",
    REPO / "goes19_c13_arthur",
]


def load_rife_checkpoint(model, ckpt_dir: Path) -> None:
    """Load flownet.pkl — handles `module.*` prefix and ignores unused teacher/caltime keys."""
    sd = torch.load(ckpt_dir / "flownet.pkl", map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    wanted = model.flownet.state_dict()
    sd = {k: v for k, v in sd.items() if k in wanted}
    missing = [k for k in wanted if k not in sd]
    if missing:
        raise RuntimeError(f"Checkpoint missing {len(missing)} keys (e.g. {missing[0]})")
    model.flownet.load_state_dict(sd, strict=True)


def save_rife_checkpoint(model, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.flownet.state_dict(), ckpt_dir / "flownet.pkl")


def verify_checkpoint_reload(model, ckpt_dir: Path, triplets, device, resize: int) -> dict:
    from train_log.RIFE_HDv3 import Model

    probe = Model()
    load_rife_checkpoint(probe, ckpt_dir)
    probe.eval()
    probe.flownet.to(device)
    return eval_triplets(probe, triplets, device, resize)


def load_nc_radiance(path: Path) -> np.ndarray:
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        return np.asarray(ds["Rad"].values, dtype=np.float64)
    finally:
        ds.close()


def rad_to_rgb(rad: np.ndarray, vmin: float, vmax: float, size: int) -> np.ndarray:
    img = np.nan_to_num(rad, nan=vmin)
    if vmax <= vmin:
        vmax = vmin + 1.0
    g = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    rgb = np.stack([(g * 255).astype(np.uint8)] * 3, axis=-1)
    return np.array(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))


def collect_triplets(resize: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for folder in NC_DIRS:
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.nc"))
        if len(files) < 3:
            continue
        for i in range(len(files) - 2):
            rads = [load_nc_radiance(files[i + k]) for k in range(3)]
            flat = np.concatenate([r[np.isfinite(r)].ravel() for r in rads])
            if flat.size > 30_000:
                idx = np.linspace(0, flat.size - 1, 30_000, dtype=np.int64)
                flat = flat[idx]
            vmin, vmax = float(np.percentile(flat, 2)), float(np.percentile(flat, 98))
            frames = [rad_to_rgb(r, vmin, vmax, resize) for r in rads]
            triplets.append((frames[0], frames[1], frames[2]))
            del rads, frames, flat
    return triplets


class GoesTripletDataset(Dataset):
    def __init__(
        self,
        triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        patch: int,
        crops_per_triplet: int = 8,
    ):
        self.triplets = triplets
        self.patch = patch
        self.crops_per_triplet = crops_per_triplet

    def __len__(self) -> int:
        return len(self.triplets) * self.crops_per_triplet

    def __getitem__(self, idx: int):
        t0, t1, t2 = self.triplets[idx // self.crops_per_triplet]
        h, w = t0.shape[:2]
        ps = self.patch
        if h == ps and w == ps:
            y, x = 0, 0
        else:
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
        t0p = t0[y : y + ps, x : x + ps]
        t1p = t1[y : y + ps, x : x + ps]
        t2p = t2[y : y + ps, x : x + ps]
        if random.random() < 0.5:
            t0p = np.ascontiguousarray(t0p[:, ::-1])
            t1p = np.ascontiguousarray(t1p[:, ::-1])
            t2p = np.ascontiguousarray(t2p[:, ::-1])
        to_t = lambda im: torch.from_numpy(im.transpose(2, 0, 1)).float() / 255.0
        return to_t(t0p), to_t(t1p), to_t(t2p)


def pad_tensor(t: torch.Tensor, tile: int = 128) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = t.shape
    ph = ((h - 1) // tile + 1) * tile
    pw = ((w - 1) // tile + 1) * tile
    if ph == h and pw == w:
        return t, h, w
    return F.pad(t, (0, pw - w, 0, ph - h)), h, w


def predict_middle(model, t0: torch.Tensor, t2: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    t0p, h, w = pad_tensor(t0)
    t2p, _, _ = pad_tensor(t2)
    out = model.inference(t0p, t2p, timestep=0.5, scale=scale)
    return out[:, :, :h, :w]


@torch.no_grad()
def eval_triplets(model, triplets, device, resize: int) -> dict:
    from skimage.metrics import structural_similarity

    model.eval()
    rife_ssim, lin_ssim = [], []
    rife_psnr, lin_psnr = [], []

    for t0_np, t1_np, t2_np in triplets:
        t0 = torch.from_numpy(t0_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t1 = torch.from_numpy(t1_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t2 = torch.from_numpy(t2_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        pred = predict_middle(model, t0, t2)
        lin = (t0 + t2) * 0.5

        def gray_metrics(a: torch.Tensor, b: torch.Tensor):
            ag = a[0, 0].cpu().numpy()
            bg = b[0, 0].cpu().numpy()
            mse = float(np.mean((ag - bg) ** 2))
            psnr = float("inf") if mse == 0 else 20 * np.log10(1.0) - 10 * np.log10(mse)
            s = float(structural_similarity(ag, bg, data_range=1.0))
            return s, psnr

        rs, rp = gray_metrics(pred, t1)
        ls, lp = gray_metrics(lin, t1)
        rife_ssim.append(rs)
        lin_ssim.append(ls)
        rife_psnr.append(rp)
        lin_psnr.append(lp)

    return {
        "rife_ssim": float(np.mean(rife_ssim)),
        "linear_ssim": float(np.mean(lin_ssim)),
        "rife_psnr": float(np.mean(rife_psnr)),
        "linear_psnr": float(np.mean(lin_psnr)),
        "delta_ssim": float(np.mean(rife_ssim) - np.mean(lin_ssim)),
        "n": len(triplets),
    }


def train(args):
    if str(PRACTICAL_RIFE) not in sys.path:
        sys.path.insert(0, str(PRACTICAL_RIFE))

    from model.pytorch_msssim import ssim as ssim_torch
    from train_log.RIFE_HDv3 import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 1: statistical fine-tuning (ISRO2026 method)")
    print(f"RIFE: {PRACTICAL_RIFE}\n")

    triplets = collect_triplets(args.resize)
    if len(triplets) < 4:
        raise SystemExit(f"Need >=4 triplets, found {len(triplets)}. Check GOES .nc folders.")

    rng = random.Random(args.seed)
    idx = list(range(len(triplets)))
    rng.shuffle(idx)
    n_val = max(2, int(len(triplets) * args.val_frac))
    val_idx = set(idx[:n_val])
    train_trips = [triplets[i] for i in range(len(triplets)) if i not in val_idx]
    val_trips = [triplets[i] for i in range(len(triplets)) if i in val_idx]
    print(f"Triplets: {len(triplets)} total, {len(train_trips)} train, {len(val_trips)} val")

    train_ds = GoesTripletDataset(train_trips, patch=args.patch, crops_per_triplet=args.crops_per_triplet)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = Model()
    load_rife_checkpoint(model, TRAIN_LOG)
    model.flownet.to(device)
    model.train()

    optim = torch.optim.AdamW(model.flownet.parameters(), lr=args.lr, weight_decay=1e-4)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = []

    baseline = eval_triplets(model, val_trips, device, args.resize)
    best_delta = baseline["delta_ssim"]
    print(
        f"Baseline val: RIFE SSIM={baseline['rife_ssim']:.4f}  "
        f"Linear={baseline['linear_ssim']:.4f}  delta={baseline['delta_ssim']:+.4f}",
        flush=True,
    )
    history.append({"epoch": 0, "phase": "baseline", **baseline})

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for t0, t1, t2 in train_dl:
            t0, t1, t2 = t0.to(device), t1.to(device), t2.to(device)
            pred = predict_middle(model, t0, t2, scale=1.0)
            lin = (t0 + t2) * 0.5

            loss_l1 = F.l1_loss(pred, t1)
            loss_ssim = 1.0 - ssim_torch(pred, t1, val_range=1.0)
            mse_pred = F.mse_loss(pred, t1)
            mse_lin = F.mse_loss(lin.detach(), t1)
            loss_beat = F.relu(mse_pred - mse_lin + args.margin)

            loss = loss_l1 + args.w_ssim * loss_ssim + args.w_beat * loss_beat

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.flownet.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.item()))

        lr_now = args.lr * (0.5 * (1 + np.cos(np.pi * epoch / args.epochs)))
        for pg in optim.param_groups:
            pg["lr"] = lr_now

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = eval_triplets(model, val_trips, device, args.resize)
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(np.mean(losses))
            history.append(metrics)

            improved = metrics["delta_ssim"] > best_delta
            if improved:
                best_delta = metrics["delta_ssim"]
                save_rife_checkpoint(model, CKPT_DIR)
                reloaded = verify_checkpoint_reload(model, CKPT_DIR, val_trips, device, args.resize)
                if abs(reloaded["rife_ssim"] - metrics["rife_ssim"]) > 0.01:
                    raise SystemExit(
                        f"Checkpoint reload mismatch: mem={metrics['rife_ssim']:.4f} "
                        f"disk={reloaded['rife_ssim']:.4f}. Training stopped."
                    )

            print(
                f"Epoch {epoch:3d}/{args.epochs}  loss={metrics['train_loss']:.4f}  "
                f"RIFE SSIM={metrics['rife_ssim']:.4f}  Linear={metrics['linear_ssim']:.4f}  "
                f"delta={metrics['delta_ssim']:+.4f}  PSNR={metrics['rife_psnr']:.1f}dB"
                + ("  *best*" if improved else ""),
                flush=True,
            )
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}  loss={float(np.mean(losses)):.4f}", flush=True)

    if not (CKPT_DIR / "flownet.pkl").exists():
        raise SystemExit("No checkpoint saved — model never improved on validation.")

    full = eval_triplets(model, triplets, device, args.resize)
    reloaded = verify_checkpoint_reload(model, CKPT_DIR, triplets, device, args.resize)
    print(f"In-memory  SSIM={full['rife_ssim']:.4f}")
    print(f"From disk   SSIM={reloaded['rife_ssim']:.4f}")
    if abs(reloaded["rife_ssim"] - full["rife_ssim"]) > 0.01:
        raise SystemExit("Saved checkpoint does not match in-memory model — aborting.")
    full = reloaded
    print(f"RIFE SSIM={full['rife_ssim']:.4f}  Linear={full['linear_ssim']:.4f}  delta={full['delta_ssim']:+.4f}")
    print(f"RIFE PSNR={full['rife_psnr']:.2f} dB  Linear PSNR={full['linear_psnr']:.2f} dB")

    pre = Model()
    load_rife_checkpoint(pre, TRAIN_LOG)
    pre.eval()
    pre.flownet.to(device)
    pre_metrics = eval_triplets(pre, triplets, device, args.resize)
    print(f"Pre-trained RIFE SSIM={pre_metrics['rife_ssim']:.4f} (reference)")

    if full["rife_ssim"] < pre_metrics["rife_ssim"]:
        print("WARNING: fine-tuned model worse than pre-trained — restoring pre-trained weights.")
        save_rife_checkpoint(pre, CKPT_DIR)
        full = pre_metrics
        full["restored_pretrained"] = True

    STAT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CKPT_DIR / "flownet.pkl", STAT_BACKUP_DIR / "flownet.pkl")

    meta = {
        "training_mode": "statistical",
        "stage": 1,
        "args": vars(args),
        "baseline": baseline,
        "best_val_delta_ssim": best_delta,
        "full_eval": full,
        "history": history,
        "statistical_backup": str(STAT_BACKUP_DIR / "flownet.pkl"),
    }
    (CKPT_DIR / "training_log.json").write_text(json.dumps(meta, indent=2))
    (STAT_BACKUP_DIR / "training_log.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {CKPT_DIR / 'flownet.pkl'}")
    print(f"Statistical backup: {STAT_BACKUP_DIR / 'flownet.pkl'}")
    print("Next: python finetune_physics.py  (stage 2 PINN)")


def main():
    p = argparse.ArgumentParser(description="Stage 1 — statistical RIFE fine-tuning (ISRO2026)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--w-ssim", type=float, default=0.3, dest="w_ssim")
    p.add_argument("--w-beat", type=float, default=0.5, dest="w_beat")
    p.add_argument("--margin", type=float, default=1e-5)
    p.add_argument("--crops-per-triplet", type=int, default=12)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
