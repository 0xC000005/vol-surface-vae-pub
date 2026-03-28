#!/usr/bin/env python
"""
160a: AR Frame Generation + CLN + E2E + K=2 (RC19-H1-S1)

Autoregressive frame-by-frame generation (FGN/AIFS pattern):
  history → encoder → condition
  for t = 1..30:
      z_t ~ N(0,1)^32  (fresh noise per step, independent)
      delta_t = FrameDecoder(condition, prev_frame, z_t)
      frame_t = prev_frame + delta_t

Per-frame afCRPS at D=25 (stronger gradient signal than D=750 one-shot).
K=2 members during training (FGN: fair CRPS unbiased at K=2, K=50 for eval).
E2E: encoder jointly trained with decoder.

Evidence: FGN (2506.10772), AIFS (2412.15832), CRPS-LAM (2510.09484) all use AR+CLN.
Old Block-AR had turb/calm=1.46 with AR — the mechanism works.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_160a_ar_frame.py \
        --epochs 80 --batch_size 8 --n_members 2 --noise_dim 32 \
        --output_dir models/backfill/flow_160a --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp, kurtosis

import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    normalize_iv, make_serializable
)


class ARFrameDecoder(nn.Module):
    """Per-frame MLP: condition + prev_frame + noise → delta.

    Each frame is generated autoregressively. Noise is fresh per step (iid).
    Zero-init output: initial prediction = repeat previous frame.
    """

    def __init__(self, cond_dim=128, frame_dim=25, noise_dim=32, hidden=128):
        super().__init__()
        input_dim = cond_dim + frame_dim + noise_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, frame_dim),
        )
        # Zero-init output: delta=0 → frame_t = prev_frame
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.frame_dim = frame_dim
        self.noise_dim = noise_dim

    def forward(self, cond, prev_frame, noise):
        """
        cond: (B*K, cond_dim)
        prev_frame: (B*K, frame_dim) — previous frame in [0, 1]
        noise: (B*K, noise_dim) — fresh noise for this step
        Returns: delta (B*K, frame_dim)
        """
        x = torch.cat([cond, prev_frame, noise], dim=-1)
        return self.mlp(x)


def load_pretrained_encoder(path, device):
    """Load pretrained GRU encoder."""
    from experiments.backfill.block_ar.train_cond_oneshot_flow import load_encoder
    encoder, cond_dim = load_encoder(path, device)
    return encoder, cond_dim


def afcrps_per_frame(samples, gt, alpha=0.95, spread_weight=0.5):
    """Per-frame afCRPS at D=25.

    Args:
        samples: (B, K, C) — K ensemble members for one frame
        gt: (B, C) — ground truth for one frame
    Returns:
        loss, mae, spread (scalars)
    """
    K = samples.shape[1]
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)
    mae = (samples - gt.unsqueeze(1)).abs().mean()
    spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()
    fcrps = mae - spread_weight * spread
    loss = alpha * fcrps + (1 - alpha) * mae
    return loss, mae, spread


def interval_score(samples, gt, alpha=0.9):
    """Interval score on (B, K, C) samples."""
    lo = samples.quantile(alpha / 2, dim=1)
    hi = samples.quantile(1 - alpha / 2, dim=1)
    width = hi - lo
    return (width + (2 / alpha) * (F.relu(lo - gt) + F.relu(gt - hi))).mean()


def evaluate_model(encoder, decoder, surfaces, start_indices,
                   n_samples=50, device='cuda', ret=None):
    """Full evaluation: run AR generation, compute metrics."""
    encoder.eval(); decoder.eval()
    H, T, C = 30, 30, 25
    N = len(start_indices)

    all_samples = []; all_gt = []
    with torch.no_grad():
        for idx in start_indices:
            hist = torch.from_numpy(
                surfaces[idx:idx+H][None].astype(np.float32)
            ).to(device)
            future = surfaces[idx+H:idx+H+T].reshape(T, C)
            all_gt.append(future)

            cond = encoder(normalize_iv(hist))  # (1, cond_dim)
            cond_K = cond.expand(n_samples, -1)  # (K, cond_dim)
            last_frame = hist[0, -1].reshape(1, C).expand(n_samples, -1)  # (K, C)

            # AR generation
            frames = []
            prev = last_frame
            for t in range(T):
                z = torch.randn(n_samples, decoder.noise_dim, device=device)
                delta = decoder(cond_K, prev, z)
                frame = (prev + delta).clamp(0, 1)
                frames.append(frame)
                prev = frame

            samples = torch.stack(frames, dim=1).cpu().numpy()  # (K, T, C)
            all_samples.append(samples)

    samples = np.array(all_samples)  # (N, K, T, C)
    gt = np.array(all_gt)  # (N, T, C)

    # CI worst cell
    worst_ci = 1.0
    for c in range(C):
        lo = np.percentile(samples[:, :, :, c], 5, axis=1)
        hi = np.percentile(samples[:, :, :, c], 95, axis=1)
        cov = ((gt[:, :, c] >= lo) & (gt[:, :, c] <= hi)).mean()
        worst_ci = min(worst_ci, cov)

    # KS on daily changes
    gen_ch = np.diff(samples[:, 0], axis=1).reshape(-1, C)
    gt_ch = np.diff(gt, axis=1).reshape(-1, C)
    ks = sum(1 for c2 in range(C) if ks_2samp(gen_ch[:, c2], gt_ch[:, c2])[0] < 0.15)

    # Correlation
    gc = np.corrcoef(gen_ch.T); gtc = np.corrcoef(gt_ch.T)
    corr = np.abs(gc).mean() / (np.abs(gtc).mean() + 1e-6)

    # Kurtosis
    kr = kurtosis(gen_ch.flatten()) / (kurtosis(gt_ch.flatten()) + 1e-6)

    # MAE
    mae = np.abs(samples.mean(axis=1) - gt).mean()

    # Spread-skill
    ss = samples.std(axis=1).mean() / (np.abs(samples.mean(axis=1) - gt).mean() + 1e-8)

    # Turb/calm
    turb_calm_ratio = None
    if ret is not None and len(ret) >= N:
        spreads = samples.std(axis=1).mean(axis=(1, 2))
        vol_30d = np.array([np.std(ret[max(0,i-30):i]) if i >= 30 else np.std(ret[:i+1])
                           for i in range(N)])
        q75, q25 = np.percentile(vol_30d, 75), np.percentile(vol_30d, 25)
        turb_mask = vol_30d >= q75
        calm_mask = vol_30d <= q25
        if turb_mask.sum() > 5 and calm_mask.sum() > 5:
            turb_calm_ratio = float(spreads[turb_mask].mean() / (spreads[calm_mask].mean() + 1e-8))

    return {
        "ci_worst": float(worst_ci), "ks": ks, "corr": float(corr),
        "kurt": float(kr), "ss": float(ss), "mae": float(mae),
        "turb_calm": turb_calm_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n_members", type=int, default=2,
                        help="K=2 during training (FGN: fair CRPS sufficient)")
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--spread_weight", type=float, default=0.5)
    parser.add_argument("--lambda_is", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    ret = data["ret"]
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25

    # Data splits (same as 158a)
    TEST_START = 4511
    max_train_idx = TEST_START - H - T
    VAL_SIZE = 441
    train_indices = np.arange(0, max_train_idx - VAL_SIZE)
    val_indices = np.arange(max_train_idx - VAL_SIZE, max_train_idx)
    test_indices = np.arange(TEST_START, N_total - H - T + 1)

    print(f"Train: {len(train_indices)} windows, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Load pretrained encoder
    encoder, cond_dim = load_pretrained_encoder(args.encoder_path, device)
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = True

    # AR frame decoder
    decoder = ARFrameDecoder(
        cond_dim=cond_dim, frame_dim=C, noise_dim=args.noise_dim, hidden=args.hidden
    ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"\n{'='*60}")
    print(f"160a: AR Frame Generation + E2E (RC19-H1-S1)")
    print(f"{'='*60}")
    print(f"  Encoder: {n_enc:,} params (warm start, lr={args.lr_encoder})")
    print(f"  AR Frame Decoder: {n_dec:,} params (lr={args.lr_decoder})")
    print(f"  Total: {n_enc + n_dec:,} params")
    print(f"  K={args.n_members} (training), noise_dim={args.noise_dim}")
    print(f"  AR steps: {T}, per-frame afCRPS at D={C}")

    # Preload surfaces
    surf_tensor = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices)),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_indices)),
        batch_size=args.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr_encoder, "weight_decay": 0.01},
        {"params": decoder.parameters(), "lr": args.lr_decoder, "weight_decay": 0.01},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        encoder.train(); decoder.train()
        ep_loss = 0; ep_mae = 0; ep_spread = 0; nb = 0

        for (idx_batch,) in train_loader:
            B = idx_batch.shape[0]
            K = args.n_members

            hist_list = []; future_list = []; last_frames = []
            for i in idx_batch:
                i = i.item()
                hist_list.append(surf_tensor[i:i+H].unsqueeze(0))
                future_list.append(surf_tensor[i+H:i+H+T])  # (T, 5, 5)
                last_frames.append(surf_tensor[i+H-1].reshape(C))

            hist = torch.cat(hist_list, dim=0)  # (B, H, 5, 5)
            gt_frames = torch.stack(future_list).reshape(B, T, C)  # (B, T, C)
            last_frame = torch.stack(last_frames)  # (B, C)

            # Encode
            cond = encoder(normalize_iv(hist))  # (B, cond_dim)
            cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

            # AR generation with per-frame loss
            prev = last_frame.unsqueeze(1).expand(B, K, C).reshape(B * K, C)
            total_loss = 0; total_mae = 0; total_spread = 0

            for t in range(T):
                z_t = torch.randn(B * K, args.noise_dim, device=device)
                delta = decoder(cond_K, prev, z_t)
                frame_t = (prev + delta).clamp(0, 1)  # (B*K, C)

                # Per-frame loss
                frame_BK = frame_t.reshape(B, K, C)
                gt_t = gt_frames[:, t, :]  # (B, C)
                loss_t, mae_t, spread_t = afcrps_per_frame(
                    frame_BK, gt_t, alpha=args.alpha, spread_weight=args.spread_weight
                )
                is_t = interval_score(frame_BK, gt_t)
                total_loss += loss_t + args.lambda_is * is_t
                total_mae += mae_t.item()
                total_spread += spread_t.item()

                # Detach for next step (no BPTT through AR chain)
                prev = frame_t.detach()

            # Average over frames
            total_loss = total_loss / T

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optimizer.step()

            ep_loss += total_loss.item(); ep_mae += total_mae / T
            ep_spread += total_spread / T; nb += 1

        scheduler.step()
        tl = ep_loss/nb; tm = ep_mae/nb; ts = ep_spread/nb
        elapsed = time.time() - t0

        # Validation
        encoder.eval(); decoder.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for (idx_batch,) in val_loader:
                B2 = idx_batch.shape[0]; K2 = args.n_members
                hist_list = []; future_list = []; last_frames = []
                for i in idx_batch:
                    i = i.item()
                    hist_list.append(surf_tensor[i:i+H].unsqueeze(0))
                    future_list.append(surf_tensor[i+H:i+H+T].reshape(T, C))
                    last_frames.append(surf_tensor[i+H-1].reshape(C))
                hist = torch.cat(hist_list, dim=0)
                gt = torch.stack(future_list)
                last_frame = torch.stack(last_frames)
                cond = encoder(normalize_iv(hist))
                cond_K = cond.unsqueeze(1).expand(B2, K2, -1).reshape(B2 * K2, -1)
                prev = last_frame.unsqueeze(1).expand(B2, K2, C).reshape(B2 * K2, C)
                vloss = 0
                for t in range(T):
                    z_t = torch.randn(B2 * K2, args.noise_dim, device=device)
                    delta = decoder(cond_K, prev, z_t)
                    frame_t = (prev + delta).clamp(0, 1)
                    frame_BK = frame_t.reshape(B2, K2, C)
                    crps_t, _, _ = afcrps_per_frame(frame_BK, gt[:, t])
                    vloss += crps_t.item()
                    prev = frame_t
                vl += (vloss / T) * B2; nv += B2
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
                "config": {
                    "cond_dim": cond_dim, "frame_dim": C, "noise_dim": args.noise_dim,
                    "hidden": args.hidden, "n_members": args.n_members,
                    "alpha": args.alpha, "spread_weight": args.spread_weight,
                    "lambda_is": args.lambda_is, "n_frames": T,
                    "type": "ar_frame_e2e",
                    "train_windows": len(train_indices),
                },
            }, f"{args.output_dir}/best_model.pt")

        # Eval every 20 epochs
        do_eval = (epoch % 20 == 0 or epoch == 1 or epoch == args.epochs)
        if do_eval:
            val_ret = ret[val_indices[0]:val_indices[0] + len(val_indices)]
            val_metrics = evaluate_model(
                encoder, decoder, surfaces,
                val_indices[:100], n_samples=50, device=device, ret=val_ret[:100]
            )
            test_ret = ret[test_indices[0]:test_indices[0] + len(test_indices)]
            test_metrics = evaluate_model(
                encoder, decoder, surfaces,
                test_indices[:100], n_samples=50, device=device, ret=test_ret[:100]
            )
            tc_v = f"{val_metrics['turb_calm']:.3f}" if val_metrics['turb_calm'] else "N/A"
            tc_t = f"{test_metrics['turb_calm']:.3f}" if test_metrics['turb_calm'] else "N/A"
            print(f"Ep {epoch:3d}  loss={tl:.4f}  val_loss={val_loss:.4f}  "
                  f"mae={tm:.4f}  spread={ts:.4f}  ({elapsed:.1f}s)")
            print(f"  VAL:  CI={val_metrics['ci_worst']:.3f}  KS={val_metrics['ks']}/25  "
                  f"corr={val_metrics['corr']:.3f}  mae={val_metrics['mae']:.4f}  "
                  f"turb/calm={tc_v}")
            print(f"  TEST: CI={test_metrics['ci_worst']:.3f}  KS={test_metrics['ks']}/25  "
                  f"corr={test_metrics['corr']:.3f}  mae={test_metrics['mae']:.4f}  "
                  f"turb/calm={tc_t}")
            history.append({
                "epoch": epoch, "train_loss": tl, "val_loss": val_loss,
                "mae": tm, "spread": ts,
                "val_metrics": val_metrics, "test_metrics": test_metrics,
            })
        else:
            print(f"Ep {epoch:3d}  loss={tl:.4f}  val_loss={val_loss:.4f}  "
                  f"mae={tm:.4f}  spread={ts:.4f}  ({elapsed:.1f}s)")
            history.append({"epoch": epoch, "train_loss": tl, "val_loss": val_loss,
                           "mae": tm, "spread": ts})

    # Save final
    torch.save({
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "epoch": args.epochs, "val_loss": val_loss,
        "config": {
            "cond_dim": cond_dim, "frame_dim": C, "noise_dim": args.noise_dim,
            "hidden": args.hidden, "n_members": args.n_members,
            "alpha": args.alpha, "spread_weight": args.spread_weight,
            "lambda_is": args.lambda_is, "n_frames": T,
            "type": "ar_frame_e2e",
            "train_windows": len(train_indices),
        },
    }, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
