#!/usr/bin/env python
"""Diagnose why factor loadings stayed small in Exp 91c.

Three analyses:
1. Gradient magnitude: factor_loadings vs frame_decoder output layer
2. Noise-delta correlation: does MLP use cell_noise at all?
3. Init scale comparison: cell_noise std vs 90d shared noise std

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_factor_noise.py \
        --model_91c models/backfill/afcrps_91c/best_model.pt \
        --model_90d models/backfill/afcrps_90d/best_model.pt \
        --output_dir results/block_ar/factor_noise_diagnosis --device cuda
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
    normalize_iv,
    afcrps_loss,
    variogram_score,
    interval_score,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from torch.utils.data import DataLoader


def load_model(model_path, device, no_ema=True):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = SinglePassConfig(**cfg)
    model = SinglePassBlockAR(cfg)
    key = "ema_state_dict" if not no_ema and "ema_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.to(device)
    return model, ckpt


def diag1_gradient_magnitudes(model, train_loader, device, n_batches=20):
    """Compare gradient norms on factor_loadings vs frame_decoder output layer."""
    print("\n" + "=" * 70)
    print("  DIAG 1: GRADIENT MAGNITUDES")
    print("=" * 70)

    model.train()
    fl_grad_norms = []
    fd_grad_norms = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        history = batch["history"].to(device)
        future = batch["future"].to(device)

        model.zero_grad()
        result = model(
            history, future, n_members=4, lambda_vs=0.1, lambda_is=0.5, n_frames=30,
        )
        result["loss"].backward()

        if hasattr(model, "factor_loadings") and model.factor_loadings.grad is not None:
            fl_grad_norms.append(model.factor_loadings.grad.norm().item())

        fd_last = model.frame_decoder.mlp[-1]
        if fd_last.weight.grad is not None:
            fd_grad_norms.append(fd_last.weight.grad.norm().item())

    model.eval()

    fl_mean = np.mean(fl_grad_norms) if fl_grad_norms else 0
    fd_mean = np.mean(fd_grad_norms) if fd_grad_norms else 0
    ratio = fd_mean / (fl_mean + 1e-12)

    print(f"  factor_loadings grad norm: {fl_mean:.6f} (n={len(fl_grad_norms)})")
    print(f"  frame_decoder.mlp[-1] grad norm: {fd_mean:.6f} (n={len(fd_grad_norms)})")
    print(f"  Ratio (fd/fl): {ratio:.1f}x")

    if fl_mean > 0 and ratio > 10:
        print(f"  → Factor loadings gradient is {ratio:.0f}x SMALLER than decoder output")
        print(f"    CRPS signal is weak for loadings")
    elif fl_mean > 0:
        print(f"  → Gradient magnitudes comparable — init/scale may be the issue")

    return {
        "fl_grad_norm_mean": fl_mean,
        "fd_grad_norm_mean": fd_mean,
        "ratio_fd_over_fl": ratio,
        "fl_grad_norms": fl_grad_norms,
        "fd_grad_norms": fd_grad_norms,
    }


@torch.no_grad()
def diag2_noise_delta_correlation(model_91c, model_90d, test_loader, device, n_batches=10):
    """Measure how much the MLP uses noise input vs ignoring it."""
    print("\n" + "=" * 70)
    print("  DIAG 2: NOISE → DELTA CORRELATION")
    print("=" * 70)

    results = {}
    for name, model in [("91c", model_91c), ("90d", model_90d)]:
        model.eval()
        cfg = model.config
        H, W = cfg.surface_h, cfg.surface_w
        rho = cfg.ar_frame_rho
        is_factor = cfg.ar_factor_noise

        all_noise_delta_corrs = []  # per-cell correlation between noise and delta

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= n_batches:
                break
            history = batch["history"].to(device)
            B = history.shape[0]

            _, vol_scale = model._compute_vol_scale(history)
            gru_outputs, h_last = model._init_gru_state(history)
            condition = model.encoder(history, mask=None)
            prev_frame = denormalize_iv(history[:, -1])

            # Collect noise and delta over 30 frames, 10 samples
            for s in range(10):
                z = model._sample_noise(B, device)
                z_t = z
                pf = prev_frame.clone()
                cond = condition.clone()
                g_out = gru_outputs.clone()
                h_l = h_last.clone()

                frame_noises = []
                frame_deltas = []

                for t in range(30):
                    if t > 0:
                        eps_t = torch.randn_like(z_t)
                        z_t = rho * z_t + math.sqrt(1 - rho**2) * eps_t

                    local_pos, horizon_bucket = model._get_ar_frame_positions(
                        step_idx=t, batch_size=B, device=device, position_mode="native",
                    )
                    prev_flat = pf.reshape(B, H * W)
                    noise_input = model._get_noise_for_decoder(z_t)
                    delta = model.frame_decoder(
                        prev_flat, cond, noise_input, local_pos, horizon_bucket
                    )  # (B, 25)

                    frame_noises.append(noise_input.cpu().numpy())  # (B, noise_dim or 25)
                    frame_deltas.append(delta.cpu().numpy())  # (B, 25)

                    vs = model._get_ar_frame_vol_scale(cond, vol_scale, None)
                    delta_hw = delta.reshape(B, H, W)
                    iv_t = (pf + vs * delta_hw).clamp(0.001, 1.0)
                    pf = iv_t
                    cond, g_out, h_l = model._gru_step(iv_t, g_out, h_l)

                # Stack: (30, B, dim)
                noises = np.stack(frame_noises, axis=0)  # (30, B, noise_dim)
                deltas = np.stack(frame_deltas, axis=0)  # (30, B, 25)

                # Per-cell correlation between noise and delta
                # For factor model: noise is 25-dim (per-cell), compute per-cell corr
                # For shared: noise is 16-dim, compute corr(noise_mean, delta_per_cell)
                for b in range(B):
                    if is_factor:
                        # noise (30, 25), delta (30, 25) → per-cell correlation
                        for c in range(25):
                            corr = np.corrcoef(noises[:, b, c], deltas[:, b, c])[0, 1]
                            if not np.isnan(corr):
                                all_noise_delta_corrs.append(abs(corr))
                    else:
                        # noise (30, 16), delta (30, 25) → correlate noise mean with each cell
                        noise_mean = noises[:, b].mean(axis=1)  # (30,)
                        for c in range(25):
                            corr = np.corrcoef(noise_mean, deltas[:, b, c])[0, 1]
                            if not np.isnan(corr):
                                all_noise_delta_corrs.append(abs(corr))

        mean_corr = np.mean(all_noise_delta_corrs)
        med_corr = np.median(all_noise_delta_corrs)
        results[name] = {
            "mean_abs_corr": float(mean_corr),
            "median_abs_corr": float(med_corr),
            "p25_abs_corr": float(np.percentile(all_noise_delta_corrs, 25)),
            "p75_abs_corr": float(np.percentile(all_noise_delta_corrs, 75)),
            "n_samples": len(all_noise_delta_corrs),
        }
        print(f"  {name}: mean |corr(noise, delta)| = {mean_corr:.4f}, "
              f"median = {med_corr:.4f}, "
              f"[P25={np.percentile(all_noise_delta_corrs, 25):.4f}, "
              f"P75={np.percentile(all_noise_delta_corrs, 75):.4f}]")

    if "91c" in results and "90d" in results:
        r = results["91c"]["mean_abs_corr"] / max(results["90d"]["mean_abs_corr"], 1e-8)
        print(f"\n  91c/90d correlation ratio: {r:.2f}x")
        if r < 0.5:
            print("  → 91c MLP uses noise LESS than 90d — factor noise is being ignored")
        else:
            print("  → Similar noise usage — issue is in correlation structure, not usage")

    return results


@torch.no_grad()
def diag3_init_scale_analysis(model_91c, model_90d, device):
    """Compare effective noise scale entering MLP."""
    print("\n" + "=" * 70)
    print("  DIAG 3: NOISE SCALE AT MLP INPUT")
    print("=" * 70)

    results = {}
    for name, model in [("91c", model_91c), ("90d", model_90d)]:
        cfg = model.config
        B = 256

        # Sample noise and compute what enters the MLP
        z = model._sample_noise(B, device)
        noise_input = model._get_noise_for_decoder(z)

        std_per_dim = noise_input.std(dim=0).cpu().numpy()
        mean_std = std_per_dim.mean()
        min_std = std_per_dim.min()
        max_std = std_per_dim.max()

        results[name] = {
            "noise_input_dim": noise_input.shape[1],
            "mean_std": float(mean_std),
            "min_std": float(min_std),
            "max_std": float(max_std),
            "noise_input_norm": float(noise_input.norm(dim=1).mean()),
        }

        print(f"  {name}:")
        print(f"    Noise input dim: {noise_input.shape[1]}")
        print(f"    Per-dim std: mean={mean_std:.4f}, min={min_std:.4f}, max={max_std:.4f}")
        print(f"    L2 norm: {noise_input.norm(dim=1).mean():.4f}")

        if hasattr(model, "factor_loadings"):
            fl = model.factor_loadings.data.cpu().numpy()
            print(f"    Factor loadings: shape={fl.shape}, "
                  f"norm/cell: min={np.linalg.norm(fl, axis=1).min():.4f}, "
                  f"max={np.linalg.norm(fl, axis=1).max():.4f}, "
                  f"mean={np.linalg.norm(fl, axis=1).mean():.4f}")

    if "91c" in results and "90d" in results:
        ratio = results["91c"]["mean_std"] / max(results["90d"]["mean_std"], 1e-8)
        print(f"\n  Scale ratio (91c/90d): {ratio:.3f}x")
        if ratio < 0.3:
            print(f"  → 91c noise is {1/ratio:.0f}x SMALLER than 90d at MLP input")
            print(f"    MLP likely learned to ignore this tiny signal")

    return results


def make_plots(diag1, diag2, diag3, output_dir):
    """Summary visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Gradient norms over batches
    ax = axes[0]
    if diag1["fl_grad_norms"]:
        ax.plot(diag1["fl_grad_norms"], label="factor_loadings", alpha=0.7)
    if diag1["fd_grad_norms"]:
        ax.plot(diag1["fd_grad_norms"], label="frame_decoder[-1]", alpha=0.7)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Grad Norm")
    ax.set_title("Gradient Magnitudes")
    ax.legend()
    ax.set_yscale("log")

    # Plot 2: Noise-delta correlation comparison
    ax = axes[1]
    names = list(diag2.keys())
    means = [diag2[n]["mean_abs_corr"] for n in names]
    p25s = [diag2[n]["p25_abs_corr"] for n in names]
    p75s = [diag2[n]["p75_abs_corr"] for n in names]
    x = range(len(names))
    ax.bar(x, means, yerr=[[m - p for m, p in zip(means, p25s)],
                            [p - m for m, p in zip(means, p75s)]],
           capsize=5, color=["steelblue", "coral"][:len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("|corr(noise, delta)|")
    ax.set_title("Noise → Delta Correlation")

    # Plot 3: Noise scale comparison
    ax = axes[2]
    names = list(diag3.keys())
    stds = [diag3[n]["mean_std"] for n in names]
    ax.bar(range(len(names)), stds, color=["steelblue", "coral"][:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Per-dim std at MLP input")
    ax.set_title("Noise Scale at MLP Input")

    plt.tight_layout()
    path = Path(output_dir) / "factor_noise_diagnosis.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved plot: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_91c", type=str, required=True)
    parser.add_argument("--model_90d", type=str, required=True)
    parser.add_argument("--no_ema", action="store_true", default=True)
    parser.add_argument("--n_grad_batches", type=int, default=20)
    parser.add_argument("--n_corr_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/block_ar/factor_noise_diagnosis")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FACTOR NOISE DIAGNOSIS")
    print("=" * 70)

    # Load models
    model_91c, _ = load_model(args.model_91c, device, no_ema=args.no_ema)
    model_90d, _ = load_model(args.model_90d, device, no_ema=args.no_ema)
    print(f"  91c: factor_noise={model_91c.config.ar_factor_noise}, "
          f"n_factors={model_91c.config.ar_n_factors}")
    print(f"  90d: factor_noise={model_90d.config.ar_factor_noise}")

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    train_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=0, end_idx=4040)
    test_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540, end_idx=5822)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Run diagnostics
    diag1 = diag1_gradient_magnitudes(model_91c, train_loader, device, n_batches=args.n_grad_batches)
    diag2 = diag2_noise_delta_correlation(model_91c, model_90d, test_loader, device, n_batches=args.n_corr_batches)
    diag3 = diag3_init_scale_analysis(model_91c, model_90d, device)

    # Save results
    summary = {"diag1_gradients": {k: v for k, v in diag1.items() if k not in ("fl_grad_norms", "fd_grad_norms")},
               "diag2_correlation": diag2,
               "diag3_scale": diag3}
    with open(Path(args.output_dir) / "diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_plots(diag1, diag2, diag3, args.output_dir)

    print(f"\n{'=' * 70}")
    print(f"All results saved to {args.output_dir}/")
    print(f"{'=' * 70}")
