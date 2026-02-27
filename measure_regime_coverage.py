"""Measure calm vs turbulent regime CI coverage for a given model checkpoint.

Usage:
    PYTHONPATH=. python measure_regime_coverage.py <model_path> [--n_windows N] [--n_samples S]
"""
import argparse
import sys
import numpy as np
import torch
from pathlib import Path

from diffusion.block_ar.block_ar_ddpm import ConditionalBlockARDDPM, BlockARConfig, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--n_windows", type=int, default=400)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Load model
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    import dataclasses
    cfg = checkpoint["config"]
    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
    config = BlockARConfig(**{k: v for k, v in cfg.items() if k in BlockARConfig.__dataclass_fields__})
    model = ConditionalBlockARDDPM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {args.model_path} (epoch {checkpoint.get('epoch', '?')})")
    print(f"  aux_regime_features={config.aux_regime_features}, ratio_target_mode={config.ratio_target_mode}")

    # Load data (test set)
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    history_len = config.history_len
    test_ds = VolSurfaceDataset(surfaces, start_idx=4540, end_idx=len(surfaces),
                                history_len=history_len, future_len=30)
    print(f"  history_len={history_len}")
    print(f"Test set: {len(test_ds)} windows")

    n_windows = min(args.n_windows, len(test_ds))
    indices = np.linspace(0, len(test_ds) - 1, n_windows, dtype=int)

    # Compute vol_of_vol for all test windows to classify calm/turb
    all_vov = []
    for i in range(len(test_ds)):
        batch = test_ds[i]
        hist = denormalize_iv(batch["history"]).numpy()
        mean_iv = hist.mean(axis=(-2, -1))
        daily_diff = np.diff(mean_iv)
        all_vov.append(daily_diff.std())
    all_vov = np.array(all_vov)

    # Quintile thresholds (Q1=calm, Q5=turb)
    q20 = np.percentile(all_vov, 20)
    q80 = np.percentile(all_vov, 80)
    print(f"Vol-of-vol quintile thresholds: Q20={q20:.5f}, Q80={q80:.5f}")

    # Generate samples and measure coverage
    horizons = [1, 7, 14, 30]
    # Store per-window results
    results = []

    for wi, idx in enumerate(indices):
        batch = test_ds[idx]
        history = batch["history"].unsqueeze(0).to(device)
        future_gt = denormalize_iv(batch["future"].to(device)).unsqueeze(0)  # (1, 30, 5, 5)

        with torch.no_grad():
            samples = model.sample_batched(
                history, n_samples=args.n_samples
            )  # (1, S, 30, 5, 5) — already denormalized

        samples = samples.squeeze(0)  # (S, 30, 5, 5)
        gt = future_gt.squeeze(0)  # (30, 5, 5)

        vov = all_vov[idx]
        regime = "calm" if vov <= q20 else ("turb" if vov >= q80 else "mid")

        # Per-horizon coverage
        for h in horizons:
            t = h - 1
            gt_slice = gt[t]  # (5, 5)
            sample_slice = samples[:, t, :, :]  # (S, 5, 5)
            lo = torch.quantile(sample_slice, 0.05, dim=0)
            hi = torch.quantile(sample_slice, 0.95, dim=0)
            covered = ((gt_slice >= lo) & (gt_slice <= hi)).float().mean().item()
            width = (hi - lo).mean().item()
            results.append({
                "idx": int(idx), "regime": regime, "horizon": h,
                "coverage": covered, "width": width, "vov": float(vov),
            })

        if (wi + 1) % 50 == 0:
            print(f"  {wi+1}/{n_windows} windows processed")

    # Aggregate
    import pandas as pd
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("REGIME COVERAGE RESULTS")
    print("=" * 70)

    for regime in ["calm", "turb", "mid"]:
        sub = df[df["regime"] == regime]
        if len(sub) == 0:
            continue
        n_wins = sub["idx"].nunique()
        print(f"\n{regime.upper()} regime ({n_wins} windows):")
        for h in horizons:
            hsub = sub[sub["horizon"] == h]
            cov = hsub["coverage"].mean()
            width = hsub["width"].mean()
            print(f"  h={h:2d}: coverage={cov:.1%}, width={width:.4f}")
        overall_cov = sub["coverage"].mean()
        overall_width = sub["width"].mean()
        print(f"  Overall: coverage={overall_cov:.1%}, width={overall_width:.4f}")

    # Width ratio turb/calm
    calm_width = df[df["regime"] == "calm"]["width"].mean()
    turb_width = df[df["regime"] == "turb"]["width"].mean()
    if calm_width > 0:
        print(f"\nWidth ratio turb/calm: {turb_width/calm_width:.3f}x")

    # Overall
    print(f"\nAll windows ({df['idx'].nunique()}):")
    for h in horizons:
        hsub = df[df["horizon"] == h]
        print(f"  h={h:2d}: coverage={hsub['coverage'].mean():.1%}")

    print(f"\nModel: {args.model_path}")


if __name__ == "__main__":
    main()
