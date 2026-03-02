"""
Exp D4: Per-Cell Z-Score Distribution Analysis.

Analyzes the generated samples per cell to understand:
- Per-cell sample spread vs GT spread
- Which cells are over/under-dispersed
- Calm vs turb spread ratios per cell
- Directly measures why some cells are over/under-covered

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_percell_zscore.py \
        --model_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --n_windows 500 --n_samples 50 --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig, ConditionalBlockARDDPM, denormalize_iv,
)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)
    model_config.device = device
    model = ConditionalBlockARDDPM(model_config).to(device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, model_config


def load_test_data(config):
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
    test_start = getattr(config, 'test_start', 4540)
    dataset = VolSurfaceDataset(surfaces, config.history_len, config.future_len,
                                start_idx=test_start)
    return dataset


def compute_vol_of_vol(history_norm):
    past_abs = denormalize_iv(history_norm)
    mean_iv = past_abs.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    return daily_chg.std(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--n_windows", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/block_ar/percell_zscore_diagnostic")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    model, config = load_model(args.model_path, device)
    dataset = load_test_data(config)

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    H, W = config.surface_h, config.surface_w
    eval_horizons = [0, 6, 13, 29]  # h=1, 7, 14, 30 (0-indexed)

    # Storage
    all_sample_spread = {h: [] for h in eval_horizons}  # std of samples per cell
    all_gt_deviation = {h: [] for h in eval_horizons}   # |GT - baseline| per cell
    all_sample_mean_bias = {h: [] for h in eval_horizons}  # median(samples) - GT per cell
    all_coverage = {h: [] for h in eval_horizons}  # per-window per-cell coverage
    all_vov = []

    n_collected = 0
    print(f"Generating {args.n_samples} samples for {args.n_windows} windows...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if n_collected >= args.n_windows:
                break

            history = batch["history"].to(device)
            future = batch["future"].to(device)
            B = history.shape[0]

            vov = compute_vol_of_vol(history)
            all_vov.append(vov.cpu().numpy())

            # Generate samples
            samples = model.sample(
                history, n_samples=args.n_samples,
            )  # (B, n_samples, T_future, H, W)

            # Denormalize to IV space
            # model.sample() already returns denormalized [0,1] values
            gt_abs = denormalize_iv(future)  # (B, T, H, W)
            samples_abs = samples  # Already denormalized by model.sample()

            # Baseline (history[-1]) — history is still in normalized [-1,1] space
            baseline = denormalize_iv(history[:, -1:])  # (B, 1, H, W)

            for h_idx in eval_horizons:
                gt_h = gt_abs[:, h_idx]  # (B, H, W)
                samples_h = samples_abs[:, :, h_idx]  # (B, n_samples, H, W)

                # Per-cell sample spread (std across samples)
                spread = samples_h.std(dim=1)  # (B, H, W)
                all_sample_spread[h_idx].append(spread.cpu().numpy())

                # GT deviation from baseline
                dev = (gt_h - baseline.squeeze(1)).abs()  # (B, H, W)
                all_gt_deviation[h_idx].append(dev.cpu().numpy())

                # Sample median bias
                median_sample = samples_h.median(dim=1).values  # (B, H, W)
                bias = median_sample - gt_h  # (B, H, W)
                all_sample_mean_bias[h_idx].append(bias.cpu().numpy())

                # Per-window per-cell coverage at 90% CI
                lower = samples_h.quantile(0.05, dim=1)  # (B, H, W)
                upper = samples_h.quantile(0.95, dim=1)  # (B, H, W)
                covered = ((gt_h >= lower) & (gt_h <= upper)).float()  # (B, H, W)
                all_coverage[h_idx].append(covered.cpu().numpy())

            n_collected += B
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {n_collected}/{args.n_windows} windows")

    # Concatenate
    vov = np.concatenate(all_vov)[:args.n_windows]
    n = len(vov)

    vov_q20 = np.percentile(vov, 20)
    vov_q80 = np.percentile(vov, 80)
    calm_mask = vov < vov_q20
    turb_mask = vov > vov_q80

    print(f"\nAnalyzing {n} windows (calm={calm_mask.sum()}, turb={turb_mask.sum()})")
    print("=" * 70)
    print("PER-CELL Z-SCORE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # GT per-cell daily change std for reference
    gmcv = np.array([
        [0.1560, 0.0591, 0.0208, 0.0288, 0.1044],
        [0.0547, 0.0281, 0.0137, 0.0079, 0.0616],
        [0.0230, 0.0147, 0.0085, 0.0056, 0.0265],
        [0.0079, 0.0064, 0.0049, 0.0044, 0.0047],
        [0.0058, 0.0048, 0.0046, 0.0046, 0.0052],
    ])

    results = {"n_windows": n, "horizons": {}}

    for h_idx in eval_horizons:
        h_label = h_idx + 1
        spread = np.concatenate(all_sample_spread[h_idx])[:n]  # (n, H, W)
        coverage = np.concatenate(all_coverage[h_idx])[:n]  # (n, H, W)
        bias = np.concatenate(all_sample_mean_bias[h_idx])[:n]  # (n, H, W)

        print(f"\n--- Horizon h={h_label} ---")

        # 1. Per-cell sample spread (mean across windows)
        mean_spread = spread.mean(axis=0)  # (H, W)
        spread_ratio = mean_spread / mean_spread.mean()
        print(f"\nPer-cell sample spread (IV points × 100):")
        for r in range(H):
            row = "  ".join(f"{mean_spread[r, c]*100:6.2f}" for c in range(W))
            print(f"  {row}")

        print(f"\nPer-cell spread / mean(spread) ratio:")
        for r in range(H):
            row = "  ".join(f"{spread_ratio[r, c]:5.3f}" for c in range(W))
            print(f"  {row}")

        # GT spread comparison (per-cell std × sqrt(h) as rough scaling)
        gt_expected = gmcv * np.sqrt(h_label)
        gt_ratio = gt_expected / gt_expected.mean()
        model_vs_gt = spread_ratio / gt_ratio
        print(f"\nModel spread ratio / GT spread ratio (1.0 = perfect recovery):")
        for r in range(H):
            row = "  ".join(f"{model_vs_gt[r, c]:5.3f}" for c in range(W))
            print(f"  {row}")

        # 2. Per-cell coverage
        cell_coverage = coverage.mean(axis=0)  # (H, W)
        print(f"\nPer-cell 90% CI coverage:")
        for r in range(H):
            row = "  ".join(f"{cell_coverage[r, c]*100:5.1f}" for c in range(W))
            print(f"  {row}")

        # 3. Calm vs Turb spread
        calm_spread = spread[calm_mask].mean(axis=0) if calm_mask.sum() > 0 else mean_spread
        turb_spread = spread[turb_mask].mean(axis=0) if turb_mask.sum() > 0 else mean_spread
        tc_ratio = turb_spread / calm_spread.clip(min=1e-6)
        print(f"\nTurb/Calm spread ratio per cell:")
        for r in range(H):
            row = "  ".join(f"{tc_ratio[r, c]:5.3f}" for c in range(W))
            print(f"  {row}")
        print(f"  Mean turb/calm: {tc_ratio.mean():.3f}")

        # 4. Calm vs Turb coverage
        calm_cov = coverage[calm_mask].mean(axis=0) if calm_mask.sum() > 0 else cell_coverage
        turb_cov = coverage[turb_mask].mean(axis=0) if turb_mask.sum() > 0 else cell_coverage
        print(f"\nCalm per-cell coverage:")
        for r in range(H):
            row = "  ".join(f"{calm_cov[r, c]*100:5.1f}" for c in range(W))
            print(f"  {row}")
        calm_under = (calm_cov < 0.70).sum()
        calm_over = (calm_cov > 0.95).sum()
        print(f"  Calm: {calm_under} under (<70%), {calm_over} over (>95%)")

        print(f"\nTurb per-cell coverage:")
        for r in range(H):
            row = "  ".join(f"{turb_cov[r, c]*100:5.1f}" for c in range(W))
            print(f"  {row}")
        turb_under = (turb_cov < 0.70).sum()
        turb_over = (turb_cov > 0.95).sum()
        print(f"  Turb: {turb_under} under (<70%), {turb_over} over (>95%)")

        # 5. Median bias per cell
        mean_bias = bias.mean(axis=0)  # (H, W)
        print(f"\nMedian bias (model - GT, IV points × 100):")
        for r in range(H):
            row = "  ".join(f"{mean_bias[r, c]*100:+6.2f}" for c in range(W))
            print(f"  {row}")

        # Correlation: spread ratio vs coverage
        from scipy.stats import spearmanr
        rho, p = spearmanr(spread_ratio.flatten(), cell_coverage.flatten())
        print(f"\n  Spearman(spread_ratio, coverage): {rho:.3f} (p={p:.3e})")

        # Store
        results["horizons"][str(h_label)] = {
            "mean_spread": mean_spread.tolist(),
            "spread_ratio": spread_ratio.tolist(),
            "model_vs_gt_ratio": model_vs_gt.tolist(),
            "cell_coverage": cell_coverage.tolist(),
            "turb_calm_spread_ratio": tc_ratio.tolist(),
            "calm_coverage": calm_cov.tolist(),
            "turb_coverage": turb_cov.tolist(),
            "median_bias": mean_bias.tolist(),
            "spearman_spread_vs_coverage": {"rho": float(rho), "p": float(p)},
            "calm_under_70": int(calm_under),
            "calm_over_95": int(calm_over),
            "turb_under_70": int(turb_under),
            "turb_over_95": int(turb_over),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_calm_under = sum(results["horizons"][str(h+1)]["calm_under_70"]
                          for h in eval_horizons)
    total_calm_over = sum(results["horizons"][str(h+1)]["calm_over_95"]
                         for h in eval_horizons)
    total_turb_under = sum(results["horizons"][str(h+1)]["turb_under_70"]
                          for h in eval_horizons)
    total_turb_over = sum(results["horizons"][str(h+1)]["turb_over_95"]
                         for h in eval_horizons)

    print(f"Calm L2 failures: {total_calm_under} under + {total_calm_over} over = {total_calm_under + total_calm_over}")
    print(f"Turb L2 failures: {total_turb_under} under + {total_turb_over} over = {total_turb_under + total_turb_over}")
    print(f"Total: {total_calm_under + total_turb_under} under + {total_calm_over + total_turb_over} over")

    results["summary"] = {
        "calm_l2_under": total_calm_under,
        "calm_l2_over": total_calm_over,
        "turb_l2_under": total_turb_under,
        "turb_l2_over": total_turb_over,
    }

    out_path = f"{args.output_dir}/percell_zscore_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
