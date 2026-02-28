"""
Checkpoint Ensemble Testing for Block-AR DDPM.

Generates samples from multiple checkpoints and combines them to test
whether ensemble diversity improves per-cell coverage.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffusion.block_ar.block_ar_ddpm import (
    ConditionalBlockARDDPM,
    BlockARConfig,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def load_model(model_path: str, device: str, **config_overrides):
    """Load a Block-AR model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    for k, v in config_overrides.items():
        config_dict[k] = v
    config = BlockARConfig(**config_dict)
    model = ConditionalBlockARDDPM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def generate_samples(model, test_loader, n_samples, max_batches, device,
                     max_global_residual=0):
    """Generate samples from a single model."""
    all_samples = []
    all_gt = []
    all_history = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))
            samples = model.sample_batched(
                history, n_samples=n_samples, max_residual=0,
                max_global_residual=max_global_residual,
            )
            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_history.append(denormalize_iv(history).cpu().numpy())

    return (
        np.concatenate(all_samples, axis=0),
        np.concatenate(all_gt, axis=0),
        np.concatenate(all_history, axis=0),
    )


def compute_per_cell_coverage(samples, gt, ci_level=0.90):
    """Compute per-cell coverage from samples and ground truth.

    Args:
        samples: (N, n_samples, T, 5, 5)
        gt: (N, T, 5, 5)
        ci_level: confidence interval level

    Returns:
        coverage_grid: (T, 5, 5) per-cell coverage
    """
    alpha = (1 - ci_level) / 2
    lower = np.quantile(samples, alpha, axis=1)       # (N, T, 5, 5)
    upper = np.quantile(samples, 1 - alpha, axis=1)   # (N, T, 5, 5)
    covered = (gt >= lower) & (gt <= upper)            # (N, T, 5, 5)
    return covered.mean(axis=0)                         # (T, 5, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Paths to model checkpoints")
    parser.add_argument("--samples_per_model", type=int, default=25,
                        help="Samples per model (total = n_models * this)")
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--max_global_residual", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/block_ar/ensemble_test")
    args = parser.parse_args()

    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_ds = VolSurfaceDataset(surfaces, start_idx=4540,
                                end_idx=len(surfaces),
                                history_len=30, future_len=30)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Ensemble of {len(args.models)} checkpoints, "
          f"{args.samples_per_model} samples each = "
          f"{len(args.models) * args.samples_per_model} total")

    # Generate samples from each model
    all_model_samples = []
    gt = None
    hist = None
    for i, model_path in enumerate(args.models):
        print(f"\n[{i+1}/{len(args.models)}] Loading {model_path}")
        model = load_model(model_path, device)
        samples, gt_i, hist_i = generate_samples(
            model, test_loader, args.samples_per_model,
            args.max_batches, device, args.max_global_residual)
        all_model_samples.append(samples)
        if gt is None:
            gt = gt_i
            hist = hist_i
        del model
        torch.cuda.empty_cache()

    # Combine samples: concatenate along sample dimension
    ensemble_samples = np.concatenate(all_model_samples, axis=1)
    N = ensemble_samples.shape[0]
    total_samples = ensemble_samples.shape[1]
    print(f"\nEnsemble: {N} windows x {total_samples} total samples")

    # Compute per-cell coverage
    horizons = [0, 6, 13, 29]  # h=1,7,14,30 (0-indexed)
    horizon_names = [1, 7, 14, 30]

    # Overall coverage
    alpha = 0.05
    lower = np.quantile(ensemble_samples, alpha, axis=1)
    upper = np.quantile(ensemble_samples, 1 - alpha, axis=1)
    covered = (gt >= lower) & (gt <= upper)
    overall_cov = covered.mean() * 100
    print(f"\nOverall 90% CI coverage: {overall_cov:.1f}%")

    # Per-horizon coverage
    for h_idx, h_name in zip(horizons, horizon_names):
        cov_h = covered[:, h_idx].mean() * 100
        print(f"  h={h_name:>2}: {cov_h:.1f}%")

    # Regime split (Q20/Q80 of vol_of_vol)
    # Compute vol_of_vol from history
    history_iv = hist  # (N, 30, 5, 5)
    daily_changes = np.diff(history_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=(1, 2, 3))  # (N,)
    q20 = np.percentile(vol_of_vol, 20)
    q80 = np.percentile(vol_of_vol, 80)
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80
    n_calm = calm_mask.sum()
    n_turb = turb_mask.sum()
    print(f"\nRegimes: {n_calm} calm (Q20), {n_turb} turb (Q80)")

    # Per-regime per-cell coverage at each horizon
    under_count = 0
    over_count = 0
    results = {}

    for regime, mask, regime_name in [
        ("calm", calm_mask, "calm"), ("turb", turb_mask, "turb")
    ]:
        results[regime] = {}
        for h_idx, h_name in zip(horizons, horizon_names):
            cov_cells = covered[mask][:, h_idx]  # (n_regime, 5, 5)
            cell_cov = cov_cells.mean(axis=0) * 100  # (5, 5)
            worst = cell_cov.min()
            best = cell_cov.max()
            worst_idx = np.unravel_index(cell_cov.argmin(), (5, 5))
            best_idx = np.unravel_index(cell_cov.argmax(), (5, 5))

            n_under = (cell_cov < 70).sum()
            n_over = (cell_cov > 95).sum()
            under_count += n_under
            over_count += n_over

            status_w = "PASS" if worst >= 70 else "FAIL"
            status_b = "PASS" if best <= 95 else "FAIL"
            print(f"  {regime_name} h={h_name:>2}: worst ({worst_idx[0]},{worst_idx[1]})="
                  f"{worst:.1f}% {status_w} | best ({best_idx[0]},{best_idx[1]})="
                  f"{best:.1f}% {status_b} | <70%: {n_under} | >95%: {n_over}")

            results[regime][str(h_name)] = {
                "grid": cell_cov.tolist(),
                "worst": float(worst),
                "best": float(best),
            }

    print(f"\n=== SUMMARY ===")
    print(f"Under 70%: {under_count}")
    print(f"Over 95%: {over_count}")
    print(f"Combined: {under_count + over_count}")
    print(f"VS bestval baseline: 27 under + 46 over = 73 combined")

    # Also show per-model coverage for comparison
    for i, model_samples in enumerate(all_model_samples):
        lower_i = np.quantile(model_samples, 0.05, axis=1)
        upper_i = np.quantile(model_samples, 0.95, axis=1)
        cov_i = ((gt >= lower_i) & (gt <= upper_i)).mean() * 100
        print(f"  Model {i}: overall={cov_i:.1f}%")

    # Save results
    results_dict = {
        "models": args.models,
        "samples_per_model": args.samples_per_model,
        "total_samples": total_samples,
        "overall_coverage": overall_cov / 100,
        "under_70": int(under_count),
        "over_95": int(over_count),
        "combined": int(under_count + over_count),
        "regime_results": results,
    }
    with open(f"{args.output_dir}/ensemble_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved to {args.output_dir}/ensemble_results.json")


if __name__ == "__main__":
    main()
