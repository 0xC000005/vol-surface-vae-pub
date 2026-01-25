#!/usr/bin/env python
"""
Comprehensive evaluation script for DDPM POC on volatility surfaces.

This script evaluates whether DDPM achieves better CI coverage than VAE (33% baseline).

Key metrics:
1. CI Coverage at multiple levels (50%, 80%, 90%, 95%)
2. Per-horizon coverage (h=1, 7, 14, 30)
3. CRPS (Continuous Ranked Probability Score)
4. Calibration curve
5. Sample diversity and explosion rate
6. ACF preservation

Usage:
    python experiments/backfill/diffusion_poc/eval_ddpm_poc.py
    python experiments/backfill/diffusion_poc/eval_ddpm_poc.py --model_path models/backfill/ddpm_poc/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config


def compute_crps(samples: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Continuous Ranked Probability Score.

    CRPS = E|X-y| - 0.5*E|X-X'|

    Lower is better. Proper scoring rule for probabilistic forecasts.

    Args:
        samples: (n_samples,) array of predictions
        gt: scalar ground truth

    Returns:
        CRPS value
    """
    n = len(samples)
    if n == 0:
        return np.nan

    # E|X-y|
    term1 = np.mean(np.abs(samples - gt))

    # E|X-X'| (can be computed efficiently using sorted samples)
    sorted_samples = np.sort(samples)
    # E|X-X'| = 2 * sum_i sum_j I(i<j) * (x_j - x_i) / n^2
    # = 2 * sum_i (2i+1-n) * x_i / n^2
    indices = np.arange(n)
    weights = 2 * indices + 1 - n
    term2 = np.sum(weights * sorted_samples) / (n * n) * 2

    return term1 - 0.5 * term2


def compute_acf(series: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return np.zeros(max_lag + 1)

    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
            acf.append(cov / var)

    return np.array(acf)


def evaluate_model(
    model: ConditionalDDPM,
    dataloader: DataLoader,
    n_samples: int = 100,
    device: str = 'cpu',
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Comprehensive evaluation of DDPM model.

    Args:
        model: Trained DDPM model
        dataloader: Test dataloader
        n_samples: Number of samples per history
        device: Device to use
        max_batches: Maximum batches to evaluate

    Returns:
        dict with all evaluation metrics
    """
    model.eval()

    # Storage for metrics
    all_coverages = {level: [] for level in [0.5, 0.8, 0.9, 0.95]}
    horizon_coverages = {h: {level: [] for level in [0.9]} for h in [1, 7, 14, 30]}
    all_crps = []
    all_diversity = []
    all_explosion_rate = []

    # For ACF comparison
    gt_series = []
    sample_series = []

    # For calibration curve
    calibration_data = {p: [] for p in np.linspace(0.1, 0.9, 9)}

    n_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=n_batches, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break

            history = batch["history"].to(device)  # (B, T_hist, 5, 5)
            future_gt = batch["future"].to(device)  # (B, T_fut, 5, 5)
            B, T_fut, H, W = future_gt.shape

            # Generate samples
            samples = model.sample(history, n_samples=n_samples)  # (B, n_samples, T_fut, 5, 5)

            # Move to CPU for numpy operations
            samples_np = samples.cpu().numpy()
            future_gt_np = future_gt.cpu().numpy()

            # 1. Overall CI Coverage
            for level in all_coverages.keys():
                alpha = (1 - level) / 2
                lower = np.quantile(samples_np, alpha, axis=1)
                upper = np.quantile(samples_np, 1 - alpha, axis=1)
                covered = (future_gt_np >= lower) & (future_gt_np <= upper)
                coverage = covered.mean()
                all_coverages[level].append(coverage)

            # 2. Per-horizon coverage (at 90% CI)
            for h in horizon_coverages.keys():
                if h <= T_fut:
                    h_idx = h - 1  # 0-indexed
                    samples_h = samples_np[:, :, h_idx, :, :]  # (B, n_samples, H, W)
                    gt_h = future_gt_np[:, h_idx, :, :]  # (B, H, W)
                    lower_h = np.quantile(samples_h, 0.05, axis=1)
                    upper_h = np.quantile(samples_h, 0.95, axis=1)
                    covered_h = (gt_h >= lower_h) & (gt_h <= upper_h)
                    horizon_coverages[h][0.9].append(covered_h.mean())

            # 3. CRPS (sample for efficiency)
            for b in range(min(B, 4)):  # Sample subset
                for t in range(0, T_fut, 10):  # Sample time steps
                    for i in range(H):
                        for j in range(W):
                            crps_val = compute_crps(
                                samples_np[b, :, t, i, j],
                                future_gt_np[b, t, i, j]
                            )
                            all_crps.append(crps_val)

            # 4. Sample diversity
            diversity = samples_np.std(axis=1).mean()
            all_diversity.append(diversity)

            # 5. Explosion rate (IV > 1.0 or < 0.01)
            explosion_high = (samples_np > 1.0).any(axis=(2, 3, 4)).mean()
            explosion_low = (samples_np < 0.01).any(axis=(2, 3, 4)).mean()
            all_explosion_rate.append(explosion_high + explosion_low)

            # 6. Calibration data
            for p in calibration_data.keys():
                alpha = (1 - p) / 2
                lower = np.quantile(samples_np, alpha, axis=1)
                upper = np.quantile(samples_np, 1 - alpha, axis=1)
                covered = (future_gt_np >= lower) & (future_gt_np <= upper)
                calibration_data[p].append(covered.mean())

            # 7. ACF data (first sample of batch)
            gt_series.extend(future_gt_np[0, :, 2, 2].tolist())  # ATM point
            sample_series.extend(samples_np[0, 0, :, 2, 2].tolist())

    # Aggregate results
    results = {
        "coverage_50": np.mean(all_coverages[0.5]),
        "coverage_80": np.mean(all_coverages[0.8]),
        "coverage_90": np.mean(all_coverages[0.9]),
        "coverage_95": np.mean(all_coverages[0.95]),
        "crps_mean": np.nanmean(all_crps),
        "crps_std": np.nanstd(all_crps),
        "sample_diversity": np.mean(all_diversity),
        "explosion_rate": np.mean(all_explosion_rate),
    }

    # Per-horizon coverage
    for h in horizon_coverages.keys():
        if horizon_coverages[h][0.9]:
            results[f"coverage_90_h{h}"] = np.mean(horizon_coverages[h][0.9])

    # Calibration curve data
    results["calibration"] = {
        "nominal": list(calibration_data.keys()),
        "empirical": [np.mean(calibration_data[p]) for p in calibration_data.keys()],
    }

    # ACF comparison
    gt_acf = compute_acf(np.array(gt_series))
    sample_acf = compute_acf(np.array(sample_series))
    acf_correlation = np.corrcoef(gt_acf, sample_acf)[0, 1]
    results["acf_correlation"] = acf_correlation
    results["gt_acf"] = gt_acf.tolist()
    results["sample_acf"] = sample_acf.tolist()

    return results


def plot_calibration(results: Dict, save_path: str):
    """Plot calibration curve."""
    plt.figure(figsize=(8, 6))

    nominal = results["calibration"]["nominal"]
    empirical = results["calibration"]["empirical"]

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(nominal, empirical, 'bo-', label='DDPM', markersize=8)

    plt.xlabel('Nominal Coverage', fontsize=12)
    plt.ylabel('Empirical Coverage', fontsize=12)
    plt.title('CI Calibration Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate calibration error
    cal_error = np.mean(np.abs(np.array(empirical) - np.array(nominal)))
    plt.text(0.05, 0.85, f'Calibration Error: {cal_error:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved calibration plot to {save_path}")


def plot_acf_comparison(results: Dict, save_path: str):
    """Plot ACF comparison between GT and samples."""
    plt.figure(figsize=(10, 5))

    lags = np.arange(len(results["gt_acf"]))
    plt.bar(lags - 0.15, results["gt_acf"], width=0.3, label='Ground Truth', alpha=0.7)
    plt.bar(lags + 0.15, results["sample_acf"], width=0.3, label='DDPM Samples', alpha=0.7)

    plt.xlabel('Lag (days)', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.title(f'ACF Comparison (Correlation: {results["acf_correlation"]:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ACF plot to {save_path}")


def plot_coverage_by_horizon(results: Dict, save_path: str):
    """Plot coverage by prediction horizon."""
    plt.figure(figsize=(8, 6))

    horizons = []
    coverages = []
    for key in results:
        if key.startswith("coverage_90_h"):
            h = int(key.split("h")[1])
            horizons.append(h)
            coverages.append(results[key])

    horizons, coverages = zip(*sorted(zip(horizons, coverages)))

    plt.bar(horizons, coverages, color='steelblue', alpha=0.7)
    plt.axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
    plt.axhline(y=0.33, color='orange', linestyle='--', label='VAE Baseline (33%)')

    plt.xlabel('Prediction Horizon (days)', fontsize=12)
    plt.ylabel('90% CI Coverage', fontsize=12)
    plt.title('Coverage vs Prediction Horizon', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved horizon coverage plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM POC")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--n_samples", type=int, default=100, help="Samples per history")
    parser.add_argument("--max_batches", type=int, default=50, help="Max batches to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    config = get_default_config()

    # Auto-detect device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Find model
    model_path = args.model_path
    if model_path is None:
        candidates = [
            f"{config.output_dir}/best_coverage_model.pt",
            f"{config.output_dir}/best_model.pt",
            f"{config.output_dir}/final_model.pt",
        ]
        for path in candidates:
            if Path(path).exists():
                model_path = path
                break

    if model_path is None or not Path(model_path).exists():
        print(f"No trained model found. Run training first:")
        print(f"  python experiments/backfill/diffusion_poc/train_ddpm_poc.py")
        return

    print("=" * 60)
    print("DDPM POC Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Samples per history: {args.n_samples}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device)
    denoiser_config = checkpoint["config"]

    model = ConditionalDDPM(
        denoiser_config,
        scheduler_config={"schedule": "cosine", "device": device},
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load test data
    print("\nLoading test data...")
    data = np.load(config.data_path)
    surfaces = data["surface"]

    test_dataset = VolSurfaceDataset(
        surfaces,
        config.history_len,
        config.future_len,
        start_idx=config.test_start,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(
        model, test_loader,
        n_samples=args.n_samples,
        device=device,
        max_batches=args.max_batches,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- CI Coverage ---")
    print(f"50% CI Coverage: {results['coverage_50']:.1%} (target: 50%)")
    print(f"80% CI Coverage: {results['coverage_80']:.1%} (target: 80%)")
    print(f"90% CI Coverage: {results['coverage_90']:.1%} (target: 90%)")
    print(f"95% CI Coverage: {results['coverage_95']:.1%} (target: 95%)")

    print("\n--- Per-Horizon Coverage (90% CI) ---")
    for key in sorted(results.keys()):
        if key.startswith("coverage_90_h"):
            h = key.split("h")[1]
            print(f"  Horizon {h}: {results[key]:.1%}")

    print("\n--- Quality Metrics ---")
    print(f"CRPS: {results['crps_mean']:.6f} +/- {results['crps_std']:.6f}")
    print(f"Sample Diversity (std): {results['sample_diversity']:.4f}")
    print(f"Explosion Rate: {results['explosion_rate']:.1%}")
    print(f"ACF Correlation: {results['acf_correlation']:.3f}")

    print("\n--- Comparison vs Baselines ---")
    vae_coverage = 0.33
    improvement = results['coverage_90'] - vae_coverage
    print(f"VAE Baseline (90% CI): {vae_coverage:.1%}")
    print(f"DDPM POC (90% CI):     {results['coverage_90']:.1%}")
    print(f"Improvement:           {improvement:+.1%}")

    # Create output directory
    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save plots
    plot_calibration(results, str(output_dir / "calibration_curve.png"))
    plot_acf_comparison(results, str(output_dir / "acf_comparison.png"))
    plot_coverage_by_horizon(results, str(output_dir / "coverage_by_horizon.png"))

    # Save results
    results_path = output_dir / "evaluation_results.npz"
    np.savez(results_path, **{k: v for k, v in results.items() if not isinstance(v, dict)})
    print(f"\nResults saved to: {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if results['coverage_90'] > 0.5:
        print("SUCCESS: DDPM achieves >50% coverage (hopeful threshold)")
    elif results['coverage_90'] > vae_coverage:
        print("PARTIAL: DDPM beats VAE baseline but coverage still low")
    else:
        print("FAILURE: DDPM does not improve over VAE baseline")

    if results['sample_diversity'] > 0.01:
        print("Sample diversity is good (>0.01)")
    else:
        print("WARNING: Low sample diversity - may have diversity collapse")

    if results['acf_correlation'] > 0.5:
        print(f"ACF correlation is acceptable ({results['acf_correlation']:.2f})")
    else:
        print(f"WARNING: ACF correlation is low ({results['acf_correlation']:.2f})")


if __name__ == "__main__":
    main()
