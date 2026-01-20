#!/usr/bin/env python
"""
Comprehensive Evaluation for Causal 3D VAE.

This script evaluates the Causal 3D VAE model with all metrics:
1. CI Coverage per grid (5x5 heatmap)
2. Explosion rate per grid
3. Distribution shape (kurtosis, skewness) per grid
4. Spatial correlation matrix (25x25)
5. Volatility smile preservation
6. Term structure preservation
7. ACF/temporal dynamics

Usage:
    python experiments/backfill/two_stage_vae/eval_causal_3d_comprehensive.py
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.causal_3d_vae import create_vol_surface_vae, create_vol_surface_vae_small, AutoencoderCausal3D
from config.causal_3d_config import Causal3DEvalConfig


@dataclass
class ComprehensiveEvalConfig:
    """Configuration for comprehensive evaluation."""
    model_path: str = "models/backfill/causal_3d/best_model.pt"
    data_path: str = "data/vol_surface_with_ret.npz"
    output_dir: str = "results/causal_3d/comprehensive"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluation parameters
    num_sequences: int = 50  # Number of test sequences to evaluate
    num_samples: int = 100   # Samples per sequence for CI estimation
    context_length: int = 60
    prediction_horizon: int = 60

    # Thresholds
    explosion_threshold_high: float = 2.0  # IV > 200%
    explosion_threshold_low: float = 0.01  # IV < 1%


def load_model(config: ComprehensiveEvalConfig) -> AutoencoderCausal3D:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)
    model_config = checkpoint.get("config", {})

    # Use small model if layers_per_block=1
    layers_per_block = model_config.get("layers_per_block", 2)
    if layers_per_block == 1:
        model = create_vol_surface_vae_small(
            latent_channels=model_config.get("latent_channels", 4),
            temporal_compression=model_config.get("temporal_compression", 2),
            block_channels=model_config.get("block_channels", (8, 16, 32)),
        )
    else:
        model = create_vol_surface_vae(
            latent_channels=model_config.get("latent_channels", 16),
            temporal_compression=model_config.get("temporal_compression", 2),
            block_channels=model_config.get("block_channels", (32, 64, 128)),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)
    model.eval()
    return model


def generate_ar_samples(
    model: AutoencoderCausal3D,
    context: torch.Tensor,
    num_steps: int,
    num_samples: int,
) -> np.ndarray:
    """Generate multiple AR samples from context."""
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            generated = model.generate_autoregressive(context, num_steps=num_steps)
            # Extract predicted future (after context)
            predicted = generated[:, :, context.shape[2]:].cpu().numpy()
            samples.append(predicted.squeeze())
    return np.array(samples)  # (num_samples, horizon, 5, 5)


def compute_per_grid_coverage(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    ci_level: float = 0.9,
) -> np.ndarray:
    """
    Compute CI coverage for each grid point.

    Args:
        samples: (num_samples, horizon, 5, 5)
        ground_truth: (horizon, 5, 5)
        ci_level: Confidence level (0.9 = 90% CI)

    Returns:
        coverage: (5, 5) coverage rate for each grid point
    """
    alpha = (1 - ci_level) / 2
    coverage = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            # Samples at this grid point: (num_samples, horizon)
            grid_samples = samples[:, :, i, j]
            gt_grid = ground_truth[:, i, j]

            # Compute coverage for each horizon, then average
            horizon_coverage = []
            for h in range(ground_truth.shape[0]):
                lower = np.quantile(grid_samples[:, h], alpha)
                upper = np.quantile(grid_samples[:, h], 1 - alpha)
                covered = (gt_grid[h] >= lower) & (gt_grid[h] <= upper)
                horizon_coverage.append(float(covered))

            coverage[i, j] = np.mean(horizon_coverage)

    return coverage


def compute_per_grid_explosion_rate(
    samples: np.ndarray,
    high_threshold: float = 2.0,
    low_threshold: float = 0.01,
) -> np.ndarray:
    """
    Compute explosion rate for each grid point.

    Args:
        samples: (num_samples, horizon, 5, 5)
        high_threshold: Upper bound for valid IV
        low_threshold: Lower bound for valid IV

    Returns:
        explosion_rate: (5, 5) explosion rate for each grid point
    """
    explosion_rate = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            grid_samples = samples[:, :, i, j]  # (num_samples, horizon)
            exploded = (grid_samples > high_threshold) | (grid_samples < low_threshold)
            # Count samples where ANY horizon exploded
            sample_exploded = exploded.any(axis=1)
            explosion_rate[i, j] = sample_exploded.mean()

    return explosion_rate


def compute_per_grid_kurtosis(
    samples: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple:
    """
    Compute kurtosis for each grid point.

    Returns:
        sample_kurtosis: (5, 5) kurtosis of samples
        gt_kurtosis: (5, 5) kurtosis of ground truth
        kurtosis_ratio: (5, 5) ratio of sample to GT kurtosis
    """
    sample_kurtosis = np.zeros((5, 5))
    gt_kurtosis = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            # Flatten samples across all horizons
            grid_samples = samples[:, :, i, j].flatten()
            gt_grid = ground_truth[:, i, j]

            sample_kurtosis[i, j] = stats.kurtosis(grid_samples, fisher=True)
            gt_kurtosis[i, j] = stats.kurtosis(gt_grid, fisher=True)

    # Avoid division by zero
    kurtosis_ratio = np.where(
        gt_kurtosis != 0,
        sample_kurtosis / gt_kurtosis,
        np.nan
    )

    return sample_kurtosis, gt_kurtosis, kurtosis_ratio


def compute_per_grid_skewness(
    samples: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple:
    """Compute skewness for each grid point."""
    sample_skewness = np.zeros((5, 5))
    gt_skewness = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            grid_samples = samples[:, :, i, j].flatten()
            gt_grid = ground_truth[:, i, j]

            sample_skewness[i, j] = stats.skew(grid_samples)
            gt_skewness[i, j] = stats.skew(gt_grid)

    return sample_skewness, gt_skewness


def compute_spatial_correlation(samples: np.ndarray) -> np.ndarray:
    """
    Compute 25x25 spatial correlation matrix.

    Args:
        samples: (num_samples, horizon, 5, 5)

    Returns:
        correlation: (25, 25) correlation matrix
    """
    # Reshape to (num_samples * horizon, 25)
    flat_samples = samples.reshape(-1, 25)
    return np.corrcoef(flat_samples.T)


def compute_smile_metrics(surfaces: np.ndarray) -> dict:
    """
    Compute volatility smile metrics.

    Smile is measured across moneyness (columns) for each maturity (row).

    Args:
        surfaces: (..., 5, 5) surfaces where columns are moneyness

    Returns:
        dict with smile curvature and amplitude
    """
    # Average across all samples/horizons
    if surfaces.ndim > 2:
        avg_surface = surfaces.mean(axis=tuple(range(surfaces.ndim - 2)))
    else:
        avg_surface = surfaces

    # Compute smile for middle maturity (row 2)
    mid_smile = avg_surface[2, :]  # 5 moneyness points

    # Curvature: second derivative approximation at ATM (index 2)
    curvature = mid_smile[0] - 2 * mid_smile[2] + mid_smile[4]

    # Amplitude: max - min
    amplitude = mid_smile.max() - mid_smile.min()

    # Skew: difference between wings
    skew = mid_smile[0] - mid_smile[4]  # ITM - OTM

    return {
        "curvature": float(curvature),
        "amplitude": float(amplitude),
        "skew": float(skew),
    }


def compute_term_structure_metrics(surfaces: np.ndarray) -> dict:
    """
    Compute term structure metrics.

    Term structure is measured across maturities (rows) for ATM (middle column).

    Args:
        surfaces: (..., 5, 5) surfaces where rows are maturities

    Returns:
        dict with term structure slope and curvature
    """
    if surfaces.ndim > 2:
        avg_surface = surfaces.mean(axis=tuple(range(surfaces.ndim - 2)))
    else:
        avg_surface = surfaces

    # ATM term structure (column 2)
    atm_term = avg_surface[:, 2]  # 5 maturity points

    # Slope: linear fit
    x = np.arange(5)
    slope = np.polyfit(x, atm_term, 1)[0]

    # Curvature: second derivative at middle
    curvature = atm_term[0] - 2 * atm_term[2] + atm_term[4]

    return {
        "slope": float(slope),
        "curvature": float(curvature),
    }


def compute_acf(series: np.ndarray, lag: int = 1) -> float:
    """Compute autocorrelation at given lag."""
    n = len(series)
    if n <= lag:
        return np.nan
    mean = series.mean()
    var = ((series - mean) ** 2).mean()
    if var == 0:
        return np.nan
    cov = ((series[:-lag] - mean) * (series[lag:] - mean)).mean()
    return cov / var


def compute_per_grid_acf(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    lag: int = 1,
) -> tuple:
    """Compute ACF at lag for each grid point."""
    sample_acf = np.zeros((5, 5))
    gt_acf = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            # Use mean trajectory for samples
            mean_trajectory = samples[:, :, i, j].mean(axis=0)
            sample_acf[i, j] = compute_acf(mean_trajectory, lag)
            gt_acf[i, j] = compute_acf(ground_truth[:, i, j], lag)

    return sample_acf, gt_acf


def plot_heatmap(data: np.ndarray, title: str, output_path: str, cmap: str = "RdYlGn", vmin=None, vmax=None):
    """Plot 5x5 heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Moneyness and maturity labels
    moneyness_labels = ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]
    maturity_labels = ["1M", "3M", "6M", "9M", "12M"]

    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=moneyness_labels,
        yticklabels=maturity_labels,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Maturity")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(corr: np.ndarray, output_path: str):
    """Plot 25x25 correlation matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Spatial Correlation Matrix (25x25)")
    ax.set_xlabel("Grid Point")
    ax.set_ylabel("Grid Point")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    config = ComprehensiveEvalConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Comprehensive Evaluation for Causal 3D VAE")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {config.model_path}...")
    model = load_model(config)

    # Load data
    print(f"Loading data from {config.data_path}...")
    data = np.load(config.data_path)
    surfaces = data["surface"].astype(np.float32)

    # Split train/test
    train_split = 0.8
    train_size = int(len(surfaces) * train_split)
    test_surfaces = surfaces[train_size:]
    print(f"Evaluating on {len(test_surfaces)} test surfaces")

    # Collect metrics across all sequences
    all_coverage = []
    all_explosion_rate = []
    all_sample_kurtosis = []
    all_gt_kurtosis = []
    all_sample_skewness = []
    all_gt_skewness = []
    all_sample_acf = []
    all_gt_acf = []
    all_sample_corr = []
    all_gt_corr = []
    all_smile_samples = []
    all_smile_gt = []
    all_term_samples = []
    all_term_gt = []

    # Select evaluation indices
    max_start = len(test_surfaces) - config.context_length - config.prediction_horizon
    eval_indices = np.linspace(0, max_start, config.num_sequences, dtype=int)

    print(f"\nEvaluating {config.num_sequences} sequences...")

    for start_idx in tqdm(eval_indices):
        # Get context and ground truth
        context = test_surfaces[start_idx:start_idx + config.context_length]
        gt_future = test_surfaces[start_idx + config.context_length:
                                   start_idx + config.context_length + config.prediction_horizon]

        if len(gt_future) < config.prediction_horizon:
            continue

        # Convert context to tensor
        context_tensor = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(0)
        context_tensor = context_tensor.to(config.device)

        # Generate samples
        samples = generate_ar_samples(
            model, context_tensor, config.prediction_horizon, config.num_samples
        )

        # Per-grid coverage
        coverage = compute_per_grid_coverage(samples, gt_future)
        all_coverage.append(coverage)

        # Per-grid explosion rate
        explosion_rate = compute_per_grid_explosion_rate(
            samples, config.explosion_threshold_high, config.explosion_threshold_low
        )
        all_explosion_rate.append(explosion_rate)

        # Per-grid kurtosis
        sample_kurt, gt_kurt, _ = compute_per_grid_kurtosis(samples, gt_future)
        all_sample_kurtosis.append(sample_kurt)
        all_gt_kurtosis.append(gt_kurt)

        # Per-grid skewness
        sample_skew, gt_skew = compute_per_grid_skewness(samples, gt_future)
        all_sample_skewness.append(sample_skew)
        all_gt_skewness.append(gt_skew)

        # Per-grid ACF
        sample_acf, gt_acf = compute_per_grid_acf(samples, gt_future)
        all_sample_acf.append(sample_acf)
        all_gt_acf.append(gt_acf)

        # Spatial correlation
        sample_corr = compute_spatial_correlation(samples)
        gt_corr = compute_spatial_correlation(gt_future[np.newaxis, :, :, :])
        all_sample_corr.append(sample_corr)
        all_gt_corr.append(gt_corr)

        # Smile metrics
        smile_sample = compute_smile_metrics(samples)
        smile_gt = compute_smile_metrics(gt_future)
        all_smile_samples.append(smile_sample)
        all_smile_gt.append(smile_gt)

        # Term structure metrics
        term_sample = compute_term_structure_metrics(samples)
        term_gt = compute_term_structure_metrics(gt_future)
        all_term_samples.append(term_sample)
        all_term_gt.append(term_gt)

    # Average metrics
    avg_coverage = np.mean(all_coverage, axis=0)
    avg_explosion = np.mean(all_explosion_rate, axis=0)
    avg_sample_kurtosis = np.mean(all_sample_kurtosis, axis=0)
    avg_gt_kurtosis = np.mean(all_gt_kurtosis, axis=0)
    avg_sample_skewness = np.mean(all_sample_skewness, axis=0)
    avg_gt_skewness = np.mean(all_gt_skewness, axis=0)
    avg_sample_acf = np.nanmean(all_sample_acf, axis=0)
    avg_gt_acf = np.nanmean(all_gt_acf, axis=0)
    avg_sample_corr = np.mean(all_sample_corr, axis=0)
    avg_gt_corr = np.mean(all_gt_corr, axis=0)

    # Print results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n1. CI COVERAGE (90% CI, target: 90%)")
    print(f"   Overall: {avg_coverage.mean():.1%}")
    print(f"   Best grid: {avg_coverage.max():.1%}")
    print(f"   Worst grid: {avg_coverage.min():.1%}")

    print(f"\n2. EXPLOSION RATE (target: 0%)")
    print(f"   Overall: {avg_explosion.mean():.1%}")
    print(f"   Max at any grid: {avg_explosion.max():.1%}")

    print(f"\n3. KURTOSIS (excess kurtosis, Gaussian=0)")
    print(f"   GT mean: {avg_gt_kurtosis.mean():.2f}")
    print(f"   Sample mean: {avg_sample_kurtosis.mean():.2f}")
    kurtosis_ratio = avg_sample_kurtosis.mean() / avg_gt_kurtosis.mean() if avg_gt_kurtosis.mean() != 0 else np.nan
    print(f"   Ratio (sample/GT): {kurtosis_ratio:.2f}")

    print(f"\n4. SKEWNESS")
    print(f"   GT mean: {avg_gt_skewness.mean():.2f}")
    print(f"   Sample mean: {avg_sample_skewness.mean():.2f}")

    print(f"\n5. AUTOCORRELATION (lag=1)")
    print(f"   GT mean: {np.nanmean(avg_gt_acf):.3f}")
    print(f"   Sample mean: {np.nanmean(avg_sample_acf):.3f}")

    print(f"\n6. SPATIAL CORRELATION")
    corr_frobenius = np.linalg.norm(avg_sample_corr - avg_gt_corr, 'fro')
    print(f"   Frobenius norm diff: {corr_frobenius:.4f}")

    # Smile and term structure
    smile_curvature_gt = np.mean([s["curvature"] for s in all_smile_gt])
    smile_curvature_sample = np.mean([s["curvature"] for s in all_smile_samples])
    term_slope_gt = np.mean([s["slope"] for s in all_term_gt])
    term_slope_sample = np.mean([s["slope"] for s in all_term_samples])

    print(f"\n7. VOLATILITY SMILE")
    print(f"   GT curvature: {smile_curvature_gt:.4f}")
    print(f"   Sample curvature: {smile_curvature_sample:.4f}")

    print(f"\n8. TERM STRUCTURE")
    print(f"   GT slope: {term_slope_gt:.4f}")
    print(f"   Sample slope: {term_slope_sample:.4f}")

    # Generate plots
    print("\nGenerating plots...")

    # Coverage heatmap
    plot_heatmap(
        avg_coverage * 100,
        "90% CI Coverage Rate (%)\n(Target: 90%)",
        os.path.join(config.output_dir, "coverage_heatmap.png"),
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
    )

    # Explosion rate heatmap
    plot_heatmap(
        avg_explosion * 100,
        "Explosion Rate (%)\n(Target: 0%)",
        os.path.join(config.output_dir, "explosion_heatmap.png"),
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(10, avg_explosion.max() * 100),
    )

    # Kurtosis heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    moneyness_labels = ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]
    maturity_labels = ["1M", "3M", "6M", "9M", "12M"]

    sns.heatmap(avg_gt_kurtosis, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0],
                xticklabels=moneyness_labels, yticklabels=maturity_labels)
    axes[0].set_title("GT Kurtosis")

    sns.heatmap(avg_sample_kurtosis, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1],
                xticklabels=moneyness_labels, yticklabels=maturity_labels)
    axes[1].set_title("Sample Kurtosis")

    kurtosis_diff = avg_sample_kurtosis - avg_gt_kurtosis
    sns.heatmap(kurtosis_diff, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[2],
                xticklabels=moneyness_labels, yticklabels=maturity_labels)
    axes[2].set_title("Kurtosis Difference (Sample - GT)")

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "kurtosis_comparison.png"), dpi=150)
    plt.close()

    # Correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(avg_gt_corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("GT Correlation")

    sns.heatmap(avg_sample_corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Sample Correlation")

    corr_diff = avg_sample_corr - avg_gt_corr
    sns.heatmap(corr_diff, cmap="RdBu_r", center=0, ax=axes[2])
    axes[2].set_title(f"Correlation Diff (Frobenius: {corr_frobenius:.4f})")

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "correlation_comparison.png"), dpi=150)
    plt.close()

    # ACF comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(avg_gt_acf, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0],
                xticklabels=moneyness_labels, yticklabels=maturity_labels)
    axes[0].set_title("GT ACF (lag=1)")

    sns.heatmap(avg_sample_acf, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1],
                xticklabels=moneyness_labels, yticklabels=maturity_labels)
    axes[1].set_title("Sample ACF (lag=1)")

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "acf_comparison.png"), dpi=150)
    plt.close()

    # Save results to JSON
    results = {
        "coverage": {
            "mean": float(avg_coverage.mean()),
            "min": float(avg_coverage.min()),
            "max": float(avg_coverage.max()),
            "per_grid": avg_coverage.tolist(),
        },
        "explosion_rate": {
            "mean": float(avg_explosion.mean()),
            "max": float(avg_explosion.max()),
            "per_grid": avg_explosion.tolist(),
        },
        "kurtosis": {
            "gt_mean": float(avg_gt_kurtosis.mean()),
            "sample_mean": float(avg_sample_kurtosis.mean()),
            "ratio": float(kurtosis_ratio) if not np.isnan(kurtosis_ratio) else None,
        },
        "skewness": {
            "gt_mean": float(avg_gt_skewness.mean()),
            "sample_mean": float(avg_sample_skewness.mean()),
        },
        "acf": {
            "gt_mean": float(np.nanmean(avg_gt_acf)),
            "sample_mean": float(np.nanmean(avg_sample_acf)),
        },
        "spatial_correlation": {
            "frobenius_norm_diff": float(corr_frobenius),
        },
        "smile": {
            "gt_curvature": float(smile_curvature_gt),
            "sample_curvature": float(smile_curvature_sample),
        },
        "term_structure": {
            "gt_slope": float(term_slope_gt),
            "sample_slope": float(term_slope_sample),
        },
    }

    with open(os.path.join(config.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw numpy arrays
    np.savez(
        os.path.join(config.output_dir, "results.npz"),
        coverage=avg_coverage,
        explosion_rate=avg_explosion,
        sample_kurtosis=avg_sample_kurtosis,
        gt_kurtosis=avg_gt_kurtosis,
        sample_skewness=avg_sample_skewness,
        gt_skewness=avg_gt_skewness,
        sample_acf=avg_sample_acf,
        gt_acf=avg_gt_acf,
        sample_corr=avg_sample_corr,
        gt_corr=avg_gt_corr,
    )

    print(f"\nResults saved to {config.output_dir}/")
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
