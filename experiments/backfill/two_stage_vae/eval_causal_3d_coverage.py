#!/usr/bin/env python
"""
Coverage and CRPS Evaluation for Causal 3D VAE.

CRITICAL: These metrics are NOT validated in video VAE literature!
Video VAE papers use "best of 100 samples" which does NOT answer:
"Does the 90% CI of the predicted distribution contain the ground truth?"

This script implements proper probabilistic forecast evaluation:
1. CI Coverage Rate: P(GT in [p5, p95]) should be ~90%
2. CRPS: Continuous Ranked Probability Score (proper scoring rule)
3. Calibration Plot: Compare nominal vs empirical coverage
4. Per-Horizon Analysis: Track coverage degradation over time

Usage:
    python experiments/backfill/two_stage_vae/eval_causal_3d_coverage.py
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.causal_3d_vae import create_vol_surface_vae, create_vol_surface_vae_small, AutoencoderCausal3D
from config.causal_3d_config import Causal3DEvalConfig


def load_model(config: Causal3DEvalConfig) -> AutoencoderCausal3D:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)

    # Get model config from checkpoint
    model_config = checkpoint.get("config", {})

    # Use small model if layers_per_block=1 (scaled-down architecture)
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


def compute_crps(samples: np.ndarray, ground_truth: float) -> float:
    """
    Compute CRPS (Continuous Ranked Probability Score).

    CRPS is a proper scoring rule for probabilistic forecasts.
    Lower is better. CRPS = E|X-y| - 0.5*E|X-X'|

    Args:
        samples: (n_samples,) array of samples from predictive distribution
        ground_truth: True value

    Returns:
        CRPS score (scalar)
    """
    n = len(samples)

    # Term 1: Expected absolute error
    term1 = np.mean(np.abs(samples - ground_truth))

    # Term 2: Expected pairwise distance (divided by 2)
    # Efficient computation using sorting
    sorted_samples = np.sort(samples)
    term2 = 0.0
    for i, x in enumerate(sorted_samples):
        # Number of samples <= x multiplied by distance to each
        term2 += (2 * i + 1 - n) * x
    term2 = 2 * term2 / (n * n)

    return term1 - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))


def compute_coverage(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    levels: tuple = (0.5, 0.8, 0.9, 0.95),
) -> dict:
    """
    Compute CI coverage at multiple levels.

    Args:
        samples: (n_samples, ...) array of samples
        ground_truth: (...) array of true values
        levels: Tuple of coverage levels to evaluate

    Returns:
        Dict with coverage rates for each level
    """
    results = {}

    for level in levels:
        alpha = (1 - level) / 2

        # Compute quantiles
        lower = np.quantile(samples, alpha, axis=0)
        upper = np.quantile(samples, 1 - alpha, axis=0)

        # Check coverage
        covered = (ground_truth >= lower) & (ground_truth <= upper)
        coverage_rate = covered.mean()

        results[f"coverage_{int(level*100)}"] = coverage_rate
        results[f"lower_{int(level*100)}"] = lower
        results[f"upper_{int(level*100)}"] = upper

    return results


def evaluate_ar_generation(
    model: AutoencoderCausal3D,
    surfaces: np.ndarray,
    config: Causal3DEvalConfig,
    start_indices: list = None,
) -> dict:
    """
    Evaluate autoregressive generation with coverage metrics.

    Args:
        model: Trained model
        surfaces: (N, 5, 5) array of ground truth surfaces
        config: Evaluation config
        start_indices: List of starting indices to evaluate

    Returns:
        Dict with evaluation metrics
    """
    if start_indices is None:
        # Use evenly spaced indices
        n_eval = 50
        max_start = len(surfaces) - config.context_length - config.prediction_horizon
        start_indices = np.linspace(0, max_start, n_eval, dtype=int)

    all_crps = []
    all_coverage = {level: [] for level in config.ci_levels}
    per_horizon_coverage = {h: {level: [] for level in config.ci_levels}
                           for h in range(config.prediction_horizon)}

    print(f"\nEvaluating {len(start_indices)} sequences...")

    for start_idx in tqdm(start_indices):
        # Get context and ground truth
        context = surfaces[start_idx:start_idx + config.context_length]
        gt_future = surfaces[start_idx + config.context_length:
                            start_idx + config.context_length + config.prediction_horizon]

        # Skip if not enough data
        if len(gt_future) < config.prediction_horizon:
            continue

        # Convert to tensor: (1, 1, C, 5, 5)
        context_tensor = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(0)
        context_tensor = context_tensor.to(config.device)

        # Generate samples
        samples = []
        with torch.no_grad():
            for _ in range(config.num_samples):
                generated = model.generate_autoregressive(
                    context_tensor,
                    num_steps=config.prediction_horizon,
                )
                # Extract predicted future (after context)
                predicted = generated[:, :, config.context_length:].cpu().numpy()
                samples.append(predicted.squeeze())

        samples = np.array(samples)  # (num_samples, horizon, 5, 5)

        # Compute CRPS for each time step and grid point
        for h in range(config.prediction_horizon):
            for i in range(5):
                for j in range(5):
                    crps = compute_crps(samples[:, h, i, j], gt_future[h, i, j])
                    all_crps.append(crps)

        # Compute coverage for each horizon
        for h in range(config.prediction_horizon):
            h_samples = samples[:, h]  # (num_samples, 5, 5)
            h_gt = gt_future[h]  # (5, 5)

            coverage_results = compute_coverage(h_samples, h_gt, config.ci_levels)

            for level in config.ci_levels:
                level_key = int(level * 100)
                per_horizon_coverage[h][level].append(
                    coverage_results[f"coverage_{level_key}"]
                )

        # Compute overall coverage for this sequence
        for level in config.ci_levels:
            level_key = int(level * 100)
            # Average coverage across all horizons
            avg_coverage = np.mean([
                per_horizon_coverage[h][level][-1]
                for h in range(config.prediction_horizon)
            ])
            all_coverage[level].append(avg_coverage)

    # Aggregate results
    results = {
        "crps_mean": np.mean(all_crps),
        "crps_std": np.std(all_crps),
    }

    # Overall coverage
    for level in config.ci_levels:
        level_key = int(level * 100)
        results[f"coverage_{level_key}_mean"] = np.mean(all_coverage[level])
        results[f"coverage_{level_key}_std"] = np.std(all_coverage[level])

    # Per-horizon coverage (for degradation analysis)
    for h in [0, 6, 13, 29]:  # Horizons 1, 7, 14, 30
        if h < config.prediction_horizon:
            for level in config.ci_levels:
                level_key = int(level * 100)
                if per_horizon_coverage[h][level]:
                    results[f"coverage_{level_key}_h{h+1}"] = np.mean(
                        per_horizon_coverage[h][level]
                    )

    # Store per-horizon data for plotting
    results["per_horizon_coverage"] = per_horizon_coverage

    return results


def plot_calibration(results: dict, output_path: str):
    """
    Create calibration plot: nominal vs empirical coverage.

    Perfect calibration = diagonal line.
    """
    plt.figure(figsize=(8, 8))

    nominal = []
    empirical = []

    for level in [0.5, 0.8, 0.9, 0.95]:
        level_key = int(level * 100)
        if f"coverage_{level_key}_mean" in results:
            nominal.append(level)
            empirical.append(results[f"coverage_{level_key}_mean"])

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(nominal, empirical, s=100, c='blue', zorder=5)
    plt.plot(nominal, empirical, 'b-', alpha=0.5)

    for n, e in zip(nominal, empirical):
        plt.annotate(f'{int(n*100)}%', (n, e), textcoords="offset points",
                    xytext=(10, 5), fontsize=10)

    plt.xlabel('Nominal Coverage', fontsize=12)
    plt.ylabel('Empirical Coverage', fontsize=12)
    plt.title('Calibration Plot: Causal 3D VAE', fontsize=14)
    plt.xlim(0.4, 1.0)
    plt.ylim(0.4, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved calibration plot to {output_path}")


def plot_coverage_vs_horizon(results: dict, output_path: str, level: float = 0.9):
    """
    Plot coverage rate vs prediction horizon.

    Shows how coverage degrades over time.
    """
    level_key = int(level * 100)
    per_horizon = results["per_horizon_coverage"]

    horizons = list(range(len(per_horizon)))
    coverage_means = []
    coverage_stds = []

    for h in horizons:
        if per_horizon[h][level]:
            coverage_means.append(np.mean(per_horizon[h][level]))
            coverage_stds.append(np.std(per_horizon[h][level]))
        else:
            coverage_means.append(np.nan)
            coverage_stds.append(np.nan)

    plt.figure(figsize=(10, 6))

    horizons_1indexed = [h + 1 for h in horizons]
    plt.fill_between(
        horizons_1indexed,
        np.array(coverage_means) - np.array(coverage_stds),
        np.array(coverage_means) + np.array(coverage_stds),
        alpha=0.3,
        label='1 std'
    )
    plt.plot(horizons_1indexed, coverage_means, 'b-', linewidth=2, label='Mean coverage')
    plt.axhline(y=level, color='r', linestyle='--', label=f'Target ({int(level*100)}%)')

    plt.xlabel('Prediction Horizon (days)', fontsize=12)
    plt.ylabel(f'{int(level*100)}% CI Coverage', fontsize=12)
    plt.title('Coverage Degradation vs Prediction Horizon', fontsize=14)
    plt.xlim(1, len(horizons))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved coverage vs horizon plot to {output_path}")


def plot_coverage_heatmap(samples: np.ndarray, ground_truth: np.ndarray,
                         level: float = 0.9, output_path: str = None):
    """
    Create 5x5 heatmap of coverage rates per grid point.

    Args:
        samples: (n_samples, 5, 5) samples
        ground_truth: (5, 5) ground truth
        level: Coverage level
        output_path: Where to save the plot
    """
    alpha = (1 - level) / 2
    lower = np.quantile(samples, alpha, axis=0)
    upper = np.quantile(samples, 1 - alpha, axis=0)

    coverage = ((ground_truth >= lower) & (ground_truth <= upper)).astype(float)

    plt.figure(figsize=(8, 6))
    plt.imshow(coverage, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Coverage Rate')

    for i in range(5):
        for j in range(5):
            plt.text(j, i, f'{coverage[i,j]:.0%}',
                    ha='center', va='center', fontsize=10)

    plt.xlabel('Moneyness')
    plt.ylabel('Tenor')
    plt.title(f'{int(level*100)}% CI Coverage by Grid Point')

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Causal 3D VAE Coverage")
    parser.add_argument("--model-path", type=str,
                        default="models/backfill/causal_3d/best_model.pt",
                        help="Path to trained model")
    parser.add_argument("--data-path", type=str,
                        default="data/vol_surface_with_ret.npz",
                        help="Path to data")
    parser.add_argument("--output-dir", type=str,
                        default="results/causal_3d/",
                        help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples per prediction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Create config
    config = Causal3DEvalConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Check if model exists
    if not os.path.exists(config.model_path):
        print(f"Model not found at {config.model_path}")
        print("Please train the model first using train_causal_3d_vae.py")
        return

    # Load model
    print(f"Loading model from {config.model_path}...")
    model = load_model(config)

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path)
    surfaces = data["surface"]

    # Use test split (last 20%)
    n_train = int(len(surfaces) * 0.8)
    test_surfaces = surfaces[n_train:]
    print(f"Evaluating on {len(test_surfaces)} test surfaces")

    # Evaluate
    results = evaluate_ar_generation(model, test_surfaces, config)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nCRPS: {results['crps_mean']:.4f} +/- {results['crps_std']:.4f}")

    print("\nOverall Coverage:")
    for level in config.ci_levels:
        level_key = int(level * 100)
        print(f"  {level_key}% CI: {results[f'coverage_{level_key}_mean']:.2%} "
              f"+/- {results[f'coverage_{level_key}_std']:.2%}")

    print("\nPer-Horizon Coverage (90% CI):")
    for h in [1, 7, 14, 30]:
        key = f"coverage_90_h{h}"
        if key in results:
            target_gap = abs(results[key] - 0.90)
            status = "OK" if target_gap < 0.05 else "DEGRADED"
            print(f"  Horizon {h}: {results[key]:.2%} [{status}]")

    # Save plots
    plot_calibration(results, os.path.join(config.output_dir, "calibration_plot.png"))
    plot_coverage_vs_horizon(results, os.path.join(config.output_dir, "coverage_vs_horizon.png"))

    # Save results
    results_file = os.path.join(config.output_dir, "evaluation_results.npz")
    np.savez(results_file, **{k: v for k, v in results.items() if not k.startswith("per_")})
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
