#!/usr/bin/env python
"""
Unconditional marginal matching analysis for DDPM volatility surface forecasting.

Compares unconditional marginal distributions of IV values between ground truth
and DDPM-generated paths. By iterating over many different conditioning histories
and stacking 30-day sequences, we obtain "unconditional" marginals.

Usage:
    python experiments/backfill/diffusion_poc/analyze_marginal_matching.py

    python experiments/backfill/diffusion_poc/analyze_marginal_matching.py \
        --model_path models/backfill/ddpm_poc/checkpoint_epoch_50.pt \
        --max_batches 50 --n_samples 30
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.simple_denoiser import ConditionalDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config


# Grid labels
TENOR_LABELS = ['1M', '2M', '3M', '6M', '1Y']
STRIKE_LABELS = ['90%', '95%', 'ATM', '105%', '110%']


# =============================================================================
# Data Collection
# =============================================================================

def collect_marginal_samples(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    device: str,
    n_samples: int = 30,
    max_batches: int = 50,
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """
    Collect pooled IV values per grid point across many conditioning histories.

    Args:
        model: DDPM model
        test_loader: Test data loader
        device: Device to run on
        n_samples: Number of samples per conditioning history
        max_batches: Maximum batches to process
        sampler: Sampling method ('ddpm' or 'ddim')
        n_inference_steps: Number of inference steps for DDIM

    Returns:
        {(tenor_idx, strike_idx): {'gt': np.array, 'gen': np.array}}
    """
    model.eval()

    # Initialize storage for 25 grid points
    grid_data = {(i, j): {'gt': [], 'gen': []}
                 for i in range(5) for j in range(5)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Collecting marginals")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"]

            # Denormalize GT
            future_gt = denormalize_iv(future_gt).numpy()

            # Generate samples (already denormalized by model.sample())
            samples = model.sample(
                history, n_samples=n_samples,
                sampler=sampler, n_inference_steps=n_inference_steps
            )
            samples = samples.cpu().numpy()  # (B, n_samples, 30, 5, 5)

            B = samples.shape[0]
            for b in range(B):
                for i in range(5):
                    for j in range(5):
                        # Stack 30 days of GT
                        gt_vals = future_gt[b, :, i, j]  # (30,)
                        grid_data[(i, j)]['gt'].extend(gt_vals.tolist())

                        # Stack 30 days × n_samples of Generated
                        for s in range(n_samples):
                            gen_vals = samples[b, s, :, i, j]  # (30,)
                            grid_data[(i, j)]['gen'].extend(gen_vals.tolist())

    # Convert to arrays
    for key in grid_data:
        grid_data[key]['gt'] = np.array(grid_data[key]['gt'])
        grid_data[key]['gen'] = np.array(grid_data[key]['gen'])

    return grid_data


def collect_per_horizon_marginals(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    device: str,
    horizons: List[int] = [1, 7, 14, 30],
    n_samples: int = 30,
    max_batches: int = 50,
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Collect marginal samples per forecast horizon (pooled across all grid points).

    Args:
        horizons: List of forecast horizons to analyze

    Returns:
        {horizon: {'gt': np.array, 'gen': np.array}}
    """
    model.eval()

    horizon_data = {h: {'gt': [], 'gen': []} for h in horizons}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Collecting per-horizon")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"]

            # Denormalize GT
            future_gt = denormalize_iv(future_gt).numpy()

            # Generate samples
            samples = model.sample(
                history, n_samples=n_samples,
                sampler=sampler, n_inference_steps=n_inference_steps
            )
            samples = samples.cpu().numpy()  # (B, n_samples, 30, 5, 5)

            B = samples.shape[0]
            for b in range(B):
                for h in horizons:
                    day_idx = h - 1  # 0-indexed

                    # GT: all 25 grid points at this horizon
                    gt_surface = future_gt[b, day_idx, :, :].flatten()  # (25,)
                    horizon_data[h]['gt'].extend(gt_surface.tolist())

                    # Generated: n_samples × 25 grid points
                    for s in range(n_samples):
                        gen_surface = samples[b, s, day_idx, :, :].flatten()  # (25,)
                        horizon_data[h]['gen'].extend(gen_surface.tolist())

    # Convert to arrays
    for h in horizons:
        horizon_data[h]['gt'] = np.array(horizon_data[h]['gt'])
        horizon_data[h]['gen'] = np.array(horizon_data[h]['gen'])

    return horizon_data


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_marginal_metrics(gt: np.ndarray, gen: np.ndarray) -> Dict:
    """
    Compute distribution comparison metrics.

    Args:
        gt: Ground truth values (1D array)
        gen: Generated values (1D array)

    Returns:
        Dictionary of metrics
    """
    ks_stat, ks_pval = stats.ks_2samp(gt, gen)
    w_dist = stats.wasserstein_distance(gt, gen)

    gt_mean = float(gt.mean())
    gen_mean = float(gen.mean())
    gt_std = float(gt.std())
    gen_std = float(gen.std())

    return {
        'ks_stat': float(ks_stat),
        'ks_pval': float(ks_pval),
        'wasserstein': float(w_dist),
        'gt_mean': gt_mean,
        'gen_mean': gen_mean,
        'mean_diff': float(gen_mean - gt_mean),
        'mean_diff_pct': float(abs(gen_mean - gt_mean) / gt_mean * 100) if gt_mean != 0 else 0.0,
        'gt_std': gt_std,
        'gen_std': gen_std,
        'std_ratio': float(gen_std / gt_std) if gt_std > 0 else 1.0,
        'gt_n': int(len(gt)),
        'gen_n': int(len(gen)),
        'match': bool(ks_pval > 0.05),
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_ks_heatmap(results: Dict[Tuple[int, int], Dict], output_path: str):
    """
    Plot 5×5 heatmap of KS statistics.
    """
    ks_matrix = np.zeros((5, 5))
    pval_matrix = np.zeros((5, 5))

    for (i, j), metrics in results.items():
        ks_matrix[i, j] = metrics['ks_stat']
        pval_matrix[i, j] = metrics['ks_pval']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KS statistic heatmap
    ax = axes[0]
    im = ax.imshow(ks_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.15)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STRIKE_LABELS)
    ax.set_yticklabels(TENOR_LABELS)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Tenor')
    ax.set_title('KS Statistic\n(Lower = Better Match)')
    plt.colorbar(im, ax=ax, label='KS Statistic')

    # Annotate values
    for i in range(5):
        for j in range(5):
            color = 'white' if ks_matrix[i, j] > 0.08 else 'black'
            ax.text(j, i, f'{ks_matrix[i, j]:.3f}', ha='center', va='center',
                    color=color, fontsize=9)

    # P-value heatmap
    ax = axes[1]
    im = ax.imshow(pval_matrix, cmap='RdYlGn', vmin=0, vmax=0.5)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STRIKE_LABELS)
    ax.set_yticklabels(TENOR_LABELS)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Tenor')
    ax.set_title('KS p-value\n(Higher = Better, >0.05 = Match)')
    plt.colorbar(im, ax=ax, label='p-value')

    # Annotate values
    for i in range(5):
        for j in range(5):
            color = 'white' if pval_matrix[i, j] < 0.1 else 'black'
            ax.text(j, i, f'{pval_matrix[i, j]:.3f}', ha='center', va='center',
                    color=color, fontsize=9)

    plt.suptitle('Unconditional Marginal Matching by Grid Point', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_marginal_histograms(
    grid_data: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """
    Plot histograms for representative grid points.
    """
    points = [
        ((0, 2), '1M ATM'),      # Short tenor ATM
        ((2, 2), '3M ATM'),      # Medium tenor ATM
        ((4, 2), '1Y ATM'),      # Long tenor ATM
        ((2, 0), '3M 90%'),      # OTM put
        ((0, 0), '1M 90%'),      # Short OTM put
        ((4, 4), '1Y 110%'),     # Long OTM call
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, ((i, j), label) in zip(axes.flat, points):
        gt = grid_data[(i, j)]['gt']
        gen = grid_data[(i, j)]['gen']
        metrics = results[(i, j)]

        bins = np.linspace(
            min(gt.min(), gen.min()),
            max(gt.max(), gen.max()),
            50
        )

        ax.hist(gt, bins=bins, alpha=0.6, density=True, label='Ground Truth',
                color='red', edgecolor='darkred')
        ax.hist(gen, bins=bins, alpha=0.6, density=True, label='Generated',
                color='blue', edgecolor='darkblue')

        # Add mean lines
        ax.axvline(gt.mean(), color='red', linestyle='--', linewidth=2)
        ax.axvline(gen.mean(), color='blue', linestyle='--', linewidth=2)

        ax.set_xlabel('Implied Volatility')
        ax.set_ylabel('Density')
        ax.set_title(f'{label}\nKS={metrics["ks_stat"]:.3f}, p={metrics["ks_pval"]:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Unconditional Marginal Distributions\n(Dashed lines = means)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_qq_grid(
    grid_data: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """
    Plot Q-Q plots for all 25 grid points.
    """
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

    n_quantiles = 100
    quantile_levels = np.linspace(0.01, 0.99, n_quantiles)

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            gt = grid_data[(i, j)]['gt']
            gen = grid_data[(i, j)]['gen']
            metrics = results[(i, j)]

            gt_quantiles = np.quantile(gt, quantile_levels)
            gen_quantiles = np.quantile(gen, quantile_levels)

            ax.scatter(gt_quantiles, gen_quantiles, alpha=0.5, s=10,
                       color='blue' if metrics['match'] else 'red')

            # Diagonal line
            min_val = min(gt_quantiles.min(), gen_quantiles.min())
            max_val = max(gt_quantiles.max(), gen_quantiles.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

            ax.set_title(f'{TENOR_LABELS[i]}/{STRIKE_LABELS[j]}\nKS={metrics["ks_stat"]:.3f}',
                         fontsize=9)

            if i == 4:
                ax.set_xlabel('GT Quantile', fontsize=8)
            if j == 0:
                ax.set_ylabel('Gen Quantile', fontsize=8)

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Q-Q Plots: Generated vs Ground Truth\n(Blue = match, Red = mismatch)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_horizon_marginals(
    horizon_data: Dict[int, Dict[str, np.ndarray]],
    output_path: str,
):
    """
    Plot marginal distributions per forecast horizon.
    """
    horizons = sorted(horizon_data.keys())
    n_horizons = len(horizons)

    fig, axes = plt.subplots(1, n_horizons, figsize=(4 * n_horizons, 5))
    if n_horizons == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        gt = horizon_data[h]['gt']
        gen = horizon_data[h]['gen']

        ks_stat, ks_pval = stats.ks_2samp(gt, gen)

        bins = np.linspace(
            min(gt.min(), gen.min()),
            max(gt.max(), gen.max()),
            50
        )

        ax.hist(gt, bins=bins, alpha=0.6, density=True, label='Ground Truth',
                color='red', edgecolor='darkred')
        ax.hist(gen, bins=bins, alpha=0.6, density=True, label='Generated',
                color='blue', edgecolor='darkblue')

        ax.axvline(gt.mean(), color='red', linestyle='--', linewidth=2)
        ax.axvline(gen.mean(), color='blue', linestyle='--', linewidth=2)

        ax.set_xlabel('Implied Volatility')
        ax.set_ylabel('Density')
        ax.set_title(f'Horizon h={h}\nKS={ks_stat:.3f}, p={ks_pval:.3f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Marginal Distributions by Forecast Horizon', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_metrics(results: Dict[Tuple[int, int], Dict], output_path: str):
    """
    Plot summary bar charts of key metrics across grid points.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    grid_labels = [f'{TENOR_LABELS[i]}/{STRIKE_LABELS[j]}' for i in range(5) for j in range(5)]
    ks_stats = [results[(i, j)]['ks_stat'] for i in range(5) for j in range(5)]
    mean_diffs = [results[(i, j)]['mean_diff_pct'] for i in range(5) for j in range(5)]
    std_ratios = [results[(i, j)]['std_ratio'] for i in range(5) for j in range(5)]
    wasserstein = [results[(i, j)]['wasserstein'] for i in range(5) for j in range(5)]

    # KS statistics
    ax = axes[0, 0]
    colors = ['green' if results[(i, j)]['match'] else 'red' for i in range(5) for j in range(5)]
    ax.bar(range(25), ks_stats, color=colors, alpha=0.7)
    ax.axhline(0.05, color='black', linestyle='--', label='Typical threshold')
    ax.set_xticks(range(25))
    ax.set_xticklabels(grid_labels, rotation=90, fontsize=7)
    ax.set_ylabel('KS Statistic')
    ax.set_title('KS Statistic by Grid Point')
    ax.legend()

    # Mean difference
    ax = axes[0, 1]
    ax.bar(range(25), mean_diffs, color='steelblue', alpha=0.7)
    ax.axhline(5, color='red', linestyle='--', label='5% threshold')
    ax.set_xticks(range(25))
    ax.set_xticklabels(grid_labels, rotation=90, fontsize=7)
    ax.set_ylabel('Mean Diff (%)')
    ax.set_title('Mean Difference by Grid Point')
    ax.legend()

    # Std ratio
    ax = axes[1, 0]
    colors = ['green' if 0.8 < r < 1.2 else 'red' for r in std_ratios]
    ax.bar(range(25), std_ratios, color=colors, alpha=0.7)
    ax.axhline(1.0, color='black', linestyle='-', linewidth=2)
    ax.axhline(0.8, color='red', linestyle='--')
    ax.axhline(1.2, color='red', linestyle='--')
    ax.set_xticks(range(25))
    ax.set_xticklabels(grid_labels, rotation=90, fontsize=7)
    ax.set_ylabel('Std Ratio (Gen/GT)')
    ax.set_title('Standard Deviation Ratio by Grid Point')

    # Wasserstein distance
    ax = axes[1, 1]
    ax.bar(range(25), wasserstein, color='purple', alpha=0.7)
    ax.axhline(0.02, color='red', linestyle='--', label='0.02 threshold')
    ax.set_xticks(range(25))
    ax.set_xticklabels(grid_labels, rotation=90, fontsize=7)
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Wasserstein Distance by Grid Point')
    ax.legend()

    plt.suptitle('Summary Metrics Across All Grid Points', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unconditional marginal matching analysis")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/ddpm_poc/checkpoint_epoch_50.pt")
    parser.add_argument("--data_path", type=str,
                        default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Number of samples per conditioning history")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/ddpm_poc/marginal_analysis")
    parser.add_argument("--sampler", type=str, default="ddim",
                        choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=20)
    args = parser.parse_args()

    # Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Unconditional Marginal Matching Analysis")
    print("=" * 70)

    # Load data
    config = get_default_config()
    data = np.load(args.data_path)
    surfaces = data["surface"]

    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    # Load model
    if not Path(args.model_path).exists():
        print(f"ERROR: Model not found at {args.model_path}")
        return

    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = ConditionalDDPM(checkpoint["config"], scheduler_config={"device": device})
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Collect marginal samples per grid point
    print("\nCollecting marginal samples per grid point...")
    grid_data = collect_marginal_samples(
        model, test_loader, device,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
    )

    # Compute metrics per grid point
    print("\nComputing metrics...")
    results = {}
    for (i, j), data in grid_data.items():
        results[(i, j)] = compute_marginal_metrics(data['gt'], data['gen'])

    # Collect per-horizon marginals
    print("\nCollecting per-horizon marginals...")
    # Reset test loader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )
    horizon_data = collect_per_horizon_marginals(
        model, test_loader, device,
        horizons=[1, 7, 14, 30],
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
    )

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_ks_heatmap(results, str(output_dir / 'ks_heatmap.png'))
    plot_marginal_histograms(grid_data, results, str(output_dir / 'marginal_histograms.png'))
    plot_qq_grid(grid_data, results, str(output_dir / 'qq_plots.png'))
    plot_horizon_marginals(horizon_data, str(output_dir / 'horizon_marginals.png'))
    plot_summary_metrics(results, str(output_dir / 'summary_metrics.png'))

    # Compute summary statistics
    n_matching = sum(1 for r in results.values() if r['match'])
    avg_ks = np.mean([r['ks_stat'] for r in results.values()])
    avg_wasserstein = np.mean([r['wasserstein'] for r in results.values()])
    avg_mean_diff = np.mean([r['mean_diff_pct'] for r in results.values()])
    avg_std_ratio = np.mean([r['std_ratio'] for r in results.values()])

    # Print summary
    print(f"\n{'='*70}")
    print("Unconditional Marginal Matching Results")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  Grid points matching (KS p > 0.05): {n_matching}/25")
    print(f"  Average KS statistic:               {avg_ks:.4f}")
    print(f"  Average Wasserstein distance:       {avg_wasserstein:.4f}")
    print(f"  Average mean difference:            {avg_mean_diff:.2f}%")
    print(f"  Average std ratio (Gen/GT):         {avg_std_ratio:.3f}")

    print(f"\nPer-Grid-Point Results:")
    print(f"{'Tenor/Strike':<12} {'KS Stat':>8} {'p-value':>8} {'W-dist':>8} {'Mean%':>8} {'StdRatio':>9} {'Match':>6}")
    print("-" * 70)
    for i in range(5):
        for j in range(5):
            r = results[(i, j)]
            label = f"{TENOR_LABELS[i]}/{STRIKE_LABELS[j]}"
            match_str = "YES" if r['match'] else "NO"
            print(f"{label:<12} {r['ks_stat']:>8.4f} {r['ks_pval']:>8.4f} {r['wasserstein']:>8.4f} "
                  f"{r['mean_diff_pct']:>7.2f}% {r['std_ratio']:>9.3f} {match_str:>6}")

    # Per-horizon results
    print(f"\nPer-Horizon Results:")
    print(f"{'Horizon':<10} {'KS Stat':>8} {'p-value':>8} {'Match':>6}")
    print("-" * 40)
    for h in [1, 7, 14, 30]:
        gt = horizon_data[h]['gt']
        gen = horizon_data[h]['gen']
        ks_stat, ks_pval = stats.ks_2samp(gt, gen)
        match_str = "YES" if ks_pval > 0.05 else "NO"
        print(f"h={h:<8} {ks_stat:>8.4f} {ks_pval:>8.4f} {match_str:>6}")

    # Save results to JSON
    json_results = {
        'summary': {
            'n_matching': n_matching,
            'total_grid_points': 25,
            'avg_ks_stat': float(avg_ks),
            'avg_wasserstein': float(avg_wasserstein),
            'avg_mean_diff_pct': float(avg_mean_diff),
            'avg_std_ratio': float(avg_std_ratio),
        },
        'per_grid_point': {
            f"{TENOR_LABELS[i]}_{STRIKE_LABELS[j]}": results[(i, j)]
            for i in range(5) for j in range(5)
        },
        'per_horizon': {
            f"h_{h}": {
                'ks_stat': float(stats.ks_2samp(horizon_data[h]['gt'], horizon_data[h]['gen'])[0]),
                'ks_pval': float(stats.ks_2samp(horizon_data[h]['gt'], horizon_data[h]['gen'])[1]),
                'gt_n': int(len(horizon_data[h]['gt'])),
                'gen_n': int(len(horizon_data[h]['gen'])),
            }
            for h in [1, 7, 14, 30]
        },
        'config': {
            'n_samples': args.n_samples,
            'max_batches': args.max_batches,
            'sampler': args.sampler,
            'ddim_steps': args.ddim_steps,
        }
    }

    with open(output_dir / 'marginal_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Save summary table as CSV
    with open(output_dir / 'summary_table.csv', 'w') as f:
        f.write("Grid Point,KS Stat,KS p-value,Wasserstein,Mean Diff %,Std Ratio,Match\n")
        for i in range(5):
            for j in range(5):
                r = results[(i, j)]
                label = f"{TENOR_LABELS[i]}/{STRIKE_LABELS[j]}"
                match_str = "YES" if r['match'] else "NO"
                f.write(f"{label},{r['ks_stat']:.4f},{r['ks_pval']:.4f},{r['wasserstein']:.4f},"
                        f"{r['mean_diff_pct']:.2f},{r['std_ratio']:.3f},{match_str}\n")

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - marginal_results.json")
    print(f"  - summary_table.csv")
    print(f"  - ks_heatmap.png")
    print(f"  - marginal_histograms.png")
    print(f"  - qq_plots.png")
    print(f"  - horizon_marginals.png")
    print(f"  - summary_metrics.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
