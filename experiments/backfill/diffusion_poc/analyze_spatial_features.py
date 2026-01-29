#!/usr/bin/env python
"""
Spatial feature analysis for DDPM volatility surface forecasting.

Quantifies how well DDPM models preserve spatial features:
1. Per-day smile RMSE (across strikes)
2. Per-day term structure RMSE (across tenors)
3. Cross-grid correlation matrix (25x25)
4. Smile width tracking (OTM put - OTM call)
5. Smile steepness at ATM (dIV/dK)

Usage:
    # Basic usage (DDPM Baseline only)
    python experiments/backfill/diffusion_poc/analyze_spatial_features.py

    # With hierarchical model
    python experiments/backfill/diffusion_poc/analyze_spatial_features.py \
        --model_path models/backfill/ddpm_poc/checkpoint_epoch_50.pt \
        --hierarchical_model_path models/backfill/ddpm_poc/hierarchical_best.pt \
        --max_batches 20 --n_samples 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.simple_denoiser import ConditionalDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config


# =============================================================================
# Grid Convention
# =============================================================================
# Rows (0-4): Tenors (1M, 2M, 3M, 6M, 1Y)
# Columns (0-4): Strikes (OTM put/90%, 95%, ATM/100%, 105%, OTM call/110%)

TENOR_LABELS = ['1M', '2M', '3M', '6M', '1Y']
STRIKE_LABELS = ['90%', '95%', 'ATM', '105%', '110%']


# =============================================================================
# Metric Functions
# =============================================================================

def compute_smile_rmse(generated: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Compute per-day smile RMSE (across strikes for each tenor).

    Args:
        generated: (n_samples, T_fut, 5, 5)
        ground_truth: (T_fut, 5, 5)

    Returns:
        dict with smile RMSE metrics
    """
    gen_mean = generated.mean(axis=0)  # (T_fut, 5, 5)

    # RMSE across strikes (axis=2) for each (day, tenor)
    rmse = np.sqrt(((gen_mean - ground_truth) ** 2).mean(axis=2))  # (T_fut, 5)

    return {
        'per_day_tenor': rmse,  # (T_fut, 5)
        'mean': float(rmse.mean()),
        'std': float(rmse.std()),
        'per_tenor_mean': rmse.mean(axis=0).tolist(),  # (5,)
    }


def compute_term_structure_rmse(generated: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Compute per-day term structure RMSE (across tenors for each strike).

    Args:
        generated: (n_samples, T_fut, 5, 5)
        ground_truth: (T_fut, 5, 5)

    Returns:
        dict with term structure RMSE metrics
    """
    gen_mean = generated.mean(axis=0)  # (T_fut, 5, 5)

    # RMSE across tenors (axis=1) for each (day, strike)
    rmse = np.sqrt(((gen_mean - ground_truth) ** 2).mean(axis=1))  # (T_fut, 5)

    return {
        'per_day_strike': rmse,  # (T_fut, 5)
        'mean': float(rmse.mean()),
        'std': float(rmse.std()),
        'per_strike_mean': rmse.mean(axis=0).tolist(),  # (5,)
    }


def compute_cross_grid_correlation(surfaces: np.ndarray) -> np.ndarray:
    """
    Compute 25x25 Pearson correlation matrix between grid points.

    Args:
        surfaces: (T, 5, 5) single trajectory

    Returns:
        corr_matrix: (25, 25) correlation matrix
    """
    T = surfaces.shape[0]
    flat = surfaces.reshape(T, 25)  # (T, 25)

    # Pearson correlation between all pairs of grid points
    corr_matrix = np.corrcoef(flat.T)  # (25, 25)

    return corr_matrix


def compute_correlation_matrix_distance(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
) -> Dict:
    """
    Compare cross-grid correlation structure between GT and generated.

    Args:
        gt_surfaces: (T, 5, 5)
        gen_surfaces: (n_samples, T, 5, 5)

    Returns:
        dict with correlation comparison metrics
    """
    gt_corr = compute_cross_grid_correlation(gt_surfaces)

    # Compute correlation for each sample
    n_samples = gen_surfaces.shape[0]
    gen_corrs = []
    for s in range(n_samples):
        gen_corrs.append(compute_cross_grid_correlation(gen_surfaces[s]))
    gen_corrs = np.stack(gen_corrs, axis=0)  # (n_samples, 25, 25)

    gen_corr_mean = gen_corrs.mean(axis=0)
    gen_corr_std = gen_corrs.std(axis=0)

    # Frobenius norm of difference
    frobenius = np.linalg.norm(gt_corr - gen_corr_mean, 'fro')

    # Mean absolute correlation difference
    mean_abs_diff = np.abs(gt_corr - gen_corr_mean).mean()

    return {
        'frobenius_norm': float(frobenius),
        'mean_abs_diff': float(mean_abs_diff),
        'gt_corr': gt_corr,
        'gen_corr_mean': gen_corr_mean,
        'gen_corr_std': gen_corr_std,
    }


def compute_smile_width(surfaces: np.ndarray) -> np.ndarray:
    """
    Compute smile width: IV[OTM put] - IV[OTM call] per tenor.

    Args:
        surfaces: (T, 5, 5) or (n_samples, T, 5, 5)

    Returns:
        width: (T, 5) or (n_samples, T, 5)
    """
    # Column 0 = OTM put (90%), Column 4 = OTM call (110%)
    if surfaces.ndim == 3:
        return surfaces[:, :, 0] - surfaces[:, :, 4]  # (T, 5)
    else:
        return surfaces[:, :, :, 0] - surfaces[:, :, :, 4]  # (n_samples, T, 5)


def compute_width_correlation(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
) -> Dict:
    """
    Compare smile width dynamics between GT and generated.

    Args:
        gt_surfaces: (T, 5, 5)
        gen_surfaces: (n_samples, T, 5, 5)

    Returns:
        dict with width correlation metrics
    """
    gt_width = compute_smile_width(gt_surfaces)  # (T, 5)
    gen_width = compute_smile_width(gen_surfaces)  # (n_samples, T, 5)

    gen_width_mean = gen_width.mean(axis=0)  # (T, 5)
    gen_width_std = gen_width.std(axis=0)  # (T, 5)

    # Correlation per tenor
    correlations = []
    for tenor_idx in range(5):
        if gt_width[:, tenor_idx].std() > 1e-8 and gen_width_mean[:, tenor_idx].std() > 1e-8:
            corr = np.corrcoef(gt_width[:, tenor_idx], gen_width_mean[:, tenor_idx])[0, 1]
        else:
            corr = 0.0
        correlations.append(float(corr) if not np.isnan(corr) else 0.0)

    return {
        'per_tenor_correlation': correlations,
        'mean_correlation': float(np.mean(correlations)),
        'gt_width': gt_width,
        'gen_width_mean': gen_width_mean,
        'gen_width_std': gen_width_std,
    }


def compute_smile_steepness(surfaces: np.ndarray) -> np.ndarray:
    """
    Compute smile steepness at ATM using central finite difference.
    steepness = (IV[strike=3] - IV[strike=1]) / 2

    Args:
        surfaces: (T, 5, 5) or (n_samples, T, 5, 5)

    Returns:
        steepness: (T, 5) or (n_samples, T, 5) per tenor
    """
    if surfaces.ndim == 3:
        return (surfaces[:, :, 3] - surfaces[:, :, 1]) / 2  # (T, 5)
    else:
        return (surfaces[:, :, :, 3] - surfaces[:, :, :, 1]) / 2  # (n_samples, T, 5)


def compute_steepness_metrics(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
) -> Dict:
    """
    Compare smile steepness between GT and generated.

    Args:
        gt_surfaces: (T, 5, 5)
        gen_surfaces: (n_samples, T, 5, 5)

    Returns:
        dict with steepness comparison metrics
    """
    gt_steep = compute_smile_steepness(gt_surfaces)  # (T, 5)
    gen_steep = compute_smile_steepness(gen_surfaces)  # (n_samples, T, 5)

    gen_steep_mean = gen_steep.mean(axis=0)  # (T, 5)
    gen_steep_std = gen_steep.std(axis=0)  # (T, 5)

    # RMSE
    rmse_per_tenor = np.sqrt(((gen_steep_mean - gt_steep) ** 2).mean(axis=0))  # (5,)
    overall_rmse = np.sqrt(((gen_steep_mean - gt_steep) ** 2).mean())

    # Correlation per tenor
    correlations = []
    for tenor_idx in range(5):
        if gt_steep[:, tenor_idx].std() > 1e-8 and gen_steep_mean[:, tenor_idx].std() > 1e-8:
            corr = np.corrcoef(gt_steep[:, tenor_idx], gen_steep_mean[:, tenor_idx])[0, 1]
        else:
            corr = 0.0
        correlations.append(float(corr) if not np.isnan(corr) else 0.0)

    return {
        'rmse': float(overall_rmse),
        'per_tenor_rmse': rmse_per_tenor.tolist(),
        'per_tenor_correlation': correlations,
        'mean_correlation': float(np.mean(correlations)),
        'gt_steepness': gt_steep,
        'gen_steepness_mean': gen_steep_mean,
        'gen_steepness_std': gen_steep_std,
    }


def compute_surface_rmse_per_point(
    generated: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """
    Compute RMSE at each grid point across all days.

    Args:
        generated: (n_samples, T_fut, 5, 5)
        ground_truth: (T_fut, 5, 5)

    Returns:
        rmse_grid: (5, 5) RMSE per grid point
    """
    gen_mean = generated.mean(axis=0)  # (T_fut, 5, 5)

    # RMSE across time for each grid point
    rmse_grid = np.sqrt(((gen_mean - ground_truth) ** 2).mean(axis=0))  # (5, 5)

    return rmse_grid


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_smile_rmse_heatmap(results: Dict, output_path: str):
    """Create heatmap of smile RMSE per (day, tenor)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    data = results['per_day_tenor']  # (T_fut, 5)
    im = ax.imshow(data.T, aspect='auto', cmap='YlOrRd')

    ax.set_xlabel('Forecast Day')
    ax.set_ylabel('Tenor')
    ax.set_yticks(range(5))
    ax.set_yticklabels(TENOR_LABELS)

    plt.colorbar(im, ax=ax, label='Smile RMSE')
    ax.set_title(f"Smile RMSE by Day and Tenor (Mean={results['mean']:.4f})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_term_rmse_heatmap(results: Dict, output_path: str):
    """Create heatmap of term structure RMSE per (day, strike)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    data = results['per_day_strike']  # (T_fut, 5)
    im = ax.imshow(data.T, aspect='auto', cmap='YlOrRd')

    ax.set_xlabel('Forecast Day')
    ax.set_ylabel('Strike')
    ax.set_yticks(range(5))
    ax.set_yticklabels(STRIKE_LABELS)

    plt.colorbar(im, ax=ax, label='Term Structure RMSE')
    ax.set_title(f"Term Structure RMSE by Day and Strike (Mean={results['mean']:.4f})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_correlation_matrix_comparison(
    gt_corr: np.ndarray,
    gen_corr: np.ndarray,
    frobenius: float,
    output_path: str,
):
    """Plot GT vs Generated correlation matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # GT correlation
    im1 = axes[0].imshow(gt_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Ground Truth Correlation (25x25)')
    axes[0].set_xlabel('Grid Point')
    axes[0].set_ylabel('Grid Point')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Generated correlation
    im2 = axes[1].imshow(gen_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Generated Correlation (25x25)')
    axes[1].set_xlabel('Grid Point')
    axes[1].set_ylabel('Grid Point')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Difference
    diff = gen_corr - gt_corr
    vmax = max(0.3, np.abs(diff).max())
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title(f'Difference (Frobenius={frobenius:.3f})')
    axes[2].set_xlabel('Grid Point')
    axes[2].set_ylabel('Grid Point')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_smile_width_tracking(results: Dict, output_path: str):
    """Plot smile width time series per tenor."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for i, (ax, label) in enumerate(zip(axes, TENOR_LABELS)):
        days = np.arange(results['gt_width'].shape[0])

        # GT
        ax.plot(days, results['gt_width'][:, i], 'r-', label='Ground Truth', linewidth=2)

        # Generated with confidence band
        mean = results['gen_width_mean'][:, i]
        std = results['gen_width_std'][:, i]
        ax.plot(days, mean, 'b-', label='Generated', linewidth=1.5)
        ax.fill_between(days, mean - std, mean + std, alpha=0.3, color='blue')

        corr = results['per_tenor_correlation'][i]
        ax.set_title(f'{label} (r={corr:.3f})')
        ax.set_xlabel('Day')
        if i == 0:
            ax.set_ylabel('Smile Width')
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Smile Width Tracking (Mean Correlation={results['mean_correlation']:.3f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_smile_steepness_comparison(results: Dict, output_path: str):
    """Plot smile steepness time series and scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series (3M tenor, index 2)
    days = np.arange(results['gt_steepness'].shape[0])
    ax = axes[0]
    ax.plot(days, results['gt_steepness'][:, 2], 'r-', label='GT (3M tenor)', linewidth=2)
    mean = results['gen_steepness_mean'][:, 2]
    std = results['gen_steepness_std'][:, 2]
    ax.plot(days, mean, 'b-', label='Generated', linewidth=1.5)
    ax.fill_between(days, mean - std, mean + std, alpha=0.3, color='blue')
    ax.set_xlabel('Day')
    ax.set_ylabel('ATM Steepness (dIV/dK)')
    ax.set_title(f"ATM Steepness Time Series (3M tenor, r={results['per_tenor_correlation'][2]:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter: all tenors
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, (label, color) in enumerate(zip(TENOR_LABELS, colors)):
        ax.scatter(
            results['gt_steepness'][:, i].flatten(),
            results['gen_steepness_mean'][:, i].flatten(),
            alpha=0.5, s=15, label=label, color=color
        )

    # Perfect prediction line
    all_vals = np.concatenate([results['gt_steepness'].flatten(), results['gen_steepness_mean'].flatten()])
    lims = [all_vals.min(), all_vals.max()]
    ax.plot(lims, lims, 'k--', label='Perfect', linewidth=1)

    ax.set_xlabel('GT Steepness')
    ax.set_ylabel('Generated Steepness')
    ax.set_title(f"Steepness Scatter (RMSE={results['rmse']:.4f})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_example_surfaces(
    gt_surface: np.ndarray,
    gen_surfaces: np.ndarray,
    day: int,
    output_path: str,
):
    """Plot GT vs Generated surface comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    gen_mean = gen_surfaces.mean(axis=0)

    # Find common color scale
    vmin = min(gt_surface.min(), gen_mean.min())
    vmax = max(gt_surface.max(), gen_mean.max())

    # GT
    im1 = axes[0].imshow(gt_surface, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth (Day {day})')
    axes[0].set_xlabel('Strike')
    axes[0].set_ylabel('Tenor')
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(STRIKE_LABELS, fontsize=8)
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(TENOR_LABELS, fontsize=8)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Generated mean
    im2 = axes[1].imshow(gen_mean, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Generated Mean')
    axes[1].set_xlabel('Strike')
    axes[1].set_ylabel('Tenor')
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(STRIKE_LABELS, fontsize=8)
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(TENOR_LABELS, fontsize=8)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Difference
    diff = gen_mean - gt_surface
    diff_max = max(0.02, np.abs(diff).max())
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    rmse = np.sqrt((diff ** 2).mean())
    axes[2].set_title(f'Difference (RMSE={rmse:.4f})')
    axes[2].set_xlabel('Strike')
    axes[2].set_ylabel('Tenor')
    axes[2].set_xticks(range(5))
    axes[2].set_xticklabels(STRIKE_LABELS, fontsize=8)
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(TENOR_LABELS, fontsize=8)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_grid_rmse_heatmap(rmse_grid: np.ndarray, output_path: str):
    """Plot 5x5 RMSE heatmap per grid point."""
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(rmse_grid, cmap='YlOrRd')

    # Annotate each cell
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{rmse_grid[i, j]:.3f}', ha='center', va='center', fontsize=9)

    ax.set_xlabel('Strike')
    ax.set_ylabel('Tenor')
    ax.set_xticks(range(5))
    ax.set_xticklabels(STRIKE_LABELS, fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(TENOR_LABELS, fontsize=9)

    plt.colorbar(im, ax=ax, label='RMSE')
    ax.set_title(f'Per-Grid-Point RMSE (Mean={rmse_grid.mean():.4f})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_smile_comparison_at_horizon(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
    horizon: int,
    output_path: str,
):
    """
    Compare volatility smile shape at a specific horizon.

    Args:
        gt_surfaces: (T_fut, 5, 5) ground truth
        gen_surfaces: (n_samples, T_fut, 5, 5) generated samples
        horizon: day index (0-indexed, so horizon=14 means day 15)
        output_path: where to save
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    strikes = np.array([0.90, 0.95, 1.00, 1.05, 1.10])

    # Get surfaces at this horizon
    gt_day = gt_surfaces[horizon]  # (5, 5) - tenors x strikes
    gen_day = gen_surfaces[:, horizon]  # (n_samples, 5, 5)

    gen_mean = gen_day.mean(axis=0)
    gen_std = gen_day.std(axis=0)
    gen_p5 = np.percentile(gen_day, 5, axis=0)
    gen_p95 = np.percentile(gen_day, 95, axis=0)

    for tenor_idx, (ax, tenor_label) in enumerate(zip(axes, TENOR_LABELS)):
        # GT smile at this tenor
        gt_smile = gt_day[tenor_idx, :]  # (5,) across strikes

        # Generated smile statistics
        gen_smile_mean = gen_mean[tenor_idx, :]
        gen_smile_std = gen_std[tenor_idx, :]
        gen_smile_p5 = gen_p5[tenor_idx, :]
        gen_smile_p95 = gen_p95[tenor_idx, :]

        # Plot GT
        ax.plot(strikes, gt_smile, 'ro-', linewidth=2, markersize=8, label='Ground Truth')

        # Plot generated mean with 90% CI
        ax.plot(strikes, gen_smile_mean, 'b.-', linewidth=1.5, markersize=6, label='Generated Mean')
        ax.fill_between(strikes, gen_smile_p5, gen_smile_p95, alpha=0.3, color='blue', label='90% CI')

        # Compute smile metrics for this tenor
        gt_skew = gt_smile[0] - gt_smile[4]  # OTM put - OTM call
        gen_skew = gen_smile_mean[0] - gen_smile_mean[4]
        gt_convex = gt_smile[0] + gt_smile[4] - 2 * gt_smile[2]  # butterfly
        gen_convex = gen_smile_mean[0] + gen_smile_mean[4] - 2 * gen_smile_mean[2]

        ax.set_title(f'{tenor_label}\nSkew: GT={gt_skew:.3f}, Gen={gen_skew:.3f}\nConvex: GT={gt_convex:.3f}, Gen={gen_convex:.3f}')
        ax.set_xlabel('Moneyness')
        ax.set_xticks(strikes)
        ax.set_xticklabels(['90%', '95%', 'ATM', '105%', '110%'], fontsize=8)
        ax.grid(True, alpha=0.3)

        if tenor_idx == 0:
            ax.set_ylabel('Implied Volatility')
            ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(f'Volatility Smile Comparison at Day {horizon + 1}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_term_structure_comparison_at_horizon(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
    horizon: int,
    output_path: str,
):
    """
    Compare term structure shape at a specific horizon.

    Args:
        gt_surfaces: (T_fut, 5, 5) ground truth
        gen_surfaces: (n_samples, T_fut, 5, 5) generated samples
        horizon: day index (0-indexed)
        output_path: where to save
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    tenors = np.array([1, 2, 3, 6, 12])  # months

    # Get surfaces at this horizon
    gt_day = gt_surfaces[horizon]  # (5, 5) - tenors x strikes
    gen_day = gen_surfaces[:, horizon]  # (n_samples, 5, 5)

    gen_mean = gen_day.mean(axis=0)
    gen_p5 = np.percentile(gen_day, 5, axis=0)
    gen_p95 = np.percentile(gen_day, 95, axis=0)

    for strike_idx, (ax, strike_label) in enumerate(zip(axes, STRIKE_LABELS)):
        # GT term structure at this strike
        gt_term = gt_day[:, strike_idx]  # (5,) across tenors

        # Generated term structure statistics
        gen_term_mean = gen_mean[:, strike_idx]
        gen_term_p5 = gen_p5[:, strike_idx]
        gen_term_p95 = gen_p95[:, strike_idx]

        # Plot GT
        ax.plot(tenors, gt_term, 'ro-', linewidth=2, markersize=8, label='Ground Truth')

        # Plot generated mean with 90% CI
        ax.plot(tenors, gen_term_mean, 'b.-', linewidth=1.5, markersize=6, label='Generated Mean')
        ax.fill_between(tenors, gen_term_p5, gen_term_p95, alpha=0.3, color='blue', label='90% CI')

        # Compute term structure metrics
        gt_slope = gt_term[0] - gt_term[4]  # short - long
        gen_slope = gen_term_mean[0] - gen_term_mean[4]

        ax.set_title(f'{strike_label}\nSlope: GT={gt_slope:.3f}, Gen={gen_slope:.3f}')
        ax.set_xlabel('Tenor (months)')
        ax.set_xticks(tenors)
        ax.set_xticklabels(['1M', '2M', '3M', '6M', '1Y'], fontsize=8)
        ax.grid(True, alpha=0.3)

        if strike_idx == 0:
            ax.set_ylabel('Implied Volatility')
            ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(f'Term Structure Comparison at Day {horizon + 1}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_shape_metrics_at_horizon(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
    horizon: int,
) -> Dict:
    """
    Compute smile and term structure shape metrics at a specific horizon.

    Returns metrics comparing GT shape to generated shape.
    """
    gt_day = gt_surfaces[horizon]  # (5, 5)
    gen_day = gen_surfaces[:, horizon]  # (n_samples, 5, 5)
    gen_mean = gen_day.mean(axis=0)

    # Smile metrics per tenor
    smile_metrics = []
    for tenor_idx in range(5):
        gt_smile = gt_day[tenor_idx, :]
        gen_smile = gen_mean[tenor_idx, :]

        # Skew: OTM put - OTM call
        gt_skew = gt_smile[0] - gt_smile[4]
        gen_skew = gen_smile[0] - gen_smile[4]

        # Convexity: butterfly spread
        gt_convex = gt_smile[0] + gt_smile[4] - 2 * gt_smile[2]
        gen_convex = gen_smile[0] + gen_smile[4] - 2 * gen_smile[2]

        # Smile RMSE
        smile_rmse = np.sqrt(((gt_smile - gen_smile) ** 2).mean())

        smile_metrics.append({
            'tenor': TENOR_LABELS[tenor_idx],
            'gt_skew': float(gt_skew),
            'gen_skew': float(gen_skew),
            'skew_error': float(gen_skew - gt_skew),
            'gt_convexity': float(gt_convex),
            'gen_convexity': float(gen_convex),
            'convexity_error': float(gen_convex - gt_convex),
            'smile_rmse': float(smile_rmse),
        })

    # Term structure metrics per strike
    term_metrics = []
    for strike_idx in range(5):
        gt_term = gt_day[:, strike_idx]
        gen_term = gen_mean[:, strike_idx]

        # Slope: short - long
        gt_slope = gt_term[0] - gt_term[4]
        gen_slope = gen_term[0] - gen_term[4]

        # Term RMSE
        term_rmse = np.sqrt(((gt_term - gen_term) ** 2).mean())

        term_metrics.append({
            'strike': STRIKE_LABELS[strike_idx],
            'gt_slope': float(gt_slope),
            'gen_slope': float(gen_slope),
            'slope_error': float(gen_slope - gt_slope),
            'term_rmse': float(term_rmse),
        })

    # Aggregate metrics
    avg_skew_error = np.mean([m['skew_error'] for m in smile_metrics])
    avg_convex_error = np.mean([m['convexity_error'] for m in smile_metrics])
    avg_slope_error = np.mean([m['slope_error'] for m in term_metrics])
    avg_smile_rmse = np.mean([m['smile_rmse'] for m in smile_metrics])
    avg_term_rmse = np.mean([m['term_rmse'] for m in term_metrics])

    # Check if signs match
    skew_sign_match = sum(1 for m in smile_metrics
                         if np.sign(m['gt_skew']) == np.sign(m['gen_skew'])) / 5
    convex_sign_match = sum(1 for m in smile_metrics
                           if np.sign(m['gt_convexity']) == np.sign(m['gen_convexity'])) / 5
    slope_sign_match = sum(1 for m in term_metrics
                          if np.sign(m['gt_slope']) == np.sign(m['gen_slope'])) / 5

    return {
        'horizon': horizon + 1,
        'smile_metrics': smile_metrics,
        'term_metrics': term_metrics,
        'summary': {
            'avg_skew_error': float(avg_skew_error),
            'avg_convexity_error': float(avg_convex_error),
            'avg_slope_error': float(avg_slope_error),
            'avg_smile_rmse': float(avg_smile_rmse),
            'avg_term_rmse': float(avg_term_rmse),
            'skew_sign_match': float(skew_sign_match),
            'convex_sign_match': float(convex_sign_match),
            'slope_sign_match': float(slope_sign_match),
        }
    }


# =============================================================================
# Main Analysis
# =============================================================================

def run_spatial_analysis(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    device: str,
    n_samples: int = 50,
    max_batches: int = 20,
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
) -> Dict:
    """Run all spatial metrics on test data."""

    model.eval()

    # Accumulators
    all_smile_rmse = []
    all_term_rmse = []
    all_corr_distances = []
    all_width_results = []
    all_steepness_results = []
    all_grid_rmse = []

    # Store first example for visualization
    example_gt = None
    example_gen = None
    first_results = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Analyzing spatial features")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            # Generate samples
            # NOTE: model.sample() already returns denormalized samples in [0, 1]
            samples = model.sample(
                history, n_samples=n_samples,
                sampler=sampler, n_inference_steps=n_inference_steps
            )  # (B, n_samples, T_fut, 5, 5) - already denormalized

            # Denormalize GT (still in [-1, 1] from dataset)
            future_gt = denormalize_iv(future_gt).cpu().numpy()
            # samples already denormalized by model.sample()
            samples = samples.cpu().numpy()

            # Process each item in batch
            B = samples.shape[0]
            for b in range(B):
                gt = future_gt[b]  # (T_fut, 5, 5)
                gen = samples[b]  # (n_samples, T_fut, 5, 5)

                # Compute metrics
                smile_rmse = compute_smile_rmse(gen, gt)
                term_rmse = compute_term_structure_rmse(gen, gt)
                corr_dist = compute_correlation_matrix_distance(gt, gen)
                width_res = compute_width_correlation(gt, gen)
                steep_res = compute_steepness_metrics(gt, gen)
                grid_rmse = compute_surface_rmse_per_point(gen, gt)

                all_smile_rmse.append(smile_rmse)
                all_term_rmse.append(term_rmse)
                all_corr_distances.append(corr_dist)
                all_width_results.append(width_res)
                all_steepness_results.append(steep_res)
                all_grid_rmse.append(grid_rmse)

                # Store first example
                if example_gt is None:
                    example_gt = gt
                    example_gen = gen
                    first_results = {
                        'smile_rmse': smile_rmse,
                        'term_rmse': term_rmse,
                        'corr_dist': corr_dist,
                        'width': width_res,
                        'steepness': steep_res,
                        'grid_rmse': grid_rmse,
                    }

    # Aggregate results
    results = {
        'smile_rmse': {
            'mean': float(np.mean([r['mean'] for r in all_smile_rmse])),
            'std': float(np.std([r['mean'] for r in all_smile_rmse])),
            'per_tenor_mean': np.mean([r['per_tenor_mean'] for r in all_smile_rmse], axis=0).tolist(),
        },
        'term_rmse': {
            'mean': float(np.mean([r['mean'] for r in all_term_rmse])),
            'std': float(np.std([r['mean'] for r in all_term_rmse])),
            'per_strike_mean': np.mean([r['per_strike_mean'] for r in all_term_rmse], axis=0).tolist(),
        },
        'correlation_distance': {
            'frobenius_mean': float(np.mean([r['frobenius_norm'] for r in all_corr_distances])),
            'frobenius_std': float(np.std([r['frobenius_norm'] for r in all_corr_distances])),
            'mean_abs_diff': float(np.mean([r['mean_abs_diff'] for r in all_corr_distances])),
        },
        'width_correlation': {
            'mean_correlation': float(np.mean([r['mean_correlation'] for r in all_width_results])),
            'per_tenor_correlation': np.mean([r['per_tenor_correlation'] for r in all_width_results], axis=0).tolist(),
        },
        'steepness': {
            'rmse_mean': float(np.mean([r['rmse'] for r in all_steepness_results])),
            'rmse_std': float(np.std([r['rmse'] for r in all_steepness_results])),
            'mean_correlation': float(np.mean([r['mean_correlation'] for r in all_steepness_results])),
            'per_tenor_correlation': np.mean([r['per_tenor_correlation'] for r in all_steepness_results], axis=0).tolist(),
        },
        'grid_rmse': {
            'mean': float(np.mean([r.mean() for r in all_grid_rmse])),
            'per_point_mean': np.mean(all_grid_rmse, axis=0).tolist(),
        },
        'n_sequences': len(all_smile_rmse),
    }

    # Store visualization data
    results['_example_gt'] = example_gt
    results['_example_gen'] = example_gen
    results['_first'] = first_results

    return results


def save_summary_table(all_results: Dict, output_path: str):
    """Save summary comparison table as CSV."""
    import csv

    metrics = [
        ('Smile RMSE (mean)', 'smile_rmse', 'mean'),
        ('Smile RMSE (std)', 'smile_rmse', 'std'),
        ('Term Structure RMSE (mean)', 'term_rmse', 'mean'),
        ('Term Structure RMSE (std)', 'term_rmse', 'std'),
        ('Cross-Grid Corr Frobenius', 'correlation_distance', 'frobenius_mean'),
        ('Cross-Grid Corr Mean Abs Diff', 'correlation_distance', 'mean_abs_diff'),
        ('Width Correlation (mean)', 'width_correlation', 'mean_correlation'),
        ('Steepness RMSE', 'steepness', 'rmse_mean'),
        ('Steepness Correlation', 'steepness', 'mean_correlation'),
        ('Grid RMSE (mean)', 'grid_rmse', 'mean'),
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Metric'] + list(all_results.keys())
        writer.writerow(header)

        for metric_name, key1, key2 in metrics:
            row = [metric_name]
            for model_name, model_results in all_results.items():
                value = model_results.get(key1, {}).get(key2, 'N/A')
                if isinstance(value, float):
                    row.append(f'{value:.4f}')
                else:
                    row.append(str(value))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Spatial feature analysis for DDPM")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/ddpm_poc/checkpoint_epoch_50.pt")
    parser.add_argument("--hierarchical_model_path", type=str, default=None,
                        help="Path to hierarchical DDPM model (optional)")
    parser.add_argument("--data_path", type=str,
                        default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/ddpm_poc/spatial_analysis")
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
    print("Spatial Feature Analysis for DDPM")
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

    # Models to analyze
    models_to_analyze = {}

    # Load baseline model
    if Path(args.model_path).exists():
        print(f"\nLoading baseline model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model = ConditionalDDPM(checkpoint["config"], scheduler_config={"device": device})
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        models_to_analyze["DDPM_Baseline"] = model
    else:
        print(f"WARNING: Model not found at {args.model_path}")

    # Load hierarchical model if provided
    if args.hierarchical_model_path and Path(args.hierarchical_model_path).exists():
        print(f"\nLoading hierarchical model from {args.hierarchical_model_path}...")
        h_checkpoint = torch.load(args.hierarchical_model_path, map_location=device, weights_only=False)
        h_model = ConditionalDDPM(h_checkpoint["config"], scheduler_config={"device": device})
        h_model.load_state_dict(h_checkpoint["model_state_dict"])
        h_model = h_model.to(device)
        models_to_analyze["DDPM_Hierarchical"] = h_model

    if not models_to_analyze:
        print("ERROR: No models found to analyze!")
        return

    # Run analysis for each model
    all_results = {}
    for model_name, model in models_to_analyze.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*60}")

        results = run_spatial_analysis(
            model, test_loader, device,
            n_samples=args.n_samples,
            max_batches=args.max_batches,
            sampler=args.sampler,
            n_inference_steps=args.ddim_steps
        )

        # Generate visualizations
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)

        # Use first example for visualizations
        first = results['_first']

        plot_smile_rmse_heatmap(
            first['smile_rmse'],
            str(model_output_dir / 'smile_rmse_heatmap.png')
        )
        plot_term_rmse_heatmap(
            first['term_rmse'],
            str(model_output_dir / 'term_rmse_heatmap.png')
        )
        plot_correlation_matrix_comparison(
            first['corr_dist']['gt_corr'],
            first['corr_dist']['gen_corr_mean'],
            first['corr_dist']['frobenius_norm'],
            str(model_output_dir / 'correlation_matrices.png')
        )
        plot_smile_width_tracking(
            first['width'],
            str(model_output_dir / 'smile_width_tracking.png')
        )
        plot_smile_steepness_comparison(
            first['steepness'],
            str(model_output_dir / 'smile_steepness.png')
        )
        plot_grid_rmse_heatmap(
            first['grid_rmse'],
            str(model_output_dir / 'grid_rmse_heatmap.png')
        )

        # Example surfaces at days 1, 15, 30
        for day in [1, 15, 30]:
            day_idx = day - 1
            if day_idx < results['_example_gt'].shape[0]:
                plot_example_surfaces(
                    results['_example_gt'][day_idx],
                    results['_example_gen'][:, day_idx],
                    day,
                    str(model_output_dir / f'example_surface_day_{day}.png')
                )

        # Smile and term structure shape comparisons at h=15 and h=30
        shape_metrics = {}
        for horizon in [14, 29]:  # 0-indexed: day 15 and day 30
            day = horizon + 1
            if horizon < results['_example_gt'].shape[0]:
                # Smile comparison
                plot_smile_comparison_at_horizon(
                    results['_example_gt'],
                    results['_example_gen'],
                    horizon,
                    str(model_output_dir / f'smile_shape_day_{day}.png')
                )
                # Term structure comparison
                plot_term_structure_comparison_at_horizon(
                    results['_example_gt'],
                    results['_example_gen'],
                    horizon,
                    str(model_output_dir / f'term_structure_day_{day}.png')
                )
                # Compute shape metrics
                shape_metrics[f'day_{day}'] = compute_shape_metrics_at_horizon(
                    results['_example_gt'],
                    results['_example_gen'],
                    horizon
                )

        # Remove internal data before storing
        clean_results = {k: v for k, v in results.items() if not k.startswith('_')}

        # Add shape metrics to results
        clean_results['shape_metrics'] = shape_metrics
        all_results[model_name] = clean_results

        # Print shape metrics summary
        print(f"\n  Shape Metrics:")
        for day_key, metrics in shape_metrics.items():
            s = metrics['summary']
            print(f"    {day_key}:")
            print(f"      Smile: RMSE={s['avg_smile_rmse']:.4f}, Skew sign match={s['skew_sign_match']*100:.0f}%, Convex sign match={s['convex_sign_match']*100:.0f}%")
            print(f"      Term:  RMSE={s['avg_term_rmse']:.4f}, Slope sign match={s['slope_sign_match']*100:.0f}%")

        # Print summary
        print(f"\n{model_name} Summary:")
        print(f"  Smile RMSE: {clean_results['smile_rmse']['mean']:.4f} +/- {clean_results['smile_rmse']['std']:.4f}")
        print(f"  Term RMSE: {clean_results['term_rmse']['mean']:.4f} +/- {clean_results['term_rmse']['std']:.4f}")
        print(f"  Cross-Grid Corr Frobenius: {clean_results['correlation_distance']['frobenius_mean']:.4f}")
        print(f"  Width Correlation: {clean_results['width_correlation']['mean_correlation']:.4f}")
        print(f"  Steepness RMSE: {clean_results['steepness']['rmse_mean']:.4f}")
        print(f"  Steepness Correlation: {clean_results['steepness']['mean_correlation']:.4f}")

    # Save summary table
    save_summary_table(all_results, str(output_dir / 'summary_table.csv'))

    # Save detailed JSON
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - summary_table.csv")
    print(f"  - detailed_results.json")
    for model_name in all_results.keys():
        print(f"  - {model_name}/ (visualizations)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
