#!/usr/bin/env python
"""
Temporal feature analysis for DDPM volatility surface forecasting.

Quantifies how well DDPM models preserve temporal dynamics:
1. Volatility clustering (ACF of squared returns per path)
2. Multi-lag ACF curves
3. Distribution comparison (KS test)

Usage:
    python experiments/backfill/diffusion_poc/analyze_temporal_features.py

    python experiments/backfill/diffusion_poc/analyze_temporal_features.py \
        --model_path models/backfill/ddpm_poc/checkpoint_epoch_50.pt \
        --max_batches 20 --n_samples 50
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


# =============================================================================
# Volatility Clustering Metrics
# =============================================================================

def compute_acf(series: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Compute autocorrelation function for a time series.

    Args:
        series: 1D array of values
        max_lag: maximum lag to compute

    Returns:
        acf: array of ACF values for lags 0 to max_lag
    """
    n = len(series)
    mean = series.mean()
    var = ((series - mean) ** 2).mean()

    if var < 1e-10:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0  # ACF at lag 0 is always 1

    for lag in range(1, min(max_lag + 1, n)):
        cov = ((series[:-lag] - mean) * (series[lag:] - mean)).mean()
        acf[lag] = cov / var

    return acf


def compute_acf_squared_returns(series: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Compute ACF of squared returns (ARCH effect measure).

    Args:
        series: 1D array of prices/IV values (T,)
        max_lag: maximum lag to compute

    Returns:
        acf: array of ACF values for squared returns
    """
    if len(series) < 3:
        return np.zeros(max_lag + 1)

    returns = np.diff(series)
    squared_returns = returns ** 2

    return compute_acf(squared_returns, max_lag)


def compute_vol_clustering_per_path(
    surfaces: np.ndarray,
    grid_point: Tuple[int, int] = (2, 2),  # ATM by default
) -> np.ndarray:
    """
    Compute ACF(1) of squared returns for each path at a specific grid point.

    Args:
        surfaces: (n_samples, T, 5, 5) or (T, 5, 5)
        grid_point: (tenor_idx, strike_idx) to extract

    Returns:
        acf_values: (n_samples,) ACF(1) for each path, or scalar if single path
    """
    tenor_idx, strike_idx = grid_point

    if surfaces.ndim == 3:
        # Single path
        series = surfaces[:, tenor_idx, strike_idx]
        acf = compute_acf_squared_returns(series, max_lag=1)
        return acf[1]
    else:
        # Multiple paths
        n_samples = surfaces.shape[0]
        acf_values = np.zeros(n_samples)

        for i in range(n_samples):
            series = surfaces[i, :, tenor_idx, strike_idx]
            acf = compute_acf_squared_returns(series, max_lag=1)
            acf_values[i] = acf[1]

        return acf_values


def compute_vol_clustering_all_points(
    surfaces: np.ndarray,
) -> np.ndarray:
    """
    Compute ACF(1) of squared returns for each path, averaged across all grid points.

    Args:
        surfaces: (n_samples, T, 5, 5)

    Returns:
        acf_values: (n_samples,) mean ACF(1) across all 25 grid points
    """
    n_samples = surfaces.shape[0]
    acf_values = np.zeros(n_samples)

    for i in range(n_samples):
        point_acfs = []
        for tenor in range(5):
            for strike in range(5):
                series = surfaces[i, :, tenor, strike]
                acf = compute_acf_squared_returns(series, max_lag=1)
                point_acfs.append(acf[1])
        acf_values[i] = np.mean(point_acfs)

    return acf_values


def compute_multi_lag_acf_per_path(
    surfaces: np.ndarray,
    grid_point: Tuple[int, int] = (2, 2),
    max_lag: int = 10,
) -> np.ndarray:
    """
    Compute multi-lag ACF curves for each path.

    Args:
        surfaces: (n_samples, T, 5, 5)
        grid_point: (tenor_idx, strike_idx)
        max_lag: maximum lag

    Returns:
        acf_curves: (n_samples, max_lag+1) ACF values
    """
    tenor_idx, strike_idx = grid_point
    n_samples = surfaces.shape[0]
    acf_curves = np.zeros((n_samples, max_lag + 1))

    for i in range(n_samples):
        series = surfaces[i, :, tenor_idx, strike_idx]
        acf_curves[i] = compute_acf_squared_returns(series, max_lag)

    return acf_curves


# =============================================================================
# Mean Reversion Metrics
# =============================================================================

def compute_mean_reversion_speed(series: np.ndarray) -> float:
    """
    Estimate mean reversion speed (kappa) from AR(1) regression.

    Model: X_{t+1} - X_t = -κ(X_t - μ) + ε
    We regress changes on deviations from mean.
    Slope β should be negative for mean reversion.
    κ = -β (mean reversion speed, positive = mean reverting)

    Args:
        series: 1D array of IV values over time

    Returns:
        kappa: Mean reversion speed (positive = mean reverting)
    """
    if len(series) < 3:
        return 0.0

    mean_val = series.mean()
    deviations = series[:-1] - mean_val  # X_t - μ
    changes = np.diff(series)  # X_{t+1} - X_t

    # Handle edge case of constant series
    if np.std(deviations) < 1e-10:
        return 0.0

    # OLS regression: changes = α + β * deviations
    slope, intercept, r_value, p_value, std_err = stats.linregress(deviations, changes)

    return -slope  # κ = -β


def compute_half_life(kappa: float) -> float:
    """
    Compute half-life of mean reversion.

    Half-life is the time for a deviation to decay to 50% of its initial value.
    t_{1/2} = ln(2) / κ

    Args:
        kappa: Mean reversion speed

    Returns:
        Half-life in time units (days)
    """
    if kappa <= 0:
        return np.inf
    return np.log(2) / kappa


def compute_mean_reversion_per_path(
    surfaces: np.ndarray,
    grid_point: Tuple[int, int] = (2, 2),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean reversion speed and half-life for each path.

    Args:
        surfaces: (n_paths, T, 5, 5) array of IV surfaces
        grid_point: (tenor_idx, strike_idx) to analyze

    Returns:
        kappas: (n_paths,) array of mean reversion speeds
        half_lives: (n_paths,) array of half-lives
    """
    tenor_idx, strike_idx = grid_point
    n_paths = surfaces.shape[0]

    kappas = np.zeros(n_paths)
    half_lives = np.zeros(n_paths)

    for i in range(n_paths):
        series = surfaces[i, :, tenor_idx, strike_idx]
        kappas[i] = compute_mean_reversion_speed(series)
        half_lives[i] = compute_half_life(kappas[i])

    return kappas, half_lives


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_vol_clustering_distribution(
    gt_acfs: np.ndarray,
    gen_acfs: np.ndarray,
    output_path: str,
    title_suffix: str = "",
):
    """
    Plot histogram and box plot comparing GT vs Generated ACF distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(-0.5, 0.5, 31)
    ax.hist(gt_acfs, bins=bins, alpha=0.6, label='Ground Truth', color='red', edgecolor='darkred')
    ax.hist(gen_acfs, bins=bins, alpha=0.6, label='Generated', color='blue', edgecolor='darkblue')
    ax.axvline(np.mean(gt_acfs), color='red', linestyle='--', linewidth=2, label=f'GT mean: {np.mean(gt_acfs):.3f}')
    ax.axvline(np.mean(gen_acfs), color='blue', linestyle='--', linewidth=2, label=f'Gen mean: {np.mean(gen_acfs):.3f}')
    ax.axvline(0.05, color='green', linestyle=':', linewidth=2, label='Clustering threshold (0.05)')
    ax.set_xlabel('ACF(1) of Squared Returns')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Volatility Clustering Distribution{title_suffix}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([gt_acfs, gen_acfs], labels=['Ground Truth', 'Generated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax.axhline(0.05, color='green', linestyle=':', linewidth=2, label='Clustering threshold')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.set_ylabel('ACF(1) of Squared Returns')
    ax.set_title(f'Vol Clustering Comparison{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_acf_curves_overlay(
    gt_acf_curves: np.ndarray,
    gen_acf_curves: np.ndarray,
    output_path: str,
    max_lag: int = 10,
):
    """
    Plot multi-lag ACF curves with individual paths overlaid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lags = np.arange(max_lag + 1)

    # GT paths
    ax = axes[0]
    for i in range(min(100, len(gt_acf_curves))):
        ax.plot(lags, gt_acf_curves[i], 'r-', alpha=0.1, linewidth=0.5)
    ax.plot(lags, gt_acf_curves.mean(axis=0), 'r-', linewidth=3, label='Mean')
    ax.fill_between(lags,
                    np.percentile(gt_acf_curves, 5, axis=0),
                    np.percentile(gt_acf_curves, 95, axis=0),
                    alpha=0.3, color='red', label='90% CI')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.axhline(0.05, color='green', linestyle=':', linewidth=1)
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('ACF of Squared Returns')
    ax.set_title('Ground Truth: ACF Curves per Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 0.6)

    # Generated paths
    ax = axes[1]
    for i in range(min(100, len(gen_acf_curves))):
        ax.plot(lags, gen_acf_curves[i], 'b-', alpha=0.1, linewidth=0.5)
    ax.plot(lags, gen_acf_curves.mean(axis=0), 'b-', linewidth=3, label='Mean')
    ax.fill_between(lags,
                    np.percentile(gen_acf_curves, 5, axis=0),
                    np.percentile(gen_acf_curves, 95, axis=0),
                    alpha=0.3, color='blue', label='90% CI')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.axhline(0.05, color='green', linestyle=':', linewidth=1)
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('ACF of Squared Returns')
    ax.set_title('Generated: ACF Curves per Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_acf_comparison_overlay(
    gt_acf_curves: np.ndarray,
    gen_acf_curves: np.ndarray,
    output_path: str,
    max_lag: int = 10,
):
    """
    Plot GT vs Generated mean ACF curves on same axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = np.arange(max_lag + 1)

    # GT
    gt_mean = gt_acf_curves.mean(axis=0)
    gt_p5 = np.percentile(gt_acf_curves, 5, axis=0)
    gt_p95 = np.percentile(gt_acf_curves, 95, axis=0)
    ax.plot(lags, gt_mean, 'r-', linewidth=2, marker='o', label='Ground Truth')
    ax.fill_between(lags, gt_p5, gt_p95, alpha=0.2, color='red')

    # Generated
    gen_mean = gen_acf_curves.mean(axis=0)
    gen_p5 = np.percentile(gen_acf_curves, 5, axis=0)
    gen_p95 = np.percentile(gen_acf_curves, 95, axis=0)
    ax.plot(lags, gen_mean, 'b-', linewidth=2, marker='s', label='Generated')
    ax.fill_between(lags, gen_p5, gen_p95, alpha=0.2, color='blue')

    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.axhline(0.05, color='green', linestyle=':', linewidth=1, label='Clustering threshold')

    ax.set_xlabel('Lag (days)', fontsize=12)
    ax.set_ylabel('ACF of Squared Returns', fontsize=12)
    ax.set_title('Volatility Clustering: GT vs Generated (Mean ± 90% CI)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_lag)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_vol_clustering_visual(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
    output_path: str,
    n_examples: int = 4,
    grid_point: Tuple[int, int] = (2, 2),
):
    """
    Visualize volatility clustering by showing absolute returns over time.
    High vol periods (large bars) cluster together, low vol periods (small bars) cluster together.
    """
    tenor_idx, strike_idx = grid_point

    fig, axes = plt.subplots(n_examples, 2, figsize=(16, 3 * n_examples))

    # Select random example paths
    np.random.seed(42)
    gt_indices = np.random.choice(len(gt_surfaces), n_examples, replace=False)
    gen_indices = np.random.choice(len(gen_surfaces), n_examples, replace=False)

    for row, (gt_idx, gen_idx) in enumerate(zip(gt_indices, gen_indices)):
        # GT path
        gt_series = gt_surfaces[gt_idx, :, tenor_idx, strike_idx]
        gt_returns = np.diff(gt_series)
        gt_abs_returns = np.abs(gt_returns)

        ax = axes[row, 0]
        colors = ['darkred' if r > np.median(gt_abs_returns) else 'lightcoral' for r in gt_abs_returns]
        ax.bar(range(len(gt_abs_returns)), gt_abs_returns, color=colors, edgecolor='none', width=0.8)
        ax.axhline(np.median(gt_abs_returns), color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel('|Return|')
        if row == 0:
            ax.set_title('Ground Truth: Absolute Returns\n(Dark = above median, Light = below)', fontsize=11)
        if row == n_examples - 1:
            ax.set_xlabel('Day')
        ax.set_xlim(-0.5, len(gt_abs_returns) - 0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate ACF(1)
        acf1 = compute_vol_clustering_per_path(gt_surfaces[gt_idx:gt_idx+1], grid_point)[0]
        ax.text(0.98, 0.95, f'ACF(1)={acf1:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Generated path
        gen_series = gen_surfaces[gen_idx, :, tenor_idx, strike_idx]
        gen_returns = np.diff(gen_series)
        gen_abs_returns = np.abs(gen_returns)

        ax = axes[row, 1]
        colors = ['darkblue' if r > np.median(gen_abs_returns) else 'lightblue' for r in gen_abs_returns]
        ax.bar(range(len(gen_abs_returns)), gen_abs_returns, color=colors, edgecolor='none', width=0.8)
        ax.axhline(np.median(gen_abs_returns), color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel('|Return|')
        if row == 0:
            ax.set_title('Generated: Absolute Returns\n(Dark = above median, Light = below)', fontsize=11)
        if row == n_examples - 1:
            ax.set_xlabel('Day')
        ax.set_xlim(-0.5, len(gen_abs_returns) - 0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate ACF(1)
        acf1 = compute_vol_clustering_per_path(gen_surfaces[gen_idx:gen_idx+1], grid_point)[0]
        ax.text(0.98, 0.95, f'ACF(1)={acf1:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.suptitle('Volatility Clustering: Large changes followed by large changes\n(3M ATM IV)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_vol_clustering_heatmap(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
    output_path: str,
    n_examples: int = 8,
    grid_point: Tuple[int, int] = (2, 2),
):
    """
    Heatmap visualization showing squared returns over time for multiple paths.
    Clustering appears as horizontal "stripes" of similar intensity.
    """
    tenor_idx, strike_idx = grid_point

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    np.random.seed(42)
    gt_indices = np.random.choice(len(gt_surfaces), min(n_examples, len(gt_surfaces)), replace=False)
    gen_indices = np.random.choice(len(gen_surfaces), min(n_examples, len(gen_surfaces)), replace=False)

    # GT heatmap
    gt_squared_returns = []
    for idx in gt_indices:
        series = gt_surfaces[idx, :, tenor_idx, strike_idx]
        returns = np.diff(series)
        gt_squared_returns.append(returns ** 2)
    gt_squared_returns = np.array(gt_squared_returns)

    ax = axes[0]
    im = ax.imshow(gt_squared_returns, aspect='auto', cmap='Reds',
                   vmin=0, vmax=np.percentile(gt_squared_returns, 95))
    ax.set_xlabel('Day')
    ax.set_ylabel('Path')
    ax.set_title('Ground Truth: Squared Returns\n(Bright = high vol, Dark = low vol)')
    plt.colorbar(im, ax=ax, label='Squared Return')

    # Gen heatmap
    gen_squared_returns = []
    for idx in gen_indices:
        series = gen_surfaces[idx, :, tenor_idx, strike_idx]
        returns = np.diff(series)
        gen_squared_returns.append(returns ** 2)
    gen_squared_returns = np.array(gen_squared_returns)

    ax = axes[1]
    im = ax.imshow(gen_squared_returns, aspect='auto', cmap='Blues',
                   vmin=0, vmax=np.percentile(gen_squared_returns, 95))
    ax.set_xlabel('Day')
    ax.set_ylabel('Path')
    ax.set_title('Generated: Squared Returns\n(Bright = high vol, Dark = low vol)')
    plt.colorbar(im, ax=ax, label='Squared Return')

    plt.suptitle('Volatility Clustering Heatmap (3M ATM)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_grid_point_acf(
    gen_surfaces: np.ndarray,
    gt_surfaces: np.ndarray,
    output_path: str,
):
    """
    Plot ACF(1) comparison at different grid points (ATM vs corners).
    """
    grid_points = [
        ((0, 0), '1M/90%'),
        ((0, 4), '1M/110%'),
        ((2, 2), '3M/ATM'),
        ((4, 0), '1Y/90%'),
        ((4, 4), '1Y/110%'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, (point, label) in zip(axes, grid_points):
        gt_acfs = compute_vol_clustering_per_path(gt_surfaces, grid_point=point)
        gen_acfs = compute_vol_clustering_per_path(gen_surfaces, grid_point=point)

        # Box plot
        bp = ax.boxplot([gt_acfs, gen_acfs], labels=['GT', 'Gen'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')

        ax.axhline(0.05, color='green', linestyle=':', linewidth=1)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

        gt_mean = np.mean(gt_acfs)
        gen_mean = np.mean(gen_acfs)
        ax.set_title(f'{label}\nGT: {gt_mean:.3f}, Gen: {gen_mean:.3f}')
        ax.set_ylabel('ACF(1)' if ax == axes[0] else '')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 0.5)

    plt.suptitle('Volatility Clustering by Grid Point', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Mean Reversion Visualization Functions
# =============================================================================

def plot_mean_reversion_scatter(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
    output_path: str,
    grid_point: Tuple[int, int] = (2, 2),
):
    """
    Scatter plot showing mean reversion: deviation from mean (x) vs next change (y).
    Negative slope indicates mean reversion.
    """
    tenor_idx, strike_idx = grid_point

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect all deviations and changes for GT
    gt_deviations = []
    gt_changes = []
    for path in gt_surfaces:
        series = path[:, tenor_idx, strike_idx]
        mean_iv = series.mean()
        gt_deviations.extend((series[:-1] - mean_iv).tolist())
        gt_changes.extend(np.diff(series).tolist())

    gt_deviations = np.array(gt_deviations)
    gt_changes = np.array(gt_changes)

    # GT scatter
    ax = axes[0]
    ax.scatter(gt_deviations, gt_changes, alpha=0.3, s=10, color='red', label='Data')

    # Fit regression line
    if len(gt_deviations) > 2 and np.std(gt_deviations) > 1e-10:
        slope, intercept, r, p, se = stats.linregress(gt_deviations, gt_changes)
        x_line = np.array([gt_deviations.min(), gt_deviations.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'Slope={slope:.3f} (κ={-slope:.3f})')

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Deviation from Mean (IV_t - μ)')
    ax.set_ylabel('Next Change (IV_{t+1} - IV_t)')
    ax.set_title('Ground Truth: Mean Reversion\n(Negative slope = mean reverting)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Collect all deviations and changes for Generated
    gen_deviations = []
    gen_changes = []
    for path in gen_surfaces:
        series = path[:, tenor_idx, strike_idx]
        mean_iv = series.mean()
        gen_deviations.extend((series[:-1] - mean_iv).tolist())
        gen_changes.extend(np.diff(series).tolist())

    gen_deviations = np.array(gen_deviations)
    gen_changes = np.array(gen_changes)

    # Gen scatter
    ax = axes[1]
    # Subsample if too many points
    n_plot = min(len(gen_deviations), 5000)
    indices = np.random.choice(len(gen_deviations), n_plot, replace=False)
    ax.scatter(gen_deviations[indices], gen_changes[indices], alpha=0.3, s=10, color='blue', label='Data')

    # Fit regression line
    if len(gen_deviations) > 2 and np.std(gen_deviations) > 1e-10:
        slope, intercept, r, p, se = stats.linregress(gen_deviations, gen_changes)
        x_line = np.array([gen_deviations.min(), gen_deviations.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'Slope={slope:.3f} (κ={-slope:.3f})')

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Deviation from Mean (IV_t - μ)')
    ax.set_ylabel('Next Change (IV_{t+1} - IV_t)')
    ax.set_title('Generated: Mean Reversion\n(Negative slope = mean reverting)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Mean Reversion Analysis (3M ATM)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mean_reversion_paths(
    gt_surfaces: np.ndarray,
    gen_surfaces: np.ndarray,
    output_path: str,
    n_examples: int = 4,
    grid_point: Tuple[int, int] = (2, 2),
):
    """
    Show example paths with their mean as horizontal line.
    Visually demonstrates reversion to mean.
    """
    tenor_idx, strike_idx = grid_point

    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 3 * n_examples))

    np.random.seed(42)
    gt_indices = np.random.choice(len(gt_surfaces), n_examples, replace=False)
    gen_indices = np.random.choice(len(gen_surfaces), n_examples, replace=False)

    for row, (gt_idx, gen_idx) in enumerate(zip(gt_indices, gen_indices)):
        # GT path
        gt_series = gt_surfaces[gt_idx, :, tenor_idx, strike_idx]
        gt_mean = gt_series.mean()
        gt_kappa = compute_mean_reversion_speed(gt_series)
        gt_half_life = compute_half_life(gt_kappa)

        ax = axes[row, 0]
        ax.plot(range(len(gt_series)), gt_series, 'r-', linewidth=2, label='IV Path')
        ax.axhline(gt_mean, color='darkred', linestyle='--', linewidth=1.5, label=f'Mean={gt_mean:.3f}')

        # Shade regions above/below mean
        ax.fill_between(range(len(gt_series)), gt_mean, gt_series,
                        where=gt_series > gt_mean, alpha=0.3, color='red', label='Above mean')
        ax.fill_between(range(len(gt_series)), gt_mean, gt_series,
                        where=gt_series < gt_mean, alpha=0.3, color='green', label='Below mean')

        ax.set_ylabel('IV')
        if row == 0:
            ax.set_title('Ground Truth Paths')
        if row == n_examples - 1:
            ax.set_xlabel('Day')
        ax.grid(True, alpha=0.3)

        # Annotate
        hl_str = f'{gt_half_life:.1f}' if gt_half_life < 100 else '∞'
        ax.text(0.98, 0.95, f'κ={gt_kappa:.3f}\nHL={hl_str}d',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Generated path
        gen_series = gen_surfaces[gen_idx, :, tenor_idx, strike_idx]
        gen_mean = gen_series.mean()
        gen_kappa = compute_mean_reversion_speed(gen_series)
        gen_half_life = compute_half_life(gen_kappa)

        ax = axes[row, 1]
        ax.plot(range(len(gen_series)), gen_series, 'b-', linewidth=2, label='IV Path')
        ax.axhline(gen_mean, color='darkblue', linestyle='--', linewidth=1.5, label=f'Mean={gen_mean:.3f}')

        # Shade regions above/below mean
        ax.fill_between(range(len(gen_series)), gen_mean, gen_series,
                        where=gen_series > gen_mean, alpha=0.3, color='blue', label='Above mean')
        ax.fill_between(range(len(gen_series)), gen_mean, gen_series,
                        where=gen_series < gen_mean, alpha=0.3, color='green', label='Below mean')

        ax.set_ylabel('IV')
        if row == 0:
            ax.set_title('Generated Paths')
        if row == n_examples - 1:
            ax.set_xlabel('Day')
        ax.grid(True, alpha=0.3)

        # Annotate
        hl_str = f'{gen_half_life:.1f}' if gen_half_life < 100 else '∞'
        ax.text(0.98, 0.95, f'κ={gen_kappa:.3f}\nHL={hl_str}d',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.suptitle('Mean Reversion in IV Paths (3M ATM)\nκ = mean reversion speed, HL = half-life', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mean_reversion_speed_comparison(
    gt_kappas: np.ndarray,
    gen_kappas: np.ndarray,
    output_path: str,
):
    """
    Box plot and histogram comparing mean reversion speed (kappa) distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(-1.0, 1.0, 41)
    ax.hist(gt_kappas, bins=bins, alpha=0.6, label='Ground Truth', color='red', edgecolor='darkred')
    ax.hist(gen_kappas, bins=bins, alpha=0.6, label='Generated', color='blue', edgecolor='darkblue')
    ax.axvline(np.mean(gt_kappas), color='red', linestyle='--', linewidth=2,
               label=f'GT mean: {np.mean(gt_kappas):.3f}')
    ax.axvline(np.mean(gen_kappas), color='blue', linestyle='--', linewidth=2,
               label=f'Gen mean: {np.mean(gen_kappas):.3f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Mean Reversion Speed (κ)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of κ\n(κ > 0 = mean reverting)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([gt_kappas, gen_kappas], tick_labels=['Ground Truth', 'Generated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Mean Reversion Speed (κ)')
    ax.set_title('κ Comparison\n(κ > 0 = mean reverting)')
    ax.grid(True, alpha=0.3)

    # Add stats
    gt_pct_pos = (gt_kappas > 0).mean() * 100
    gen_pct_pos = (gen_kappas > 0).mean() * 100
    ax.text(0.98, 0.02, f'GT: {gt_pct_pos:.0f}% with κ>0\nGen: {gen_pct_pos:.0f}% with κ>0',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Mean Reversion Speed Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_half_life_distribution(
    gt_kappas: np.ndarray,
    gen_kappas: np.ndarray,
    output_path: str,
):
    """
    Histogram of half-lives for mean-reverting paths only.
    """
    # Only compute half-life for mean-reverting paths (kappa > 0)
    gt_half_lives = np.array([compute_half_life(k) for k in gt_kappas if k > 0.01])
    gen_half_lives = np.array([compute_half_life(k) for k in gen_kappas if k > 0.01])

    # Cap at reasonable value for visualization
    gt_half_lives = np.clip(gt_half_lives, 0, 50)
    gen_half_lives = np.clip(gen_half_lives, 0, 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, 30, 31)
    ax.hist(gt_half_lives, bins=bins, alpha=0.6, label='Ground Truth', color='red', edgecolor='darkred')
    ax.hist(gen_half_lives, bins=bins, alpha=0.6, label='Generated', color='blue', edgecolor='darkblue')

    if len(gt_half_lives) > 0:
        ax.axvline(np.median(gt_half_lives), color='red', linestyle='--', linewidth=2,
                   label=f'GT median: {np.median(gt_half_lives):.1f}d')
    if len(gen_half_lives) > 0:
        ax.axvline(np.median(gen_half_lives), color='blue', linestyle='--', linewidth=2,
                   label=f'Gen median: {np.median(gen_half_lives):.1f}d')

    ax.set_xlabel('Half-Life (days)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Mean Reversion Half-Life\n(Only paths with κ > 0.01)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    data_to_plot = []
    labels = []
    if len(gt_half_lives) > 0:
        data_to_plot.append(gt_half_lives)
        labels.append('Ground Truth')
    if len(gen_half_lives) > 0:
        data_to_plot.append(gen_half_lives)
        labels.append('Generated')

    if len(data_to_plot) > 0:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)

    ax.set_ylabel('Half-Life (days)')
    ax.set_title('Half-Life Comparison')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Mean Reversion Half-Life Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

def run_temporal_analysis(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    device: str,
    n_samples: int = 50,
    max_batches: int = 20,
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    max_lag: int = 10,
) -> Dict:
    """Run temporal analysis on test data."""

    model.eval()

    # Collect all GT and Generated paths
    all_gt_paths = []  # List of (T, 5, 5) arrays
    all_gen_paths = []  # List of (T, 5, 5) arrays

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating samples")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            # Generate samples (already denormalized)
            samples = model.sample(
                history, n_samples=n_samples,
                sampler=sampler, n_inference_steps=n_inference_steps
            )  # (B, n_samples, T, 5, 5)

            # Denormalize GT
            future_gt = denormalize_iv(future_gt).cpu().numpy()
            samples = samples.cpu().numpy()

            B = samples.shape[0]
            for b in range(B):
                all_gt_paths.append(future_gt[b])
                for s in range(n_samples):
                    all_gen_paths.append(samples[b, s])

    # Convert to arrays
    all_gt_paths = np.array(all_gt_paths)  # (N_gt, T, 5, 5)
    all_gen_paths = np.array(all_gen_paths)  # (N_gen, T, 5, 5)

    print(f"Collected {len(all_gt_paths)} GT paths, {len(all_gen_paths)} generated paths")

    # Compute ACF(1) for ATM grid point
    gt_acf1_atm = compute_vol_clustering_per_path(all_gt_paths, grid_point=(2, 2))
    gen_acf1_atm = compute_vol_clustering_per_path(all_gen_paths, grid_point=(2, 2))

    # Compute ACF(1) averaged across all grid points
    gt_acf1_all = compute_vol_clustering_all_points(all_gt_paths)
    gen_acf1_all = compute_vol_clustering_all_points(all_gen_paths)

    # Compute multi-lag ACF curves
    gt_acf_curves = compute_multi_lag_acf_per_path(all_gt_paths, grid_point=(2, 2), max_lag=max_lag)
    gen_acf_curves = compute_multi_lag_acf_per_path(all_gen_paths, grid_point=(2, 2), max_lag=max_lag)

    # KS test
    ks_stat_atm, ks_pval_atm = stats.ks_2samp(gt_acf1_atm, gen_acf1_atm)
    ks_stat_all, ks_pval_all = stats.ks_2samp(gt_acf1_all, gen_acf1_all)

    results = {
        'vol_clustering_atm': {
            'gt_mean': float(np.mean(gt_acf1_atm)),
            'gt_std': float(np.std(gt_acf1_atm)),
            'gt_median': float(np.median(gt_acf1_atm)),
            'gt_pct_positive': float((gt_acf1_atm > 0.05).mean() * 100),
            'gen_mean': float(np.mean(gen_acf1_atm)),
            'gen_std': float(np.std(gen_acf1_atm)),
            'gen_median': float(np.median(gen_acf1_atm)),
            'gen_pct_positive': float((gen_acf1_atm > 0.05).mean() * 100),
            'ks_statistic': float(ks_stat_atm),
            'ks_pvalue': float(ks_pval_atm),
        },
        'vol_clustering_all_points': {
            'gt_mean': float(np.mean(gt_acf1_all)),
            'gt_std': float(np.std(gt_acf1_all)),
            'gen_mean': float(np.mean(gen_acf1_all)),
            'gen_std': float(np.std(gen_acf1_all)),
            'ks_statistic': float(ks_stat_all),
            'ks_pvalue': float(ks_pval_all),
        },
        'multi_lag_acf': {
            'gt_mean_curve': gt_acf_curves.mean(axis=0).tolist(),
            'gen_mean_curve': gen_acf_curves.mean(axis=0).tolist(),
        },
        'n_gt_paths': len(all_gt_paths),
        'n_gen_paths': len(all_gen_paths),
    }

    # Compute mean reversion metrics
    gt_kappas, gt_half_lives = compute_mean_reversion_per_path(all_gt_paths, grid_point=(2, 2))
    gen_kappas, gen_half_lives = compute_mean_reversion_per_path(all_gen_paths, grid_point=(2, 2))

    # Filter valid half-lives (kappa > 0)
    gt_valid_hl = gt_half_lives[gt_kappas > 0.01]
    gen_valid_hl = gen_half_lives[gen_kappas > 0.01]

    results['mean_reversion'] = {
        'gt_kappa_mean': float(np.mean(gt_kappas)),
        'gt_kappa_std': float(np.std(gt_kappas)),
        'gt_kappa_median': float(np.median(gt_kappas)),
        'gt_pct_mean_reverting': float((gt_kappas > 0).mean() * 100),
        'gen_kappa_mean': float(np.mean(gen_kappas)),
        'gen_kappa_std': float(np.std(gen_kappas)),
        'gen_kappa_median': float(np.median(gen_kappas)),
        'gen_pct_mean_reverting': float((gen_kappas > 0).mean() * 100),
        'gt_half_life_median': float(np.median(gt_valid_hl)) if len(gt_valid_hl) > 0 else None,
        'gen_half_life_median': float(np.median(gen_valid_hl)) if len(gen_valid_hl) > 0 else None,
    }

    # Store arrays for visualization
    results['_gt_acf1_atm'] = gt_acf1_atm
    results['_gen_acf1_atm'] = gen_acf1_atm
    results['_gt_acf_curves'] = gt_acf_curves
    results['_gen_acf_curves'] = gen_acf_curves
    results['_all_gt_paths'] = all_gt_paths
    results['_all_gen_paths'] = all_gen_paths
    results['_gt_kappas'] = gt_kappas
    results['_gen_kappas'] = gen_kappas

    return results


def main():
    parser = argparse.ArgumentParser(description="Temporal feature analysis for DDPM")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/ddpm_poc/checkpoint_epoch_50.pt")
    parser.add_argument("--data_path", type=str,
                        default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/ddpm_poc/temporal_analysis")
    parser.add_argument("--sampler", type=str, default="ddim",
                        choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--max_lag", type=int, default=10)
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
    print("Temporal Feature Analysis for DDPM")
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

    # Run analysis
    print("\nRunning temporal analysis...")
    results = run_temporal_analysis(
        model, test_loader, device,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
        max_lag=args.max_lag,
    )

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_vol_clustering_distribution(
        results['_gt_acf1_atm'],
        results['_gen_acf1_atm'],
        str(output_dir / 'vol_clustering_distribution.png'),
        title_suffix=' (ATM)'
    )

    plot_acf_curves_overlay(
        results['_gt_acf_curves'],
        results['_gen_acf_curves'],
        str(output_dir / 'acf_curves_overlay.png'),
        max_lag=args.max_lag,
    )

    plot_acf_comparison_overlay(
        results['_gt_acf_curves'],
        results['_gen_acf_curves'],
        str(output_dir / 'acf_comparison.png'),
        max_lag=args.max_lag,
    )

    plot_per_grid_point_acf(
        results['_all_gen_paths'],
        results['_all_gt_paths'],
        str(output_dir / 'vol_clustering_by_grid_point.png'),
    )

    plot_vol_clustering_visual(
        results['_all_gt_paths'],
        results['_all_gen_paths'],
        str(output_dir / 'vol_clustering_visual.png'),
        n_examples=4,
    )

    plot_vol_clustering_heatmap(
        results['_all_gt_paths'],
        results['_all_gen_paths'],
        str(output_dir / 'vol_clustering_heatmap.png'),
        n_examples=20,
    )

    # Mean reversion visualizations
    plot_mean_reversion_scatter(
        results['_all_gt_paths'],
        results['_all_gen_paths'],
        str(output_dir / 'mean_reversion_scatter.png'),
    )

    plot_mean_reversion_paths(
        results['_all_gt_paths'],
        results['_all_gen_paths'],
        str(output_dir / 'mean_reversion_paths.png'),
        n_examples=4,
    )

    plot_mean_reversion_speed_comparison(
        results['_gt_kappas'],
        results['_gen_kappas'],
        str(output_dir / 'mean_reversion_speed.png'),
    )

    plot_half_life_distribution(
        results['_gt_kappas'],
        results['_gen_kappas'],
        str(output_dir / 'half_life_distribution.png'),
    )

    # Print summary
    vc = results['vol_clustering_atm']
    print(f"\n{'='*60}")
    print("Volatility Clustering Results (ATM Grid Point)")
    print(f"{'='*60}")
    print(f"  Ground Truth:")
    print(f"    ACF(1) mean:  {vc['gt_mean']:.4f}")
    print(f"    ACF(1) std:   {vc['gt_std']:.4f}")
    print(f"    % > 0.05:     {vc['gt_pct_positive']:.1f}%")
    print(f"  Generated:")
    print(f"    ACF(1) mean:  {vc['gen_mean']:.4f}")
    print(f"    ACF(1) std:   {vc['gen_std']:.4f}")
    print(f"    % > 0.05:     {vc['gen_pct_positive']:.1f}%")
    print(f"  KS Test:")
    print(f"    Statistic:    {vc['ks_statistic']:.4f}")
    print(f"    p-value:      {vc['ks_pvalue']:.4f}")
    print(f"    Match:        {'YES' if vc['ks_pvalue'] > 0.05 else 'NO'} (p > 0.05)")

    # Mean reversion summary
    mr = results['mean_reversion']
    print(f"\n{'='*60}")
    print("Mean Reversion Results (ATM Grid Point)")
    print(f"{'='*60}")
    print(f"  Ground Truth:")
    print(f"    κ mean:       {mr['gt_kappa_mean']:.4f}")
    print(f"    κ median:     {mr['gt_kappa_median']:.4f}")
    print(f"    % κ > 0:      {mr['gt_pct_mean_reverting']:.1f}%")
    if mr['gt_half_life_median'] is not None:
        print(f"    Half-life:    {mr['gt_half_life_median']:.1f} days (median)")
    print(f"  Generated:")
    print(f"    κ mean:       {mr['gen_kappa_mean']:.4f}")
    print(f"    κ median:     {mr['gen_kappa_median']:.4f}")
    print(f"    % κ > 0:      {mr['gen_pct_mean_reverting']:.1f}%")
    if mr['gen_half_life_median'] is not None:
        print(f"    Half-life:    {mr['gen_half_life_median']:.1f} days (median)")

    # Save results (without internal arrays)
    clean_results = {k: v for k, v in results.items() if not k.startswith('_')}

    with open(output_dir / 'temporal_results.json', 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - temporal_results.json")
    print(f"  - vol_clustering_distribution.png")
    print(f"  - acf_curves_overlay.png")
    print(f"  - acf_comparison.png")
    print(f"  - vol_clustering_by_grid_point.png")
    print(f"  - vol_clustering_visual.png")
    print(f"  - vol_clustering_heatmap.png")
    print(f"  - mean_reversion_scatter.png")
    print(f"  - mean_reversion_paths.png")
    print(f"  - mean_reversion_speed.png")
    print(f"  - half_life_distribution.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
