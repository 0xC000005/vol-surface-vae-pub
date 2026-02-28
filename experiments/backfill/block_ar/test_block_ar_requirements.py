#!/usr/bin/env python
"""
Comprehensive validation tests for Block-AR DDPM.

Tests six requirement categories adapted for the Block-AR model:
1. Surface Validity: No explosions, proper term structure, smile convexity
2. CI Coverage: Variation large enough to include ground truth at multiple horizons
3. Conditionality: Conditional samples are tighter/better than unconditional baseline
4. Time Series Properties: ACF, kurtosis preserved
5. Block-AR Specific: Block boundary smoothness, growing uncertainty monotonicity
6. Cointegration: IV-EWMA realized vol relationship preserved (Engle-Granger)

Usage:
    # Quick test (default model, small run)
    python experiments/backfill/block_ar/test_block_ar_requirements.py \
        --n_samples 20 --max_batches 10

    # Full test
    python experiments/backfill/block_ar/test_block_ar_requirements.py \
        --n_samples 50 --max_batches 30

    # Specify model path and device
    python experiments/backfill/block_ar/test_block_ar_requirements.py \
        --model_path models/backfill/block_ar/best_coverage_model.pt \
        --device cuda --n_samples 50 --max_batches 30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# =============================================================================
# Helpers
# =============================================================================

def compute_acf(series: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute autocorrelation function."""
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


def convert_to_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    return obj


# =============================================================================
# Sample Generation
# =============================================================================

def _apply_post_hoc_scale(samples: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply post-hoc multiplicative scaling to ensemble spread.

    samples: (B, n_samples, T, 5, 5) in [0, 1]
    scale: multiplicative factor (1.0 = no change, 1.3 = 30% wider)

    Returns scaled samples clamped to [0, 1].
    """
    if scale == 1.0:
        return samples
    mean = samples.mean(dim=1, keepdim=True)
    return (mean + scale * (samples - mean)).clamp(0.0, 1.0)


def generate_all_samples(
    model: ConditionalBlockARDDPM,
    test_loader: DataLoader,
    n_samples: int,
    max_batches: int,
    max_residual: int,
    device: str,
    max_global_residual: Optional[int] = None,
    post_hoc_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate conditioned samples and ground truth for all batches.

    Returns:
        cond_samples: (N, n_samples, T, 5, 5) denormalized [0, 1]
        ground_truth: (N, T, 5, 5) denormalized [0, 1]
        history: (N, H, 5, 5) denormalized [0, 1]
    """
    all_samples = []
    all_gt = []
    all_history = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Generating samples", total=max_batches)
        ):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            # model.sample_batched() returns (B, n_samples, T, 5, 5) in [0, 1]
            samples = model.sample_batched(
                history, n_samples=n_samples, max_residual=max_residual,
                max_global_residual=max_global_residual,
            )

            if post_hoc_scale != 1.0:
                samples = _apply_post_hoc_scale(samples, post_hoc_scale)

            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_history.append(denormalize_iv(history).cpu().numpy())

    cond_samples = np.concatenate(all_samples, axis=0)
    ground_truth = np.concatenate(all_gt, axis=0)
    history_arr = np.concatenate(all_history, axis=0)
    return cond_samples, ground_truth, history_arr


# =============================================================================
# Test Suite 1: Surface Validity
# =============================================================================

def test_explosion_rate(
    samples: np.ndarray, iv_min: float = 0.0, iv_max: float = 1.0
) -> Dict:
    """Check for explosion (IV values outside valid range).

    Args:
        samples: (N, T, 5, 5) generated surfaces

    Target: < 5%
    """
    high_explosions = (samples > iv_max).any(axis=(1, 2, 3))
    low_explosions = (samples < iv_min).any(axis=(1, 2, 3))
    any_explosion = high_explosions | low_explosions
    total_rate = float(any_explosion.mean())
    return {
        'explosion_high_rate': float(high_explosions.mean()),
        'explosion_low_rate': float(low_explosions.mean()),
        'explosion_total_rate': total_rate,
        'max_iv_observed': float(samples.max()),
        'min_iv_observed': float(samples.min()),
        'pass': total_rate < 0.05,
    }


def test_calendar_arbitrage(samples: np.ndarray) -> Dict:
    """Check calendar spread arbitrage (total variance should increase with tenor).

    Row index = tenor (0=short, 4=long). Total variance = IV^2 * tau.

    Target: avg < 15%, worst strike < 25%
    """
    tenors = np.array([1, 2, 4, 8, 12])
    violations = []
    per_strike_violations = {k: [] for k in range(5)}
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]  # (N, 5, 5)
        total_var = surf ** 2 * tenors[:, None]  # (N, 5, 5)
        for i in range(4):
            violation = (total_var[:, i, :] > total_var[:, i + 1, :] * 1.001)
            violations.append(violation.mean())
            for k in range(5):
                per_strike_violations[k].append(float(violation[:, k].mean()))
    avg_violation_rate = float(np.mean(violations))
    per_strike_rates = [float(np.mean(per_strike_violations[k])) for k in range(5)]
    worst_strike_rate = max(per_strike_rates)
    worst_strike_pass = worst_strike_rate < 0.25
    return {
        'calendar_avg_violation_rate': avg_violation_rate,
        'calendar_max_violation_rate': float(np.max(violations)),
        'per_strike_rates': per_strike_rates,
        'worst_strike_rate': worst_strike_rate,
        'worst_strike_pass': worst_strike_pass,
        'pass': avg_violation_rate < 0.15 and worst_strike_pass,
    }


def test_butterfly_arbitrage(samples: np.ndarray) -> Dict:
    """Check butterfly spread arbitrage (smile should be convex).

    Column index = moneyness (0=ITM, 2=ATM, 4=OTM).
    Second derivative d^2 sigma / dK^2 should be non-negative.

    Target: avg < 40%, worst tenor < 50%
    """
    violations = []
    per_tenor_violations = {r: [] for r in range(5)}
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]  # (N, 5, 5)
        d2_dk2 = surf[:, :, :-2] - 2 * surf[:, :, 1:-1] + surf[:, :, 2:]
        violation = (d2_dk2 < -0.005)  # (N, 5, 3)
        violations.append(float(violation.mean()))
        for r in range(5):
            per_tenor_violations[r].append(float(violation[:, r].mean()))
    avg_violation_rate = float(np.mean(violations))
    per_tenor_rates = [float(np.mean(per_tenor_violations[r])) for r in range(5)]
    worst_tenor_rate = max(per_tenor_rates)
    worst_tenor_pass = worst_tenor_rate < 0.50
    return {
        'butterfly_avg_violation_rate': avg_violation_rate,
        'butterfly_max_violation_rate': float(np.max(violations)),
        'per_tenor_rates': per_tenor_rates,
        'worst_tenor_rate': worst_tenor_rate,
        'worst_tenor_pass': worst_tenor_pass,
        'pass': avg_violation_rate < 0.40 and worst_tenor_pass,
    }


def run_surface_validity_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict:
    """Run all surface validity tests.

    Args:
        cond_samples: (N, n_samples, T, 5, 5)
        ground_truth: (N, T, 5, 5)
    """
    print("\n" + "=" * 60)
    print("TEST SUITE 1: SURFACE VALIDITY")
    print("=" * 60)

    N, S, T, H, W = cond_samples.shape
    all_samples = cond_samples.reshape(N * S, T, H, W)

    print(f"  Total generated trajectories: {all_samples.shape[0]}")
    print(f"  Ground truth windows: {ground_truth.shape[0]}")

    explosion_results = test_explosion_rate(all_samples)
    print(
        f"  Explosion rate: {explosion_results['explosion_total_rate']:.1%} "
        f"(target <5%) {'PASS' if explosion_results['pass'] else 'FAIL'}"
    )

    calendar_results = test_calendar_arbitrage(all_samples)
    print(
        f"  Calendar arbitrage: {calendar_results['calendar_avg_violation_rate']:.1%} "
        f"(target <15%) {'PASS' if calendar_results['pass'] else 'FAIL'}"
    )

    butterfly_results = test_butterfly_arbitrage(all_samples)
    print(
        f"  Butterfly arbitrage: {butterfly_results['butterfly_avg_violation_rate']:.1%} "
        f"(target <40%) {'PASS' if butterfly_results['pass'] else 'FAIL'}"
    )

    # Per-cell breakdown
    print(f"  Calendar worst strike: {calendar_results['worst_strike_rate']:.1%} "
          f"(target <25%) {'PASS' if calendar_results['worst_strike_pass'] else 'FAIL'}")
    print(f"  Butterfly worst tenor: {butterfly_results['worst_tenor_rate']:.1%} "
          f"(target <50%) {'PASS' if butterfly_results['worst_tenor_pass'] else 'FAIL'}")

    overall_pass = all([
        explosion_results['pass'],
        calendar_results['pass'],
        butterfly_results['pass'],
    ])

    return {
        'explosion': explosion_results,
        'calendar': calendar_results,
        'butterfly': butterfly_results,
        'overall_pass': overall_pass,
    }


# =============================================================================
# Test Suite 2: CI Coverage
# =============================================================================

def run_ci_coverage_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    horizons: List[int] = None,
    ci_levels: List[float] = None,
) -> Dict:
    """Compute CI coverage at multiple horizons and confidence levels.

    Per-horizon targets (90% CI):
        h=1  > 80%
        h=7  > 75%
        h=14 > 70%
        h=30 > 65%

    Also computes 50%, 80%, 95% coverage and calibration curve.

    Args:
        cond_samples: (N, n_samples, T, 5, 5)
        ground_truth: (N, T, 5, 5)
    """
    if horizons is None:
        horizons = [1, 7, 14, 30]
    if ci_levels is None:
        ci_levels = [0.5, 0.8, 0.9, 0.95]

    print("\n" + "=" * 60)
    print("TEST SUITE 2: CI COVERAGE")
    print("=" * 60)

    T_fut = ground_truth.shape[1]

    # Overall coverage across all horizons
    overall_coverage = {}
    for level in ci_levels:
        alpha = (1 - level) / 2
        lower = np.quantile(cond_samples, alpha, axis=1)
        upper = np.quantile(cond_samples, 1 - alpha, axis=1)
        covered = (ground_truth >= lower) & (ground_truth <= upper)
        overall_coverage[level] = float(covered.mean())

    # Per-horizon coverage
    horizon_coverage = {h: {} for h in horizons}
    per_cell_coverage = {}  # h -> (5, 5) array
    for h in horizons:
        if h <= T_fut:
            h_idx = h - 1
            samples_h = cond_samples[:, :, h_idx]  # (N, n_samples, 5, 5)
            gt_h = ground_truth[:, h_idx]  # (N, 5, 5)
            for level in ci_levels:
                alpha = (1 - level) / 2
                lower = np.quantile(samples_h, alpha, axis=1)
                upper = np.quantile(samples_h, 1 - alpha, axis=1)
                covered = (gt_h >= lower) & (gt_h <= upper)
                horizon_coverage[h][level] = float(covered.mean())
                if level == 0.9:
                    per_cell_coverage[h] = covered.mean(axis=0)  # (5, 5)

    # Calibration curve
    calibration_levels = np.linspace(0.1, 0.9, 9)
    calibration_nominal = []
    calibration_empirical = []
    for p in calibration_levels:
        alpha = (1 - p) / 2
        lower = np.quantile(cond_samples, alpha, axis=1)
        upper = np.quantile(cond_samples, 1 - alpha, axis=1)
        covered = (ground_truth >= lower) & (ground_truth <= upper)
        calibration_nominal.append(round(float(p), 1))
        calibration_empirical.append(float(covered.mean()))

    calibration_error = float(np.mean(np.abs(
        np.array(calibration_nominal) - np.array(calibration_empirical)
    )))

    # Print overall
    print(f"  Overall 90% CI Coverage: {overall_coverage[0.9]:.1%}")
    for level in ci_levels:
        print(f"    {level:.0%} CI: {overall_coverage[level]:.1%}")

    # Print per-horizon with pass/fail
    horizon_targets = {1: 0.80, 7: 0.75, 14: 0.70, 30: 0.65}
    horizon_pass = {}

    print(f"\n  Per-Horizon Coverage (90% CI):")
    for h in horizons:
        if h in horizon_coverage and 0.9 in horizon_coverage[h]:
            cov = horizon_coverage[h][0.9]
            target = horizon_targets.get(h, 0.65)
            passed = cov > target
            horizon_pass[h] = passed
            print(
                f"    h={h:2d}: {cov:.1%} (target >{target:.0%}) "
                f"{'PASS' if passed else 'FAIL'}"
            )

    # Per-cell worst coverage (90% CI)
    worst_cell_per_horizon = {}
    worst_cell_pass_all = True
    print(f"\n  Per-Cell Worst Coverage (90% CI):")
    for h in horizons:
        if h in per_cell_coverage:
            worst = float(per_cell_coverage[h].min())
            worst_cell_per_horizon[h] = worst
            passed = worst > 0.60
            if not passed:
                worst_cell_pass_all = False
            worst_idx = np.unravel_index(per_cell_coverage[h].argmin(), (5, 5))
            print(
                f"    h={h:2d}: worst cell ({worst_idx[0]},{worst_idx[1]}) = {worst:.1%} "
                f"(target >60%) {'PASS' if passed else 'FAIL'}"
            )

    print(f"  Calibration Error: {calibration_error:.3f}")

    # Pass if all per-horizon targets met AND worst cell passes
    all_horizons_pass = all(horizon_pass.values()) if horizon_pass else False

    return {
        'overall': overall_coverage,
        'per_horizon': horizon_coverage,
        'per_cell_coverage': {
            h: per_cell_coverage[h].tolist() for h in per_cell_coverage
        },
        'worst_cell_per_horizon': worst_cell_per_horizon,
        'worst_cell_pass': worst_cell_pass_all,
        'calibration': {
            'nominal': calibration_nominal,
            'empirical': calibration_empirical,
        },
        'calibration_error': calibration_error,
        'horizon_pass': horizon_pass,
        'pass': all_horizons_pass and worst_cell_pass_all,
    }


# =============================================================================
# Test Suite 3: Conditionality
# =============================================================================

def run_conditionality_tests(
    model: ConditionalBlockARDDPM,
    test_loader: DataLoader,
    n_samples: int = 50,
    max_batches: int = 15,
    max_residual: int = 20,
    device: str = "cpu",
    max_global_residual: Optional[int] = None,
    post_hoc_scale: float = 1.0,
) -> Dict:
    """Test that conditioning on history actually matters.

    Three sub-tests:
    a) Width ratio: conditional CI width / unconditional CI width < 0.95
    b) MAE reduction: conditional MAE < unconditional MAE (>5% reduction)
    c) Growing uncertainty: Var(h=1) < Var(h=10) < Var(h=20) < Var(h=30)

    Unconditional baseline: use zero-history (all zeros in [-1,1] space) which
    produces near-null conditioning, giving a truly unconditional baseline.
    """
    print("\n" + "=" * 60)
    print("TEST SUITE 3: CONDITIONALITY")
    print("=" * 60)

    model.eval()

    cond_widths = []
    uncond_widths = []
    cond_maes = []
    uncond_maes = []
    per_horizon_var = {h: [] for h in [1, 10, 20, 30]}

    # Per-cell accumulators: (5, 5) sums across batches
    cond_width_sum = np.zeros((5, 5))
    uncond_width_sum = np.zeros((5, 5))
    cond_mae_sum = np.zeros((5, 5))
    uncond_mae_sum = np.zeros((5, 5))
    n_cell_samples = 0
    n_uncond_batches = 0

    # Per-horizon accumulators
    cond_horizons = [1, 7, 14, 30]
    per_h_cond_width = {h: [] for h in cond_horizons}
    per_h_uncond_width = {h: [] for h in cond_horizons}
    per_h_cond_mae = {h: [] for h in cond_horizons}
    per_h_uncond_mae = {h: [] for h in cond_horizons}

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Conditionality tests", total=max_batches)
        ):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))
            B = history.shape[0]

            if B < 2:
                continue

            # --- Conditional samples ---
            cond_samples = model.sample_batched(
                history, n_samples=n_samples, max_residual=max_residual,
                max_global_residual=max_global_residual,
            )  # (B, n_samples, T, 5, 5)
            if post_hoc_scale != 1.0:
                cond_samples = _apply_post_hoc_scale(cond_samples, post_hoc_scale)

            # --- Unconditional baseline: zero history (near-null conditioning) ---
            # Only run unconditional for first 5 batches (enough for stable estimate,
            # saves ~60% of Suite 3 runtime since uncond is half the cost per batch)
            MAX_UNCOND_BATCHES = 5
            if batch_idx < MAX_UNCOND_BATCHES:
                zero_history = torch.zeros_like(history)
                uncond_samples = model.sample_batched(
                    zero_history, n_samples=n_samples, max_residual=max_residual,
                    max_global_residual=max_global_residual,
                )  # (B, n_samples, T, 5, 5)
            else:
                uncond_samples = None

            cond_np = cond_samples.cpu().numpy()
            gt_np = future_gt.cpu().numpy()

            # 90% CI width (conditional)
            cond_lower = np.quantile(cond_np, 0.05, axis=1)
            cond_upper = np.quantile(cond_np, 0.95, axis=1)
            cond_widths.append((cond_upper - cond_lower).mean())

            # MAE: median sample vs GT (conditional)
            cond_median = np.median(cond_np, axis=1)
            cond_maes.append(np.abs(cond_median - gt_np).mean())

            # Per-cell conditional accumulators
            cond_cell_width = (cond_upper - cond_lower).mean(axis=(0, 1))  # (5, 5)
            cond_cell_mae = np.abs(cond_median - gt_np).mean(axis=(0, 1))  # (5, 5)
            cond_width_sum += cond_cell_width
            cond_mae_sum += cond_cell_mae
            n_cell_samples += 1

            # Per-horizon conditional width and MAE
            T_batch = cond_np.shape[2]
            for h in cond_horizons:
                if h - 1 < T_batch:
                    t = h - 1
                    per_h_cond_width[h].append(float((cond_upper[:, t] - cond_lower[:, t]).mean()))
                    per_h_cond_mae[h].append(float(np.abs(cond_median[:, t] - gt_np[:, t]).mean()))

            # Unconditional metrics (only for first MAX_UNCOND_BATCHES batches)
            if uncond_samples is not None:
                uncond_np = uncond_samples.cpu().numpy()
                uncond_lower = np.quantile(uncond_np, 0.05, axis=1)
                uncond_upper = np.quantile(uncond_np, 0.95, axis=1)
                uncond_widths.append((uncond_upper - uncond_lower).mean())

                uncond_median = np.median(uncond_np, axis=1)
                uncond_maes.append(np.abs(uncond_median - gt_np).mean())

                uncond_cell_width = (uncond_upper - uncond_lower).mean(axis=(0, 1))
                uncond_cell_mae = np.abs(uncond_median - gt_np).mean(axis=(0, 1))
                uncond_width_sum += uncond_cell_width
                uncond_mae_sum += uncond_cell_mae
                n_uncond_batches += 1

                for h in cond_horizons:
                    if h - 1 < T_batch:
                        t = h - 1
                        per_h_uncond_width[h].append(float((uncond_upper[:, t] - uncond_lower[:, t]).mean()))
                        per_h_uncond_mae[h].append(float(np.abs(uncond_median[:, t] - gt_np[:, t]).mean()))

            # Per-horizon variance (conditional only)
            for h in per_horizon_var:
                if h <= cond_np.shape[2]:
                    var_h = cond_np[:, :, h - 1].var(axis=1).mean()
                    per_horizon_var[h].append(float(var_h))

    # Aggregate
    avg_cond_width = float(np.mean(cond_widths))
    avg_uncond_width = float(np.mean(uncond_widths))
    width_ratio = avg_cond_width / avg_uncond_width if avg_uncond_width > 0 else 1.0

    avg_cond_mae = float(np.mean(cond_maes))
    avg_uncond_mae = float(np.mean(uncond_maes))
    mae_reduction_pct = (
        (avg_uncond_mae - avg_cond_mae) / avg_uncond_mae * 100
        if avg_uncond_mae > 0
        else 0.0
    )

    avg_horizon_var = {
        h: float(np.mean(per_horizon_var[h]))
        for h in per_horizon_var
        if per_horizon_var[h]
    }

    # Sub-test a: width ratio < 0.95
    width_pass = width_ratio < 0.95
    print(
        f"  Width ratio (cond/uncond): {width_ratio:.3f} "
        f"(target <0.95) {'PASS' if width_pass else 'FAIL'}"
    )
    print(f"    Cond width:   {avg_cond_width:.4f}")
    print(f"    Uncond width: {avg_uncond_width:.4f}")

    # Sub-test b: MAE reduction > 5%
    mae_pass = mae_reduction_pct > 5.0
    print(
        f"  MAE reduction: {mae_reduction_pct:.1f}% "
        f"(target >5%) {'PASS' if mae_pass else 'FAIL'}"
    )
    print(f"    Cond MAE:   {avg_cond_mae:.4f}")
    print(f"    Uncond MAE: {avg_uncond_mae:.4f}")

    # Sub-test c: growing uncertainty (monotonic)
    horizon_keys = sorted(avg_horizon_var.keys())
    monotonic = True
    for i in range(len(horizon_keys) - 1):
        if avg_horizon_var[horizon_keys[i]] >= avg_horizon_var[horizon_keys[i + 1]]:
            monotonic = False
            break

    print(
        f"  Growing uncertainty (monotonic variance): "
        f"{'PASS' if monotonic else 'FAIL'} (informational, not gated)"
    )
    for h in horizon_keys:
        print(f"    Var(h={h:2d}): {avg_horizon_var[h]:.6f}")

    # --- Per-horizon conditionality ---
    print("\n  --- Test 3d: Per-Horizon Conditionality ---")
    per_horizon_cond = {}
    for h in cond_horizons:
        if per_h_cond_width[h]:
            h_cond_w = float(np.mean(per_h_cond_width[h]))
            h_uncond_w = float(np.mean(per_h_uncond_width[h]))
            h_wr = h_cond_w / h_uncond_w if h_uncond_w > 0 else 1.0
            h_cond_m = float(np.mean(per_h_cond_mae[h]))
            h_uncond_m = float(np.mean(per_h_uncond_mae[h]))
            h_mae_red = (h_uncond_m - h_cond_m) / h_uncond_m * 100 if h_uncond_m > 0 else 0.0
            per_horizon_cond[h] = {
                'width_ratio': h_wr,
                'mae_reduction_pct': h_mae_red,
                'cond_width': h_cond_w,
                'uncond_width': h_uncond_w,
            }
            print(
                f"    h={h:2d}: width_ratio={h_wr:.3f}, "
                f"MAE_reduction={h_mae_red:.1f}%"
            )

    # --- Per-cell conditionality ---
    print("\n  --- Test 3e: Per-Cell Conditionality ---")
    if n_cell_samples > 0 and n_uncond_batches > 0:
        avg_cond_cell_width = cond_width_sum / n_cell_samples  # (5, 5)
        avg_uncond_cell_width = uncond_width_sum / n_uncond_batches  # (5, 5)
        avg_cond_cell_mae = cond_mae_sum / n_cell_samples  # (5, 5)
        avg_uncond_cell_mae = uncond_mae_sum / n_uncond_batches  # (5, 5)

        # Per-cell width ratio: cond / uncond (< 1.0 means conditioning narrows CI)
        cell_width_ratio = avg_cond_cell_width / np.maximum(avg_uncond_cell_width, 1e-8)
        worst_cell_width_ratio = float(cell_width_ratio.max())
        worst_cell_wr_pass = worst_cell_width_ratio < 3.0

        # Per-cell MAE reduction: (uncond - cond) / uncond * 100
        cell_mae_reduction = np.where(
            avg_uncond_cell_mae > 1e-8,
            (avg_uncond_cell_mae - avg_cond_cell_mae) / avg_uncond_cell_mae * 100,
            0.0,
        )
        worst_cell_mae_reduction = float(cell_mae_reduction.min())
        worst_cell_mae_pass = worst_cell_mae_reduction > -10.0

        print(f"  Per-cell width ratio (cond/uncond):")
        for r in range(5):
            row_str = "    " + " ".join(f"{cell_width_ratio[r,c]:.3f}" for c in range(5))
            print(row_str)
        print(f"    Worst cell: {worst_cell_width_ratio:.3f} "
              f"(target <3.0) {'PASS' if worst_cell_wr_pass else 'FAIL'}")

        print(f"  Per-cell MAE reduction (%):")
        for r in range(5):
            row_str = "    " + " ".join(f"{cell_mae_reduction[r,c]:6.1f}" for c in range(5))
            print(row_str)
        print(f"    Worst cell: {worst_cell_mae_reduction:.1f}% "
              f"(target >-10%) {'PASS' if worst_cell_mae_pass else 'FAIL'}")
    else:
        worst_cell_width_ratio = 1.0
        worst_cell_wr_pass = True
        worst_cell_mae_reduction = 0.0
        worst_cell_mae_pass = True
        cell_width_ratio = np.ones((5, 5))
        cell_mae_reduction = np.zeros((5, 5))

    # NOTE: growing uncertainty disabled from gate — draft feature, not confirmed from data
    overall_pass = width_pass and mae_pass and worst_cell_wr_pass and worst_cell_mae_pass

    return {
        'width_ratio': float(width_ratio),
        'avg_cond_width': avg_cond_width,
        'avg_uncond_width': avg_uncond_width,
        'width_pass': width_pass,
        'mae_reduction_pct': float(mae_reduction_pct),
        'avg_cond_mae': avg_cond_mae,
        'avg_uncond_mae': avg_uncond_mae,
        'mae_pass': mae_pass,
        'per_horizon_var': avg_horizon_var,
        'per_horizon_conditionality': {str(h): v for h, v in per_horizon_cond.items()},
        'growing_uncertainty_monotonic': monotonic,
        'per_cell_width_ratio': cell_width_ratio.tolist(),
        'worst_cell_width_ratio': float(worst_cell_width_ratio),
        'worst_cell_wr_pass': worst_cell_wr_pass,
        'per_cell_mae_reduction': cell_mae_reduction.tolist(),
        'worst_cell_mae_reduction': float(worst_cell_mae_reduction),
        'worst_cell_mae_pass': worst_cell_mae_pass,
        'pass': overall_pass,
    }


# =============================================================================
# Test Suite 4: Time Series Properties
# =============================================================================

def run_time_series_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    max_lag: int = 20,
) -> Dict:
    """Test time series properties: ACF correlation and kurtosis matching.

    ACF correlation target: > 0.5
    Kurtosis ratio target: 0.5 - 2.0

    Args:
        cond_samples: (N, n_samples, T, 5, 5)
        ground_truth: (N, T, 5, 5)
    """
    print("\n" + "=" * 60)
    print("TEST SUITE 4: TIME SERIES PROPERTIES")
    print("=" * 60)

    # --- ACF test ---
    print("\n  --- Test 4a: ACF Preservation ---")

    # ATM series for ACF
    gt_atm = ground_truth[:, :, 2, 2].flatten()
    gen_atm = cond_samples[:, 0, :, 2, 2].flatten()  # first sample per window

    gt_acf = compute_acf(gt_atm, max_lag)
    gen_acf = compute_acf(gen_atm, max_lag)
    acf_correlation = float(np.corrcoef(gt_acf, gen_acf)[0, 1])
    acf_mae = float(np.mean(np.abs(gt_acf - gen_acf)))
    acf_pass = acf_correlation > 0.5

    print(
        f"  ACF correlation: {acf_correlation:.3f} "
        f"(target >0.5) {'PASS' if acf_pass else 'FAIL'}"
    )
    print(f"  ACF MAE: {acf_mae:.4f}")

    # --- Kurtosis test ---
    print("\n  --- Test 4b: Kurtosis Matching ---")

    gt_diff = np.diff(ground_truth, axis=1)  # (N, T-1, 5, 5)
    gen_diff = np.diff(cond_samples[:, 0], axis=1)  # (N, T-1, 5, 5)

    gt_changes = gt_diff.flatten()
    gen_changes = gen_diff.flatten()

    gt_kurt = float(kurtosis(gt_changes, fisher=True))
    gen_kurt = float(kurtosis(gen_changes, fisher=True))
    kurt_ratio = gen_kurt / gt_kurt if gt_kurt != 0 else float("inf")
    kurt_pass = 0.5 <= kurt_ratio <= 2.0

    gt_skew_val = float(skew(gt_changes))
    gen_skew_val = float(skew(gen_changes))
    skew_ratio = gen_skew_val / gt_skew_val if gt_skew_val != 0 else float("inf")
    skew_pass = skew_ratio >= 0.25  # recover at least 25% of GT skewness

    print(f"  GT kurtosis:  {gt_kurt:.3f}")
    print(f"  Gen kurtosis: {gen_kurt:.3f}")
    print(
        f"  Kurtosis ratio: {kurt_ratio:.3f} "
        f"(target 0.5-2.0) {'PASS' if kurt_pass else 'FAIL'}"
    )
    print(f"  GT skewness:  {gt_skew_val:.3f}")
    print(f"  Gen skewness: {gen_skew_val:.3f}")
    print(
        f"  Skewness ratio: {skew_ratio:.3f} "
        f"(target >=0.25) {'PASS' if skew_pass else 'FAIL'}"
    )

    # Per-cell kurtosis
    print("\n  --- Test 4c: Per-Cell Kurtosis ---")
    per_cell_kurt_ratio = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            gt_k = kurtosis(gt_diff[:, :, r, c].flatten(), fisher=True)
            gen_k = kurtosis(gen_diff[:, :, r, c].flatten(), fisher=True)
            per_cell_kurt_ratio[r, c] = gen_k / gt_k if gt_k != 0 else float("inf")
    worst_kurt = float(per_cell_kurt_ratio.min())
    best_kurt = float(per_cell_kurt_ratio.max())
    worst_idx = np.unravel_index(per_cell_kurt_ratio.argmin(), (5, 5))
    best_idx = np.unravel_index(per_cell_kurt_ratio.argmax(), (5, 5))
    print(
        f"  Worst cell ({worst_idx[0]},{worst_idx[1]}): {worst_kurt:.3f}, "
        f"Best cell ({best_idx[0]},{best_idx[1]}): {best_kurt:.3f}"
    )
    print(
        f"  Per-cell kurtosis range: [{worst_kurt:.3f}, {best_kurt:.3f}] (informational)"
    )

    # Per-cell skewness
    print("\n  --- Test 4d: Per-Cell Skewness ---")
    per_cell_skew_ratio = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            gt_s = skew(gt_diff[:, :, r, c].flatten())
            gen_s = skew(gen_diff[:, :, r, c].flatten())
            per_cell_skew_ratio[r, c] = gen_s / gt_s if gt_s != 0 else float("inf")
    worst_skew = float(per_cell_skew_ratio.min())
    best_skew = float(per_cell_skew_ratio.max())
    worst_skew_idx = np.unravel_index(per_cell_skew_ratio.argmin(), (5, 5))
    best_skew_idx = np.unravel_index(per_cell_skew_ratio.argmax(), (5, 5))
    print(
        f"  Worst cell ({worst_skew_idx[0]},{worst_skew_idx[1]}): {worst_skew:.3f}, "
        f"Best cell ({best_skew_idx[0]},{best_skew_idx[1]}): {best_skew:.3f}"
    )
    print(f"  Per-cell skewness grid:")
    for r in range(5):
        row_str = "    " + " ".join(f"{per_cell_skew_ratio[r,c]:6.3f}" for c in range(5))
        print(row_str)
    print(
        f"  Per-cell skewness range: [{worst_skew:.3f}, {best_skew:.3f}] (informational)"
    )

    overall_pass = acf_pass and kurt_pass

    return {
        'acf': {
            'acf_correlation': acf_correlation,
            'acf_mae': acf_mae,
            'gt_acf': gt_acf.tolist(),
            'gen_acf': gen_acf.tolist(),
            'pass': acf_pass,
        },
        'kurtosis': {
            'gt_kurtosis': gt_kurt,
            'gen_kurtosis': gen_kurt,
            'kurtosis_ratio': kurt_ratio,
            'gt_skewness': gt_skew_val,
            'gen_skewness': gen_skew_val,
            'skewness_ratio': skew_ratio,
            'skewness_pass': skew_pass,
            'per_cell_ratio': per_cell_kurt_ratio.tolist(),
            'worst_cell_ratio': worst_kurt,
            'best_cell_ratio': best_kurt,
            'per_cell_skew_ratio': per_cell_skew_ratio.tolist(),
            'worst_cell_skew_ratio': worst_skew,
            'best_cell_skew_ratio': best_skew,
            'pass': kurt_pass,
        },
        'overall_pass': overall_pass,
    }


# =============================================================================
# Test Suite 5: Block-AR Specific
# =============================================================================

def run_block_ar_tests(
    cond_samples: np.ndarray,
    block_size: int = 10,
) -> Dict:
    """Block-AR specific tests.

    a) Block boundary smoothness: the mean absolute frame-to-frame jump at a
       block boundary (e.g. frame 9->10) should be similar to intra-block jumps
       (e.g. frame 8->9). Measured as boundary/interior ratio; target < 2.0.

    b) Growing uncertainty monotonicity: variance across samples should increase
       with horizon. Checked at key horizons h=1, h=10, h=20, h=30.

    Args:
        cond_samples: (N, n_samples, T, 5, 5)
        block_size: frames per block (default 10)
    """
    print("\n" + "=" * 60)
    print("TEST SUITE 5: BLOCK-AR SPECIFIC")
    print("=" * 60)

    N, S, T, H, W = cond_samples.shape
    n_blocks = T // block_size

    # ---- Test 5a: Block Boundary Smoothness ----
    print("\n  --- Test 5a: Block Boundary Smoothness ---")

    # Use first sample per window for boundary analysis
    trajectories = cond_samples[:, 0]  # (N, T, 5, 5)
    diffs = np.abs(np.diff(trajectories, axis=1))  # (N, T-1, 5, 5)
    mean_diff_per_frame = diffs.mean(axis=(0, 2, 3))  # (T-1,)

    # Boundary indices: last diff of each block transition
    # e.g. block_size=10: boundaries at diff indices 9 (frame 9->10) and 19 (frame 19->20)
    boundary_indices = [block_size * i - 1 for i in range(1, n_blocks)]
    boundary_indices = [b for b in boundary_indices if b < T - 1]

    interior_indices = [i for i in range(T - 1) if i not in boundary_indices]

    if boundary_indices:
        boundary_diff = float(mean_diff_per_frame[boundary_indices].mean())
        interior_diff = float(mean_diff_per_frame[interior_indices].mean())
        boundary_ratio = boundary_diff / interior_diff if interior_diff > 0 else 1.0
    else:
        boundary_diff = 0.0
        interior_diff = 0.0
        boundary_ratio = 1.0

    boundary_pass = boundary_ratio < 2.0

    print(f"  Avg boundary jump:    {boundary_diff:.6f}")
    print(f"  Avg intra-block jump: {interior_diff:.6f}")
    print(
        f"  Boundary/Intra ratio: {boundary_ratio:.3f} "
        f"(target <2.0) {'PASS' if boundary_pass else 'FAIL'}"
    )

    # ---- Test 5b: Growing Uncertainty Monotonicity ----
    print("\n  --- Test 5b: Growing Uncertainty Monotonicity ---")

    # Variance across samples at each horizon, averaged over windows and grid points
    var_per_time = cond_samples.var(axis=1).mean(axis=(0, 2, 3))  # (T,)

    key_horizons = [1, 10, 20, 30]
    valid_key_horizons = [h for h in key_horizons if h <= T]
    key_vars = [float(var_per_time[h - 1]) for h in valid_key_horizons]

    monotonic = all(key_vars[i] < key_vars[i + 1] for i in range(len(key_vars) - 1))

    for h, v in zip(valid_key_horizons, key_vars):
        print(f"    Var(h={h:2d}): {v:.6f}")
    print(
        f"  Monotonically increasing: {'PASS' if monotonic else 'FAIL'} (informational, not gated)"
    )

    # NOTE: growing uncertainty disabled from gate — draft feature, not confirmed from data
    overall_pass = boundary_pass

    return {
        'boundary_smoothness': {
            'avg_boundary_jump': boundary_diff,
            'avg_intra_jump': interior_diff,
            'boundary_ratio': float(boundary_ratio),
            'pass': boundary_pass,
        },
        'growing_uncertainty': {
            'key_horizon_vars': {
                h: v for h, v in zip(valid_key_horizons, key_vars)
            },
            'monotonic': monotonic,
            'pass': monotonic,
        },
        'overall_pass': overall_pass,
    }


# =============================================================================
# Test Suite 6: Cointegration (IV ~ EWMA Realized Vol)
# =============================================================================

def run_cointegration_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    returns: np.ndarray,
    test_start: int,
    history_len: int = 30,
    future_len: int = 30,
    ewma_lambda: float = 0.94,
    adf_lags: int = 3,
    adf_alpha: float = 0.10,
) -> Dict:
    """Test IV-EWMA cointegration preservation in generated samples.

    Tests whether generated IV surfaces maintain the Engle-Granger
    cointegration relationship with EWMA realized volatility. This is a
    fundamental economic relationship: IV should track realized vol.

    For each test window, we:
    1. Compute EWMA vol from returns over the future period
    2. Test cointegration between generated median IV and EWMA vol
    3. Compare gen pass rate to GT pass rate

    Args:
        cond_samples: (N, n_samples, T, 5, 5) denormalized [0, 1]
        ground_truth: (N, T, 5, 5) denormalized [0, 1]
        returns: (total_days,) daily returns array
        test_start: starting index in the full surfaces array
        history_len: number of history frames
        future_len: number of future frames
        ewma_lambda: EWMA decay parameter (0.94 = RiskMetrics standard)
        adf_lags: ADF test lags (3 for short sequences)
        adf_alpha: significance level for ADF test
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    print("\n" + "=" * 60)
    print("TEST SUITE 6: IV-EWMA COINTEGRATION")
    print("=" * 60)

    N, S, T, H, W = cond_samples.shape

    # Use median of samples as the generated IV trajectory
    gen_median = np.median(cond_samples, axis=1)  # (N, T, 5, 5)

    def compute_ewma_vol(ret_window, lambda_=ewma_lambda):
        """Compute EWMA vol for a window of returns."""
        n = len(ret_window)
        variance = np.zeros(n)
        variance[0] = ret_window[0] ** 2
        for t in range(1, n):
            variance[t] = lambda_ * variance[t - 1] + (1 - lambda_) * ret_window[t] ** 2
        return np.sqrt(variance * 252)  # annualized

    def test_cointegration(iv_series, ewma_series):
        """Engle-Granger cointegration test: IV ~ EWMA."""
        if len(iv_series) < 10 or np.std(iv_series) < 1e-8 or np.std(ewma_series) < 1e-8:
            return {'cointegrated': False, 'adf_pvalue': 1.0, 'rsquared': 0.0, 'alpha1': 0.0}
        try:
            X = add_constant(ewma_series)
            model = OLS(iv_series, X).fit()
            residuals = model.resid
            adf_result = adfuller(residuals, maxlag=adf_lags, regression='c')
            return {
                'cointegrated': adf_result[1] < adf_alpha,
                'adf_pvalue': float(adf_result[1]),
                'rsquared': float(model.rsquared),
                'alpha1': float(model.params[1]),
            }
        except Exception:
            return {'cointegrated': False, 'adf_pvalue': 1.0, 'rsquared': 0.0, 'alpha1': 0.0}

    # Test cointegration for each window and grid point
    gen_pass_counts = np.zeros((H, W))
    gt_pass_counts = np.zeros((H, W))
    gen_rsq_sums = np.zeros((H, W))
    gt_rsq_sums = np.zeros((H, W))
    n_valid = 0

    for win_idx in range(N):
        # Global index of the future period start
        future_start_global = test_start + win_idx + history_len

        # Check we have returns for this window
        if future_start_global + future_len > len(returns):
            continue

        # EWMA vol for this window's future period
        ret_window = returns[future_start_global:future_start_global + future_len]
        ewma_vol = compute_ewma_vol(ret_window)

        n_valid += 1

        for i in range(H):
            for j in range(W):
                # Generated IV (median across samples)
                gen_iv = gen_median[win_idx, :, i, j]
                gt_iv = ground_truth[win_idx, :, i, j]

                # Test generated
                gen_result = test_cointegration(gen_iv, ewma_vol)
                if gen_result['cointegrated']:
                    gen_pass_counts[i, j] += 1
                gen_rsq_sums[i, j] += gen_result['rsquared']

                # Test ground truth
                gt_result = test_cointegration(gt_iv, ewma_vol)
                if gt_result['cointegrated']:
                    gt_pass_counts[i, j] += 1
                gt_rsq_sums[i, j] += gt_result['rsquared']

    if n_valid == 0:
        print("  WARNING: No valid windows for cointegration test")
        return {'pass': False, 'n_valid': 0}

    gen_pass_rates = gen_pass_counts / n_valid
    gt_pass_rates = gt_pass_counts / n_valid
    gen_mean_rsq = gen_rsq_sums / n_valid
    gt_mean_rsq = gt_rsq_sums / n_valid

    gen_overall_pass_rate = float(gen_pass_rates.mean())
    gt_overall_pass_rate = float(gt_pass_rates.mean())
    gen_overall_rsq = float(gen_mean_rsq.mean())
    gt_overall_rsq = float(gt_mean_rsq.mean())

    # Pass criterion: gen pass rate >= 50% of GT pass rate
    # (30-day sequences have low ADF power, so absolute rates are low)
    ratio = gen_overall_pass_rate / gt_overall_pass_rate if gt_overall_pass_rate > 0 else 0.0
    coint_pass = ratio >= 0.5

    # Per-cell worst: gen/GT ratio per cell, worst cell >= 0.3
    per_cell_ratio = np.where(
        gt_pass_rates > 0,
        gen_pass_rates / gt_pass_rates,
        np.where(gen_pass_rates > 0, np.inf, 1.0),
    )
    worst_cell_ratio = float(per_cell_ratio.min())
    worst_cell_idx = np.unravel_index(per_cell_ratio.argmin(), (H, W))
    worst_cell_pass = worst_cell_ratio >= 0.3
    coint_pass = coint_pass and worst_cell_pass

    print(f"\n  Windows tested: {n_valid}")
    print(f"  GT cointegration pass rate:  {gt_overall_pass_rate:.1%}")
    print(f"  Gen cointegration pass rate: {gen_overall_pass_rate:.1%}")
    print(f"  Gen/GT ratio: {ratio:.3f} (target >=0.50) {'PASS' if ratio >= 0.5 else 'FAIL'}")
    print(f"  Worst cell ({worst_cell_idx[0]},{worst_cell_idx[1]}): "
          f"gen/GT={worst_cell_ratio:.3f} (target >=0.30) "
          f"{'PASS' if worst_cell_pass else 'FAIL'}")
    print(f"  GT mean R²:  {gt_overall_rsq:.4f}")
    print(f"  Gen mean R²: {gen_overall_rsq:.4f}")

    # Per-grid summary
    print(f"\n  Per-grid gen pass rates (%):")
    for i in range(H):
        row = " ".join(f"{gen_pass_rates[i, j]*100:5.1f}" for j in range(W))
        print(f"    [{row}]")

    return {
        'gen_pass_rate': gen_overall_pass_rate,
        'gt_pass_rate': gt_overall_pass_rate,
        'gen_gt_ratio': float(ratio),
        'gen_mean_rsq': gen_overall_rsq,
        'gt_mean_rsq': gt_overall_rsq,
        'gen_pass_rates_grid': gen_pass_rates.tolist(),
        'gt_pass_rates_grid': gt_pass_rates.tolist(),
        'per_cell_ratio_grid': per_cell_ratio.tolist(),
        'worst_cell_ratio': worst_cell_ratio,
        'worst_cell_idx': list(worst_cell_idx),
        'worst_cell_pass': worst_cell_pass,
        'n_valid_windows': n_valid,
        'adf_alpha': adf_alpha,
        'adf_lags': adf_lags,
        'pass': coint_pass,
    }


# =============================================================================
# Test Suite 7: Three-Layer Regime Coverage
# =============================================================================

def run_regime_coverage_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
    horizons: List[int] = None,
) -> Dict:
    """Three-layer regime coverage tests.

    Computes a single boolean tensor covered[w, h, r, c] and derives:
      Layer 1: Per-regime (calm Q20 / turb Q80) per-horizon coverage
      Layer 2: Per-cell coverage within each regime (worst cell gate)
      Layer 3: Catastrophic window-cell detection

    Args:
        cond_samples: (N, n_samples, T, 5, 5)
        ground_truth: (N, T, 5, 5)
        history: (N, H, 5, 5)
        horizons: horizons to test (default [1, 7, 14, 30])
    """
    if horizons is None:
        horizons = [1, 7, 14, 30]

    print("\n" + "=" * 60)
    print("TEST SUITE 7: REGIME COVERAGE (THREE-LAYER)")
    print("=" * 60)

    N, S, T, H, W = cond_samples.shape

    # --- Base boolean tensor: covered[w, h, r, c] at 90% CI ---
    lower = np.quantile(cond_samples, 0.05, axis=1)  # (N, T, 5, 5)
    upper = np.quantile(cond_samples, 0.95, axis=1)  # (N, T, 5, 5)
    covered = (ground_truth >= lower) & (ground_truth <= upper)  # (N, T, 5, 5)
    ci_width = upper - lower  # (N, T, 5, 5)
    median_pred = np.median(cond_samples, axis=1)  # (N, T, 5, 5)

    # --- Regime classification from history vol_of_vol ---
    mean_iv = history.mean(axis=(2, 3))  # (N, hist_len)
    daily_changes = np.diff(mean_iv, axis=1)  # (N, hist_len-1)
    vol_of_vol = daily_changes.std(axis=1)  # (N,)

    q20 = np.quantile(vol_of_vol, 0.20)
    q80 = np.quantile(vol_of_vol, 0.80)
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80
    n_calm = int(calm_mask.sum())
    n_turb = int(turb_mask.sum())

    print(f"  Windows: {N} total, {n_calm} calm (Q20), {n_turb} turb (Q80)")
    print(f"  Vol-of-vol thresholds: Q20={q20:.5f}, Q80={q80:.5f}")

    # =================================================================
    # Layer 1: Per-regime per-horizon coverage + directional bias
    # =================================================================
    print("\n  --- Layer 1: Per-Regime Per-Horizon Coverage ---")

    layer1_results = {}
    layer1_pass = True
    LAYER1_GATE = 0.65

    for regime_name, regime_mask in [("calm", calm_mask), ("turb", turb_mask)]:
        layer1_results[regime_name] = {}
        n_regime = int(regime_mask.sum())
        if n_regime == 0:
            print(f"  WARNING: no {regime_name} windows")
            continue
        for h in horizons:
            h_idx = h - 1
            if h_idx >= T:
                continue
            regime_covered = covered[regime_mask, h_idx]
            cov = float(regime_covered.mean())

            # Directional bias: how does coverage fail?
            regime_gt = ground_truth[regime_mask, h_idx]
            regime_upper = upper[regime_mask, h_idx]
            regime_lower = lower[regime_mask, h_idx]
            regime_median = median_pred[regime_mask, h_idx]
            gt_above_pct = float((regime_gt > regime_upper).mean())
            gt_below_pct = float((regime_gt < regime_lower).mean())
            median_above_gt_pct = float((regime_median > regime_gt).mean())

            layer1_results[regime_name][h] = {
                'coverage': cov,
                'gt_above_pct': gt_above_pct,
                'gt_below_pct': gt_below_pct,
                'median_above_gt_pct': median_above_gt_pct,
            }
            passed = cov > LAYER1_GATE
            if not passed:
                layer1_pass = False
            bias_dir = "UP" if median_above_gt_pct > 0.55 else (
                "DOWN" if median_above_gt_pct < 0.45 else "~0"
            )
            print(
                f"    {regime_name:5s} h={h:2d}: {cov:.1%} "
                f"(target >{LAYER1_GATE:.0%}) {'PASS' if passed else 'FAIL'}  "
                f"bias={bias_dir} (gt>upper={gt_above_pct:.1%}, "
                f"gt<lower={gt_below_pct:.1%})"
            )

    # --- Whole-path directional bias ---
    # For each window: what fraction of the 30 time steps have median below GT?
    # A window with >80% same-sign is "persistently biased" across the trajectory.
    PATH_BIAS_THRESHOLD = 0.80
    print(f"\n  --- Path-Level Directional Bias (>{PATH_BIAS_THRESHOLD:.0%} same-sign) ---")

    path_bias_results = {}
    # median_pred: (N, T, 5, 5), ground_truth: (N, T, 5, 5)
    # Average across cells to get per-window per-timestep scalar
    median_mean = median_pred.mean(axis=(2, 3))  # (N, T)
    gt_mean = ground_truth.mean(axis=(2, 3))  # (N, T)
    median_below_gt = (median_mean < gt_mean)  # (N, T) boolean

    for regime_name, regime_mask in [("calm", calm_mask), ("turb", turb_mask)]:
        n_regime = int(regime_mask.sum())
        if n_regime == 0:
            continue
        regime_below = median_below_gt[regime_mask]  # (n_regime, T)
        frac_below = regime_below.mean(axis=1)  # (n_regime,) fraction of steps with median < GT

        # Persistently biased: median below GT for >80% of steps, or above for >80%
        persistent_low = (frac_below > PATH_BIAS_THRESHOLD).mean()  # CI sits below GT
        persistent_high = ((1 - frac_below) > PATH_BIAS_THRESHOLD).mean()  # CI sits above GT
        mean_frac_below = float(frac_below.mean())

        path_bias_results[regime_name] = {
            'persistent_low_pct': float(persistent_low),
            'persistent_high_pct': float(persistent_high),
            'mean_frac_median_below_gt': mean_frac_below,
        }

        print(
            f"    {regime_name:5s}: {persistent_low:.1%} windows persistently LOW "
            f"(median<GT >{PATH_BIAS_THRESHOLD:.0%}% of steps), "
            f"{persistent_high:.1%} persistently HIGH, "
            f"avg frac_below={mean_frac_below:.1%}"
        )

    # CI width vs vol_of_vol correlation (informational)
    # Measures: does the model produce wider CIs when vol_of_vol is higher?
    # Spearman rank correlation avoids binning artifacts (Q20/Q80 averages wash out signal)
    from scipy.stats import spearmanr
    width_regime_results = {}
    print(f"\n  CI Width ~ Vol-of-Vol Correlation:")
    for h in horizons:
        h_idx = h - 1
        if h_idx >= T:
            continue
        # Per-window mean CI width at this horizon
        per_window_width = ci_width[:, h_idx].mean(axis=(1, 2))  # (N,)
        rho, pval = spearmanr(vol_of_vol, per_window_width)

        # Also report P10 vs P90 width ratio (matches fan chart comparison)
        p10_mask = vol_of_vol <= np.percentile(vol_of_vol, 10)
        p90_mask = vol_of_vol >= np.percentile(vol_of_vol, 90)
        p10_w = float(per_window_width[p10_mask].mean()) if p10_mask.any() else 0
        p90_w = float(per_window_width[p90_mask].mean()) if p90_mask.any() else 0
        p90_p10_ratio = p90_w / p10_w if p10_w > 0 else 1.0

        width_regime_results[h] = {
            'spearman_rho': float(rho),
            'spearman_pval': float(pval),
            'p90_p10_width_ratio': p90_p10_ratio,
            'p10_mean_width': p10_w,
            'p90_mean_width': p90_w,
        }
        print(
            f"    h={h:2d}: Spearman={rho:.3f} (p={pval:.1e}), "
            f"P90/P10 width={p90_p10_ratio:.3f}x "
            f"(P10={p10_w:.4f}, P90={p90_w:.4f})"
        )

    # =================================================================
    # Layer 2: Per-regime per-cell coverage (worst cell gate)
    # =================================================================
    print("\n  --- Layer 2: Per-Regime Per-Cell Worst Coverage ---")

    layer2_results = {}
    layer2_pass = True
    LAYER2_GATE = 0.55

    for regime_name, regime_mask in [("calm", calm_mask), ("turb", turb_mask)]:
        layer2_results[regime_name] = {}
        n_regime = int(regime_mask.sum())
        if n_regime == 0:
            continue
        for h in horizons:
            h_idx = h - 1
            if h_idx >= T:
                continue
            # Per-cell coverage: (5, 5)
            cell_cov = covered[regime_mask, h_idx].mean(axis=0)  # (5, 5)
            worst = float(cell_cov.min())
            worst_idx = np.unravel_index(cell_cov.argmin(), (5, 5))
            layer2_results[regime_name][h] = {
                'grid': cell_cov.tolist(),
                'worst': worst,
                'worst_cell': list(worst_idx),
            }
            passed = worst > LAYER2_GATE
            if not passed:
                layer2_pass = False
            print(
                f"    {regime_name:5s} h={h:2d}: worst ({worst_idx[0]},{worst_idx[1]}) "
                f"= {worst:.1%} (target >{LAYER2_GATE:.0%}) "
                f"{'PASS' if passed else 'FAIL'}"
            )

    # =================================================================
    # Layer 3: Catastrophic window-cell detection
    # =================================================================
    print("\n  --- Layer 3: Catastrophic Window-Cell Detection ---")

    # For each (window, cell): mean coverage across all horizons
    window_cell_cov = covered.mean(axis=1)  # (N, 5, 5) — fraction of horizons covered
    catastrophic = window_cell_cov < 0.30  # (N, 5, 5)
    catastrophic_rate = float(catastrophic.mean())
    n_catastrophic = int(catastrophic.sum())
    total_pairs = N * H * W

    LAYER3_GATE = 0.05
    layer3_pass = catastrophic_rate < LAYER3_GATE

    print(
        f"  Catastrophic (window,cell) pairs: {n_catastrophic}/{total_pairs} "
        f"({catastrophic_rate:.1%})"
    )
    print(
        f"  Gate: < {LAYER3_GATE:.0%} catastrophic "
        f"{'PASS' if layer3_pass else 'FAIL'}"
    )

    # Show worst windows if any catastrophic
    if n_catastrophic > 0:
        # Find windows with most catastrophic cells
        cats_per_window = catastrophic.sum(axis=(1, 2))  # (N,)
        worst_windows = np.argsort(cats_per_window)[-3:][::-1]
        print(f"  Worst windows:")
        for w in worst_windows:
            if cats_per_window[w] > 0:
                bad_cells = list(zip(*np.where(catastrophic[w])))
                print(
                    f"    Window {w}: {cats_per_window[w]} catastrophic cells, "
                    f"vol_of_vol={vol_of_vol[w]:.5f}, "
                    f"cells={bad_cells[:5]}{'...' if len(bad_cells) > 5 else ''}"
                )

    overall_pass = layer1_pass and layer2_pass and layer3_pass
    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

    return {
        'layer1_regime_horizon': layer1_results,
        'layer1_pass': layer1_pass,
        'path_bias': path_bias_results,
        'width_vs_vov': width_regime_results,
        'layer2_regime_cell': layer2_results,
        'layer2_pass': layer2_pass,
        'layer3_catastrophic_rate': catastrophic_rate,
        'layer3_n_catastrophic': n_catastrophic,
        'layer3_pass': layer3_pass,
        'n_calm': n_calm,
        'n_turb': n_turb,
        'vol_of_vol_q20': float(q20),
        'vol_of_vol_q80': float(q80),
        'overall_pass': overall_pass,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_calibration_curve(results: Dict, output_path: Optional[str] = None):
    """Plot CI calibration curve."""
    plt.figure(figsize=(8, 6))
    nominal = results['calibration']['nominal']
    empirical = results['calibration']['empirical']
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    plt.plot(nominal, empirical, 'bo-', label='Block-AR', markersize=8, linewidth=2)
    plt.xlabel('Nominal Coverage', fontsize=12)
    plt.ylabel('Empirical Coverage', fontsize=12)
    plt.title('CI Calibration Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    cal_error = results['calibration_error']
    plt.text(0.05, 0.85, f'Calibration Error: {cal_error:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved calibration plot to {output_path}")
    plt.close()


def plot_acf_comparison(acf_results: Dict, output_path: Optional[str] = None):
    """Plot ACF comparison between GT and generated samples."""
    plt.figure(figsize=(10, 5))
    lags = np.arange(len(acf_results['gt_acf']))
    width = 0.35
    plt.bar(lags - width / 2, acf_results['gt_acf'], width,
            label='Ground Truth', alpha=0.7)
    plt.bar(lags + width / 2, acf_results['gen_acf'], width,
            label='Generated', alpha=0.7)
    plt.xlabel('Lag (days)', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.title(
        f'ACF Comparison - ATM (Correlation: {acf_results["acf_correlation"]:.3f})',
        fontsize=14,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved ACF comparison to {output_path}")
    plt.close()


def visualize_generated_paths(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    idx: int = 0,
    output_path: Optional[str] = None,
):
    """Visualize multiple generated trajectories vs ground truth for one window."""
    sample_gt = ground_truth[idx]  # (T, 5, 5)
    sample_paths = cond_samples[idx]  # (n_samples, T, 5, 5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    grid_points = [
        ((0, 2), 'Short-term ATM'),
        ((0, 0), 'Short-term ITM'),
        ((0, 4), 'Short-term OTM'),
        ((4, 2), 'Long-term ATM'),
        ((4, 0), 'Long-term ITM'),
        ((4, 4), 'Long-term OTM'),
    ]
    for ax, ((tenor, strike), title) in zip(axes.flat, grid_points):
        for i in range(min(20, sample_paths.shape[0])):
            ax.plot(sample_paths[i, :, tenor, strike],
                    alpha=0.3, color='blue', linewidth=0.5)
        ax.plot(sample_gt[:, tenor, strike],
                color='red', linewidth=2, label='Ground Truth')
        p5 = np.percentile(sample_paths[:, :, tenor, strike], 5, axis=0)
        p95 = np.percentile(sample_paths[:, :, tenor, strike], 95, axis=0)
        ax.fill_between(range(len(p5)), p5, p95,
                        alpha=0.2, color='blue', label='90% CI')
        ax.set_title(title)
        ax.set_xlabel('Days ahead')
        ax.set_ylabel('IV')
        if ax == axes[0, 0]:
            ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved path visualization to {output_path}")
    plt.close()
    return fig


def plot_growing_uncertainty(
    cond_samples: np.ndarray, output_path: Optional[str] = None
):
    """Plot variance across samples vs horizon."""
    var_per_time = cond_samples.var(axis=1).mean(axis=(0, 2, 3))  # (T,)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(var_per_time) + 1), var_per_time, 'b-o', markersize=3)
    plt.xlabel('Horizon (days)', fontsize=12)
    plt.ylabel('Mean Variance across Samples', fontsize=12)
    plt.title('Uncertainty Growth with Horizon', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved uncertainty growth plot to {output_path}")
    plt.close()


# =============================================================================
# Summary
# =============================================================================

def print_summary(results: Dict) -> bool:
    """Print overall pass/fail summary. Returns True if all tests pass."""
    print("\n" + "=" * 60)
    print("SUMMARY - PASS/FAIL")
    print("=" * 60)

    # Test Suite 1: Surface Validity
    s = results['surface']
    print("\nTest Suite 1: Surface Validity")
    print(f"  Explosion rate:        {s['explosion']['explosion_total_rate']:.1%} "
          f"{'PASS' if s['explosion']['pass'] else 'FAIL'}")
    print(f"  Calendar arbitrage:    {s['calendar']['calendar_avg_violation_rate']:.1%} "
          f"(worst strike: {s['calendar']['worst_strike_rate']:.1%}) "
          f"{'PASS' if s['calendar']['pass'] else 'FAIL'}")
    print(f"  Butterfly arbitrage:   {s['butterfly']['butterfly_avg_violation_rate']:.1%} "
          f"(worst tenor: {s['butterfly']['worst_tenor_rate']:.1%}) "
          f"{'PASS' if s['butterfly']['pass'] else 'FAIL'}")
    print(f"  Overall:               {'PASS' if s['overall_pass'] else 'FAIL'}")

    # Test Suite 2: CI Coverage
    c = results['coverage']
    print("\nTest Suite 2: CI Coverage")
    print(f"  Overall 90% CI:        {c['overall'][0.9]:.1%}")
    for h, passed in c.get('horizon_pass', {}).items():
        cov = c['per_horizon'].get(h, {}).get(0.9, 0.0)
        worst = c.get('worst_cell_per_horizon', {}).get(h, 0.0)
        print(f"    h={h:2d}: {cov:.1%} (worst cell: {worst:.1%}) {'PASS' if passed else 'FAIL'}")
    print(f"  Worst cell pass:       {'PASS' if c.get('worst_cell_pass', True) else 'FAIL'}")
    print(f"  Calibration error:     {c['calibration_error']:.3f}")
    print(f"  Overall:               {'PASS' if c['pass'] else 'FAIL'}")

    # Test Suite 3: Conditionality
    d = results['conditionality']
    print("\nTest Suite 3: Conditionality")
    print(f"  Width ratio:         {d['width_ratio']:.3f} "
          f"{'PASS' if d['width_pass'] else 'FAIL'}")
    print(f"  MAE reduction:       {d['mae_reduction_pct']:.1f}% "
          f"{'PASS' if d['mae_pass'] else 'FAIL'}")
    print(f"  Growing uncertainty: "
          f"{'PASS' if d['growing_uncertainty_monotonic'] else 'FAIL'}")
    print(f"  Worst cell width:    {d.get('worst_cell_width_ratio', 0):.3f} "
          f"{'PASS' if d.get('worst_cell_wr_pass', True) else 'FAIL'}")
    print(f"  Worst cell MAE red:  {d.get('worst_cell_mae_reduction', 0):.1f}% "
          f"{'PASS' if d.get('worst_cell_mae_pass', True) else 'FAIL'}")
    print(f"  Overall:             {'PASS' if d['pass'] else 'FAIL'}")

    # Test Suite 4: Time Series
    ts = results['time_series']
    print("\nTest Suite 4: Time Series Properties")
    print(f"  ACF correlation:     {ts['acf']['acf_correlation']:.3f} "
          f"{'PASS' if ts['acf']['pass'] else 'FAIL'}")
    print(f"  Kurtosis ratio:      {ts['kurtosis']['kurtosis_ratio']:.3f} "
          f"(per-cell: [{ts['kurtosis'].get('worst_cell_ratio', 0):.3f}, "
          f"{ts['kurtosis'].get('best_cell_ratio', 0):.3f}]) "
          f"{'PASS' if ts['kurtosis']['pass'] else 'FAIL'}")
    print(f"  Overall:             {'PASS' if ts['overall_pass'] else 'FAIL'}")

    # Test Suite 5: Block-AR Specific
    ba = results['block_ar']
    print("\nTest Suite 5: Block-AR Specific")
    print(f"  Boundary smoothness: {ba['boundary_smoothness']['boundary_ratio']:.3f} "
          f"{'PASS' if ba['boundary_smoothness']['pass'] else 'FAIL'}")
    print(f"  Growing uncertainty: "
          f"{'PASS' if ba['growing_uncertainty']['pass'] else 'FAIL'}")
    print(f"  Overall:             {'PASS' if ba['overall_pass'] else 'FAIL'}")

    # Test Suite 6: Cointegration (if available)
    if 'cointegration' in results:
        co = results['cointegration']
        print("\nTest Suite 6: IV-EWMA Cointegration")
        print(f"  Gen pass rate:       {co['gen_pass_rate']:.1%}")
        print(f"  GT pass rate:        {co['gt_pass_rate']:.1%}")
        print(f"  Gen/GT ratio:        {co['gen_gt_ratio']:.3f} "
              f"(worst cell: {co.get('worst_cell_ratio', 0):.3f}) "
              f"{'PASS' if co['pass'] else 'FAIL'}")
        print(f"  Gen mean R²:         {co['gen_mean_rsq']:.4f}")

    # Test Suite 7: Regime Coverage
    if 'regime_coverage' in results:
        rc = results['regime_coverage']
        print("\nTest Suite 7: Regime Coverage (Three-Layer)")
        print(f"  Layer 1 (regime×horizon): {'PASS' if rc['layer1_pass'] else 'FAIL'}")
        print(f"  Layer 2 (regime×cell):    {'PASS' if rc['layer2_pass'] else 'FAIL'}")
        print(f"  Layer 3 (catastrophic):   {rc['layer3_catastrophic_rate']:.1%} "
              f"{'PASS' if rc['layer3_pass'] else 'FAIL'}")
        print(f"  Overall:                  {'PASS' if rc['overall_pass'] else 'FAIL'}")

    # Overall
    print("\n" + "=" * 60)
    all_pass = all([
        s['overall_pass'],
        c['pass'],
        d['pass'],
        ts['overall_pass'],
        ba['overall_pass'],
    ])
    if 'regime_coverage' in results:
        all_pass = all_pass and results['regime_coverage']['overall_pass']
    # Cointegration is informational — doesn't affect overall pass/fail yet
    if 'cointegration' in results and not results['cointegration']['pass']:
        print("  NOTE: Cointegration test FAILED (informational)")
    if all_pass:
        print("OVERALL: ALL TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED")
    print("=" * 60)

    return all_pass


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Block-AR DDPM Validation Tests"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained Block-AR model checkpoint",
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of samples per history for CI / conditionality tests",
    )
    parser.add_argument(
        "--max_batches", type=int, default=20,
        help="Maximum number of test batches to evaluate",
    )
    parser.add_argument(
        "--max_residual", type=int, default=20,
        help="Max residual timestep for staggered DDPM sampling",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda / cpu)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results (default: results/block_ar/validation_tests)",
    )
    parser.add_argument(
        "--no_ema", action="store_true",
        help="Use regular model weights instead of EMA parameters",
    )
    parser.add_argument(
        "--max_global_residual", type=int, default=None,
        help="Global horizon-dependent residual noise. "
             "Frame h stops at t_min=mgr*h/(T-1). 0=off, 10=recommended.",
    )
    parser.add_argument(
        "--sampling_mode", type=str, default=None, choices=["pyramid", "uniform"],
        help="Override checkpoint's sampling mode for inference (pyramid or uniform)",
    )
    parser.add_argument(
        "--no_clamp_output", action="store_true",
        help="Disable output clamping to [0,1] (overrides checkpoint config)",
    )
    parser.add_argument(
        "--post_hoc_scale", type=float, default=1.0,
        help="Post-hoc multiplicative scaling of ensemble spread (1.0=off, 1.3=30%% wider)",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=None,
        help="Override CFG guidance scale at inference (None=use checkpoint config)",
    )
    args = parser.parse_args()

    config = get_default_config()

    # Device
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
        print(f"No trained model found at {model_path}. Run training first:")
        print("  python experiments/backfill/block_ar/train_block_ar.py")
        return

    # Output directory
    output_dir = args.output_dir or f"{config.results_dir}/validation_tests"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Header
    print("=" * 60)
    print("Block-AR DDPM Validation Tests")
    print("=" * 60)
    print(f"Model:         {model_path}")
    print(f"Device:        {device}")
    print(f"Samples/hist:  {args.n_samples}")
    print(f"Max batches:   {args.max_batches}")
    print(f"Max residual:  {args.max_residual}")
    if args.sampling_mode:
        print(f"Sampling mode: {args.sampling_mode} (override)")
    if args.post_hoc_scale != 1.0:
        print(f"Post-hoc scale: {args.post_hoc_scale}")
    print(f"Output:        {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]

    # Support both BlockARConfig instance and dict
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)

    # Override sampling mode if requested (allows testing same weights with different inference)
    if args.sampling_mode is not None:
        model_config.sampling_mode = args.sampling_mode
        print(f"  Sampling mode override: {args.sampling_mode}")

    # Override clamp_output if requested
    if args.no_clamp_output:
        model_config.clamp_output = False
        print("  Output clamping: DISABLED (override)")

    # Override guidance scale if requested
    if args.guidance_scale is not None:
        model_config.guidance_scale = args.guidance_scale
        print(f"  Guidance scale override: {args.guidance_scale}")

    model = ConditionalBlockARDDPM(model_config)

    # Load EMA params if available (and not disabled), otherwise regular state dict
    if "ema_params" in checkpoint and not args.no_ema:
        print("  Loading EMA parameters...")
        state_dict = model.state_dict()
        for name in state_dict:
            if name in checkpoint["ema_params"]:
                state_dict[name] = checkpoint["ema_params"][name]
        model.load_state_dict(state_dict)
    else:
        print("  Loading regular model weights...")
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Block size: {model_config.block_size}, "
          f"Future len: {model_config.future_len}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Load test data
    print("\nLoading test data...")
    data = np.load(config.data_path)
    surfaces = data["surface"]
    returns = data["ret"] if "ret" in data else None

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

    print(f"  Test set: {len(test_dataset)} windows")

    # =========================================================================
    # Generate samples (shared across test suites 1, 2, 4, 5, 7)
    # =========================================================================
    print("\nGenerating samples for validation tests...")
    cond_samples, ground_truth, history_arr = generate_all_samples(
        model, test_loader,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        max_residual=args.max_residual,
        device=device,
        max_global_residual=args.max_global_residual,
        post_hoc_scale=args.post_hoc_scale,
    )
    print(f"  Conditioned samples: {cond_samples.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    print(f"  History: {history_arr.shape}")

    # =========================================================================
    # Run all test suites
    # =========================================================================
    results = {}

    # Test Suite 1: Surface Validity
    results['surface'] = run_surface_validity_tests(cond_samples, ground_truth)

    # Test Suite 2: CI Coverage
    results['coverage'] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Test Suite 3: Conditionality (needs fresh data loader iteration + shuffled)
    cond_test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
    )
    results['conditionality'] = run_conditionality_tests(
        model, cond_test_loader,
        n_samples=args.n_samples,
        max_batches=min(args.max_batches, 15),
        max_residual=args.max_residual,
        device=device,
        max_global_residual=args.max_global_residual,
        post_hoc_scale=args.post_hoc_scale,
    )

    # Test Suite 4: Time Series Properties
    results['time_series'] = run_time_series_tests(cond_samples, ground_truth)

    # Test Suite 5: Block-AR Specific
    results['block_ar'] = run_block_ar_tests(
        cond_samples,
        block_size=model_config.block_size,
    )

    # Test Suite 6: Cointegration (IV ~ EWMA Realized Vol)
    if returns is not None:
        results['cointegration'] = run_cointegration_tests(
            cond_samples, ground_truth,
            returns=returns,
            test_start=config.test_start,
            history_len=config.history_len,
            future_len=config.future_len,
        )
    else:
        print("\n  Skipping cointegration test (no returns data)")

    # Test Suite 7: Three-Layer Regime Coverage
    results['regime_coverage'] = run_regime_coverage_tests(
        cond_samples, ground_truth, history_arr,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    all_pass = print_summary(results)

    # =========================================================================
    # Visualizations
    # =========================================================================
    print("\nGenerating visualizations...")
    plot_calibration_curve(
        results['coverage'], f"{output_dir}/calibration_curve.png"
    )
    plot_acf_comparison(
        results['time_series']['acf'], f"{output_dir}/acf_comparison.png"
    )
    visualize_generated_paths(
        cond_samples, ground_truth, idx=0,
        output_path=f"{output_dir}/path_visualization.png",
    )
    plot_growing_uncertainty(
        cond_samples, f"{output_dir}/uncertainty_growth.png"
    )

    # =========================================================================
    # Eval provenance — record everything needed to reproduce this evaluation
    # =========================================================================
    import hashlib, dataclasses
    config_dict = dataclasses.asdict(model_config)
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]

    results['eval_config'] = {
        'checkpoint_path': str(model_path),
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'n_samples': args.n_samples,
        'max_batches': args.max_batches,
        'max_residual': args.max_residual,
        'max_global_residual': args.max_global_residual,
        'sampling_mode': model_config.sampling_mode,
        'clamp_output': model_config.clamp_output,
        'use_ema': ("ema_params" in checkpoint and not args.no_ema),
        'forward_only': model_config.forward_only,
        'post_hoc_scale': args.post_hoc_scale,
        'model_config_hash': config_hash,
        'model_config': config_dict,
    }

    # =========================================================================
    # Save results JSON
    # =========================================================================
    results_serializable = convert_to_serializable(results)
    json_path = f"{output_dir}/summary.json"
    with open(json_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
