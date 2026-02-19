#!/usr/bin/env python
"""
Comprehensive validation tests for Block-AR DDPM.

Tests five requirement categories adapted for the Block-AR model:
1. Surface Validity: No explosions, proper term structure, smile convexity
2. CI Coverage: Variation large enough to include ground truth at multiple horizons
3. Conditionality: Conditional samples are tighter/better than unconditional baseline
4. Time Series Properties: ACF, kurtosis preserved
5. Block-AR Specific: Block boundary smoothness, growing uncertainty monotonicity

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

def generate_all_samples(
    model: ConditionalBlockARDDPM,
    test_loader: DataLoader,
    n_samples: int,
    max_batches: int,
    max_residual: int,
    device: str,
    max_global_residual: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate conditioned samples and ground truth for all batches.

    Returns:
        cond_samples: (N, n_samples, T, 5, 5) denormalized [0, 1]
        ground_truth: (N, T, 5, 5) denormalized [0, 1]
    """
    all_samples = []
    all_gt = []

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

            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())

    cond_samples = np.concatenate(all_samples, axis=0)
    ground_truth = np.concatenate(all_gt, axis=0)
    return cond_samples, ground_truth


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

    Target: < 15% (GT data floor is ~7% full / ~10% val set)
    """
    tenors = np.array([1, 2, 4, 8, 12])
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]  # (N, 5, 5)
        total_var = surf ** 2 * tenors[:, None]  # (N, 5, 5)
        for i in range(4):
            violation = (total_var[:, i, :] > total_var[:, i + 1, :] * 1.001)
            violations.append(violation.mean())
    avg_violation_rate = float(np.mean(violations))
    return {
        'calendar_avg_violation_rate': avg_violation_rate,
        'calendar_max_violation_rate': float(np.max(violations)),
        'pass': avg_violation_rate < 0.15,
    }


def test_butterfly_arbitrage(samples: np.ndarray) -> Dict:
    """Check butterfly spread arbitrage (smile should be convex).

    Column index = moneyness (0=ITM, 2=ATM, 4=OTM).
    Second derivative d^2 sigma / dK^2 should be non-negative.

    Target: < 40% (GT data floor is ~20% full / ~23% val set)
    """
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]  # (N, 5, 5)
        d2_dk2 = surf[:, :, :-2] - 2 * surf[:, :, 1:-1] + surf[:, :, 2:]
        violation = (d2_dk2 < -0.005).mean()
        violations.append(float(violation))
    avg_violation_rate = float(np.mean(violations))
    return {
        'butterfly_avg_violation_rate': avg_violation_rate,
        'butterfly_max_violation_rate': float(np.max(violations)),
        'pass': avg_violation_rate < 0.40,
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

    print(f"  Calibration Error: {calibration_error:.3f}")

    # Pass if all per-horizon targets met
    all_horizons_pass = all(horizon_pass.values()) if horizon_pass else False

    return {
        'overall': overall_coverage,
        'per_horizon': horizon_coverage,
        'calibration': {
            'nominal': calibration_nominal,
            'empirical': calibration_empirical,
        },
        'calibration_error': calibration_error,
        'horizon_pass': horizon_pass,
        'pass': all_horizons_pass,
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

            # --- Unconditional baseline: zero history (near-null conditioning) ---
            zero_history = torch.zeros_like(history)
            uncond_samples = model.sample_batched(
                zero_history, n_samples=n_samples, max_residual=max_residual,
                max_global_residual=max_global_residual,
            )  # (B, n_samples, T, 5, 5)

            cond_np = cond_samples.cpu().numpy()
            uncond_np = uncond_samples.cpu().numpy()
            gt_np = future_gt.cpu().numpy()

            # 90% CI width
            cond_lower = np.quantile(cond_np, 0.05, axis=1)
            cond_upper = np.quantile(cond_np, 0.95, axis=1)
            cond_widths.append((cond_upper - cond_lower).mean())

            uncond_lower = np.quantile(uncond_np, 0.05, axis=1)
            uncond_upper = np.quantile(uncond_np, 0.95, axis=1)
            uncond_widths.append((uncond_upper - uncond_lower).mean())

            # MAE: median sample vs GT
            cond_median = np.median(cond_np, axis=1)
            uncond_median = np.median(uncond_np, axis=1)
            cond_maes.append(np.abs(cond_median - gt_np).mean())
            uncond_maes.append(np.abs(uncond_median - gt_np).mean())

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
        f"{'PASS' if monotonic else 'FAIL'}"
    )
    for h in horizon_keys:
        print(f"    Var(h={h:2d}): {avg_horizon_var[h]:.6f}")

    overall_pass = width_pass and mae_pass and monotonic

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
        'growing_uncertainty_monotonic': monotonic,
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

    print(f"  GT kurtosis:  {gt_kurt:.3f}")
    print(f"  Gen kurtosis: {gen_kurt:.3f}")
    print(
        f"  Kurtosis ratio: {kurt_ratio:.3f} "
        f"(target 0.5-2.0) {'PASS' if kurt_pass else 'FAIL'}"
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
        f"  Monotonically increasing: {'PASS' if monotonic else 'FAIL'}"
    )

    overall_pass = boundary_pass and monotonic

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
    print(f"  Explosion rate:      {s['explosion']['explosion_total_rate']:.1%} "
          f"{'PASS' if s['explosion']['pass'] else 'FAIL'}")
    print(f"  Calendar arbitrage:  {s['calendar']['calendar_avg_violation_rate']:.1%} "
          f"{'PASS' if s['calendar']['pass'] else 'FAIL'}")
    print(f"  Butterfly arbitrage: {s['butterfly']['butterfly_avg_violation_rate']:.1%} "
          f"{'PASS' if s['butterfly']['pass'] else 'FAIL'}")
    print(f"  Overall:             {'PASS' if s['overall_pass'] else 'FAIL'}")

    # Test Suite 2: CI Coverage
    c = results['coverage']
    print("\nTest Suite 2: CI Coverage")
    print(f"  Overall 90% CI:      {c['overall'][0.9]:.1%}")
    for h, passed in c.get('horizon_pass', {}).items():
        cov = c['per_horizon'].get(h, {}).get(0.9, 0.0)
        print(f"    h={h:2d}: {cov:.1%} {'PASS' if passed else 'FAIL'}")
    print(f"  Calibration error:   {c['calibration_error']:.3f}")
    print(f"  Overall:             {'PASS' if c['pass'] else 'FAIL'}")

    # Test Suite 3: Conditionality
    d = results['conditionality']
    print("\nTest Suite 3: Conditionality")
    print(f"  Width ratio:         {d['width_ratio']:.3f} "
          f"{'PASS' if d['width_pass'] else 'FAIL'}")
    print(f"  MAE reduction:       {d['mae_reduction_pct']:.1f}% "
          f"{'PASS' if d['mae_pass'] else 'FAIL'}")
    print(f"  Growing uncertainty: "
          f"{'PASS' if d['growing_uncertainty_monotonic'] else 'FAIL'}")
    print(f"  Overall:             {'PASS' if d['pass'] else 'FAIL'}")

    # Test Suite 4: Time Series
    ts = results['time_series']
    print("\nTest Suite 4: Time Series Properties")
    print(f"  ACF correlation:     {ts['acf']['acf_correlation']:.3f} "
          f"{'PASS' if ts['acf']['pass'] else 'FAIL'}")
    print(f"  Kurtosis ratio:      {ts['kurtosis']['kurtosis_ratio']:.3f} "
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

    # Overall
    print("\n" + "=" * 60)
    all_pass = all([
        s['overall_pass'],
        c['pass'],
        d['pass'],
        ts['overall_pass'],
        ba['overall_pass'],
    ])
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
    # Generate samples (shared across test suites 1, 2, 4, 5)
    # =========================================================================
    print("\nGenerating samples for validation tests...")
    cond_samples, ground_truth = generate_all_samples(
        model, test_loader,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        max_residual=args.max_residual,
        device=device,
        max_global_residual=args.max_global_residual,
    )
    print(f"  Conditioned samples: {cond_samples.shape}")
    print(f"  Ground truth: {ground_truth.shape}")

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
    )

    # Test Suite 4: Time Series Properties
    results['time_series'] = run_time_series_tests(cond_samples, ground_truth)

    # Test Suite 5: Block-AR Specific
    results['block_ar'] = run_block_ar_tests(
        cond_samples,
        block_size=model_config.block_size,
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
