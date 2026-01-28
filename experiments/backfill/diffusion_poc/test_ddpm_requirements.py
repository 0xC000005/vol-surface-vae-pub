#!/usr/bin/env python
"""
Comprehensive validation tests for DDPM POC.
Tests all four original requirements before full training.

Requirements tested:
1. Valid Surfaces: No explosions, proper term structure, smile convexity
2. CI Coverage: Variation large enough to include ground truth
3. Marginal Recovery: Pooled conditional samples match unconditional distribution
4. Time Series Properties: ACF, vol clustering, kurtosis preserved

Usage:
    # Quick test (current model, fast config)
    python experiments/backfill/diffusion_poc/test_ddpm_requirements.py \
        --n_samples 50 --max_batches 20

    # Full test (after longer training)
    python experiments/backfill/diffusion_poc/test_ddpm_requirements.py \
        --n_samples 100 --max_batches 100

    # Specify model path
    python experiments/backfill/diffusion_poc/test_ddpm_requirements.py \
        --model_path models/backfill/ddpm_poc/best_coverage_model.pt
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, kurtosis, skew

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config


# =============================================================================
# Test 1: Surface Validity
# =============================================================================

def test_explosion_rate(samples: np.ndarray, iv_min: float = 0.0, iv_max: float = 1.0) -> Dict:
    """
    Check for explosion (IV values outside valid range).

    Args:
        samples: (n_samples, T_fut, 5, 5) generated surfaces
        iv_min: Minimum valid IV (default 0.0 = 0%)
        iv_max: Maximum valid IV (default 1.0 = 100%)

    Returns:
        dict with explosion metrics
    """
    n_samples = samples.shape[0]

    # Check high explosions (IV > 1.0)
    high_explosions = (samples > iv_max).any(axis=(1, 2, 3))
    high_rate = high_explosions.mean()

    # Check low explosions (IV < 0.0)
    low_explosions = (samples < iv_min).any(axis=(1, 2, 3))
    low_rate = low_explosions.mean()

    # Combined
    any_explosion = high_explosions | low_explosions
    total_rate = any_explosion.mean()

    # Max/min values observed
    max_iv = samples.max()
    min_iv = samples.min()

    return {
        'explosion_high_rate': float(high_rate),
        'explosion_low_rate': float(low_rate),
        'explosion_total_rate': float(total_rate),
        'max_iv_observed': float(max_iv),
        'min_iv_observed': float(min_iv),
        'pass': total_rate < 0.05,  # Target: <5%
    }


def test_calendar_arbitrage(samples: np.ndarray) -> Dict:
    """
    Check calendar spread arbitrage (total variance should increase with tenor).

    For IV surfaces, row index corresponds to tenor (0=short, 4=long).
    Total variance = σ²τ should increase with τ.

    Approximation: IV should not decrease faster than would make total variance negative.
    Since total variance = IV² * τ, we check if IV[i+1]² * τ[i+1] >= IV[i]² * τ[i].

    Args:
        samples: (n_samples, T_fut, 5, 5) generated surfaces

    Returns:
        dict with calendar arbitrage metrics
    """
    # Approximate tenors (relative, assuming exponential spacing)
    # Row 0 = shortest, Row 4 = longest
    tenors = np.array([1, 2, 4, 8, 12])  # Relative tenor weights

    violations = []

    for t_idx in range(samples.shape[1]):  # For each time step
        surf = samples[:, t_idx]  # (n_samples, 5, 5)

        # Compute total variance for each tenor
        total_var = surf ** 2 * tenors[:, None]  # (n_samples, 5, 5)

        # Check if total variance increases with tenor
        for i in range(4):  # Compare adjacent tenors
            violation = (total_var[:, i, :] > total_var[:, i+1, :] * 1.001)  # Small tolerance
            violations.append(violation.mean())

    avg_violation_rate = np.mean(violations)
    max_violation_rate = np.max(violations)

    return {
        'calendar_avg_violation_rate': float(avg_violation_rate),
        'calendar_max_violation_rate': float(max_violation_rate),
        'pass': avg_violation_rate < 0.10,  # Target: <10%
    }


def test_butterfly_arbitrage(samples: np.ndarray) -> Dict:
    """
    Check butterfly spread arbitrage (smile should be convex).

    For IV surfaces, column index corresponds to moneyness (0=ITM, 2=ATM, 4=OTM).
    Second derivative in strike direction should be non-negative.

    Args:
        samples: (n_samples, T_fut, 5, 5) generated surfaces

    Returns:
        dict with butterfly arbitrage metrics
    """
    violations = []

    for t_idx in range(samples.shape[1]):  # For each time step
        surf = samples[:, t_idx]  # (n_samples, 5, 5)

        # Compute second derivative in strike direction (columns)
        # d²σ/dK² ≈ σ[k-1] - 2*σ[k] + σ[k+1]
        d2_dk2 = surf[:, :, :-2] - 2 * surf[:, :, 1:-1] + surf[:, :, 2:]  # (n_samples, 5, 3)

        # Butterfly violation if second derivative is negative
        # Small tolerance for numerical errors
        violation = (d2_dk2 < -0.005).float().mean() if hasattr(d2_dk2, 'float') else (d2_dk2 < -0.005).mean()
        violations.append(float(violation))

    avg_violation_rate = np.mean(violations)
    max_violation_rate = np.max(violations)

    return {
        'butterfly_avg_violation_rate': float(avg_violation_rate),
        'butterfly_max_violation_rate': float(max_violation_rate),
        'pass': avg_violation_rate < 0.10,  # Target: <10%
    }


def test_smile_symmetry(samples: np.ndarray, gt_samples: np.ndarray) -> Dict:
    """
    Check if smile asymmetry (skew sign) matches historical data.

    Equity options typically have negative skew (OTM puts more expensive).

    Args:
        samples: (n_samples, T_fut, 5, 5) generated surfaces
        gt_samples: (n_gt, T_fut, 5, 5) ground truth surfaces

    Returns:
        dict with smile symmetry metrics
    """
    # Compute skew: IV(OTM) - IV(ITM) = col 4 - col 0
    # Negative skew means OTM puts > OTM calls (typical for equity)

    gen_skew = samples[:, :, :, 4] - samples[:, :, :, 0]  # (n_samples, T, 5)
    gt_skew = gt_samples[:, :, :, 4] - gt_samples[:, :, :, 0]

    # Historical skew should be negative on average
    gt_skew_sign = (gt_skew.mean() < 0)
    gen_skew_sign = (gen_skew.mean() < 0)

    # Match rate: how often does generated skew have same sign as GT
    gen_signs = np.sign(gen_skew.flatten())
    gt_signs = np.sign(gt_skew.flatten())
    sign_match_rate = (gen_signs == gt_signs[0]).mean()  # Compare to overall GT sign

    return {
        'gt_skew_mean': float(gt_skew.mean()),
        'gen_skew_mean': float(gen_skew.mean()),
        'skew_sign_match': gen_skew_sign == gt_skew_sign,
        'pass': gen_skew_sign == gt_skew_sign,
    }


def run_surface_validity_tests(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    n_samples: int = 50,
    max_batches: int = 20,
    device: str = 'cpu',
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
) -> Dict:
    """Run all surface validity tests."""
    print("\n--- Test 1: Surface Validity ---")

    all_samples = []
    all_gt = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating samples", total=max_batches)):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"]

            # Denormalize ground truth from [-1, 1] to [0.05, 1.0]
            future_gt = denormalize_iv(future_gt)

            # Generate samples (model.sample() returns denormalized values)
            samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                   n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)

            # Store (flatten batch dim into samples)
            B = samples.shape[0]
            for b in range(B):
                all_samples.append(samples[b].cpu().numpy())  # (n_samples, T_fut, 5, 5)
                all_gt.append(future_gt[b].numpy())  # (T_fut, 5, 5)

    # Concatenate all samples
    all_samples = np.concatenate(all_samples, axis=0)  # (total_samples, T_fut, 5, 5)
    all_gt = np.stack(all_gt, axis=0)  # (n_gt, T_fut, 5, 5)

    print(f"  Total samples: {all_samples.shape[0]}, GT samples: {all_gt.shape[0]}")

    # Run tests
    explosion_results = test_explosion_rate(all_samples)
    print(f"  Explosion rate: {explosion_results['explosion_total_rate']:.1%} "
          f"({'PASS' if explosion_results['pass'] else 'FAIL'})")

    calendar_results = test_calendar_arbitrage(all_samples)
    print(f"  Calendar arbitrage: {calendar_results['calendar_avg_violation_rate']:.1%} "
          f"({'PASS' if calendar_results['pass'] else 'FAIL'})")

    butterfly_results = test_butterfly_arbitrage(all_samples)
    print(f"  Butterfly arbitrage: {butterfly_results['butterfly_avg_violation_rate']:.1%} "
          f"({'PASS' if butterfly_results['pass'] else 'FAIL'})")

    symmetry_results = test_smile_symmetry(all_samples, all_gt)
    print(f"  Smile symmetry: {'PASS' if symmetry_results['pass'] else 'FAIL'} "
          f"(gen skew={symmetry_results['gen_skew_mean']:.4f}, gt skew={symmetry_results['gt_skew_mean']:.4f})")

    return {
        'explosion': explosion_results,
        'calendar': calendar_results,
        'butterfly': butterfly_results,
        'symmetry': symmetry_results,
        'overall_pass': all([
            explosion_results['pass'],
            calendar_results['pass'],
            butterfly_results['pass'],
            symmetry_results['pass'],
        ]),
    }


# =============================================================================
# Test 2: CI Coverage
# =============================================================================

def compute_ci_coverage_detailed(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    n_samples: int = 100,
    horizons: List[int] = [1, 7, 14, 30],
    ci_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
    max_batches: int = 50,
    device: str = 'cpu',
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
) -> Dict:
    """
    Compute detailed CI coverage metrics.

    Returns:
        dict with overall coverage, per-horizon coverage, calibration data
    """
    print("\n--- Test 2: CI Coverage ---")

    model.eval()

    # Storage
    overall_coverage = {level: [] for level in ci_levels}
    horizon_coverage = {h: {level: [] for level in ci_levels} for h in horizons}
    calibration_levels = np.linspace(0.1, 0.9, 9)
    calibration_data = {round(p, 1): [] for p in calibration_levels}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating CI", total=max_batches)):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            # Denormalize ground truth from [-1, 1] to [0.05, 1.0]
            future_gt = denormalize_iv(future_gt)

            # Generate samples (model.sample() returns denormalized values)
            samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                   n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)  # (B, n_samples, T_fut, 5, 5)

            samples_np = samples.cpu().numpy()
            future_gt_np = future_gt.cpu().numpy()
            T_fut = future_gt.shape[1]

            # Overall coverage
            for level in ci_levels:
                alpha = (1 - level) / 2
                lower = np.quantile(samples_np, alpha, axis=1)
                upper = np.quantile(samples_np, 1 - alpha, axis=1)
                covered = (future_gt_np >= lower) & (future_gt_np <= upper)
                overall_coverage[level].append(covered.mean())

            # Per-horizon coverage
            for h in horizons:
                if h <= T_fut:
                    h_idx = h - 1
                    samples_h = samples_np[:, :, h_idx]
                    gt_h = future_gt_np[:, h_idx]

                    for level in ci_levels:
                        alpha = (1 - level) / 2
                        lower = np.quantile(samples_h, alpha, axis=1)
                        upper = np.quantile(samples_h, 1 - alpha, axis=1)
                        covered = (gt_h >= lower) & (gt_h <= upper)
                        horizon_coverage[h][level].append(covered.mean())

            # Calibration curve data
            for p in calibration_levels:
                alpha = (1 - p) / 2
                lower = np.quantile(samples_np, alpha, axis=1)
                upper = np.quantile(samples_np, 1 - alpha, axis=1)
                covered = (future_gt_np >= lower) & (future_gt_np <= upper)
                calibration_data[round(p, 1)].append(covered.mean())

    # Aggregate results
    results = {
        'overall': {level: np.mean(overall_coverage[level]) for level in ci_levels},
        'per_horizon': {h: {level: np.mean(horizon_coverage[h][level])
                          for level in ci_levels if horizon_coverage[h][level]}
                       for h in horizons},
        'calibration': {
            'nominal': list(calibration_data.keys()),
            'empirical': [np.mean(calibration_data[p]) for p in calibration_data.keys()],
        },
    }

    # Calibration error
    nominal = np.array(results['calibration']['nominal'])
    empirical = np.array(results['calibration']['empirical'])
    results['calibration_error'] = float(np.mean(np.abs(nominal - empirical)))

    # Print results
    print(f"  Overall 90% CI Coverage: {results['overall'][0.9]:.1%} (target: 90%)")
    print(f"  Per-Horizon Coverage (90% CI):")
    for h in horizons:
        if h in results['per_horizon'] and 0.9 in results['per_horizon'][h]:
            print(f"    h={h}: {results['per_horizon'][h][0.9]:.1%}")
    print(f"  Calibration Error: {results['calibration_error']:.3f}")

    # Pass/fail
    results['pass'] = results['overall'][0.9] > 0.70  # Hopeful threshold
    results['pass_strong'] = results['overall'][0.9] > 0.85

    return results


# =============================================================================
# Test 3: Marginal Recovery
# =============================================================================

def test_marginal_recovery(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    n_samples_per_condition: int = 10,
    max_batches: int = 100,
    device: str = 'cpu',
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Test if pooled conditional samples match unconditional distribution.

    Method:
    1. Generate samples from many different histories
    2. Pool all generated samples
    3. Compare distribution to ground truth

    Returns:
        dict with metrics, generated array, gt array
    """
    print("\n--- Test 3: Unconditional Marginal Recovery ---")

    model.eval()

    all_generated = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Sampling for marginal", total=max_batches)):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"]

            # Denormalize ground truth from [-1, 1] to [0.05, 1.0]
            future_gt = denormalize_iv(future_gt)

            # Generate samples (model.sample() returns denormalized values)
            samples = model.sample(history, n_samples=n_samples_per_condition, sampler=sampler,
                                   n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)

            # Pool all samples
            all_generated.append(samples.cpu().numpy().flatten())
            all_gt.append(future_gt.numpy().flatten())

    generated = np.concatenate(all_generated)
    gt = np.concatenate(all_gt)

    print(f"  Generated samples: {len(generated):,}, GT samples: {len(gt):,}")

    # K-S test
    ks_stat, ks_pvalue = ks_2samp(generated, gt)

    # Moment comparison
    gen_mean = generated.mean()
    gt_mean = gt.mean()
    mean_diff_pct = abs(gen_mean - gt_mean) / gt_mean * 100

    gen_std = generated.std()
    gt_std = gt.std()
    std_diff_pct = abs(gen_std - gt_std) / gt_std * 100

    results = {
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'gen_mean': float(gen_mean),
        'gt_mean': float(gt_mean),
        'mean_diff_pct': float(mean_diff_pct),
        'gen_std': float(gen_std),
        'gt_std': float(gt_std),
        'std_diff_pct': float(std_diff_pct),
    }

    # Print results
    print(f"  K-S statistic: {ks_stat:.4f} (p={ks_pvalue:.4f})")
    print(f"  Mean: gen={gen_mean:.4f}, gt={gt_mean:.4f} ({mean_diff_pct:.1f}% diff)")
    print(f"  Std:  gen={gen_std:.4f}, gt={gt_std:.4f} ({std_diff_pct:.1f}% diff)")

    # Pass/fail
    results['pass'] = (ks_stat < 0.15) and (mean_diff_pct < 10) and (std_diff_pct < 30)

    return results, generated, gt


# =============================================================================
# Test 4: Time Series Properties
# =============================================================================

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


def test_acf_preservation(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    n_samples: int = 10,
    max_batches: int = 50,
    max_lag: int = 20,
    device: str = 'cpu',
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
) -> Dict:
    """
    Test if autocorrelation structure is preserved.

    Compares ACF of generated series vs real data.
    """
    print("\n--- Test 4a: ACF Preservation ---")

    model.eval()

    gt_atm_series = []
    gen_atm_series = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Collecting series", total=max_batches)):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"]

            # Denormalize ground truth from [-1, 1] to [0.05, 1.0]
            future_gt = denormalize_iv(future_gt)

            # Generate samples (model.sample() returns denormalized values)
            samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                   n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)

            # Extract ATM point (center of grid)
            gt_atm = future_gt[:, :, 2, 2].numpy()  # (B, T_fut)
            gen_atm = samples[:, 0, :, 2, 2].cpu().numpy()  # (B, T_fut) - first sample

            gt_atm_series.append(gt_atm.flatten())
            gen_atm_series.append(gen_atm.flatten())

    gt_series = np.concatenate(gt_atm_series)
    gen_series = np.concatenate(gen_atm_series)

    # Compute ACFs
    gt_acf = compute_acf(gt_series, max_lag)
    gen_acf = compute_acf(gen_series, max_lag)

    # ACF correlation (similarity measure)
    acf_correlation = np.corrcoef(gt_acf, gen_acf)[0, 1]

    # ACF MAE
    acf_mae = np.mean(np.abs(gt_acf - gen_acf))

    results = {
        'acf_correlation': float(acf_correlation),
        'acf_mae': float(acf_mae),
        'gt_acf': gt_acf.tolist(),
        'gen_acf': gen_acf.tolist(),
    }

    print(f"  ACF correlation: {acf_correlation:.3f} (target: >0.7)")
    print(f"  ACF MAE: {acf_mae:.4f}")

    results['pass'] = acf_correlation > 0.5  # Hopeful threshold

    return results


def test_vol_clustering(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    n_samples: int = 10,
    max_batches: int = 50,
    max_lag: int = 10,
    device: str = 'cpu',
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
) -> Dict:
    """
    Test if ARCH effects (volatility clustering) are preserved.

    Computes ACF of squared returns.
    """
    print("\n--- Test 4b: Volatility Clustering ---")

    model.eval()

    gt_atm_series = []
    gen_atm_series = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Collecting series", total=max_batches)):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"]

            # Denormalize ground truth from [-1, 1] to [0.0, 1.0]
            future_gt = denormalize_iv(future_gt)

            # Generate samples (model.sample() returns denormalized values)
            samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                   n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)

            # Extract ATM point
            gt_atm = future_gt[:, :, 2, 2].numpy().flatten()
            gen_atm = samples[:, 0, :, 2, 2].cpu().numpy().flatten()

            gt_atm_series.extend(gt_atm)
            gen_atm_series.extend(gen_atm)

    gt_series = np.array(gt_atm_series)
    gen_series = np.array(gen_atm_series)

    # Compute squared returns
    gt_returns = np.diff(gt_series)
    gen_returns = np.diff(gen_series)

    gt_sq_returns = gt_returns ** 2
    gen_sq_returns = gen_returns ** 2

    # ACF of squared returns
    gt_sq_acf = compute_acf(gt_sq_returns, max_lag)
    gen_sq_acf = compute_acf(gen_sq_returns, max_lag)

    # ARCH effect = ACF(1) of squared returns
    gt_arch = gt_sq_acf[1] if len(gt_sq_acf) > 1 else 0
    gen_arch = gen_sq_acf[1] if len(gen_sq_acf) > 1 else 0

    results = {
        'gt_arch_lag1': float(gt_arch),
        'gen_arch_lag1': float(gen_arch),
        'gt_sq_acf': gt_sq_acf.tolist(),
        'gen_sq_acf': gen_sq_acf.tolist(),
    }

    # ARCH preserved if both show persistence or both don't
    arch_preserved = (gen_arch > 0.05) if (gt_arch > 0.05) else True
    results['arch_preserved'] = arch_preserved

    print(f"  GT ARCH effect (lag-1): {gt_arch:.4f}")
    print(f"  Gen ARCH effect (lag-1): {gen_arch:.4f}")
    print(f"  ARCH preserved: {'YES' if arch_preserved else 'NO'}")

    results['pass'] = arch_preserved

    return results


def test_kurtosis_matching(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    n_samples: int = 10,
    max_batches: int = 50,
    device: str = 'cpu',
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
    use_hierarchical: bool = False,
    atm_only: bool = False,
) -> Dict:
    """
    Test if kurtosis of one-step changes is preserved.

    Fat tails in financial data should be captured.
    """
    mode_parts = []
    if use_hierarchical:
        mode_parts.append("hierarchical")
    if atm_only:
        mode_parts.append("ATM-only")
    mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
    print(f"\n--- Test 4c: Kurtosis Matching{mode_str} ---")

    model.eval()

    gt_changes = []
    gen_changes = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Computing changes", total=max_batches)):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"]

            # Denormalize ground truth from [-1, 1] to [0.0, 1.0]
            future_gt = denormalize_iv(future_gt)

            # Generate samples (model.sample() returns denormalized values)
            if use_hierarchical:
                samples, _regimes = model.sample_hierarchical(
                    history, n_samples=n_samples, n_inference_steps=n_inference_steps)
            else:
                samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                       n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)

            # Compute one-step changes
            gt_diff = np.diff(future_gt.numpy(), axis=1)  # (B, T-1, 5, 5)
            gen_diff = np.diff(samples[:, 0].cpu().numpy(), axis=1)  # (B, T-1, 5, 5)

            if atm_only:
                gt_changes.append(gt_diff[:, :, 2, 2].flatten())
                gen_changes.append(gen_diff[:, :, 2, 2].flatten())
            else:
                gt_changes.append(gt_diff.flatten())
                gen_changes.append(gen_diff.flatten())

    gt_changes = np.concatenate(gt_changes)
    gen_changes = np.concatenate(gen_changes)

    # Compute kurtosis (Fisher definition: normal = 0)
    gt_kurt = kurtosis(gt_changes, fisher=True)
    gen_kurt = kurtosis(gen_changes, fisher=True)

    # Kurtosis ratio
    kurt_ratio = gen_kurt / gt_kurt if gt_kurt != 0 else np.inf

    # Also compute skewness
    gt_skew = skew(gt_changes)
    gen_skew = skew(gen_changes)

    results = {
        'gt_kurtosis': float(gt_kurt),
        'gen_kurtosis': float(gen_kurt),
        'kurtosis_ratio': float(kurt_ratio),
        'gt_skewness': float(gt_skew),
        'gen_skewness': float(gen_skew),
    }

    print(f"  GT kurtosis: {gt_kurt:.3f}")
    print(f"  Gen kurtosis: {gen_kurt:.3f}")
    print(f"  Kurtosis ratio: {kurt_ratio:.3f} (target: 0.5-2.0)")

    # Pass if ratio is within reasonable bounds
    results['pass'] = 0.5 <= kurt_ratio <= 2.0

    return results


def run_time_series_tests(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    n_samples: int = 10,
    max_batches: int = 50,
    device: str = 'cpu',
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
    use_hierarchical: bool = False,
    atm_only: bool = False,
) -> Dict:
    """Run all time series property tests."""
    print("\n" + "=" * 40)
    print("TIME SERIES PROPERTY TESTS")
    print("=" * 40)

    acf_results = test_acf_preservation(model, test_loader, n_samples, max_batches, device=device, sampler=sampler, n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)
    vol_results = test_vol_clustering(model, test_loader, n_samples, max_batches, device=device, sampler=sampler, n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)
    kurt_results = test_kurtosis_matching(model, test_loader, n_samples, max_batches, device=device, sampler=sampler, n_inference_steps=n_inference_steps, guidance_scale=guidance_scale, use_hierarchical=use_hierarchical, atm_only=atm_only)

    return {
        'acf': acf_results,
        'vol_clustering': vol_results,
        'kurtosis': kurt_results,
        'overall_pass': all([
            acf_results['pass'],
            vol_results['pass'],
            kurt_results['pass'],
        ]),
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_generated_paths(
    model: ConditionalDDPM,
    history: torch.Tensor,
    future_gt: torch.Tensor,
    n_samples: int = 20,
    device: str = 'cpu',
    output_path: Optional[str] = None,
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
    guidance_scale: float = 1.0,
):
    """Visualize multiple generated trajectories vs ground truth."""
    model.eval()

    with torch.no_grad():
        samples = model.sample(history.unsqueeze(0).to(device), n_samples=n_samples, sampler=sampler,
                               n_inference_steps=n_inference_steps, guidance_scale=guidance_scale)
        samples = samples[0].cpu().numpy()  # (n_samples, T_fut, 5, 5)

    future_gt = future_gt.numpy()  # (T_fut, 5, 5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot ATM (center), ITM (left), OTM (right) at short and long tenor
    grid_points = [
        ((0, 2), 'Short-term ATM'),
        ((0, 0), 'Short-term ITM'),
        ((0, 4), 'Short-term OTM'),
        ((4, 2), 'Long-term ATM'),
        ((4, 0), 'Long-term ITM'),
        ((4, 4), 'Long-term OTM'),
    ]

    for ax, ((tenor, strike), title) in zip(axes.flat, grid_points):
        # Plot generated samples (faded)
        for i in range(n_samples):
            ax.plot(samples[i, :, tenor, strike], alpha=0.3, color='blue', linewidth=0.5)

        # Plot ground truth (bold)
        ax.plot(future_gt[:, tenor, strike], color='red', linewidth=2, label='Ground Truth')

        # Plot 90% CI
        p5 = np.percentile(samples[:, :, tenor, strike], 5, axis=0)
        p95 = np.percentile(samples[:, :, tenor, strike], 95, axis=0)
        ax.fill_between(range(len(p5)), p5, p95, alpha=0.2, color='blue', label='90% CI')

        ax.set_title(title)
        ax.set_xlabel('Days ahead')
        ax.set_ylabel('IV')
        if ax == axes[0, 0]:
            ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved path visualization to {output_path}")
    plt.close()
    return fig


def plot_calibration_curve(results: Dict, output_path: Optional[str] = None):
    """Plot CI calibration curve."""
    plt.figure(figsize=(8, 6))

    nominal = results['calibration']['nominal']
    empirical = results['calibration']['empirical']

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    plt.plot(nominal, empirical, 'bo-', label='DDPM', markersize=8, linewidth=2)

    plt.xlabel('Nominal Coverage', fontsize=12)
    plt.ylabel('Empirical Coverage', fontsize=12)
    plt.title('CI Calibration Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate calibration error
    cal_error = results['calibration_error']
    plt.text(0.05, 0.85, f'Calibration Error: {cal_error:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved calibration plot to {output_path}")
    plt.close()


def plot_marginal_comparison(generated: np.ndarray, gt: np.ndarray, output_path: Optional[str] = None):
    """Plot histogram comparison of generated vs ground truth distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    bins = np.linspace(min(gt.min(), generated.min()), max(gt.max(), generated.max()), 50)
    axes[0].hist(gt, bins=bins, alpha=0.5, density=True, label='Ground Truth', color='blue')
    axes[0].hist(generated, bins=bins, alpha=0.5, density=True, label='Generated', color='orange')
    axes[0].set_xlabel('IV', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Marginal Distribution Comparison', fontsize=14)
    axes[0].legend()

    # Q-Q plot
    gt_sorted = np.sort(gt)
    gen_sorted = np.sort(generated)
    # Subsample for Q-Q plot if too many points
    n_points = min(1000, len(gt_sorted), len(gen_sorted))
    gt_quantiles = np.quantile(gt_sorted, np.linspace(0, 1, n_points))
    gen_quantiles = np.quantile(gen_sorted, np.linspace(0, 1, n_points))

    axes[1].plot([gt_quantiles.min(), gt_quantiles.max()],
                 [gt_quantiles.min(), gt_quantiles.max()], 'r--', label='Perfect match')
    axes[1].scatter(gt_quantiles, gen_quantiles, alpha=0.5, s=10)
    axes[1].set_xlabel('Ground Truth Quantiles', fontsize=12)
    axes[1].set_ylabel('Generated Quantiles', fontsize=12)
    axes[1].set_title('Q-Q Plot', fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved marginal comparison to {output_path}")
    plt.close()


def plot_acf_comparison(acf_results: Dict, output_path: Optional[str] = None):
    """Plot ACF comparison between GT and generated samples."""
    plt.figure(figsize=(10, 5))

    lags = np.arange(len(acf_results['gt_acf']))
    width = 0.35

    plt.bar(lags - width/2, acf_results['gt_acf'], width, label='Ground Truth', alpha=0.7)
    plt.bar(lags + width/2, acf_results['gen_acf'], width, label='Generated', alpha=0.7)

    plt.xlabel('Lag (days)', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.title(f'ACF Comparison (Correlation: {acf_results["acf_correlation"]:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved ACF comparison to {output_path}")
    plt.close()


def plot_kurtosis_heatmap(results: Dict, gt_surfaces: np.ndarray, gen_surfaces: np.ndarray,
                          output_path: Optional[str] = None):
    """Plot per-grid kurtosis heatmap."""
    # Compute per-grid kurtosis
    gt_changes = np.diff(gt_surfaces, axis=0)  # (N-1, T, 5, 5)
    gen_changes = np.diff(gen_surfaces, axis=0)

    gt_kurt_grid = np.zeros((5, 5))
    gen_kurt_grid = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_kurt_grid[i, j] = kurtosis(gt_changes[:, :, i, j].flatten(), fisher=True)
            gen_kurt_grid[i, j] = kurtosis(gen_changes[:, :, i, j].flatten(), fisher=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # GT kurtosis
    im1 = axes[0].imshow(gt_kurt_grid, cmap='RdYlBu_r')
    axes[0].set_title('Ground Truth Kurtosis')
    plt.colorbar(im1, ax=axes[0])

    # Gen kurtosis
    im2 = axes[1].imshow(gen_kurt_grid, cmap='RdYlBu_r')
    axes[1].set_title('Generated Kurtosis')
    plt.colorbar(im2, ax=axes[1])

    # Ratio
    ratio_grid = gen_kurt_grid / (gt_kurt_grid + 1e-6)
    im3 = axes[2].imshow(ratio_grid, cmap='RdYlGn', vmin=0.5, vmax=2.0)
    axes[2].set_title('Kurtosis Ratio (Gen/GT)')
    plt.colorbar(im3, ax=axes[2])

    # Add labels
    for ax in axes:
        ax.set_xlabel('Moneyness (ITM→OTM)')
        ax.set_ylabel('Tenor (Short→Long)')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved kurtosis heatmap to {output_path}")
    plt.close()


# =============================================================================
# Main Test Script
# =============================================================================

def print_summary(results: Dict):
    """Print pass/fail summary."""
    print("\n" + "=" * 60)
    print("SUMMARY - PASS/FAIL")
    print("=" * 60)

    # Test 1: Surface Validity
    print("\nTest 1: Surface Validity")
    print(f"  Explosion rate:      {'PASS' if results['surface']['explosion']['pass'] else 'FAIL'}")
    print(f"  Calendar arbitrage:  {'PASS' if results['surface']['calendar']['pass'] else 'FAIL'}")
    print(f"  Butterfly arbitrage: {'PASS' if results['surface']['butterfly']['pass'] else 'FAIL'}")
    print(f"  Smile symmetry:      {'PASS' if results['surface']['symmetry']['pass'] else 'FAIL'}")
    print(f"  Overall:             {'PASS' if results['surface']['overall_pass'] else 'FAIL'}")

    # Test 2: CI Coverage
    print("\nTest 2: CI Coverage")
    print(f"  90% CI Coverage:     {results['coverage']['overall'][0.9]:.1%} "
          f"({'PASS' if results['coverage']['pass'] else 'FAIL'} - target: >70%)")
    print(f"  Calibration error:   {results['coverage']['calibration_error']:.3f}")

    # Test 3: Marginal Recovery
    print("\nTest 3: Marginal Recovery")
    print(f"  K-S statistic:       {results['marginal']['ks_stat']:.4f} "
          f"({'PASS' if results['marginal']['ks_stat'] < 0.15 else 'FAIL'})")
    print(f"  Mean diff:           {results['marginal']['mean_diff_pct']:.1f}% "
          f"({'PASS' if results['marginal']['mean_diff_pct'] < 10 else 'FAIL'})")
    print(f"  Std diff:            {results['marginal']['std_diff_pct']:.1f}% "
          f"({'PASS' if results['marginal']['std_diff_pct'] < 30 else 'FAIL'})")
    print(f"  Overall:             {'PASS' if results['marginal']['pass'] else 'FAIL'}")

    # Test 4: Time Series
    print("\nTest 4: Time Series Properties")
    print(f"  ACF correlation:     {results['time_series']['acf']['acf_correlation']:.3f} "
          f"({'PASS' if results['time_series']['acf']['pass'] else 'FAIL'} - target: >0.5)")
    print(f"  Vol clustering:      {'PASS' if results['time_series']['vol_clustering']['pass'] else 'FAIL'}")
    print(f"  Kurtosis ratio:      {results['time_series']['kurtosis']['kurtosis_ratio']:.3f} "
          f"({'PASS' if results['time_series']['kurtosis']['pass'] else 'FAIL'} - target: 0.5-2.0)")
    print(f"  Overall:             {'PASS' if results['time_series']['overall_pass'] else 'FAIL'}")

    # Overall
    print("\n" + "=" * 60)
    all_pass = all([
        results['surface']['overall_pass'],
        results['coverage']['pass'],
        results['marginal']['pass'],
        results['time_series']['overall_pass'],
    ])
    if all_pass:
        print("OVERALL: ALL TESTS PASSED - Proceed to full training!")
    else:
        print("OVERALL: SOME TESTS FAILED - Review and adjust before full training")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive DDPM POC Validation Tests")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--n_samples", type=int, default=50, help="Samples per history")
    parser.add_argument("--max_batches", type=int, default=20, help="Max batches to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim", "ddim_staggered", "ddpm_staggered"], default="ddpm",
                        help="Sampling method: ddpm (slow), ddim (fast), ddim_staggered (causal DDIM), ddpm_staggered (exact causal DDPM)")
    parser.add_argument("--ddim_steps", type=int, default=20,
                        help="Number of denoising steps for DDIM/staggered (default: 20)")
    parser.add_argument("--max_residual", type=int, default=20,
                        help="For ddim_staggered: t_min for last frame (default: 20). Higher = more uncertainty growth")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="CFG guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical regime sampling for kurtosis test (requires regime-conditioned model)")
    parser.add_argument("--atm_only", action="store_true",
                        help="Compute kurtosis on ATM grid point [2,2] only instead of all 25 points")
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
        print(f"No trained model found at {model_path}. Run training first:")
        print(f"  python experiments/backfill/diffusion_poc/train_ddpm_poc.py")
        return

    # Output directory
    output_dir = args.output_dir or f"{config.results_dir}/validation_tests"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DDPM POC Validation Tests")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Samples per history: {args.n_samples}")
    print(f"Max batches: {args.max_batches}")
    sampler_info = f"Sampler: {args.sampler}"
    if args.sampler in ['ddim', 'ddim_staggered']:
        sampler_info += f" ({args.ddim_steps} steps)"
        if args.sampler == 'ddim_staggered':
            sampler_info += f", max_residual={args.max_residual}"
    else:
        sampler_info += " (all steps)"
    if args.guidance_scale != 1.0:
        sampler_info += f", guidance_scale={args.guidance_scale}"
    print(sampler_info)
    if args.hierarchical:
        print("Hierarchical regime sampling: ENABLED (for kurtosis test)")
    if args.atm_only:
        print("ATM-only kurtosis: ENABLED (grid point [2,2] only)")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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

    # Run all tests
    results = {}

    # Test 1: Surface Validity
    results['surface'] = run_surface_validity_tests(
        model, test_loader,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        device=device,
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )

    # Test 2: CI Coverage
    results['coverage'] = compute_ci_coverage_detailed(
        model, test_loader,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        device=device,
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )

    # Test 3: Marginal Recovery
    results['marginal'], generated, gt = test_marginal_recovery(
        model, test_loader,
        n_samples_per_condition=min(10, args.n_samples),
        max_batches=args.max_batches,
        device=device,
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )

    # Test 4: Time Series Properties
    results['time_series'] = run_time_series_tests(
        model, test_loader,
        n_samples=min(10, args.n_samples),
        max_batches=args.max_batches,
        device=device,
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        use_hierarchical=args.hierarchical,
        atm_only=args.atm_only,
    )

    # Print summary
    print_summary(results)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Calibration curve
    plot_calibration_curve(results['coverage'], f"{output_dir}/calibration_curve.png")

    # Marginal comparison
    plot_marginal_comparison(generated, gt, f"{output_dir}/marginal_comparison.png")

    # ACF comparison
    plot_acf_comparison(results['time_series']['acf'], f"{output_dir}/acf_comparison.png")

    # Path visualization (sample one history)
    sample_batch = next(iter(test_loader))
    # Denormalize ground truth for visualization
    sample_future_gt = denormalize_iv(sample_batch["future"][0])
    visualize_generated_paths(
        model,
        sample_batch["history"][0],
        sample_future_gt,
        n_samples=20,
        device=device,
        output_path=f"{output_dir}/path_visualization.png",
        sampler=args.sampler,
        n_inference_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )

    # Save results to JSON
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    results_serializable = convert_to_serializable(results)

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_dir}/summary.json")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
