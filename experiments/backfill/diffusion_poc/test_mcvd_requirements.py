#!/usr/bin/env python
"""
Comprehensive validation tests for MCVD POC.

Tests:
1. Surface Validity: No explosions, calendar/butterfly arbitrage
2. CI Coverage & Sharpness: Coverage + interval width at all levels/horizons
3. Conditionality: Does model actually use history conditioning?
4. CRPS: Proper scoring rule per horizon
5. Time Series: ACF preservation (expanded), kurtosis matching
6. Sample Quality: Diversity

Usage:
    python experiments/backfill/diffusion_poc/test_mcvd_requirements.py
    python experiments/backfill/diffusion_poc/test_mcvd_requirements.py \
        --model_path models/backfill/mcvd_poc/best_coverage_model.pt \
        --sampler ddim --ddim_steps 20 --n_samples 50 --max_batches 20
    python experiments/backfill/diffusion_poc/test_mcvd_requirements.py --skip_conditionality
"""

import os
import sys
import argparse
import json
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

from diffusion.mcvd_wrapper import MCVDModel, denormalize_iv, pad_surface, crop_surface
from experiments.backfill.diffusion_poc.config_mcvd_poc import (
    MCVDPOCConfig,
    build_mcvd_config,
    get_default_config,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# =============================================================================
# Model Loading
# =============================================================================

def load_mcvd_model(model_path: str, device: str) -> Tuple[MCVDModel, MCVDPOCConfig, dict]:
    """Load MCVD model from checkpoint. Uses EMA weights if available."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    poc_config = MCVDPOCConfig(**checkpoint["mcvd_config"])
    mcvd_config = build_mcvd_config(poc_config)
    model = MCVDModel(mcvd_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Apply EMA weights if saved in checkpoint
    if "ema_state_dict" in checkpoint:
        from diffusion.mcvd.models.ema import EMAHelper
        ema_helper = EMAHelper()
        ema_helper.shadow = checkpoint["ema_state_dict"]
        ema_helper.ema(model)
        print("  Loaded EMA weights")
    model = model.to(device)
    model.eval()
    return model, poc_config, checkpoint


# =============================================================================
# Autoregressive Sampling Wrapper
# =============================================================================

class ARModelWrapper:
    """Wraps MCVDModel so .sample() transparently uses autoregressive rollout.

    When a model was trained with future_len=5 (paper-aligned config), tests
    still expect 30-frame outputs. This wrapper makes model.sample() return
    30 frames via n_blocks autoregressive steps.
    """

    def __init__(self, model: MCVDModel, n_blocks: int, n_inference_steps: int = 100):
        self._model = model
        self.n_blocks = n_blocks
        self.n_inference_steps = n_inference_steps

    def sample(self, history, n_samples=1, sampler='ddim', n_inference_steps=None, **kwargs):
        steps = n_inference_steps or self.n_inference_steps
        return self._model.sample_autoregressive(
            history, n_samples=n_samples, n_blocks=self.n_blocks,
            sampler=sampler, n_inference_steps=steps,
        )

    def __getattr__(self, name):
        return getattr(self._model, name)


# =============================================================================
# Unconditioned Sampling Helper
# =============================================================================

@torch.no_grad()
def _sample_unconditioned_block(
    raw_model: MCVDModel,
    B: int, device: torch.device,
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
) -> torch.Tensor:
    """Generate one unconditional block of T frames in [-1, 1], padded 8×8.

    Uses zeroed conditioning and cond_mask=0.
    Returns: (B, T, 8, 8) in [-1, 1]
    """
    T = raw_model.num_frames
    H, W = 8, 8
    zero_cond = torch.zeros(B, raw_model.num_frames_cond, H, W, device=device)
    cond_mask = torch.zeros(B, device=device, dtype=torch.int32)

    if sampler == 'ddim':
        return raw_model._sample_ddim(
            zero_cond, cond_mask, B, T, H, W, device,
            n_inference_steps=n_inference_steps,
        )
    else:
        return raw_model._sample_ddpm(
            zero_cond, cond_mask, B, T, H, W, device,
        )


@torch.no_grad()
def sample_unconditioned(
    model: MCVDModel,
    history: torch.Tensor,
    n_samples: int = 1,
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
) -> torch.Tensor:
    """Generate unconditional samples (zeroed history + cond_mask=0).

    Simulates prob_mask_cond=1.0 (fully masked conditioning).
    Used by the conditionality test to compare conditioned vs unconditioned.
    Handles both direct (30-frame) and AR (5-frame × n_blocks) models.

    Returns:
        samples: (B, n_samples, T_fut, 5, 5) denormalized to [0, 1]
    """
    is_ar = isinstance(model, ARModelWrapper)
    raw_model = model._model if is_ar else model

    B = history.shape[0]
    device = history.device

    samples = []
    for _ in range(n_samples):
        if is_ar:
            # AR rollout: generate n_blocks of unconditioned blocks
            blocks = []
            for _ in range(model.n_blocks):
                x_block = _sample_unconditioned_block(
                    raw_model, B, device, sampler, n_inference_steps,
                )
                blocks.append(crop_surface(x_block))
            x_0_cropped = torch.cat(blocks, dim=1)  # (B, n_blocks*T, 5, 5)
        else:
            x_0 = _sample_unconditioned_block(
                raw_model, B, device, sampler, n_inference_steps,
            )
            x_0_cropped = crop_surface(x_0)

        samples.append(x_0_cropped)

    samples = torch.stack(samples, dim=1)
    samples = denormalize_iv(samples)
    return samples


# =============================================================================
# CRPS (from eval_ddpm_poc.py)
# =============================================================================

def compute_crps(samples: np.ndarray, gt: float) -> float:
    """CRPS = E|X-y| - 0.5*E|X-X'|. Lower is better."""
    n = len(samples)
    if n == 0:
        return np.nan
    term1 = np.mean(np.abs(samples - gt))
    sorted_samples = np.sort(samples)
    indices = np.arange(n)
    weights = 2 * indices + 1 - n
    term2 = np.sum(weights * sorted_samples) / (n * n) * 2
    return term1 - 0.5 * term2


# =============================================================================
# ACF Helper
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


# =============================================================================
# Test 1: Surface Validity
# =============================================================================

def test_explosion_rate(samples: np.ndarray, iv_min: float = 0.0, iv_max: float = 1.0) -> Dict:
    """Check for explosion (IV values outside valid range)."""
    high_explosions = (samples > iv_max).any(axis=(1, 2, 3))
    low_explosions = (samples < iv_min).any(axis=(1, 2, 3))
    any_explosion = high_explosions | low_explosions
    total_rate = any_explosion.mean()
    return {
        'explosion_high_rate': float((samples > iv_max).any(axis=(1, 2, 3)).mean()),
        'explosion_low_rate': float((samples < iv_min).any(axis=(1, 2, 3)).mean()),
        'explosion_total_rate': float(total_rate),
        'max_iv_observed': float(samples.max()),
        'min_iv_observed': float(samples.min()),
        'pass': total_rate < 0.05,
    }


def test_calendar_arbitrage(samples: np.ndarray) -> Dict:
    """Check calendar spread arbitrage (total variance should increase with tenor)."""
    tenors = np.array([1, 2, 4, 8, 12])
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]
        total_var = surf ** 2 * tenors[:, None]
        for i in range(4):
            violation = (total_var[:, i, :] > total_var[:, i+1, :] * 1.001)
            violations.append(violation.mean())
    avg_violation_rate = np.mean(violations)
    return {
        'calendar_avg_violation_rate': float(avg_violation_rate),
        'calendar_max_violation_rate': float(np.max(violations)),
        'pass': avg_violation_rate < 0.10,
    }


def test_butterfly_arbitrage(samples: np.ndarray) -> Dict:
    """Check butterfly spread arbitrage (smile should be convex)."""
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]
        d2_dk2 = surf[:, :, :-2] - 2 * surf[:, :, 1:-1] + surf[:, :, 2:]
        violation = (d2_dk2 < -0.005).mean()
        violations.append(float(violation))
    avg_violation_rate = np.mean(violations)
    return {
        'butterfly_avg_violation_rate': float(avg_violation_rate),
        'butterfly_max_violation_rate': float(np.max(violations)),
        'pass': avg_violation_rate < 0.10,
    }


def run_surface_validity_tests(
    model: MCVDModel = None,
    test_loader: DataLoader = None,
    n_samples: int = 50,
    max_batches: int = 20,
    device: str = 'cpu',
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    precomputed: tuple = None,
) -> Dict:
    """Run all surface validity tests.

    Args:
        precomputed: Optional (cond_samples, ground_truth) numpy arrays.
            cond_samples: (N, n_samples, T, 5, 5), ground_truth: (N, T, 5, 5)
    """
    print("\n--- Test 1: Surface Validity ---")

    if precomputed is not None:
        cond_samples_np, gt_np = precomputed
        N, S, T, H, W = cond_samples_np.shape
        all_samples = cond_samples_np.reshape(N * S, T, H, W)
        all_gt = gt_np
    else:
        all_samples = []
        all_gt = []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating samples", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"])
                samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                       n_inference_steps=n_inference_steps)
                B = samples.shape[0]
                for b in range(B):
                    all_samples.append(samples[b].cpu().numpy())
                    all_gt.append(future_gt[b].numpy())
        all_samples = np.concatenate(all_samples, axis=0)
        all_gt = np.stack(all_gt, axis=0)

    print(f"  Total samples: {all_samples.shape[0]}, GT samples: {all_gt.shape[0]}")

    explosion_results = test_explosion_rate(all_samples)
    print(f"  Explosion rate: {explosion_results['explosion_total_rate']:.1%} "
          f"({'PASS' if explosion_results['pass'] else 'FAIL'})")

    calendar_results = test_calendar_arbitrage(all_samples)
    print(f"  Calendar arbitrage: {calendar_results['calendar_avg_violation_rate']:.1%} "
          f"({'PASS' if calendar_results['pass'] else 'FAIL'})")

    butterfly_results = test_butterfly_arbitrage(all_samples)
    print(f"  Butterfly arbitrage: {butterfly_results['butterfly_avg_violation_rate']:.1%} "
          f"({'PASS' if butterfly_results['pass'] else 'FAIL'})")

    return {
        'explosion': explosion_results,
        'calendar': calendar_results,
        'butterfly': butterfly_results,
        'overall_pass': all([
            explosion_results['pass'],
            calendar_results['pass'],
            butterfly_results['pass'],
        ]),
    }


# =============================================================================
# Test 2: CI Coverage & Sharpness
# =============================================================================

def compute_ci_coverage_and_sharpness(
    model: MCVDModel = None,
    test_loader: DataLoader = None,
    n_samples: int = 100,
    horizons: List[int] = [1, 7, 14, 30],
    ci_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
    max_batches: int = 50,
    device: str = 'cpu',
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    precomputed: tuple = None,
) -> Dict:
    """Compute CI coverage and interval width (sharpness).

    Args:
        precomputed: Optional (cond_samples, ground_truth) numpy arrays.
            cond_samples: (N, n_samples, T, 5, 5), ground_truth: (N, T, 5, 5)
    """
    print("\n--- Test 2: CI Coverage & Sharpness ---")

    overall_coverage = {level: [] for level in ci_levels}
    overall_width = {level: [] for level in ci_levels}
    horizon_coverage = {h: {level: [] for level in ci_levels} for h in horizons}
    horizon_width = {h: {level: [] for level in ci_levels} for h in horizons}
    calibration_levels = np.linspace(0.1, 0.9, 9)
    calibration_data = {round(p, 1): [] for p in calibration_levels}

    if precomputed is not None:
        cond_samples_np, future_gt_np = precomputed
        batches = [(cond_samples_np, future_gt_np)]
    else:
        model.eval()
        batches = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating CI", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"].to(device))
                samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                       n_inference_steps=n_inference_steps)
                batches.append((samples.cpu().numpy(), future_gt.cpu().numpy()))

    for samples_np, future_gt_np in batches:
        T_fut = future_gt_np.shape[1]

        # Overall coverage and width
        for level in ci_levels:
            alpha = (1 - level) / 2
            lower = np.quantile(samples_np, alpha, axis=1)
            upper = np.quantile(samples_np, 1 - alpha, axis=1)
            covered = (future_gt_np >= lower) & (future_gt_np <= upper)
            overall_coverage[level].append(covered.mean())
            overall_width[level].append((upper - lower).mean())

        # Per-horizon
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
                    horizon_width[h][level].append((upper - lower).mean())

        # Calibration curve
        for p in calibration_levels:
            alpha = (1 - p) / 2
            lower = np.quantile(samples_np, alpha, axis=1)
            upper = np.quantile(samples_np, 1 - alpha, axis=1)
            covered = (future_gt_np >= lower) & (future_gt_np <= upper)
            calibration_data[round(p, 1)].append(covered.mean())

    results = {
        'overall': {level: np.mean(overall_coverage[level]) for level in ci_levels},
        'overall_width': {level: np.mean(overall_width[level]) for level in ci_levels},
        'per_horizon': {h: {level: np.mean(horizon_coverage[h][level])
                          for level in ci_levels if horizon_coverage[h][level]}
                       for h in horizons},
        'per_horizon_width': {h: {level: np.mean(horizon_width[h][level])
                                for level in ci_levels if horizon_width[h][level]}
                             for h in horizons},
        'calibration': {
            'nominal': list(calibration_data.keys()),
            'empirical': [np.mean(calibration_data[p]) for p in calibration_data.keys()],
        },
    }

    nominal = np.array(results['calibration']['nominal'])
    empirical = np.array(results['calibration']['empirical'])
    results['calibration_error'] = float(np.mean(np.abs(nominal - empirical)))

    # Print coverage + width table
    print(f"  {'Level':>6} | {'Coverage':>10} | {'Width':>8} | Status")
    print(f"  {'-'*45}")
    for level in ci_levels:
        cov = results['overall'][level]
        wid = results['overall_width'][level]
        print(f"  {level:>5.0%}  | {cov:>9.1%}  | {wid:>7.4f}  | [target: {level:.0%}]")

    print(f"\n  Per-Horizon (90% CI):")
    for h in horizons:
        if h in results['per_horizon'] and 0.9 in results['per_horizon'][h]:
            cov = results['per_horizon'][h][0.9]
            wid = results['per_horizon_width'][h][0.9]
            print(f"    h={h:>2}: {cov:.1%} (width: {wid:.4f})")

    print(f"  Calibration Error: {results['calibration_error']:.3f}")

    results['pass'] = results['overall'][0.9] > 0.70
    results['pass_strong'] = results['overall'][0.9] > 0.85
    return results


# =============================================================================
# Test 3: Conditionality
# =============================================================================

def test_conditionality(
    model: MCVDModel = None,
    test_loader: DataLoader = None,
    n_samples: int = 50,
    max_batches: int = 10,
    device: str = 'cpu',
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    precomputed: tuple = None,
) -> Dict:
    """Test if model uses conditioning information.

    Compares conditioned (normal) vs unconditioned (zeroed history + cond_mask=0)
    generation. If conditioning works, conditioned samples should be sharper
    (narrower CI) and more accurate (lower MAE).

    Args:
        precomputed: Optional (cond_samples, uncond_samples, ground_truth) numpy arrays.
            cond_samples: (N, n_samples, T, 5, 5)
            uncond_samples: (N, n_samples, T, 5, 5)
            ground_truth: (N, T, 5, 5)
    """
    print("\n--- Test 3: Conditionality ---")

    cond_widths = []
    uncond_widths = []
    cond_maes = []
    uncond_maes = []

    if precomputed is not None:
        cond_np, uncond_np, gt_np = precomputed
        # Process all at once using numpy
        for samples_np, width_list, mae_list in [
            (cond_np, cond_widths, cond_maes),
            (uncond_np, uncond_widths, uncond_maes),
        ]:
            p05 = np.quantile(samples_np, 0.05, axis=1)
            p95 = np.quantile(samples_np, 0.95, axis=1)
            width = (p95 - p05).mean()
            width_list.append(float(width))
            median = np.quantile(samples_np, 0.5, axis=1)
            mae = np.abs(median - gt_np).mean()
            mae_list.append(float(mae))
    else:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Conditionality test", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"].to(device))

                # Conditioned samples (normal)
                samples_cond = model.sample(history, n_samples=n_samples, sampler=sampler,
                                            n_inference_steps=n_inference_steps)
                # Unconditioned samples (zeroed history, cond_mask=0)
                samples_uncond = sample_unconditioned(model, history, n_samples=n_samples,
                                                       sampler=sampler,
                                                       n_inference_steps=n_inference_steps)

                # 90% CI width
                for samples, width_list, mae_list in [
                    (samples_cond, cond_widths, cond_maes),
                    (samples_uncond, uncond_widths, uncond_maes),
                ]:
                    p05 = torch.quantile(samples, 0.05, dim=1)
                    p95 = torch.quantile(samples, 0.95, dim=1)
                    width = (p95 - p05).mean().item()
                    width_list.append(width)

                    median = torch.quantile(samples, 0.5, dim=1)
                    mae = (median - future_gt).abs().mean().item()
                    mae_list.append(mae)

    cond_width = np.mean(cond_widths)
    uncond_width = np.mean(uncond_widths)
    width_ratio = cond_width / uncond_width if uncond_width > 0 else 1.0
    cond_mae = np.mean(cond_maes)
    uncond_mae = np.mean(uncond_maes)
    mae_reduction = 1 - (cond_mae / uncond_mae) if uncond_mae > 0 else 0.0

    passed = (width_ratio < 0.95) and (mae_reduction > 0.05)

    print(f"  Conditioned CI Width:   {cond_width:.4f}")
    print(f"  Unconditioned CI Width: {uncond_width:.4f}")
    print(f"  Width Ratio:            {width_ratio:.3f} (< 1.0 = conditioning helps)")
    print(f"  Conditioned MAE:        {cond_mae:.4f}")
    print(f"  Unconditioned MAE:      {uncond_mae:.4f}")
    print(f"  MAE Reduction:          {mae_reduction:.1%}")
    print(f"  Conditionality:         {'PASS' if passed else 'FAIL'}")

    return {
        'cond_width': float(cond_width),
        'uncond_width': float(uncond_width),
        'width_ratio': float(width_ratio),
        'cond_mae': float(cond_mae),
        'uncond_mae': float(uncond_mae),
        'mae_reduction': float(mae_reduction),
        'pass': passed,
    }


# =============================================================================
# Test 4: CRPS per Horizon
# =============================================================================

def test_crps_per_horizon(
    model: MCVDModel = None,
    test_loader: DataLoader = None,
    n_samples: int = 50,
    horizons: List[int] = [1, 7, 14, 30],
    max_batches: int = 20,
    device: str = 'cpu',
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    precomputed: tuple = None,
) -> Dict:
    """Compute CRPS (proper scoring rule) per horizon.

    Args:
        precomputed: Optional (cond_samples, ground_truth) numpy arrays.
    """
    print("\n--- Test 4: CRPS per Horizon ---")

    horizon_crps = {h: [] for h in horizons}
    all_crps = []

    if precomputed is not None:
        samples_np, gt_np = precomputed
        N = samples_np.shape[0]
        for h in horizons:
            h_idx = h - 1
            for b in range(N):
                for i in range(5):
                    for j in range(5):
                        crps_val = compute_crps(
                            samples_np[b, :, h_idx, i, j],
                            gt_np[b, h_idx, i, j],
                        )
                        horizon_crps[h].append(crps_val)
                        all_crps.append(crps_val)
    else:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Computing CRPS", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"].to(device))
                samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                       n_inference_steps=n_inference_steps)

                samples_np = samples.cpu().numpy()
                gt_np = future_gt.cpu().numpy()
                B = samples_np.shape[0]

                for h in horizons:
                    h_idx = h - 1
                    for b in range(B):
                        for i in range(5):
                            for j in range(5):
                                crps_val = compute_crps(
                                    samples_np[b, :, h_idx, i, j],
                                    gt_np[b, h_idx, i, j],
                                )
                                horizon_crps[h].append(crps_val)
                                all_crps.append(crps_val)

    results = {
        'crps_per_horizon': {h: float(np.nanmean(vals)) for h, vals in horizon_crps.items()},
        'crps_overall': float(np.nanmean(all_crps)),
    }

    for h in horizons:
        print(f"  h={h:>2}: {results['crps_per_horizon'][h]:.6f}")
    print(f"  Overall: {results['crps_overall']:.6f}")

    return results


# =============================================================================
# Test 5: Time Series Properties
# =============================================================================

def test_acf_preservation(
    model: MCVDModel = None,
    test_loader: DataLoader = None,
    n_samples: int = 10,
    max_batches: int = 50,
    max_lag: int = 20,
    device: str = 'cpu',
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    precomputed: tuple = None,
) -> Dict:
    """Test ACF preservation across 9 representative grid points.

    Args:
        precomputed: Optional (cond_samples, ground_truth) numpy arrays.
    """
    print("\n--- Test 5a: ACF Preservation (9 grid points) ---")

    grid_points = [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4), (4, 0), (4, 2), (4, 4)]

    gt_series = {pt: [] for pt in grid_points}
    gen_series = {pt: [] for pt in grid_points}

    if precomputed is not None:
        cond_samples_np, gt_np = precomputed
        for pt in grid_points:
            r, c = pt
            gt_series[pt].append(gt_np[:, :, r, c].flatten())
            gen_series[pt].append(cond_samples_np[:, 0, :, r, c].flatten())
    else:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Collecting series", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"])
                samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                       n_inference_steps=n_inference_steps)

                for pt in grid_points:
                    r, c = pt
                    gt_series[pt].append(future_gt[:, :, r, c].numpy().flatten())
                    gen_series[pt].append(samples[:, 0, :, r, c].cpu().numpy().flatten())

    # Compute per-point ACF correlation
    acf_correlations = {}
    for pt in grid_points:
        gt_flat = np.concatenate(gt_series[pt])
        gen_flat = np.concatenate(gen_series[pt])
        gt_acf = compute_acf(gt_flat, max_lag)
        gen_acf = compute_acf(gen_flat, max_lag)
        corr = np.corrcoef(gt_acf, gen_acf)[0, 1]
        acf_correlations[f"{pt[0]}_{pt[1]}"] = float(corr)

    # Also store ATM ACF for plotting
    atm_gt = np.concatenate(gt_series[(2, 2)])
    atm_gen = np.concatenate(gen_series[(2, 2)])
    gt_acf_atm = compute_acf(atm_gt, max_lag)
    gen_acf_atm = compute_acf(atm_gen, max_lag)

    corr_values = list(acf_correlations.values())
    mean_corr = np.mean(corr_values)
    min_corr = np.min(corr_values)
    atm_corr = acf_correlations["2_2"]

    print(f"  ACF Correlation (ATM):  {atm_corr:.3f}")
    print(f"  ACF Correlation (mean): {mean_corr:.3f}")
    print(f"  ACF Correlation (min):  {min_corr:.3f}")

    return {
        'acf_correlation_atm': float(atm_corr),
        'acf_correlation_mean': float(mean_corr),
        'acf_correlation_min': float(min_corr),
        'per_point': acf_correlations,
        'gt_acf_atm': gt_acf_atm.tolist(),
        'gen_acf_atm': gen_acf_atm.tolist(),
        'pass': mean_corr > 0.5,
    }


def test_kurtosis_matching(
    model: MCVDModel = None,
    test_loader: DataLoader = None,
    n_samples: int = 10,
    max_batches: int = 50,
    device: str = 'cpu',
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    atm_only: bool = False,
    precomputed: tuple = None,
) -> Dict:
    """Test if kurtosis of one-step changes is preserved.

    Args:
        precomputed: Optional (cond_samples, ground_truth) numpy arrays.
    """
    mode_str = " (ATM-only)" if atm_only else ""
    print(f"\n--- Test 5b: Kurtosis Matching{mode_str} ---")

    gt_changes = []
    gen_changes = []

    if precomputed is not None:
        cond_samples_np, gt_np = precomputed
        gt_diff = np.diff(gt_np, axis=1)
        gen_diff = np.diff(cond_samples_np[:, 0], axis=1)
        if atm_only:
            gt_changes.append(gt_diff[:, :, 2, 2].flatten())
            gen_changes.append(gen_diff[:, :, 2, 2].flatten())
        else:
            gt_changes.append(gt_diff.flatten())
            gen_changes.append(gen_diff.flatten())
    else:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Computing changes", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"])
                samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                       n_inference_steps=n_inference_steps)

                gt_diff = np.diff(future_gt.numpy(), axis=1)
                gen_diff = np.diff(samples[:, 0].cpu().numpy(), axis=1)

                if atm_only:
                    gt_changes.append(gt_diff[:, :, 2, 2].flatten())
                    gen_changes.append(gen_diff[:, :, 2, 2].flatten())
                else:
                    gt_changes.append(gt_diff.flatten())
                    gen_changes.append(gen_diff.flatten())

    gt_changes = np.concatenate(gt_changes)
    gen_changes = np.concatenate(gen_changes)

    gt_kurt = kurtosis(gt_changes, fisher=True)
    gen_kurt = kurtosis(gen_changes, fisher=True)
    kurt_ratio = gen_kurt / gt_kurt if gt_kurt != 0 else np.inf
    gt_skew_val = skew(gt_changes)
    gen_skew_val = skew(gen_changes)

    print(f"  GT kurtosis:  {gt_kurt:.3f}")
    print(f"  Gen kurtosis: {gen_kurt:.3f}")
    print(f"  Ratio:        {kurt_ratio:.3f} (target: 0.5-2.0)")

    return {
        'gt_kurtosis': float(gt_kurt),
        'gen_kurtosis': float(gen_kurt),
        'kurtosis_ratio': float(kurt_ratio),
        'gt_skewness': float(gt_skew_val),
        'gen_skewness': float(gen_skew_val),
        'pass': 0.5 <= kurt_ratio <= 2.0,
    }


# =============================================================================
# Test 6: Sample Quality
# =============================================================================

def compute_sample_diversity(
    model: MCVDModel = None,
    test_loader: DataLoader = None,
    n_samples: int = 50,
    max_batches: int = 20,
    device: str = 'cpu',
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
    precomputed: tuple = None,
) -> Dict:
    """Compute sample diversity (std across samples).

    Args:
        precomputed: Optional (cond_samples,) numpy array. (N, n_samples, T, 5, 5)
    """
    print("\n--- Test 6: Sample Quality ---")

    if precomputed is not None:
        cond_samples_np = precomputed[0] if isinstance(precomputed, tuple) else precomputed
        mean_diversity = float(cond_samples_np.std(axis=1).mean())
    else:
        model.eval()
        all_diversity = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Computing diversity", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                history = batch["history"].to(device)
                samples = model.sample(history, n_samples=n_samples, sampler=sampler,
                                       n_inference_steps=n_inference_steps)
                diversity = samples.std(dim=1).mean().item()
                all_diversity.append(diversity)
        mean_diversity = np.mean(all_diversity)

    print(f"  Diversity: {mean_diversity:.4f}")
    return {'diversity': float(mean_diversity)}


# =============================================================================
# Visualization
# =============================================================================

def plot_calibration_curve(results: Dict, output_path: Optional[str] = None):
    """Plot CI calibration curve."""
    plt.figure(figsize=(8, 6))
    nominal = results['calibration']['nominal']
    empirical = results['calibration']['empirical']
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    plt.plot(nominal, empirical, 'bo-', label='MCVD', markersize=8, linewidth=2)
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
    lags = np.arange(len(acf_results['gt_acf_atm']))
    width = 0.35
    plt.bar(lags - width/2, acf_results['gt_acf_atm'], width, label='Ground Truth', alpha=0.7)
    plt.bar(lags + width/2, acf_results['gen_acf_atm'], width, label='Generated', alpha=0.7)
    plt.xlabel('Lag (days)', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.title(f'ACF Comparison - ATM (Correlation: {acf_results["acf_correlation_atm"]:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved ACF comparison to {output_path}")
    plt.close()


def visualize_generated_paths(
    model: MCVDModel,
    history: torch.Tensor,
    future_gt: torch.Tensor,
    n_samples: int = 20,
    device: str = 'cpu',
    output_path: Optional[str] = None,
    sampler: str = 'ddim',
    n_inference_steps: int = 20,
):
    """Visualize multiple generated trajectories vs ground truth."""
    model.eval()
    with torch.no_grad():
        samples = model.sample(history.unsqueeze(0).to(device), n_samples=n_samples,
                               sampler=sampler, n_inference_steps=n_inference_steps)
        samples = samples[0].cpu().numpy()
    future_gt = future_gt.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    grid_points = [
        ((0, 2), 'Short-term ATM'), ((0, 0), 'Short-term ITM'), ((0, 4), 'Short-term OTM'),
        ((4, 2), 'Long-term ATM'), ((4, 0), 'Long-term ITM'), ((4, 4), 'Long-term OTM'),
    ]
    for ax, ((tenor, strike), title) in zip(axes.flat, grid_points):
        for i in range(n_samples):
            ax.plot(samples[i, :, tenor, strike], alpha=0.3, color='blue', linewidth=0.5)
        ax.plot(future_gt[:, tenor, strike], color='red', linewidth=2, label='Ground Truth')
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
        print(f"  Saved path visualization to {output_path}")
    plt.close()
    return fig


# =============================================================================
# Summary & Main
# =============================================================================

def print_summary(results: Dict):
    """Print pass/fail summary."""
    print("\n" + "=" * 60)
    print("MCVD POC VALIDATION RESULTS")
    print("=" * 60)

    # Surface Validity
    print("\n--- Surface Validity ---")
    s = results['surface']
    print(f"  Explosion Rate:      {s['explosion']['explosion_total_rate']:.1%} "
          f"[{'PASS' if s['explosion']['pass'] else 'FAIL'} < 5%]")
    print(f"  Calendar Arbitrage:  {s['calendar']['calendar_avg_violation_rate']:.1%} "
          f"[{'PASS' if s['calendar']['pass'] else 'FAIL'} < 10%]")
    print(f"  Butterfly Arbitrage: {s['butterfly']['butterfly_avg_violation_rate']:.1%} "
          f"[{'PASS' if s['butterfly']['pass'] else 'FAIL'} < 10%]")

    # CI Coverage & Sharpness
    c = results['coverage']
    print(f"\n--- CI Coverage & Sharpness ---")
    print(f"  {'Level':>6} | {'Coverage':>10} | {'Width':>8} | Status")
    print(f"  {'-'*45}")
    for level in [0.5, 0.8, 0.9, 0.95]:
        cov = c['overall'][level]
        wid = c['overall_width'][level]
        print(f"  {level:>5.0%}  | {cov:>9.1%}  | {wid:>7.4f}  | [target: {level:.0%}]")

    print(f"\n  Per-Horizon (90% CI):")
    for h in [1, 7, 14, 30]:
        if h in c['per_horizon'] and 0.9 in c['per_horizon'][h]:
            cov = c['per_horizon'][h][0.9]
            wid = c['per_horizon_width'][h][0.9]
            print(f"    h={h:>2}: {cov:.1%} (width: {wid:.4f})")

    # Conditionality
    if 'conditionality' in results:
        d = results['conditionality']
        print(f"\n--- Conditionality ---")
        print(f"  Conditioned CI Width:   {d['cond_width']:.4f}")
        print(f"  Unconditioned CI Width: {d['uncond_width']:.4f}")
        print(f"  Width Ratio:            {d['width_ratio']:.3f} (< 1.0 = conditioning helps)")
        print(f"  MAE Reduction:          {d['mae_reduction']:.1%}")
        print(f"  Conditionality:         [{'PASS' if d['pass'] else 'FAIL'}]")

    # CRPS
    if 'crps' in results:
        print(f"\n--- CRPS (lower is better) ---")
        for h in [1, 7, 14, 30]:
            print(f"  h={h:>2}: {results['crps']['crps_per_horizon'][h]:.6f}")
        print(f"  Overall: {results['crps']['crps_overall']:.6f}")

    # Time Series
    ts = results['time_series']
    print(f"\n--- Time Series ---")
    print(f"  ACF Correlation (mean 9pts): {ts['acf']['acf_correlation_mean']:.3f} "
          f"[{'PASS' if ts['acf']['pass'] else 'FAIL'} > 0.5]")
    print(f"  Kurtosis Ratio:              {ts['kurtosis']['kurtosis_ratio']:.3f} "
          f"[{'PASS' if ts['kurtosis']['pass'] else 'FAIL'} 0.5-2.0]")

    # Sample Quality
    print(f"\n--- Sample Quality ---")
    print(f"  Diversity: {results['diversity']['diversity']:.4f}")

    # Overall
    print("\n" + "=" * 60)
    all_pass = all([
        results['surface']['overall_pass'],
        results['coverage']['pass'],
        results.get('conditionality', {}).get('pass', True),
        ts['acf']['pass'],
        ts['kurtosis']['pass'],
    ])
    if all_pass:
        print("OVERALL: ALL TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED")
    print("=" * 60)


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


def main():
    parser = argparse.ArgumentParser(description="MCVD POC Validation Tests")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--precomputed", type=str, default=None,
                        help="Path to pre-generated .npz from generate_test_samples.py")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="DDIM steps (default: min(100, n_steps))")
    parser.add_argument("--atm_only", action="store_true",
                        help="Compute kurtosis on ATM point only")
    parser.add_argument("--skip_conditionality", action="store_true",
                        help="Skip conditionality test (saves time)")
    parser.add_argument("--conditionality_batches", type=int, default=10,
                        help="Max batches for conditionality test")
    args = parser.parse_args()

    output_dir = args.output_dir or "results/mcvd_poc/validation_tests"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Mode 1: Load pre-computed samples from .npz
    # ================================================================
    if args.precomputed:
        print("=" * 60)
        print("MCVD POC Validation Tests (precomputed)")
        print("=" * 60)
        print(f"Loading: {args.precomputed}")
        data = np.load(args.precomputed, allow_pickle=True)
        cond_samples = data['cond_samples']
        uncond_samples = data['uncond_samples']
        ground_truth = data['ground_truth']
        print(f"  Conditioned: {cond_samples.shape}")
        print(f"  Ground truth: {ground_truth.shape}")
        has_uncond = uncond_samples.size > 0
        if has_uncond:
            print(f"  Unconditioned: {uncond_samples.shape}")
        print(f"Output: {output_dir}")
        print("=" * 60)

        results = {}

        # Test 1: Surface Validity
        results['surface'] = run_surface_validity_tests(
            precomputed=(cond_samples, ground_truth),
        )

        # Test 2: CI Coverage & Sharpness
        results['coverage'] = compute_ci_coverage_and_sharpness(
            precomputed=(cond_samples, ground_truth),
        )

        # Test 3: Conditionality
        if has_uncond and not args.skip_conditionality:
            results['conditionality'] = test_conditionality(
                precomputed=(cond_samples, uncond_samples, ground_truth),
            )

        # Test 4: CRPS
        results['crps'] = test_crps_per_horizon(
            precomputed=(cond_samples, ground_truth),
        )

        # Test 5: Time Series
        ts_results = {}
        ts_results['acf'] = test_acf_preservation(
            precomputed=(cond_samples, ground_truth),
        )
        ts_results['kurtosis'] = test_kurtosis_matching(
            precomputed=(cond_samples, ground_truth),
            atm_only=args.atm_only,
        )
        results['time_series'] = ts_results

        # Test 6: Sample Quality
        results['diversity'] = compute_sample_diversity(
            precomputed=(cond_samples,),
        )

        # Print summary
        print_summary(results)

        # Visualizations
        print("\nGenerating visualizations...")
        plot_calibration_curve(results['coverage'], f"{output_dir}/calibration_curve.png")
        plot_acf_comparison(results['time_series']['acf'], f"{output_dir}/acf_comparison.png")

        # Path visualization from precomputed data
        idx = 0
        sample_gt = ground_truth[idx]  # (T, 5, 5)
        sample_paths = cond_samples[idx]  # (n_samples, T, 5, 5)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        grid_points = [
            ((0, 2), 'Short-term ATM'), ((0, 0), 'Short-term ITM'), ((0, 4), 'Short-term OTM'),
            ((4, 2), 'Long-term ATM'), ((4, 0), 'Long-term ITM'), ((4, 4), 'Long-term OTM'),
        ]
        for ax, ((tenor, strike), title) in zip(axes.flat, grid_points):
            for i in range(min(20, sample_paths.shape[0])):
                ax.plot(sample_paths[i, :, tenor, strike], alpha=0.3, color='blue', linewidth=0.5)
            ax.plot(sample_gt[:, tenor, strike], color='red', linewidth=2, label='Ground Truth')
            p5 = np.percentile(sample_paths[:, :, tenor, strike], 5, axis=0)
            p95 = np.percentile(sample_paths[:, :, tenor, strike], 95, axis=0)
            ax.fill_between(range(len(p5)), p5, p95, alpha=0.2, color='blue', label='90% CI')
            ax.set_title(title)
            ax.set_xlabel('Days ahead')
            ax.set_ylabel('IV')
            if ax == axes[0, 0]:
                ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/path_visualization.png", dpi=150)
        print(f"  Saved path visualization to {output_dir}/path_visualization.png")
        plt.close()

        # Save results JSON
        results_serializable = convert_to_serializable(results)
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nResults saved to {output_dir}/summary.json")
        print(f"All outputs saved to: {output_dir}")
        return

    # ================================================================
    # Mode 2: Generate samples live from model
    # ================================================================
    config = get_default_config()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Find model
    model_path = args.model_path
    if model_path is None:
        search_dirs = [config.output_dir, "models/backfill/mcvd_paper_aligned"]
        candidates = []
        for d in search_dirs:
            candidates.extend([
                f"{d}/best_coverage_model.pt",
                f"{d}/best_model.pt",
                f"{d}/final_model.pt",
            ])
        for path in candidates:
            if Path(path).exists():
                model_path = path
                break

    if model_path is None or not Path(model_path).exists():
        print(f"No trained model found. Run training first:")
        print(f"  python experiments/backfill/diffusion_poc/train_mcvd_poc.py")
        return

    print("=" * 60)
    print("MCVD POC Validation Tests")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Samples per history: {args.n_samples}")
    print(f"Max batches: {args.max_batches}")
    print(f"Sampler: {args.sampler} ({args.ddim_steps} steps)")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, poc_config, checkpoint = load_mcvd_model(model_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')} ({n_params:,} params)")

    # Default DDIM steps based on model's noise schedule
    if args.ddim_steps is None:
        args.ddim_steps = min(100, poc_config.n_steps)
    print(f"  Using {args.ddim_steps} DDIM steps (T={poc_config.n_steps})")

    # Detect short-frame models and set up autoregressive evaluation
    eval_future_len = 30  # Always evaluate at 30-day horizon
    use_ar = poc_config.future_len < eval_future_len
    if use_ar:
        n_blocks = eval_future_len // poc_config.future_len
        print(f"\n  Short-frame model (future_len={poc_config.future_len})")
        print(f"  Using autoregressive rollout: {n_blocks} blocks × {poc_config.future_len} frames = {n_blocks * poc_config.future_len} days")
        model = ARModelWrapper(model, n_blocks=n_blocks, n_inference_steps=args.ddim_steps)

    # Load test data — always use eval_future_len=30 for ground truth
    print("\nLoading test data...")
    data = np.load(poc_config.data_path)
    surfaces = data["surface"]
    test_dataset = VolSurfaceDataset(
        surfaces, poc_config.history_len, eval_future_len,
        start_idx=poc_config.test_start,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=poc_config.batch_size,
        shuffle=False, num_workers=2,
    )

    # Run all tests
    results = {}

    # Test 1: Surface Validity
    results['surface'] = run_surface_validity_tests(
        model, test_loader, n_samples=args.n_samples, max_batches=args.max_batches,
        device=device, sampler=args.sampler, n_inference_steps=args.ddim_steps,
    )

    # Test 2: CI Coverage & Sharpness
    results['coverage'] = compute_ci_coverage_and_sharpness(
        model, test_loader, n_samples=args.n_samples, max_batches=args.max_batches,
        device=device, sampler=args.sampler, n_inference_steps=args.ddim_steps,
    )

    # Test 3: Conditionality
    if not args.skip_conditionality:
        results['conditionality'] = test_conditionality(
            model, test_loader, n_samples=args.n_samples,
            max_batches=args.conditionality_batches,
            device=device, sampler=args.sampler, n_inference_steps=args.ddim_steps,
        )

    # Test 4: CRPS per Horizon
    results['crps'] = test_crps_per_horizon(
        model, test_loader, n_samples=args.n_samples, max_batches=args.max_batches,
        device=device, sampler=args.sampler, n_inference_steps=args.ddim_steps,
    )

    # Test 5: Time Series Properties
    ts_results = {}
    ts_results['acf'] = test_acf_preservation(
        model, test_loader, n_samples=min(10, args.n_samples),
        max_batches=args.max_batches, device=device,
        sampler=args.sampler, n_inference_steps=args.ddim_steps,
    )
    ts_results['kurtosis'] = test_kurtosis_matching(
        model, test_loader, n_samples=min(10, args.n_samples),
        max_batches=args.max_batches, device=device,
        sampler=args.sampler, n_inference_steps=args.ddim_steps,
        atm_only=args.atm_only,
    )
    results['time_series'] = ts_results

    # Test 6: Sample Quality
    results['diversity'] = compute_sample_diversity(
        model, test_loader, n_samples=args.n_samples, max_batches=args.max_batches,
        device=device, sampler=args.sampler, n_inference_steps=args.ddim_steps,
    )

    # Print summary
    print_summary(results)

    # Visualizations
    print("\nGenerating visualizations...")
    plot_calibration_curve(results['coverage'], f"{output_dir}/calibration_curve.png")
    plot_acf_comparison(results['time_series']['acf'], f"{output_dir}/acf_comparison.png")

    sample_batch = next(iter(test_loader))
    sample_future_gt = denormalize_iv(sample_batch["future"][0])
    visualize_generated_paths(
        model, sample_batch["history"][0], sample_future_gt,
        n_samples=20, device=device,
        output_path=f"{output_dir}/path_visualization.png",
        sampler=args.sampler, n_inference_steps=args.ddim_steps,
    )

    # Save results JSON
    results_serializable = convert_to_serializable(results)
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_dir}/summary.json")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
