#!/usr/bin/env python
"""
225b: Split-conformal prediction on 183c.

Provides distribution-free coverage guarantee while preserving 183c's
cross-cell correlation structure. Standard industry practice for
production risk systems.

Method:
1. Split validation windows into calibration (50%) and test (50%)
2. On calibration set: compute nonconformity scores per horizon
3. Find quantile that guarantees target coverage
4. On test set: inflate 183c's CIs by that quantile
5. Verify worst-window coverage meets target
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import build_rollout_windows
from experiments.backfill.block_ar.analyze_183c_best_mechanism import load_model
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


def generate_samples(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    n_samples: int = 48,
    batch_size: int = 16,
    chunk_size: int = 8,
) -> np.ndarray:
    """Generate multi-day samples from 183c."""
    N = history_01.shape[0]
    outputs = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        hist_norm = normalize_iv(history_01[start:end])
        with torch.no_grad():
            samp = model.sample_batched(
                hist_norm, n_samples=n_samples, n_steps=30, chunk_size=chunk_size
            )
        outputs.append(samp.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def compute_conformal_quantiles(
    samples: np.ndarray,
    gt: np.ndarray,
    target_coverage: float = 0.90,
    per_horizon: bool = True,
) -> np.ndarray:
    """Compute conformal quantiles from calibration set.

    Args:
        samples: (N, n_samples, 30, 5, 5)
        gt: (N, 30, 5, 5)
        target_coverage: target coverage level
        per_horizon: if True, compute separate quantile per horizon

    Returns:
        q_hat: (30,) or scalar — conformal inflation factor per horizon
    """
    N, K, T = samples.shape[0], samples.shape[1], samples.shape[2]
    gt_flat = gt.reshape(N, T, 25)
    samp_flat = samples.reshape(N, K, T, 25)

    # Compute prediction intervals
    lo = np.quantile(samp_flat, 0.05, axis=1)  # (N, T, 25)
    hi = np.quantile(samp_flat, 0.95, axis=1)  # (N, T, 25)
    width = hi - lo  # (N, T, 25)

    # Nonconformity score: how far GT falls outside the CI, normalized by width
    below = np.maximum(lo - gt_flat, 0) / np.maximum(width, 1e-8)
    above = np.maximum(gt_flat - hi, 0) / np.maximum(width, 1e-8)
    scores = np.maximum(below, above)  # (N, T, 25)

    # Per-window max nonconformity (worst cell at each horizon)
    if per_horizon:
        scores_per_h = scores.max(axis=2)  # (N, T) — worst cell per window per horizon
        # Quantile at target coverage level (with finite-sample correction)
        n_cal = scores_per_h.shape[0]
        q_level = min(target_coverage * (1 + 1 / n_cal), 1.0)
        q_hat = np.quantile(scores_per_h, q_level, axis=0)  # (T,)
    else:
        scores_all = scores.max(axis=(1, 2))  # (N,) — worst across all horizons and cells
        n_cal = len(scores_all)
        q_level = min(target_coverage * (1 + 1 / n_cal), 1.0)
        q_hat = np.quantile(scores_all, q_level)  # scalar

    return q_hat


def apply_conformal(
    samples: np.ndarray,
    q_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conformal inflation to prediction intervals.

    Returns inflated (lo, hi) arrays.
    """
    N, K, T = samples.shape[0], samples.shape[1], samples.shape[2]
    samp_flat = samples.reshape(N, K, T, 25)

    lo = np.quantile(samp_flat, 0.05, axis=1)  # (N, T, 25)
    hi = np.quantile(samp_flat, 0.95, axis=1)  # (N, T, 25)
    width = hi - lo

    # Inflate by conformal quantile
    if q_hat.ndim == 1:
        # Per-horizon: q_hat is (T,)
        inflation = q_hat[np.newaxis, :, np.newaxis] * width  # (N, T, 25)
    else:
        inflation = q_hat * width

    lo_conf = lo - inflation
    hi_conf = hi + inflation
    return lo_conf, hi_conf


def per_window_coverage(lo: np.ndarray, hi: np.ndarray, gt_flat: np.ndarray) -> np.ndarray:
    """Compute per-window coverage: fraction of (horizon, cell) covered."""
    covered = (gt_flat >= lo) & (gt_flat <= hi)
    return covered.mean(axis=(1, 2))


def main():
    parser = argparse.ArgumentParser(description="225b: Conformal prediction on 183c")
    parser.add_argument("--checkpoint", type=str,
                        default="models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=192)
    parser.add_argument("--n_samples", type=int, default=48)
    parser.add_argument("--target_coverage", type=float, default=0.90)
    parser.add_argument("--output_dir", type=str, default="results/block_ar/225b_conformal")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, _ = load_model(args.checkpoint, device)
    model.eval()

    # Build validation windows
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=30, future_len=30,
        test_start=args.test_start, val_size=args.val_size,
        max_windows=None, device=device, split="val",
    )
    history_01 = batch.history_01
    future_01 = batch.future_01
    N = history_01.shape[0]

    # Generate samples
    print(f"Generating {args.n_samples} samples for {N} windows...")
    samples = generate_samples(model, history_01, n_samples=args.n_samples)
    gt = future_01.cpu().numpy()
    gt_flat = gt.reshape(N, 30, 25)
    print(f"Samples shape: {samples.shape}")

    # Split: first half calibration, second half test
    n_cal = N // 2
    cal_samples, test_samples = samples[:n_cal], samples[n_cal:]
    cal_gt, test_gt = gt[:n_cal], gt[n_cal:]
    cal_gt_flat, test_gt_flat = gt_flat[:n_cal], gt_flat[n_cal:]
    print(f"Calibration: {n_cal} windows, Test: {N - n_cal} windows")

    # Raw coverage (before conformal)
    samp_flat = samples.reshape(N, args.n_samples, 30, 25)
    raw_lo = np.quantile(samp_flat, 0.05, axis=1)
    raw_hi = np.quantile(samp_flat, 0.95, axis=1)
    raw_cov = per_window_coverage(raw_lo, raw_hi, gt_flat)

    print(f"\n=== Raw 183c (before conformal) ===")
    print(f"  Mean coverage: {raw_cov.mean():.3f}")
    print(f"  Worst window:  {raw_cov.min():.3f} (win {raw_cov.argmin()})")
    print(f"  Windows < 70%: {(raw_cov < 0.70).sum()}")
    print(f"  Windows < 80%: {(raw_cov < 0.80).sum()}")

    # Compute conformal quantiles on calibration set
    q_hat = compute_conformal_quantiles(
        cal_samples, cal_gt,
        target_coverage=args.target_coverage,
        per_horizon=True,
    )
    print(f"\n=== Conformal Quantiles (per horizon) ===")
    for h in [0, 4, 9, 14, 29]:
        print(f"  h={h+1:2d}: q_hat={q_hat[h]:.4f} (inflate by {q_hat[h]*100:.1f}% of width)")

    # Apply to TEST set
    test_lo_conf, test_hi_conf = apply_conformal(test_samples, q_hat)
    test_cov_conf = per_window_coverage(test_lo_conf, test_hi_conf, test_gt_flat)

    # Also compute raw test coverage for comparison
    test_lo_raw = np.quantile(test_samples.reshape(N - n_cal, args.n_samples, 30, 25), 0.05, axis=1)
    test_hi_raw = np.quantile(test_samples.reshape(N - n_cal, args.n_samples, 30, 25), 0.95, axis=1)
    test_cov_raw = per_window_coverage(test_lo_raw, test_hi_raw, test_gt_flat)

    # Width comparison
    raw_width = (test_hi_raw - test_lo_raw).mean()
    conf_width = (test_hi_conf - test_lo_conf).mean()
    width_increase = (conf_width - raw_width) / raw_width * 100

    print(f"\n=== Test Set Results (conformal vs raw) ===")
    print(f"  {'Metric':<25s} {'Raw':>10s} {'Conformal':>10s}")
    print(f"  {'-'*47}")
    print(f"  {'Mean coverage':<25s} {test_cov_raw.mean():10.3f} {test_cov_conf.mean():10.3f}")
    print(f"  {'Worst window':<25s} {test_cov_raw.min():10.3f} {test_cov_conf.min():10.3f}")
    print(f"  {'Windows < 70%':<25s} {(test_cov_raw < 0.70).sum():10d} {(test_cov_conf < 0.70).sum():10d}")
    print(f"  {'Windows < 80%':<25s} {(test_cov_raw < 0.80).sum():10d} {(test_cov_conf < 0.80).sum():10d}")
    print(f"  {'P10 coverage':<25s} {np.quantile(test_cov_raw, 0.10):10.3f} {np.quantile(test_cov_conf, 0.10):10.3f}")
    print(f"  {'Mean CI width':<25s} {raw_width:10.5f} {conf_width:10.5f}")
    print(f"  {'Width increase':<25s} {'':>10s} {width_increase:+9.1f}%")

    # Save results
    results = {
        "conformal_quantiles": q_hat.tolist(),
        "target_coverage": args.target_coverage,
        "n_calibration": n_cal,
        "n_test": N - n_cal,
        "raw": {
            "mean_coverage": float(raw_cov.mean()),
            "worst_window": float(raw_cov.min()),
            "windows_below_70": int((raw_cov < 0.70).sum()),
            "windows_below_80": int((raw_cov < 0.80).sum()),
        },
        "conformal_test": {
            "mean_coverage": float(test_cov_conf.mean()),
            "worst_window": float(test_cov_conf.min()),
            "worst_window_idx": int(test_cov_conf.argmin()) + n_cal,
            "windows_below_70": int((test_cov_conf < 0.70).sum()),
            "windows_below_80": int((test_cov_conf < 0.80).sum()),
            "p10_coverage": float(np.quantile(test_cov_conf, 0.10)),
            "mean_width": float(conf_width),
            "width_increase_pct": float(width_increase),
        },
        "raw_test": {
            "mean_coverage": float(test_cov_raw.mean()),
            "worst_window": float(test_cov_raw.min()),
            "windows_below_70": int((test_cov_raw < 0.70).sum()),
            "windows_below_80": int((test_cov_raw < 0.80).sum()),
        },
    }
    with open(out_dir / "conformal_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save conformal quantiles for deployment
    np.save(out_dir / "conformal_quantiles.npy", q_hat)
    print(f"\nResults saved to {out_dir}")
    print(f"Conformal quantiles saved to {out_dir / 'conformal_quantiles.npy'}")


if __name__ == "__main__":
    main()
