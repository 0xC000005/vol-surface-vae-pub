#!/usr/bin/env python
"""
Pure-GT oracle test: does GT data itself, assembled into a conditional ensemble,
pass the 11-suite requirements?

If even the GT oracle fails some suites, those failures are METRIC-DESIGN artefacts
(the metrics are in tension) or the oracle method is too weak. If the GT oracle
passes all testable suites, the 241b/241c trade-off we observed is purely a
MODEL-CAPACITY limitation — real data satisfies every requirement simultaneously.

Two oracles for comparison:

ORACLE A — k-NN conditional oracle:
  For each val window i, find K=48 train windows whose HISTORY is closest to val
  history (L2 distance on last 10 days flattened). Use their GT futures as the
  "ensemble". This is the best a pure-retrieval model using only GT data can do.

ORACLE B — unconditional pool:
  For each val window, randomly sample K=48 GT futures from the entire train pool
  (no history matching). Isolates whether conditioning matters.

BASELINE — shuffled GT:
  For each val window, shuffle K=48 random GT futures from val set itself.

Runs all 10 suites that don't require a live model (conditionality needs model
sample_fn callbacks, so it's skipped; that gives 10/11 testable).

Suites testable here:
  1. surface_validity    4. block_ar            7. distributional_fidelity
  2. ci_coverage         5. cointegration       8. cross_cell_correlation
  3. time_series         6. regime_coverage     9. mean_reversion
                                                10. pathwise_jump_realism
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_surface_validity_tests,
    run_ci_coverage_tests,
    run_time_series_tests,
    run_block_ar_tests,
    run_cointegration_tests,
    run_regime_coverage_tests,
    run_distributional_fidelity_tests,
    run_cross_cell_correlation_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
)


def build_knn_oracle(
    val_hist: np.ndarray,
    val_future: np.ndarray,
    train_hist: np.ndarray,
    train_future: np.ndarray,
    K: int,
    hist_window: int = 10,
) -> np.ndarray:
    """For each val window, find K nearest train windows by history L2, return GT futures.

    Returns (N_val, K, T, 5, 5).
    """
    # Flatten last hist_window days of history into a feature vector
    val_feat = val_hist[:, -hist_window:].reshape(len(val_hist), -1)     # (N_val, hist_window*25)
    train_feat = train_hist[:, -hist_window:].reshape(len(train_hist), -1)
    # L2 distance matrix (N_val, N_train) — compute in chunks to save memory
    out = np.zeros((len(val_hist), K, val_future.shape[1], 5, 5), dtype=np.float32)
    chunk = 32
    for i in range(0, len(val_hist), chunk):
        j = min(i + chunk, len(val_hist))
        # Pairwise distances
        diff = val_feat[i:j, None, :] - train_feat[None, :, :]  # (chunk, N_train, D)
        dist = np.sum(diff * diff, axis=-1)  # (chunk, N_train)
        # Top-K smallest
        idx = np.argpartition(dist, kth=K, axis=1)[:, :K]  # (chunk, K)
        for k in range(j - i):
            neigh_idx = idx[k]
            out[i + k] = train_future[neigh_idx].reshape(K, val_future.shape[1], 5, 5)
    return out


def build_pool_oracle(
    val_future: np.ndarray,
    train_future: np.ndarray,
    K: int,
    seed: int = 42,
) -> np.ndarray:
    """For each val window, pick K random train futures (unconditional)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(val_future), K, val_future.shape[1], 5, 5), dtype=np.float32)
    for i in range(len(val_future)):
        idx = rng.choice(len(train_future), size=K, replace=False)
        out[i] = train_future[idx].reshape(K, val_future.shape[1], 5, 5)
    return out


def run_10_suites(cond_samples, ground_truth, history, returns, rollout_start,
                  history_len, future_len):
    """Run all 10 model-less suites on (cond_samples, ground_truth, history)."""
    results = {}
    results["surface"] = run_surface_validity_tests(cond_samples, ground_truth)
    results["coverage"] = run_ci_coverage_tests(cond_samples, ground_truth)
    results["time_series"] = run_time_series_tests(cond_samples, ground_truth)
    results["block_ar"] = run_block_ar_tests(cond_samples)
    results["cointegration"] = run_cointegration_tests(
        cond_samples, ground_truth, returns=returns,
        test_start=rollout_start, history_len=history_len, future_len=future_len,
    )
    results["regime_coverage"] = run_regime_coverage_tests(
        cond_samples, ground_truth, history
    )
    results["distributional_fidelity"] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history
    )
    results["cross_cell_correlation"] = run_cross_cell_correlation_tests(
        cond_samples, ground_truth
    )
    results["mean_reversion"] = run_mean_reversion_tests(
        cond_samples, ground_truth, history
    )
    results["pathwise_jump_realism"] = run_pathwise_jump_realism_tests(
        cond_samples, ground_truth
    )
    return results


def count_pass(results):
    n_pass = 0
    failed = []
    for name, d in results.items():
        if d.get("overall_pass", False):
            n_pass += 1
        else:
            failed.append(name)
    return n_pass, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--history_len", type=int, default=30)
    ap.add_argument("--future_len", type=int, default=30)
    ap.add_argument("--test_start", type=int, default=4511)
    ap.add_argument("--val_size", type=int, default=441)
    ap.add_argument("--knn_hist_window", type=int, default=10)
    ap.add_argument("--output_json", type=str, required=True)
    args = ap.parse_args()

    data = np.load("data/vol_surface_with_ret.npz")
    surf = data["surface"].astype(np.float32)
    returns = data["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len

    # Train: [0, max_train_idx - val_size); Val: [max_train_idx - val_size, max_train_idx)
    rollout_start = max_train_idx - args.val_size
    train_indices = np.arange(0, rollout_start)
    val_indices = np.arange(rollout_start, max_train_idx)
    surf_t = torch.from_numpy(surf).cpu()

    # Build windows (shape (N, T, 5, 5) each)
    train_hist, train_future = build_multistep_windows(train_indices, surf_t, args.history_len, args.future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_t, args.history_len, args.future_len)
    train_hist = train_hist.cpu().numpy()
    train_future = train_future.cpu().numpy().reshape(len(train_indices), args.future_len, 5, 5)
    val_hist_np = val_hist.cpu().numpy()
    val_future_np = val_future.cpu().numpy().reshape(len(val_indices), args.future_len, 5, 5)

    print(f"Train windows: {len(train_indices)}, Val: {len(val_indices)}")
    print(f"K = {args.K}")

    # --- ORACLE A: k-NN conditional ---
    print("\n[1/3] Building k-NN conditional oracle...")
    samples_knn = build_knn_oracle(
        val_hist_np, val_future_np, train_hist, train_future,
        K=args.K, hist_window=args.knn_hist_window,
    )
    print(f"  k-NN samples shape: {samples_knn.shape}")
    print("\nRunning 10 suites on k-NN oracle...")
    knn_results = run_10_suites(
        samples_knn, val_future_np, val_hist_np,
        returns=returns, rollout_start=rollout_start,
        history_len=args.history_len, future_len=args.future_len,
    )
    knn_pass, knn_fail = count_pass(knn_results)
    print(f"\nk-NN oracle: {knn_pass}/10 PASS  |  failed: {knn_fail}")

    # --- ORACLE B: Unconditional pool ---
    print("\n[2/3] Building unconditional pool oracle...")
    samples_pool = build_pool_oracle(val_future_np, train_future, K=args.K, seed=42)
    print("\nRunning 10 suites on pool oracle...")
    pool_results = run_10_suites(
        samples_pool, val_future_np, val_hist_np,
        returns=returns, rollout_start=rollout_start,
        history_len=args.history_len, future_len=args.future_len,
    )
    pool_pass, pool_fail = count_pass(pool_results)
    print(f"\nPool oracle: {pool_pass}/10 PASS  |  failed: {pool_fail}")

    # --- ORACLE C: GT replicated K (no spread) ---
    print("\n[3/4] Running GT-replicated oracle (should fail coverage)...")
    samples_repl = np.tile(val_future_np[:, np.newaxis], (1, args.K, 1, 1, 1))
    repl_results = run_10_suites(
        samples_repl, val_future_np, val_hist_np,
        returns=returns, rollout_start=rollout_start,
        history_len=args.history_len, future_len=args.future_len,
    )
    repl_pass, repl_fail = count_pass(repl_results)
    print(f"\nGT-replicated oracle: {repl_pass}/10 PASS  |  failed: {repl_fail}")

    # --- ORACLE D: GT-centered with GT-matched Gaussian spread (clean oracle) ---
    # pred_mean = GT exactly. Spread = calibrated per-cell-per-horizon std so the
    # MARGINAL distribution of samples matches GT marginal across windows.
    # This is the "perfect conditional predictor + correct marginal variance" test —
    # if this fails suites, the 11-suite is fundamentally tensioned.
    print("\n[4/5] Running GT-Gaussian oracle (calibrated spread + GT center)...")
    # σ[t, i, j] = std across val windows of GT[:, t, i, j] (marginal std).
    # Using the sample variance of GT rather than innovation variance because the
    # "ensemble marginal" we want to match is the GT marginal (which is what
    # level-KS/kurtosis/distributional_fidelity compare against).
    gt_std_per_hc = val_future_np.std(axis=0)  # (T, 5, 5)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((len(val_future_np), args.K, args.future_len, 5, 5)).astype(np.float32)
    noise = noise * gt_std_per_hc[np.newaxis, np.newaxis]
    samples_gaussian = val_future_np[:, np.newaxis] + noise  # (N, K, T, 5, 5)
    # Clip to support [0.01, 1.0] to avoid surface violations
    samples_gaussian = np.clip(samples_gaussian, 0.01, 0.999)
    gaussian_results = run_10_suites(
        samples_gaussian, val_future_np, val_hist_np,
        returns=returns, rollout_start=rollout_start,
        history_len=args.history_len, future_len=args.future_len,
    )
    gaussian_pass, gaussian_fail = count_pass(gaussian_results)
    print(f"\nGT-Gaussian oracle: {gaussian_pass}/10 PASS  |  failed: {gaussian_fail}")

    # --- ORACLE E: GT-centered with STRUCTURED spread from other val windows ---
    # For each val window i, use K=48 OTHER val windows' anomalies (GT[j] - GT[j]_mean)
    # as the noise pattern. This preserves joint structure (cross-cell, temporal) from
    # real data while centering on the correct pred_mean.
    print("\n[5/5] Running GT-Structured oracle (real-data anomaly noise + GT center)...")
    rng = np.random.default_rng(43)
    gt_mean_per_window = val_future_np.mean(axis=(1, 2, 3), keepdims=True)  # (N,1,1,1) scalar per window
    anomalies = val_future_np - gt_mean_per_window  # (N, T, 5, 5) — zero-mean paths with real joint structure
    structured_noise = np.zeros((len(val_future_np), args.K, args.future_len, 5, 5), dtype=np.float32)
    for i in range(len(val_future_np)):
        # Pick K different val windows as noise sources (exclude i itself when possible)
        choices = list(range(len(val_future_np)))
        if i in choices:
            choices.remove(i)
        pick = rng.choice(choices, size=args.K, replace=True)
        structured_noise[i] = anomalies[pick]  # (K, T, 5, 5)
    samples_struct = val_future_np[:, np.newaxis] + structured_noise
    samples_struct = np.clip(samples_struct, 0.01, 0.999)
    struct_results = run_10_suites(
        samples_struct, val_future_np, val_hist_np,
        returns=returns, rollout_start=rollout_start,
        history_len=args.history_len, future_len=args.future_len,
    )
    struct_pass, struct_fail = count_pass(struct_results)
    print(f"\nGT-Structured oracle: {struct_pass}/10 PASS  |  failed: {struct_fail}")

    summary = {
        "args": vars(args),
        "knn_oracle":      {"n_pass": knn_pass, "failed": knn_fail, "results": knn_results},
        "pool_oracle":     {"n_pass": pool_pass, "failed": pool_fail, "results": pool_results},
        "repl_oracle":     {"n_pass": repl_pass, "failed": repl_fail, "results": repl_results},
        "gaussian_oracle": {"n_pass": gaussian_pass, "failed": gaussian_fail, "results": gaussian_results},
        "struct_oracle":   {"n_pass": struct_pass, "failed": struct_fail, "results": struct_results},
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {args.output_json}")

    print("\n" + "="*80)
    print("PURE-GT ORACLE SUITE RESULTS")
    print("="*80)
    print(f"  k-NN conditional oracle: {knn_pass}/10  (fail: {knn_fail})")
    print(f"  Pool unconditional oracle: {pool_pass}/10  (fail: {pool_fail})")
    print(f"  GT-replicated K oracle:    {repl_pass}/10  (fail: {repl_fail})")
    print("="*80)
    print("INTERPRETATION:")
    if knn_pass >= 8:
        print("  k-NN oracle passes most suites → GT data can satisfy the metrics")
        print("  simultaneously. 241b/241c trade-off is a MODEL capacity issue.")
    else:
        print("  k-NN oracle passes few suites → metrics may have fundamental tension,")
        print("  OR k-NN retrieval isn't good enough at capturing conditioning.")


if __name__ == "__main__":
    main()
