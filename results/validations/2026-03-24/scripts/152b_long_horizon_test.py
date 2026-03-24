#!/usr/bin/env python
"""
Long-horizon verification for 152b AR Flow Matching model.

Generates 252-day paths by overriding model.future_len and checks:
  1. CI coverage at horizons d=30, 60, 120, 180, 252
  2. Monotonic spread growth with horizon
  3. Kurtosis at 252d horizon
  4. Cross-cell correlation at 252d
  5. Surface validity (explosion rate)

Usage:
    PYTHONPATH=. python results/validations/2026-03-24/scripts/152b_long_horizon_test.py \
        --model_path models/backfill/flow_152b/best_model.pt \
        --max_batches 5 --n_samples 50 --device cuda
"""

import argparse
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

from experiments.backfill.block_ar.eval_ar_flow import load_ar_flow_model
from diffusion.block_ar.single_pass_ar import denormalize_iv, normalize_iv
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from torch.utils.data import DataLoader


# Cell labels
LABELS_K = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
LABELS_T = ["1M", "3M", "6M", "1Y", "2Y"]
CELL_NAMES = [f"{t}/{k}" for t in LABELS_T for k in LABELS_K]
CHECK_HORIZONS = [30, 60, 120, 180, 252]


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def compute_ci_coverage(samples, ground_truth, horizon_idx, alpha=0.1):
    """Compute per-cell 90% CI coverage at a given horizon index.

    Args:
        samples: (N, S, T, 5, 5) generated samples
        ground_truth: (N, T, 5, 5) actual future
        horizon_idx: which time step to check (0-indexed)
        alpha: significance level (0.1 = 90% CI)

    Returns:
        dict with per-cell coverage and mean coverage
    """
    lo = alpha / 2
    hi = 1 - alpha / 2

    # samples at horizon: (N, S, 5, 5)
    s = samples[:, :, horizon_idx, :, :]
    gt = ground_truth[:, horizon_idx, :, :]  # (N, 5, 5)

    # Quantiles: (N, 5, 5)
    q_lo = np.quantile(s, lo, axis=1)
    q_hi = np.quantile(s, hi, axis=1)

    # Coverage: (N, 5, 5)
    covered = (gt >= q_lo) & (gt <= q_hi)
    per_cell_cov = covered.mean(axis=0)  # (5, 5)
    mean_cov = per_cell_cov.mean()

    return {
        "mean_coverage": float(mean_cov),
        "per_cell_coverage": per_cell_cov.tolist(),
        "worst_cell": float(per_cell_cov.min()),
        "best_cell": float(per_cell_cov.max()),
    }


def compute_spread_by_horizon(samples, horizons):
    """Compute mean ensemble spread (std) at each horizon.

    Args:
        samples: (N, S, T, 5, 5)
        horizons: list of horizon indices

    Returns:
        dict mapping horizon to mean spread
    """
    result = {}
    for h in horizons:
        if h - 1 < samples.shape[2]:
            std = samples[:, :, h - 1, :, :].std(axis=1)  # (N, 5, 5)
            result[h] = float(std.mean())
    return result


def check_monotonic_spread(spread_dict):
    """Check if spread grows monotonically with horizon."""
    horizons = sorted(spread_dict.keys())
    values = [spread_dict[h] for h in horizons]
    monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    return monotonic, values


def compute_kurtosis_at_horizon(samples, horizon_range=None):
    """Compute kurtosis of daily changes at specified horizon range.

    Args:
        samples: (N, S, T, 5, 5)
        horizon_range: (start, end) horizon indices for daily changes

    Returns:
        dict with per-cell kurtosis
    """
    if horizon_range is None:
        horizon_range = (0, samples.shape[2])

    start, end = horizon_range
    # Daily changes: (N, S, T-1, 5, 5)
    if end > 1:
        changes = samples[:, :, start + 1:end, :, :] - samples[:, :, start:end - 1, :, :]
        changes_flat = changes.reshape(-1, 5, 5)

        kurtosis_map = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                cell_changes = changes_flat[:, i, j]
                kurtosis_map[i, j] = float(sp_stats.kurtosis(cell_changes, fisher=True))

        return {
            "kurtosis_map": kurtosis_map.tolist(),
            "mean_kurtosis": float(kurtosis_map.mean()),
            "min_kurtosis": float(kurtosis_map.min()),
            "max_kurtosis": float(kurtosis_map.max()),
        }
    return {"kurtosis_map": None, "mean_kurtosis": None}


def compute_cross_cell_correlation(samples, horizon_idx):
    """Compute cross-cell correlation at a given horizon.

    Args:
        samples: (N, S, T, 5, 5)
        horizon_idx: which time step

    Returns:
        dict with correlation matrix stats
    """
    # Daily changes up to horizon
    if horizon_idx < 1:
        return {}

    # Clamp horizon_idx to valid range (max = T-1 since we need pairs)
    T = samples.shape[2]
    h = min(horizon_idx, T - 1)
    changes = samples[:, :, 1:h + 1, :, :] - samples[:, :, :h, :, :]
    # Flatten spatial: (N*S*(T-1), 25)
    flat = changes.reshape(-1, 25)

    # Correlation matrix
    corr_mat = np.corrcoef(flat.T)  # (25, 25)

    # Effective rank (from eigenvalues)
    eigvals = np.linalg.eigvalsh(corr_mat)
    eigvals = np.maximum(eigvals, 1e-10)
    eigvals_norm = eigvals / eigvals.sum()
    eff_rank = float(np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm))))

    return {
        "effective_rank": eff_rank,
        "mean_off_diagonal": float((corr_mat.sum() - 25) / (25 * 24)),
        "min_corr": float(corr_mat.min()),
        "max_corr": float(corr_mat.max()),
    }


def compute_explosion_rate(samples):
    """Check for exploded surfaces (IV values > 2 or < -0.5)."""
    max_iv = samples.max(axis=(1, 2, 3, 4))  # (N,)
    min_iv = samples.min(axis=(1, 2, 3, 4))  # (N,)
    exploded = (max_iv > 2.0) | (min_iv < -0.5)
    return float(exploded.mean())


def main():
    parser = argparse.ArgumentParser(description="152b Long-Horizon Verification")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/validations/2026-03-24/analysis/152b_long_horizon")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("152b AR Flow Matching — Long-Horizon (252d) Verification")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, ckpt = load_ar_flow_model(args.model_path, device)
    original_future_len = model.future_len
    print(f"  Original future_len: {original_future_len}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint val_loss: {ckpt.get('val_loss', '?')}")

    # Override future_len to 252
    model.future_len = 252
    print(f"  Overridden future_len: {model.future_len}")

    # Load test data
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]

    # Use original future_len=30 for dataset (we only need history windows)
    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"  Test windows: {len(test_dataset)}")
    print(f"  Max batches: {args.max_batches}")
    print(f"  N samples: {args.n_samples}")

    # Generate 252-day samples
    print("\nGenerating 252-day samples...")
    t0 = time.time()
    all_samples = []
    all_gt_30d = []  # GT is only 30 days
    all_history = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= args.max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            print(f"  Batch {batch_idx + 1}/{args.max_batches}...", end=" ", flush=True)
            bt0 = time.time()

            # Generate 252-day samples
            samples = model.sample(history, n_samples=args.n_samples)
            # samples: (B, n_samples, 252, 5, 5) in [0,1]

            history_denorm = denormalize_iv(history)

            all_samples.append(samples.cpu().numpy())
            all_gt_30d.append(future_gt.cpu().numpy())
            all_history.append(history_denorm.cpu().numpy())

            bt1 = time.time()
            print(f"{bt1 - bt0:.1f}s")

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s")

    cond_samples = np.concatenate(all_samples)  # (N, S, 252, 5, 5)
    gt_30d = np.concatenate(all_gt_30d)  # (N, 30, 5, 5)
    history_arr = np.concatenate(all_history)  # (N, 30, 5, 5)

    N, S, T, H, W = cond_samples.shape
    print(f"  Samples shape: {cond_samples.shape}")
    print(f"  GT (30d) shape: {gt_30d.shape}")

    # ── Analysis ──
    results = {
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "hash": hash_file(args.model_path),
            "original_future_len": original_future_len,
            "overridden_future_len": 252,
            "epoch": ckpt.get("epoch", None),
            "val_loss": float(ckpt.get("val_loss", -1)),
        },
        "generation": {
            "n_windows": N,
            "n_samples": S,
            "n_horizons": T,
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    # 1. Explosion rate
    print("\n--- Explosion Rate ---")
    explosion_rate = compute_explosion_rate(cond_samples)
    print(f"  Explosion rate: {explosion_rate:.4f}")
    results["explosion_rate"] = explosion_rate

    # 2. CI coverage at checkpoints (only d=30 has GT)
    print("\n--- CI Coverage (d=30 only, GT available) ---")
    ci_30 = compute_ci_coverage(cond_samples, gt_30d, horizon_idx=29)
    print(f"  d=30 mean coverage: {ci_30['mean_coverage']:.3f}")
    print(f"  d=30 worst cell:    {ci_30['worst_cell']:.3f}")
    print(f"  d=30 best cell:     {ci_30['best_cell']:.3f}")
    results["ci_coverage_30d"] = ci_30

    # 3. Spread growth
    print("\n--- Spread Growth ---")
    spread = compute_spread_by_horizon(cond_samples, CHECK_HORIZONS)
    monotonic, spread_values = check_monotonic_spread(spread)
    print(f"  Spread by horizon:")
    for h, v in sorted(spread.items()):
        print(f"    d={h:3d}: {v:.6f}")
    print(f"  Monotonic: {monotonic}")
    results["spread"] = {
        "by_horizon": {str(k): v for k, v in spread.items()},
        "monotonic": monotonic,
    }

    # 4. Kurtosis at various horizons
    print("\n--- Kurtosis (daily changes) ---")
    kurtosis_results = {}
    for h in CHECK_HORIZONS:
        if h <= T:
            # Compute kurtosis over daily changes in the last 30 days before horizon
            start = max(0, h - 30)
            kurt = compute_kurtosis_at_horizon(cond_samples, horizon_range=(start, h))
            kurtosis_results[str(h)] = kurt
            print(f"  d={h:3d}: mean_kurtosis={kurt['mean_kurtosis']:.3f} "
                  f"(min={kurt['min_kurtosis']:.3f}, max={kurt['max_kurtosis']:.3f})")
    results["kurtosis"] = kurtosis_results

    # 5. Cross-cell correlation at various horizons
    print("\n--- Cross-Cell Correlation ---")
    corr_results = {}
    for h in CHECK_HORIZONS:
        if h <= T:
            cc = compute_cross_cell_correlation(cond_samples, h)
            corr_results[str(h)] = cc
            print(f"  d={h:3d}: eff_rank={cc['effective_rank']:.3f}, "
                  f"mean_off_diag={cc['mean_off_diagonal']:.3f}")
    results["cross_cell_correlation"] = corr_results

    # 6. Per-cell spread at d=252 (full heatmap)
    print("\n--- Per-Cell Spread at d=252 ---")
    spread_252 = cond_samples[:, :, -1, :, :].std(axis=1).mean(axis=0)  # (5, 5)
    print("  " + "  ".join(LABELS_K))
    for i, tenor_label in enumerate(LABELS_T):
        row = "  ".join(f"{spread_252[i, j]:.4f}" for j in range(5))
        print(f"  {tenor_label}: {row}")
    results["per_cell_spread_252d"] = spread_252.tolist()

    # 7. Mean IV level at d=252 (check for drift)
    print("\n--- Mean IV Level at d=252 ---")
    mean_iv_252 = cond_samples[:, :, -1, :, :].mean(axis=(0, 1))  # (5, 5)
    mean_iv_1 = cond_samples[:, :, 0, :, :].mean(axis=(0, 1))  # (5, 5)
    print("  d=1 mean IV:")
    print("  " + "  ".join(LABELS_K))
    for i, tenor_label in enumerate(LABELS_T):
        row = "  ".join(f"{mean_iv_1[i, j]:.4f}" for j in range(5))
        print(f"  {tenor_label}: {row}")
    print("  d=252 mean IV:")
    print("  " + "  ".join(LABELS_K))
    for i, tenor_label in enumerate(LABELS_T):
        row = "  ".join(f"{mean_iv_252[i, j]:.4f}" for j in range(5))
        print(f"  {tenor_label}: {row}")
    results["mean_iv"] = {
        "d1": mean_iv_1.tolist(),
        "d252": mean_iv_252.tolist(),
    }

    # 8. Path stationarity: rolling std of daily changes
    print("\n--- Path Stationarity (rolling std of daily changes) ---")
    # Take ATM 3M cell (index [1,2])
    atm_3m = cond_samples[:, :, :, 1, 2]  # (N, S, 252)
    daily_changes = atm_3m[:, :, 1:] - atm_3m[:, :, :-1]  # (N, S, 251)
    # Rolling std over 30-day windows
    rolling_points = list(range(29, T - 1, 30))  # every 30 days
    stationarity = {}
    for end in rolling_points:
        start = end - 29
        window = daily_changes[:, :, start:end + 1]
        roll_std = float(window.std())
        stationarity[end + 1] = roll_std
    print(f"  ATM 3M rolling std (30-day windows):")
    for d, s in sorted(stationarity.items()):
        print(f"    d={d:3d}: {s:.6f}")
    results["stationarity_atm3m"] = {str(k): v for k, v in stationarity.items()}

    # ── Summary verdict ──
    print("\n" + "=" * 70)
    print("SUMMARY VERDICT")
    print("=" * 70)

    checks = {
        "generation_successful": T == 252,
        "explosion_rate_ok": explosion_rate < 0.05,
        "spread_monotonic": monotonic,
        "ci_30d_adequate": ci_30["mean_coverage"] >= 0.80,
        "kurtosis_252d_finite": (
            kurtosis_results.get("252", {}).get("mean_kurtosis") is not None
            and np.isfinite(kurtosis_results["252"]["mean_kurtosis"])
        ),
        "eff_rank_252d_above_1": (
            corr_results.get("252", {}).get("effective_rank", 0) > 1.0
        ),
    }
    results["checks"] = checks

    all_pass = all(checks.values())
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Overall: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    results["overall_pass"] = all_pass

    # Save results
    out_path = Path(args.output_dir) / "long_horizon_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Also save raw numpy arrays for further analysis
    np.savez_compressed(
        Path(args.output_dir) / "samples_summary.npz",
        spread_252=spread_252,
        mean_iv_1=mean_iv_1,
        mean_iv_252=mean_iv_252,
    )
    print(f"Numpy summary saved to {Path(args.output_dir) / 'samples_summary.npz'}")


if __name__ == "__main__":
    main()
