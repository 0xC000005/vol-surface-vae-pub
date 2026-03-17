#!/usr/bin/env python
"""Extended evaluation script for model comparison.

Computes 6 metric categories beyond the standard 8-suite test:
  A. Trading P&L (ATM vega, calendar spread, butterfly)
  B. Sample Diversity (mean pairwise L2 at h=30)
  C. Factor Structure (correlation, eigenvalues, participation ratio)
  D. Marginal Quality (kurtosis, Wasserstein, Anderson-Darling, KS per cell)
  E. Inference Speed (seconds per 50 samples)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/evaluate_comparison.py \
        --model_path models/backfill/afcrps_99m_v2/best_model.pt \
        --model_name 99m_v2 --no_ema --output_dir results/comparison \
        --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _to_list(x):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    return x


def save_json(data, path):
    """Save dict to JSON with numpy conversion."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_list(data), f, indent=2)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Sample generation
# ─────────────────────────────────────────────────────────────────────

def generate_samples(model, loader, n_samples, max_batches, device, qmapper=None,
                     qmap_alpha=0.3, qmap_reflect=False):
    """Generate samples and collect GT across batches.

    Returns:
        all_gen: (N, K, 30, 5, 5) numpy in IV space [0, 1]
        all_gt:  (N, 30, 5, 5) numpy in IV space [0, 1]
        all_hist: (N, 30, 5, 5) numpy in IV space [0, 1]
    """
    all_gen = []
    all_gt = []
    all_hist = []

    torch.manual_seed(42)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        history = batch["history"].to(device)
        future_gt = denormalize_iv(batch["future"].to(device))

        extra_hist = batch.get("history_returns")
        if extra_hist is not None:
            extra_hist = extra_hist.to(device)

        with torch.no_grad():
            samples = model.sample(history, n_samples=n_samples, extra_hist=extra_hist)
            # samples: (B, K, 30, 5, 5) in [0, 1]

        samples_np = samples.cpu().numpy()
        gt_np = future_gt.cpu().numpy()
        hist_np = denormalize_iv(history).cpu().numpy()

        # Apply quantile mapping if provided
        if qmapper is not None:
            samples_np = qmapper.apply(samples_np, hist_np, reflect=qmap_reflect)

        all_gen.append(samples_np)
        all_gt.append(gt_np)
        all_hist.append(hist_np)

        print(f"  Batch {batch_idx + 1}/{max_batches}: "
              f"gen {samples_np.shape}, gt {gt_np.shape}")

    all_gen = np.concatenate(all_gen, axis=0)   # (N, K, 30, 5, 5)
    all_gt = np.concatenate(all_gt, axis=0)     # (N, 30, 5, 5)
    all_hist = np.concatenate(all_hist, axis=0) # (N, 30, 5, 5)
    print(f"  Total: {all_gen.shape[0]} windows, {all_gen.shape[1]} samples each")
    return all_gen, all_gt, all_hist


# ─────────────────────────────────────────────────────────────────────
# A. Trading P&L
# ─────────────────────────────────────────────────────────────────────

def compute_trading_pnl(all_gen, all_gt):
    """Compute trading P&L metrics for 3 strategies at h=1 and h=30.

    Args:
        all_gen: (N, K, 30, 5, 5) IV surfaces
        all_gt:  (N, 30, 5, 5) IV surfaces
    """
    print("\n[A] Computing Trading P&L...")
    N, K, T, H, W = all_gen.shape

    strategies = {
        "atm_vega": {
            "desc": "ATM vega (1M/ATM)",
            "fn": lambda iv_change: iv_change[..., 0, 2],
        },
        "calendar": {
            "desc": "Calendar spread (1M-1Y ATM)",
            "fn": lambda iv_change: iv_change[..., 0, 2] - iv_change[..., 3, 2],
        },
        "butterfly": {
            "desc": "Butterfly (1M smile)",
            "fn": lambda iv_change: (
                iv_change[..., 0, 0] + iv_change[..., 0, 4]
                - 2.0 * iv_change[..., 0, 2]
            ),
        },
    }

    results = {}
    for strat_name, strat in strategies.items():
        results[strat_name] = {}
        for h_label, h_idx in [("h1", 0), ("h30", 29)]:
            # Generated P&L: IV change from t=0 to t=h for all windows x samples
            # Anchor is the last history frame (same as gen[:, :, 0] at h=0)
            # IV change = gen[:, :, h] - gen[:, :, 0]
            gen_iv_change = all_gen[:, :, h_idx] - all_gen[:, :, 0]  # (N, K, 5, 5)
            gen_pnl = strat["fn"](gen_iv_change).flatten()  # (N*K,)

            gt_iv_change = all_gt[:, h_idx] - all_gt[:, 0]  # (N, 5, 5)
            gt_pnl = strat["fn"](gt_iv_change).flatten()  # (N,)

            # Metrics
            mean_bias = float(np.mean(gen_pnl) - np.mean(gt_pnl))
            gen_std = float(np.std(gen_pnl))
            gt_std = float(np.std(gt_pnl))
            std_ratio = gen_std / gt_std if gt_std > 1e-10 else float("inf")

            ks_stat, ks_pval = sp_stats.ks_2samp(gen_pnl, gt_pnl)

            gen_var99 = float(np.percentile(gen_pnl, 1))
            gt_var99 = float(np.percentile(gt_pnl, 1))
            var99_ratio = (
                gen_var99 / gt_var99
                if abs(gt_var99) > 1e-10
                else float("inf")
            )

            results[strat_name][h_label] = {
                "mean_pnl_bias": mean_bias,
                "gen_mean": float(np.mean(gen_pnl)),
                "gt_mean": float(np.mean(gt_pnl)),
                "std_ratio": std_ratio,
                "gen_std": gen_std,
                "gt_std": gt_std,
                "ks_stat": float(ks_stat),
                "ks_pval": float(ks_pval),
                "var99_ratio": var99_ratio,
                "gen_var99": gen_var99,
                "gt_var99": gt_var99,
            }
            print(f"  {strat['desc']} {h_label}: bias={mean_bias:.6f}, "
                  f"std_ratio={std_ratio:.3f}, KS={ks_stat:.4f}, "
                  f"VaR99_ratio={var99_ratio:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# B. Sample Diversity
# ─────────────────────────────────────────────────────────────────────

def compute_sample_diversity(all_gen):
    """Compute mean pairwise L2 distance between samples at h=30.

    Args:
        all_gen: (N, K, 30, 5, 5)
    """
    print("\n[B] Computing Sample Diversity...")
    N, K = all_gen.shape[0], all_gen.shape[1]
    samples_h30 = all_gen[:, :, -1]  # (N, K, 5, 5)

    diversities = []
    for i in range(N):
        flat = samples_h30[i].reshape(K, 25)  # (K, 25)
        dists = pdist(flat)  # pairwise L2 distances
        diversities.append(float(dists.mean()))

    diversities = np.array(diversities)
    results = {
        "mean": float(diversities.mean()),
        "std": float(diversities.std()),
        "min": float(diversities.min()),
        "max": float(diversities.max()),
        "n_windows": N,
    }
    print(f"  Mean pairwise L2 at h=30: {results['mean']:.6f} "
          f"(std={results['std']:.6f}, range=[{results['min']:.6f}, {results['max']:.6f}])")
    return results


# ─────────────────────────────────────────────────────────────────────
# C. Factor Structure
# ─────────────────────────────────────────────────────────────────────

def compute_factor_structure(all_gen, all_gt):
    """Compute correlation matrices and eigenvalue decomposition.

    Args:
        all_gen: (N, K, 30, 5, 5)
        all_gt:  (N, 30, 5, 5)
    """
    print("\n[C] Computing Factor Structure...")

    def _factor_metrics(deltas_flat):
        """Compute factor metrics from flat daily changes (M, 25)."""
        corr = np.corrcoef(deltas_flat.T)  # (25, 25)
        # Handle NaN from constant columns
        corr = np.nan_to_num(corr, nan=0.0)

        eigvals = np.linalg.eigvalsh(corr)[::-1]
        eigvals = np.maximum(eigvals, 0.0)  # clip small negatives

        total = eigvals.sum()
        participation_ratio = (
            (total ** 2) / (eigvals ** 2).sum()
            if (eigvals ** 2).sum() > 1e-10 else 0.0
        )

        # Variance explained by top 5 PCs
        pc_var = (eigvals[:5] / total * 100).tolist() if total > 1e-10 else [0.0] * 5

        # Mean off-diagonal correlation
        mask = ~np.eye(25, dtype=bool)
        mean_corr = float(corr[mask].mean())

        return {
            "mean_cross_corr": mean_corr,
            "effective_rank": float(participation_ratio),
            "pc_var_explained": pc_var,
            "corr_matrix": corr.tolist(),
        }

    # Generated daily changes
    gen_deltas = all_gen[:, :, 1:] - all_gen[:, :, :-1]  # (N, K, 29, 5, 5)
    gen_flat = gen_deltas.reshape(-1, 25)  # (N*K*29, 25)

    # GT daily changes
    gt_deltas = all_gt[:, 1:] - all_gt[:, :-1]  # (N, 29, 5, 5)
    gt_flat = gt_deltas.reshape(-1, 25)  # (N*29, 25)

    gen_metrics = _factor_metrics(gen_flat)
    gt_metrics = _factor_metrics(gt_flat)

    gen_corr = np.array(gen_metrics["corr_matrix"])
    gt_corr = np.array(gt_metrics["corr_matrix"])
    frob = float(np.linalg.norm(gen_corr - gt_corr, "fro"))

    results = {
        "gen": gen_metrics,
        "gt": gt_metrics,
        "corr_frobenius": frob,
    }

    print(f"  Gen: mean_corr={gen_metrics['mean_cross_corr']:.3f}, "
          f"eff_rank={gen_metrics['effective_rank']:.2f}, "
          f"PC1={gen_metrics['pc_var_explained'][0]:.1f}%")
    print(f"  GT:  mean_corr={gt_metrics['mean_cross_corr']:.3f}, "
          f"eff_rank={gt_metrics['effective_rank']:.2f}, "
          f"PC1={gt_metrics['pc_var_explained'][0]:.1f}%")
    print(f"  Frobenius(gen-gt): {frob:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# D. Marginal Quality
# ─────────────────────────────────────────────────────────────────────

def compute_marginal_quality(all_gen, all_gt):
    """Per-cell distributional metrics on daily changes.

    Args:
        all_gen: (N, K, 30, 5, 5)
        all_gt:  (N, 30, 5, 5)
    """
    print("\n[D] Computing Marginal Quality...")

    gen_deltas = all_gen[:, :, 1:] - all_gen[:, :, :-1]  # (N, K, 29, 5, 5)
    gen_flat = gen_deltas.reshape(-1, 25)  # (N*K*29, 25)

    gt_deltas = all_gt[:, 1:] - all_gt[:, :-1]  # (N, 29, 5, 5)
    gt_flat = gt_deltas.reshape(-1, 25)  # (N*29, 25)

    kurt_grid = np.zeros((5, 5))
    wass_grid = np.zeros((5, 5))
    ad_stat_grid = np.zeros((5, 5))
    ad_pval_grid = np.zeros((5, 5))
    ks_stat_grid = np.zeros((5, 5))

    worst_kurt_ratio = 0.0
    worst_cell = [0, 0]
    cells_above_3x = 0
    cells_above_5x = 0
    ks_pass_count = 0

    for cell_idx in range(25):
        r, c = divmod(cell_idx, 5)
        gen_c = gen_flat[:, cell_idx]
        gt_c = gt_flat[:, cell_idx]

        # Kurtosis ratio
        gen_kurt = float(sp_stats.kurtosis(gen_c))  # excess kurtosis
        gt_kurt = float(sp_stats.kurtosis(gt_c))
        if abs(gt_kurt) > 0.01:
            kurt_ratio = gen_kurt / gt_kurt
        else:
            kurt_ratio = float("inf")
        kurt_grid[r, c] = kurt_ratio

        abs_ratio = abs(kurt_ratio)
        if abs_ratio > abs(worst_kurt_ratio):
            worst_kurt_ratio = kurt_ratio
            worst_cell = [r, c]
        if abs_ratio > 3.0:
            cells_above_3x += 1
        if abs_ratio > 5.0:
            cells_above_5x += 1

        # Wasserstein distance
        wass = float(sp_stats.wasserstein_distance(gen_c, gt_c))
        wass_grid[r, c] = wass

        # Anderson-Darling k-sample test
        try:
            ad_result = sp_stats.anderson_ksamp([gen_c, gt_c])
            ad_stat_grid[r, c] = float(ad_result.statistic)
            ad_pval_grid[r, c] = float(ad_result.pvalue)
        except Exception:
            ad_stat_grid[r, c] = float("nan")
            ad_pval_grid[r, c] = float("nan")

        # KS test
        ks_stat, ks_pval = sp_stats.ks_2samp(gen_c, gt_c)
        ks_stat_grid[r, c] = float(ks_stat)
        if ks_stat < 0.15:
            ks_pass_count += 1

    results = {
        "kurtosis_ratio_grid": kurt_grid.tolist(),
        "max_kurtosis_ratio": float(worst_kurt_ratio),
        "cells_above_3x": cells_above_3x,
        "cells_above_5x": cells_above_5x,
        "worst_cell": worst_cell,
        "wasserstein_grid": wass_grid.tolist(),
        "mean_wasserstein": float(wass_grid.mean()),
        "ad_statistic_grid": ad_stat_grid.tolist(),
        "ad_pvalue_grid": ad_pval_grid.tolist(),
        "ks_pass_count": ks_pass_count,
    }

    print(f"  KS pass (<0.15): {ks_pass_count}/25")
    print(f"  Mean Wasserstein: {wass_grid.mean():.6f}")
    print(f"  Worst kurtosis ratio: {worst_kurt_ratio:.3f} at cell {worst_cell}")
    print(f"  Cells above 3x kurtosis: {cells_above_3x}, above 5x: {cells_above_5x}")

    return results


# ─────────────────────────────────────────────────────────────────────
# E. Inference Speed
# ─────────────────────────────────────────────────────────────────────

def compute_inference_speed(model, loader, n_samples, device):
    """Time model.sample() for a single history window.

    Args:
        model: loaded model
        loader: data loader (uses first batch)
        n_samples: number of samples to generate
        device: "cuda" or "cpu"
    """
    print("\n[E] Computing Inference Speed...")

    batch = next(iter(loader))
    hist = batch["history"].to(device)[:1]  # single window
    extra_hist = batch.get("history_returns")
    if extra_hist is not None:
        extra_hist = extra_hist.to(device)[:1]

    # Warm-up
    with torch.no_grad():
        model.sample(hist, n_samples=n_samples, extra_hist=extra_hist)
    if device == "cuda":
        torch.cuda.synchronize()

    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model.sample(hist, n_samples=n_samples, extra_hist=extra_hist)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    results = {
        "seconds_per_50_samples": elapsed,
        "samples_per_second": n_samples / elapsed if elapsed > 0 else float("inf"),
    }
    print(f"  {elapsed:.4f}s per {n_samples} samples "
          f"({results['samples_per_second']:.1f} samples/sec)")
    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extended model evaluation for comparison"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name used in output filenames")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quantile_map", type=str, default=None,
                        help="Path to quantile_map.npz for post-hoc mapping")
    parser.add_argument("--qmap_alpha", type=float, default=0.3)
    parser.add_argument("--qmap_reflect", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Extended Evaluation: {args.model_name} ===")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Batches: {args.max_batches}, Samples: {args.n_samples}")

    # ── Load model ──
    print("\nLoading model...")
    checkpoint = torch.load(args.model_path, weights_only=False, map_location="cpu")
    config = SinglePassConfig(**checkpoint["config"])
    model = SinglePassBlockAR(config)

    if args.no_ema or "ema_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Loaded model_state_dict (no EMA)")
    else:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("  Loaded ema_state_dict")

    model.eval().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Load data ──
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    returns = data.get("ret")

    # Use test split (same as test_block_ar_requirements.py: start_idx=4540)
    test_start = 4540
    # Only pass returns if model was trained with extra_features
    use_returns = returns if (returns is not None and getattr(config, 'extra_features', 0) > 0) else None
    dataset = VolSurfaceDataset(
        surfaces,
        history_len=config.history_len,
        future_len=config.future_len,
        start_idx=test_start,
        returns=use_returns,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # ── Load quantile mapper (optional) ──
    qmapper = None
    if args.quantile_map:
        try:
            from experiments.backfill.block_ar.quantile_mapper import QuantileMapper
            qmapper = QuantileMapper(args.quantile_map, alpha=args.qmap_alpha)
            print(f"  Quantile mapper loaded: {args.quantile_map} "
                  f"(alpha={args.qmap_alpha}, reflect={args.qmap_reflect})")
        except Exception as e:
            print(f"  WARNING: Could not load quantile mapper: {e}")
            qmapper = None

    # ── Generate samples ──
    print("\nGenerating samples...")
    all_gen, all_gt, all_hist = generate_samples(
        model, loader, args.n_samples, args.max_batches, args.device,
        qmapper=qmapper, qmap_alpha=args.qmap_alpha,
        qmap_reflect=args.qmap_reflect,
    )

    # ── Compute metrics ──
    name = args.model_name
    out = args.output_dir

    # A. Trading P&L
    pnl_results = compute_trading_pnl(all_gen, all_gt)
    save_json(pnl_results, f"{out}/trading_pnl_{name}.json")

    # B. Sample Diversity
    diversity_results = compute_sample_diversity(all_gen)
    save_json(diversity_results, f"{out}/sample_diversity_{name}.json")

    # C. Factor Structure
    factor_results = compute_factor_structure(all_gen, all_gt)
    save_json(factor_results, f"{out}/factor_structure_{name}.json")

    # D. Marginal Quality
    marginal_results = compute_marginal_quality(all_gen, all_gt)
    save_json(marginal_results, f"{out}/marginal_quality_{name}.json")

    # E. Inference Speed
    speed_results = compute_inference_speed(model, loader, args.n_samples, args.device)
    save_json(speed_results, f"{out}/inference_speed_{name}.json")

    print(f"\n=== Done: {name} ===")
    print(f"  All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
