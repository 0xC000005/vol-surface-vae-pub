"""Diagnostic: rank histogram, spread-skill ratio, conditional KS, PIT histogram.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_marginal.py \
        --model_path models/backfill/afcrps_89n_shared_z/best_coverage_model.pt \
        --no_ema --max_batches 20 --n_samples 50 --device cuda
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, uniform
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def classify_regime(history: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Classify each window as calm (0) or turb (1) based on vol-of-vol.

    Args:
        history: (N, T_hist, 5, 5) denormalized IV
    Returns:
        regime: (N,) boolean, True = turb
    """
    mean_iv = history.mean(axis=(-1, -2))  # (N, T)
    daily_chg = np.diff(mean_iv, axis=1)   # (N, T-1)
    vov = daily_chg.std(axis=1)            # (N,)
    return vov > np.median(vov)


def rank_histogram(cond_samples, ground_truth):
    """Rank histogram per cell, pooled across horizons and windows.

    For each (window, horizon, cell): where does GT fall among ensemble members?
    If calibrated, rank should be uniform over {0, 1, ..., n_samples}.

    Args:
        cond_samples: (N, S, T, 5, 5)
        ground_truth: (N, T, 5, 5)
    Returns:
        rank_counts: (5, 5, S+1) histogram counts per cell
    """
    N, S, T = cond_samples.shape[:3]
    # Rank: count how many members are below GT
    # (N, S, T, 5, 5) < (N, 1, T, 5, 5) → sum over S
    gt_expanded = ground_truth[:, np.newaxis, :, :, :]  # (N, 1, T, 5, 5)
    ranks = (cond_samples < gt_expanded).sum(axis=1)  # (N, T, 5, 5) values in [0, S]

    rank_counts = np.zeros((5, 5, S + 1), dtype=int)
    for r in range(5):
        for c in range(5):
            cell_ranks = ranks[:, :, r, c].ravel()  # (N*T,)
            rank_counts[r, c] = np.bincount(cell_ranks, minlength=S + 1)

    return rank_counts


def spread_skill_ratio(cond_samples, ground_truth):
    """Spread-skill ratio per cell, per horizon.

    spread = mean ensemble std (across members)
    skill = RMSE of ensemble mean vs GT
    SSR = spread / skill. SSR=1.0 = perfectly calibrated.

    Args:
        cond_samples: (N, S, T, 5, 5)
        ground_truth: (N, T, 5, 5)
    Returns:
        ssr_per_cell: (5, 5) pooled across all horizons
        ssr_per_horizon: (T, 5, 5) per horizon
    """
    # Ensemble mean and spread per window
    ens_mean = cond_samples.mean(axis=1)  # (N, T, 5, 5)
    ens_std = cond_samples.std(axis=1)    # (N, T, 5, 5)

    # Per horizon
    T = ground_truth.shape[1]
    ssr_per_horizon = np.zeros((T, 5, 5))
    for t in range(T):
        for r in range(5):
            for c in range(5):
                spread = ens_std[:, t, r, c].mean()
                rmse = np.sqrt(((ens_mean[:, t, r, c] - ground_truth[:, t, r, c]) ** 2).mean())
                ssr_per_horizon[t, r, c] = spread / max(rmse, 1e-8)

    # Pooled across horizons
    ssr_per_cell = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            spread = ens_std[:, :, r, c].mean()
            rmse = np.sqrt(((ens_mean[:, :, r, c] - ground_truth[:, :, r, c]) ** 2).mean())
            ssr_per_cell[r, c] = spread / max(rmse, 1e-8)

    return ssr_per_cell, ssr_per_horizon


def conditional_ks(cond_samples, ground_truth, regime_mask):
    """KS test on daily changes, split by regime.

    Args:
        cond_samples: (N, S, T, 5, 5)
        ground_truth: (N, T, 5, 5)
        regime_mask: (N,) boolean, True = turb
    Returns:
        dict with calm/turb KS grids
    """
    n_samp = min(5, cond_samples.shape[1])

    gt_diff = np.diff(ground_truth, axis=1)           # (N, T-1, 5, 5)
    gen_diff = np.diff(cond_samples[:, :n_samp], axis=2)  # (N, n_samp, T-1, 5, 5)

    results = {}
    for regime_name, mask in [("calm", ~regime_mask), ("turb", regime_mask)]:
        ks_grid = np.zeros((5, 5))
        n_windows = mask.sum()
        for r in range(5):
            for c in range(5):
                gt_vals = gt_diff[mask, :, r, c].ravel()
                gen_vals = gen_diff[mask, :, :, r, c].ravel()
                if len(gt_vals) < 10:
                    ks_grid[r, c] = np.nan
                    continue
                stat, _ = ks_2samp(gt_vals, gen_vals[:len(gt_vals)])
                ks_grid[r, c] = stat
        results[regime_name] = {
            "ks_grid": ks_grid,
            "n_pass": int((ks_grid < 0.15).sum()),
            "n_windows": int(n_windows),
        }

    # Also unconditional for comparison
    ks_grid_all = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            gt_vals = gt_diff[:, :, r, c].ravel()
            gen_vals = gen_diff[:, :, :, r, c].ravel()
            stat, _ = ks_2samp(gt_vals, gen_vals[:len(gt_vals)])
            ks_grid_all[r, c] = stat
    results["unconditional"] = {
        "ks_grid": ks_grid_all,
        "n_pass": int((ks_grid_all < 0.15).sum()),
    }

    return results


def pit_histogram(cond_samples, ground_truth):
    """Probability Integral Transform per cell.

    For each (window, horizon, cell): fraction of ensemble below GT.
    If calibrated, PIT values should be U(0,1).

    Args:
        cond_samples: (N, S, T, 5, 5)
        ground_truth: (N, T, 5, 5)
    Returns:
        pit_values: (5, 5, N*T) PIT values per cell
        pit_ks: (5, 5) KS statistic vs U(0,1)
    """
    N, S, T = cond_samples.shape[:3]
    gt_expanded = ground_truth[:, np.newaxis, :, :, :]
    # Fraction of members below GT (with jitter for ties)
    pit = (cond_samples < gt_expanded).mean(axis=1)  # (N, T, 5, 5) in [0, 1]

    pit_values = np.zeros((5, 5, N * T))
    pit_ks = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            vals = pit[:, :, r, c].ravel()
            pit_values[r, c] = vals
            stat, _ = ks_2samp(vals, np.random.uniform(0, 1, len(vals)))
            # Better: use KS test against uniform CDF directly
            stat2, _ = ks_2samp(vals, uniform.rvs(size=len(vals)))
            pit_ks[r, c] = stat

    return pit_values, pit_ks


def plot_rank_histograms(rank_counts, output_dir):
    """Plot 5x5 grid of rank histograms."""
    S_plus_1 = rank_counts.shape[2]
    fig, axes = plt.subplots(5, 5, figsize=(15, 12))
    fig.suptitle("Rank Histograms per Cell (uniform = calibrated)", fontsize=14)

    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            counts = rank_counts[r, c]
            expected = counts.sum() / S_plus_1
            bars = ax.bar(range(S_plus_1), counts, color="steelblue", alpha=0.7, width=1.0)
            ax.axhline(expected, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"({r},{c})", fontsize=9)
            ax.set_xticks([])
            if c == 0:
                ax.set_ylabel("Count", fontsize=8)
            # Color-code: U-shape = underdispersed, dome = overdispersed
            edge_frac = (counts[0] + counts[-1]) / (2 * expected) if expected > 0 else 1
            if edge_frac > 1.3:
                ax.set_facecolor("#fff0f0")  # red tint = underdispersed
            elif edge_frac < 0.7:
                ax.set_facecolor("#f0fff0")  # green tint = overdispersed

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rank_histograms.png"), dpi=150)
    plt.close()
    print(f"  Saved rank_histograms.png")


def plot_pit_histograms(pit_values, output_dir):
    """Plot 5x5 grid of PIT histograms."""
    fig, axes = plt.subplots(5, 5, figsize=(15, 12))
    fig.suptitle("PIT Histograms per Cell (uniform = calibrated)", fontsize=14)

    n_bins = 20
    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            vals = pit_values[r, c]
            counts, edges = np.histogram(vals, bins=n_bins, range=(0, 1))
            expected = len(vals) / n_bins
            ax.bar(edges[:-1], counts, width=1.0/n_bins, color="steelblue", alpha=0.7, align="edge")
            ax.axhline(expected, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"({r},{c})", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            if c == 0:
                ax.set_ylabel("Count", fontsize=8)
            # Color-code
            edge_frac = (counts[0] + counts[-1]) / (2 * expected) if expected > 0 else 1
            if edge_frac > 1.3:
                ax.set_facecolor("#fff0f0")
            elif edge_frac < 0.7:
                ax.set_facecolor("#f0fff0")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pit_histograms.png"), dpi=150)
    plt.close()
    print(f"  Saved pit_histograms.png")


def plot_ssr_grid(ssr_per_cell, ssr_per_horizon, output_dir):
    """Plot SSR heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pooled SSR
    ax = axes[0]
    im = ax.imshow(ssr_per_cell, cmap="RdYlGn", vmin=0.5, vmax=1.5)
    ax.set_title("Spread-Skill Ratio (pooled)")
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{ssr_per_cell[r,c]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)

    # Per-horizon at h=1,7,14,30
    horizons = [0, 6, 13, 29]
    horizon_labels = ["h=1", "h=7", "h=14", "h=30"]
    ax = axes[1]
    ssr_selected = np.array([ssr_per_horizon[h].mean() for h in range(ssr_per_horizon.shape[0])])
    ax.plot(range(1, len(ssr_selected)+1), ssr_selected, "b-o", markersize=3)
    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("SSR (mean across cells)")
    ax.set_title("SSR by Horizon")
    ax.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_skill_ratio.png"), dpi=150)
    plt.close()
    print(f"  Saved spread_skill_ratio.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        model_dir = os.path.basename(os.path.dirname(args.model_path))
        args.output_dir = f"results/block_ar/diag_marginal_{model_dir}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    sp_cfg = {k: v for k, v in checkpoint["config"].items()
              if k in SinglePassConfig.__dataclass_fields__}
    sp_config = SinglePassConfig(**sp_cfg)
    model = SinglePassBlockAR(sp_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load test data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_dataset = VolSurfaceDataset(
        surfaces, sp_config.history_len, sp_config.future_len,
        start_idx=4540,  # test split
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    print(f"Test set: {len(test_dataset)} windows")

    # Generate samples
    print("Generating samples...")
    all_samples, all_gt, all_history = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Generating", total=args.max_batches)
        ):
            if batch_idx >= args.max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))
            samples = model.sample_batched(
                history, n_samples=args.n_samples,
                max_residual=0, max_global_residual=0,
            )
            history_denorm = denormalize_iv(history)
            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_history.append(history_denorm.cpu().numpy())

    cond_samples = np.concatenate(all_samples, axis=0)
    ground_truth = np.concatenate(all_gt, axis=0)
    history_arr = np.concatenate(all_history, axis=0)

    N, S, T = cond_samples.shape[:3]
    print(f"Data: {N} windows, {S} samples, {T} horizons")

    # Regime classification
    regime_mask = classify_regime(history_arr)
    n_calm = (~regime_mask).sum()
    n_turb = regime_mask.sum()
    print(f"Regimes: {n_calm} calm, {n_turb} turb")

    # ================================================================
    # 1. Rank Histogram
    # ================================================================
    print("\n" + "=" * 60)
    print("1. RANK HISTOGRAM")
    print("=" * 60)
    rc = rank_histogram(cond_samples, ground_truth)
    expected = (N * T) / (S + 1)

    print(f"  Expected count per bin: {expected:.1f}")
    print(f"  Edge excess ratio per cell (>1 = underdispersed, <1 = overdispersed):")
    edge_ratio = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            counts = rc[r, c]
            edge_ratio[r, c] = (counts[0] + counts[-1]) / (2 * expected)

    for r in range(5):
        row = "    " + "  ".join(f"{edge_ratio[r,c]:.2f}" for c in range(5))
        print(row)

    print(f"\n  Worst underdispersed: ({np.unravel_index(edge_ratio.argmax(), (5,5))}) "
          f"ratio={edge_ratio.max():.2f}")
    print(f"  Worst overdispersed:  ({np.unravel_index(edge_ratio.argmin(), (5,5))}) "
          f"ratio={edge_ratio.min():.2f}")

    plot_rank_histograms(rc, args.output_dir)

    # ================================================================
    # 2. Spread-Skill Ratio
    # ================================================================
    print("\n" + "=" * 60)
    print("2. SPREAD-SKILL RATIO (SSR=1.0 = calibrated)")
    print("=" * 60)
    ssr_cell, ssr_horizon = spread_skill_ratio(cond_samples, ground_truth)

    print("  Pooled SSR per cell:")
    for r in range(5):
        row = "    " + "  ".join(f"{ssr_cell[r,c]:.2f}" for c in range(5))
        print(row)

    print(f"\n  Overall mean SSR: {ssr_cell.mean():.3f}")
    print(f"  Min SSR: ({np.unravel_index(ssr_cell.argmin(), (5,5))}) = {ssr_cell.min():.3f}")
    print(f"  Max SSR: ({np.unravel_index(ssr_cell.argmax(), (5,5))}) = {ssr_cell.max():.3f}")

    # SSR at key horizons
    for h_idx, h_label in [(0, "h=1"), (6, "h=7"), (13, "h=14"), (29, "h=30")]:
        if h_idx < T:
            mean_ssr = ssr_horizon[h_idx].mean()
            min_ssr = ssr_horizon[h_idx].min()
            min_cell = np.unravel_index(ssr_horizon[h_idx].argmin(), (5, 5))
            print(f"  {h_label}: mean={mean_ssr:.3f}, min={min_ssr:.3f} at {min_cell}")

    plot_ssr_grid(ssr_cell, ssr_horizon, args.output_dir)

    # ================================================================
    # 3. Conditional KS by Regime
    # ================================================================
    print("\n" + "=" * 60)
    print("3. CONDITIONAL KS (daily changes, split by regime)")
    print("=" * 60)
    cks = conditional_ks(cond_samples, ground_truth, regime_mask)

    for regime_name in ["unconditional", "calm", "turb"]:
        data = cks[regime_name]
        ks_grid = data["ks_grid"]
        n_pass = data["n_pass"]
        n_win = data.get("n_windows", N)
        print(f"\n  {regime_name.upper()} ({n_win} windows, {n_pass}/25 pass D<0.15):")
        for r in range(5):
            row = "    " + "  ".join(
                f"{ks_grid[r,c]:.3f}{'*' if ks_grid[r,c] >= 0.15 else ' '}"
                for c in range(5)
            )
            print(row)

    # Key comparison
    calm_pass = cks["calm"]["n_pass"]
    turb_pass = cks["turb"]["n_pass"]
    uncond_pass = cks["unconditional"]["n_pass"]
    print(f"\n  Summary: uncond={uncond_pass}/25, calm={calm_pass}/25, turb={turb_pass}/25")
    if calm_pass > uncond_pass or turb_pass > uncond_pass:
        print("  → Conditional KS better than unconditional: anchoring artifact present")

    # ================================================================
    # 4. PIT Histogram
    # ================================================================
    print("\n" + "=" * 60)
    print("4. PIT HISTOGRAM")
    print("=" * 60)
    pit_vals, pit_ks = pit_histogram(cond_samples, ground_truth)

    print("  PIT KS vs U(0,1) per cell (lower = better):")
    for r in range(5):
        row = "    " + "  ".join(f"{pit_ks[r,c]:.3f}" for c in range(5))
        print(row)

    # Compute PIT bias: mean PIT should be 0.5 if calibrated
    pit_mean = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            pit_mean[r, c] = pit_vals[r, c].mean()

    print(f"\n  PIT mean per cell (0.5 = unbiased):")
    for r in range(5):
        row = "    " + "  ".join(f"{pit_mean[r,c]:.3f}" for c in range(5))
        print(row)

    # PIT edge fraction (underdispersion indicator)
    print(f"\n  PIT edge fraction (PIT<0.05 or PIT>0.95, expect 10%):")
    for r in range(5):
        row_vals = []
        for c in range(5):
            vals = pit_vals[r, c]
            edge_frac = ((vals < 0.05) | (vals > 0.95)).mean()
            row_vals.append(f"{edge_frac:.1%}")
        print("    " + "  ".join(row_vals))

    plot_pit_histograms(pit_vals, args.output_dir)

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "rank_histogram_edge_ratio": edge_ratio.tolist(),
        "ssr_per_cell": ssr_cell.tolist(),
        "ssr_per_horizon_mean": [float(ssr_horizon[t].mean()) for t in range(T)],
        "conditional_ks": {
            regime: {
                "ks_grid": data["ks_grid"].tolist(),
                "n_pass": data["n_pass"],
            }
            for regime, data in cks.items()
        },
        "pit_ks": pit_ks.tolist(),
        "pit_mean": pit_mean.tolist(),
    }

    out_path = os.path.join(args.output_dir, "marginal_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
