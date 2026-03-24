#!/usr/bin/env python
"""
Long-Horizon 252-Day Test for flow_153a (Conditional One-Shot Flow Matching)

The model generates 30-day windows. To reach 252 days, we chain 9 windows
(9 * 30 = 270, truncated to 252). For each window:
  1. Encode the most recent 30 days of history -> condition (128-dim)
  2. ODE from N(0,I) -> 30-day future (8-step Euler)
  3. Denormalize: x * train_std + train_mean, clamp [0,1]
  4. Use last 30 days of generated future as new "history" for next window

Metrics:
  1. Explosion check: mean IV level at each 30-day chunk
  2. Surface validity: NaN, out-of-range, calendar/butterfly arbitrage
  3. Spread growth: ensemble spread evolution over 252 days
  4. Arbitrage rates at various horizons
  5. Cross-cell correlation preservation

Usage:
    PYTHONPATH=. python results/validations/2026-03-24/scripts/153a_long_horizon.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")

from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv
)

# ── Constants ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/backfill/flow_153a/final_model.pt"
ENCODER_PATH = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_DIR = Path("results/validations/2026-03-24/analysis/153a_longhorizon")
RESULT_PATH = Path("results/validations/2026-03-24/verification_results/153a_long_horizon.json")

N_SAMPLES = 10        # ensemble members
N_WINDOWS = 5         # different starting points
N_CHAINS = 9          # 9 * 30 = 270 days, truncate to 252
TOTAL_DAYS = 252
ODE_STEPS = 8         # per checkpoint config
HORIZONS = [30, 60, 90, 120, 150, 180, 210, 240, 252]

# Cell labels
LABELS_K = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
LABELS_T = ["1M", "3M", "6M", "1Y", "2Y"]
CELL_NAMES = [f"{t}/{k}" for t in LABELS_T for k in LABELS_K]


def make_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def check_calendar_arbitrage(surfaces):
    """
    Calendar arbitrage: total variance must be non-decreasing across tenors.
    surfaces: (..., 5, 5) -- tenor x moneyness
    Tenors in years: [1/12, 3/12, 6/12, 1, 2]
    Returns violation rate.
    """
    tenor_years = np.array([1/12, 3/12, 6/12, 1.0, 2.0])
    # total variance = IV^2 * T
    # surfaces are in IV space (0-1 range), but we treat them as IV levels
    total_var = surfaces ** 2 * tenor_years[None, :, None]  # (..., 5, 5)
    # Check if total_var is non-decreasing across tenors (axis=-2)
    diffs = np.diff(total_var, axis=-2)  # (..., 4, 5)
    violations = (diffs < -1e-6).any(axis=(-2, -1))  # (...,)
    return float(violations.mean())


def check_butterfly_arbitrage(surfaces):
    """
    Butterfly arbitrage: smile must be convex (second derivative >= 0 in moneyness).
    surfaces: (..., 5, 5) -- tenor x moneyness
    Returns violation rate.
    """
    # Second difference along moneyness axis (last axis)
    d2 = np.diff(surfaces, n=2, axis=-1)  # (..., 5, 3)
    violations = (d2 < -1e-6).any(axis=(-2, -1))
    return float(violations.mean())


@torch.no_grad()
def generate_one_chain(model, encoder, history_30d, train_mean_t, train_std_t,
                       n_samples, n_chains, device):
    """
    Generate a long-horizon chain of 30-day windows.

    Args:
        model: ConditionalFactoredVelocityTransformer
        encoder: GRU encoder (frozen)
        history_30d: (30, 5, 5) raw IV history in [0,1]
        train_mean_t: (1, 750) tensor on device
        train_std_t: (1, 750) tensor on device
        n_samples: number of ensemble members
        n_chains: number of 30-day windows to chain
        device: torch device

    Returns:
        all_futures: (n_samples, n_chains*30, 5, 5) generated IV surfaces
    """
    all_futures = []  # list of (n_samples, 30, 5, 5) arrays

    # Current history: (30, 5, 5)
    # For ensemble: we propagate each sample independently
    # Start with shared history for all samples
    current_histories = np.tile(history_30d, (n_samples, 1, 1, 1))  # (S, 30, 5, 5)

    for chain_idx in range(n_chains):
        # Encode history for each sample -> condition
        hist_tensor = torch.from_numpy(current_histories).float().to(device)
        hist_norm = normalize_iv(hist_tensor)  # [-1, 1]
        cond = encoder(hist_norm)  # (S, 128)

        # ODE from N(0,I)
        x = torch.randn(n_samples, 750, device=device)
        dt = 1.0 / ODE_STEPS
        for step in range(ODE_STEPS):
            t = torch.full((n_samples,), step * dt, device=device)
            x = x + model(x, t, cond=cond) * dt

        # Denormalize
        samples = x * train_std_t + train_mean_t  # (S, 750)
        samples = torch.clamp(samples, 0, 1)
        samples_np = samples.cpu().numpy().reshape(n_samples, 30, 5, 5)

        all_futures.append(samples_np)

        # Update history: last 30 days of generated future
        current_histories = samples_np.copy()

    # Stack: (n_samples, n_chains*30, 5, 5)
    result = np.concatenate(all_futures, axis=1)
    return result[:, :TOTAL_DAYS]  # Truncate to 252


def compute_spread_by_horizon(all_samples, horizons):
    """
    Compute ensemble spread (std across samples) at various horizons.

    all_samples: (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    Returns dict of horizon -> mean spread across cells and windows
    """
    result = {}
    for h in horizons:
        if h > all_samples.shape[2]:
            continue
        # std across samples at day h-1
        std_at_h = all_samples[:, :, h-1].std(axis=1)  # (N_WINDOWS, 5, 5)
        result[h] = {
            "mean_spread": float(std_at_h.mean()),
            "min_spread": float(std_at_h.min()),
            "max_spread": float(std_at_h.max()),
            "atm_6m_spread": float(std_at_h[:, 2, 2].mean()),
        }
    return result


def compute_iv_level_by_chunk(all_samples):
    """
    Track mean IV level at each 30-day chunk boundary.

    all_samples: (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    Returns list of chunk-level statistics.
    """
    chunks = []
    for chunk_start in range(0, TOTAL_DAYS, 30):
        chunk_end = min(chunk_start + 30, TOTAL_DAYS)
        chunk_data = all_samples[:, :, chunk_start:chunk_end]
        chunks.append({
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "mean_iv": float(chunk_data.mean()),
            "std_iv": float(chunk_data.std()),
            "min_iv": float(chunk_data.min()),
            "max_iv": float(chunk_data.max()),
            "nan_count": int(np.isnan(chunk_data).sum()),
            "fraction_above_1": float((chunk_data > 1.0).mean()),
            "fraction_below_0": float((chunk_data < 0.0).mean()),
        })
    return chunks


def compute_arbitrage_by_horizon(all_samples, horizons):
    """
    Calendar and butterfly arbitrage rates at various horizons.

    all_samples: (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    """
    result = {}
    for h in horizons:
        if h > all_samples.shape[2]:
            continue
        # All surfaces at this horizon across windows and samples
        surfaces = all_samples[:, :, h-1]  # (N_WINDOWS, N_SAMPLES, 5, 5)
        surfaces_flat = surfaces.reshape(-1, 5, 5)
        cal_arb = check_calendar_arbitrage(surfaces_flat)
        but_arb = check_butterfly_arbitrage(surfaces_flat)
        result[h] = {
            "calendar_arb_rate": cal_arb,
            "butterfly_arb_rate": but_arb,
        }
    return result


def compute_cross_cell_correlation(all_samples):
    """
    Mean pairwise correlation across cells for generated samples.

    all_samples: (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    """
    # Use daily changes
    daily_changes = np.diff(all_samples, axis=2)  # (W, S, 251, 5, 5)
    flat = daily_changes.reshape(-1, 25)  # pool all windows, samples, days
    corr = np.corrcoef(flat, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]

    # Also compute for levels
    flat_levels = all_samples.reshape(-1, 25)
    corr_levels = np.corrcoef(flat_levels, rowvar=False)
    upper_levels = corr_levels[np.triu_indices_from(corr_levels, k=1)]
    upper_levels = upper_levels[np.isfinite(upper_levels)]

    return {
        "daily_change_corr_mean": float(upper.mean()) if len(upper) else 0.0,
        "daily_change_corr_std": float(upper.std()) if len(upper) else 0.0,
        "level_corr_mean": float(upper_levels.mean()) if len(upper_levels) else 0.0,
        "level_corr_std": float(upper_levels.std()) if len(upper_levels) else 0.0,
    }


def plot_fan_chart(all_samples, gt_future, output_dir, cell=(2, 2), cell_name="ATM 6M"):
    """
    Fan chart: median + 10th/90th percentile for a single cell across 252 days.

    all_samples: (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    gt_future: (N_WINDOWS, 252, 5, 5) or None
    """
    r, c = cell
    # Pool all windows for each sample path
    for win_idx in range(all_samples.shape[0]):
        fig, ax = plt.subplots(figsize=(12, 5))
        paths = all_samples[win_idx, :, :, r, c]  # (S, 252)
        median = np.median(paths, axis=0)
        p10 = np.percentile(paths, 10, axis=0)
        p90 = np.percentile(paths, 90, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p75 = np.percentile(paths, 75, axis=0)

        days = np.arange(1, paths.shape[1] + 1)
        ax.fill_between(days, p10, p90, alpha=0.2, color="blue", label="10-90% CI")
        ax.fill_between(days, p25, p75, alpha=0.3, color="blue", label="25-75% CI")
        ax.plot(days, median, color="blue", linewidth=1.5, label="Median")

        if gt_future is not None and win_idx < gt_future.shape[0]:
            T = min(252, gt_future.shape[1])
            ax.plot(range(1, T+1), gt_future[win_idx, :T, r, c],
                    color="red", linewidth=1, alpha=0.8, label="Ground Truth")

        # Mark chain boundaries
        for ch in range(30, 252, 30):
            ax.axvline(ch, color="gray", alpha=0.3, ls="--")

        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("IV Level")
        ax.set_title(f"252-Day Fan Chart: {cell_name} — Window {win_idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"fan_chart_win{win_idx}_{cell_name.replace('/', '_').replace(' ', '_')}.png",
                    dpi=150)
        plt.close(fig)


def plot_spread_evolution(all_samples, output_dir):
    """Plot ensemble spread (mean std) over time for key cells."""
    # (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    std_over_time = all_samples.std(axis=1).mean(axis=0)  # (252, 5, 5)

    fig, ax = plt.subplots(figsize=(12, 5))
    cells = [(2, 2, "ATM 6M"), (0, 2, "ATM 1M"), (4, 2, "ATM 2Y"),
             (2, 0, "6M K=0.70"), (2, 4, "6M K=1.30")]
    for r, c, name in cells:
        ax.plot(range(1, 253), std_over_time[:, r, c], label=name)
    for ch in range(30, 252, 30):
        ax.axvline(ch, color="gray", alpha=0.3, ls="--")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Ensemble std (IV pts)")
    ax.set_title("Ensemble Spread Evolution Over 252 Days")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "spread_evolution.png", dpi=150)
    plt.close(fig)


def plot_mean_iv_evolution(all_samples, gt_future, output_dir):
    """Plot mean IV level over time."""
    # Mean across samples, then across cells
    gen_mean_iv = all_samples.mean(axis=(1, 3, 4))  # (N_WINDOWS, 252)

    fig, ax = plt.subplots(figsize=(12, 5))
    for win_idx in range(gen_mean_iv.shape[0]):
        ax.plot(range(1, 253), gen_mean_iv[win_idx], alpha=0.5, label=f"Gen win {win_idx}")
    if gt_future is not None:
        for win_idx in range(gt_future.shape[0]):
            T = min(252, gt_future.shape[1])
            gt_mean = gt_future[win_idx, :T].reshape(T, -1).mean(axis=1)
            ax.plot(range(1, T+1), gt_mean, '--', alpha=0.5, label=f"GT win {win_idx}")
    for ch in range(30, 252, 30):
        ax.axvline(ch, color="gray", alpha=0.3, ls="--")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean IV Level")
    ax.set_title("Mean IV Level Over 252 Days (Generated vs GT)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_iv_evolution.png", dpi=150)
    plt.close(fig)


def plot_arbitrage_over_horizon(arb_by_horizon, output_dir):
    """Plot arbitrage rates vs horizon."""
    horizons = sorted(arb_by_horizon.keys())
    cal_rates = [arb_by_horizon[h]["calendar_arb_rate"] for h in horizons]
    but_rates = [arb_by_horizon[h]["butterfly_arb_rate"] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(horizons, cal_rates, 'o-', label="Calendar arb rate")
    ax.plot(horizons, but_rates, 's-', label="Butterfly arb rate")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Arbitrage Rate")
    ax.set_title("Arbitrage Rates vs Horizon (252-Day Test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "arbitrage_vs_horizon.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("153a Long-Horizon 252-Day Test")
    print("=" * 70)

    device = DEVICE
    print(f"Device: {device}")

    # ── Load model ──
    print("\nLoading model...")
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Config: {cfg}")

    train_mean = ckpt["train_mean"]  # (1, 750)
    train_std = ckpt["train_std"]    # (1, 750)
    train_mean_t = torch.from_numpy(train_mean).float().to(device)
    train_std_t = torch.from_numpy(train_std).float().to(device)

    # ── Load encoder ──
    print("Loading encoder...")
    encoder, cond_dim = load_encoder(ENCODER_PATH, device)
    print(f"  Encoder cond_dim: {cond_dim}")

    # ── Load data ──
    print("Loading data...")
    data = np.load(DATA_PATH)
    surfaces = data["surface"]  # (5822, 5, 5)
    print(f"  Surfaces shape: {surfaces.shape}")

    # Select test windows: spread across test set (after index 4040)
    # Need 30 history + 252 future = 282 consecutive points
    test_start = 4040
    max_start = len(surfaces) - 282
    # 5 evenly spaced windows from test set
    window_starts = np.linspace(test_start, max_start, N_WINDOWS, dtype=int)
    print(f"  Window starts: {window_starts}")

    # ── Generate long-horizon samples ──
    print(f"\nGenerating {N_SAMPLES} samples x {N_WINDOWS} windows x {N_CHAINS} chains...")
    all_samples = []  # (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    gt_futures = []   # (N_WINDOWS, 252, 5, 5)

    for win_i, start_idx in enumerate(window_starts):
        print(f"\n  Window {win_i}: start={start_idx}, "
              f"history=[{start_idx}:{start_idx+30}], "
              f"future=[{start_idx+30}:{start_idx+30+252}]")

        history = surfaces[start_idx:start_idx+30]  # (30, 5, 5)
        gt_future = surfaces[start_idx+30:start_idx+30+252]  # (252, 5, 5)
        gt_futures.append(gt_future)

        t_win = time.time()
        chain_result = generate_one_chain(
            model, encoder, history, train_mean_t, train_std_t,
            n_samples=N_SAMPLES, n_chains=N_CHAINS, device=device
        )
        elapsed_win = time.time() - t_win
        all_samples.append(chain_result)

        print(f"    Generated in {elapsed_win:.1f}s")
        print(f"    Shape: {chain_result.shape}")
        print(f"    IV range: [{chain_result.min():.4f}, {chain_result.max():.4f}]")
        print(f"    NaN count: {np.isnan(chain_result).sum()}")
        print(f"    Mean IV at day 30: {chain_result[:, 29].mean():.4f}")
        print(f"    Mean IV at day 252: {chain_result[:, -1].mean():.4f}")

    all_samples = np.array(all_samples)  # (N_WINDOWS, N_SAMPLES, 252, 5, 5)
    gt_futures = np.array(gt_futures)    # (N_WINDOWS, 252, 5, 5)
    print(f"\nAll samples shape: {all_samples.shape}")
    print(f"GT futures shape: {gt_futures.shape}")

    # ── Compute metrics ──
    print("\n" + "=" * 50)
    print("Computing metrics...")
    print("=" * 50)

    # 1. Explosion check
    print("\n1. IV Level by Chunk:")
    chunk_stats = compute_iv_level_by_chunk(all_samples)
    for chunk in chunk_stats:
        print(f"   Day {chunk['chunk_start']:3d}-{chunk['chunk_end']:3d}: "
              f"mean={chunk['mean_iv']:.4f}, std={chunk['std_iv']:.4f}, "
              f"range=[{chunk['min_iv']:.4f}, {chunk['max_iv']:.4f}], "
              f"NaN={chunk['nan_count']}")

    # GT comparison
    print("\n   GT IV levels:")
    for chunk_start in range(0, TOTAL_DAYS, 30):
        chunk_end = min(chunk_start + 30, TOTAL_DAYS)
        gt_chunk = gt_futures[:, chunk_start:chunk_end]
        print(f"   Day {chunk_start:3d}-{chunk_end:3d}: "
              f"mean={gt_chunk.mean():.4f}, range=[{gt_chunk.min():.4f}, {gt_chunk.max():.4f}]")

    # 2. Surface validity
    print("\n2. Surface Validity:")
    nan_total = int(np.isnan(all_samples).sum())
    fraction_above_1 = float((all_samples > 1.0).mean())
    fraction_below_0 = float((all_samples < 0.0).mean())
    exploded = float(all_samples[:, :, -1].mean() > 1.0)
    print(f"   Total NaN: {nan_total}")
    print(f"   Fraction > 1.0: {fraction_above_1:.6f}")
    print(f"   Fraction < 0.0: {fraction_below_0:.6f}")
    print(f"   Exploded (mean IV at day 252 > 1.0): {exploded}")

    # 3. Spread growth
    print("\n3. Ensemble Spread by Horizon:")
    spread_by_horizon = compute_spread_by_horizon(all_samples, HORIZONS)
    for h, s in sorted(spread_by_horizon.items()):
        print(f"   Day {h:3d}: mean_spread={s['mean_spread']:.6f}, "
              f"ATM_6M={s['atm_6m_spread']:.6f}")

    # Check for growing uncertainty (monotonic spread growth)
    spread_values = [spread_by_horizon[h]["mean_spread"] for h in sorted(spread_by_horizon.keys())]
    spread_monotonic = all(spread_values[i] <= spread_values[i+1] * 1.1
                           for i in range(len(spread_values)-1))
    print(f"   Approximately monotonic spread growth: {spread_monotonic}")

    # 4. Arbitrage
    print("\n4. Arbitrage Rates by Horizon:")
    arb_by_horizon = compute_arbitrage_by_horizon(all_samples, HORIZONS)
    for h, a in sorted(arb_by_horizon.items()):
        print(f"   Day {h:3d}: cal_arb={a['calendar_arb_rate']:.3f}, "
              f"but_arb={a['butterfly_arb_rate']:.3f}")

    # GT arbitrage for comparison
    print("\n   GT Arbitrage:")
    for h in HORIZONS:
        if h <= gt_futures.shape[1]:
            gt_surf = gt_futures[:, h-1]  # (N_WINDOWS, 5, 5)
            cal_gt = check_calendar_arbitrage(gt_surf)
            but_gt = check_butterfly_arbitrage(gt_surf)
            print(f"   Day {h:3d}: cal_arb={cal_gt:.3f}, but_arb={but_gt:.3f}")

    # 5. Cross-cell correlation
    print("\n5. Cross-Cell Correlation:")
    corr_stats = compute_cross_cell_correlation(all_samples)
    print(f"   Daily change corr (gen): {corr_stats['daily_change_corr_mean']:.4f} "
          f"+/- {corr_stats['daily_change_corr_std']:.4f}")
    print(f"   Level corr (gen): {corr_stats['level_corr_mean']:.4f} "
          f"+/- {corr_stats['level_corr_std']:.4f}")

    # GT cross-cell correlation
    gt_daily = np.diff(gt_futures, axis=1).reshape(-1, 25)
    gt_corr = np.corrcoef(gt_daily, rowvar=False)
    gt_upper = gt_corr[np.triu_indices_from(gt_corr, k=1)]
    gt_upper = gt_upper[np.isfinite(gt_upper)]
    gt_corr_mean = float(gt_upper.mean()) if len(gt_upper) else 0.0
    print(f"   Daily change corr (GT): {gt_corr_mean:.4f}")

    # 6. Mean/Median bias at horizon 252
    gen_median_252 = np.median(all_samples[:, :, -1], axis=1)  # (N_WINDOWS, 5, 5)
    gt_252 = gt_futures[:, -1]  # (N_WINDOWS, 5, 5)
    bias_252 = float((gen_median_252 - gt_252).mean())
    abs_bias_252 = float(np.abs(gen_median_252 - gt_252).mean())
    print(f"\n6. Bias at Day 252:")
    print(f"   Mean bias: {bias_252:.6f}")
    print(f"   Mean absolute bias: {abs_bias_252:.6f}")

    # 7. Spread collapse check: does spread go to zero?
    final_spread = float(all_samples[:, :, -1].std(axis=1).mean())
    initial_spread = float(all_samples[:, :, 0].std(axis=1).mean())
    spread_ratio = final_spread / (initial_spread + 1e-10)
    print(f"\n7. Spread Collapse Check:")
    print(f"   Initial spread (day 1): {initial_spread:.6f}")
    print(f"   Final spread (day 252): {final_spread:.6f}")
    print(f"   Ratio (final/initial): {spread_ratio:.4f}")

    # ── Generate plots ──
    print("\nGenerating plots...")
    plot_fan_chart(all_samples, gt_futures, OUTPUT_DIR, cell=(2, 2), cell_name="ATM 6M")
    plot_fan_chart(all_samples, gt_futures, OUTPUT_DIR, cell=(0, 2), cell_name="ATM 1M")
    plot_fan_chart(all_samples, gt_futures, OUTPUT_DIR, cell=(4, 2), cell_name="ATM 2Y")
    plot_spread_evolution(all_samples, OUTPUT_DIR)
    plot_mean_iv_evolution(all_samples, gt_futures, OUTPUT_DIR)
    plot_arbitrage_over_horizon(arb_by_horizon, OUTPUT_DIR)

    # ── Overall pass/fail ──
    elapsed_total = time.time() - t0

    # Determine pass/fail criteria
    no_explosion = all(c["max_iv"] <= 1.01 for c in chunk_stats)
    no_nan = nan_total == 0
    no_collapse = final_spread > 0.001  # spread didn't collapse to zero
    cal_arb_ok = arb_by_horizon.get(252, {}).get("calendar_arb_rate", 1.0) < 0.50
    but_arb_ok = arb_by_horizon.get(252, {}).get("butterfly_arb_rate", 1.0) < 0.50
    spread_grows = spread_ratio > 1.0

    all_pass = no_explosion and no_nan and no_collapse and cal_arb_ok and but_arb_ok

    print("\n" + "=" * 50)
    print("PASS/FAIL Summary:")
    print("=" * 50)
    print(f"  No explosion (max IV <= 1.01):    {'PASS' if no_explosion else 'FAIL'}")
    print(f"  No NaN:                           {'PASS' if no_nan else 'FAIL'}")
    print(f"  No spread collapse (>0.001):      {'PASS' if no_collapse else 'FAIL'}")
    print(f"  Calendar arb at 252d < 50%:       {'PASS' if cal_arb_ok else 'FAIL'}")
    print(f"  Butterfly arb at 252d < 50%:      {'PASS' if but_arb_ok else 'FAIL'}")
    print(f"  Spread grows over 252d:           {'PASS' if spread_grows else 'FAIL'}")
    print(f"  OVERALL:                          {'PASS' if all_pass else 'FAIL'}")
    print(f"\n  Total time: {elapsed_total:.1f}s")

    # ── Build result JSON ──
    result = {
        "verification": "153a_long_horizon_252d",
        "model_path": MODEL_PATH,
        "encoder_path": ENCODER_PATH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": make_serializable(cfg),
        "n_params": n_params,
        "n_samples": N_SAMPLES,
        "n_windows": N_WINDOWS,
        "n_chains": N_CHAINS,
        "total_days": TOTAL_DAYS,
        "window_starts": window_starts.tolist(),
        "elapsed_seconds": round(elapsed_total, 1),

        "pass_fail": {
            "no_explosion": no_explosion,
            "no_nan": no_nan,
            "no_spread_collapse": no_collapse,
            "calendar_arb_ok": cal_arb_ok,
            "butterfly_arb_ok": but_arb_ok,
            "spread_grows": spread_grows,
            "overall": all_pass,
        },

        "iv_level_by_chunk": make_serializable(chunk_stats),
        "spread_by_horizon": make_serializable(spread_by_horizon),
        "arbitrage_by_horizon": make_serializable(arb_by_horizon),
        "cross_cell_correlation": make_serializable(corr_stats),
        "gt_daily_change_corr": gt_corr_mean,
        "bias_at_252": {
            "mean_bias": bias_252,
            "mean_absolute_bias": abs_bias_252,
        },
        "spread_collapse_check": {
            "initial_spread": initial_spread,
            "final_spread": final_spread,
            "ratio": spread_ratio,
        },

        "plots": [
            str(OUTPUT_DIR / f) for f in [
                "fan_chart_win0_ATM_6M.png",
                "fan_chart_win0_ATM_1M.png",
                "fan_chart_win0_ATM_2Y.png",
                "spread_evolution.png",
                "mean_iv_evolution.png",
                "arbitrage_vs_horizon.png",
            ]
        ],
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {RESULT_PATH}")

    return result


if __name__ == "__main__":
    main()
