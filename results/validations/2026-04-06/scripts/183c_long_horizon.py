#!/usr/bin/env python
"""
Standalone 252-day long-horizon test for experiment 183c.

Model: StateMetricTransportModel (183c)
  - Joint distribution model that generates 30 frames at once
  - For 252 days: chain 9 blocks of 30 frames autoregressively
    (use last 30 frames of output as new history for next block)
  - Trim to 252 total frames at the end

Evaluates at horizons h=30, 60, 90, 180, 252:
  1. Ensemble spread (per-cell std) -- monotonic growth check
  2. Surface validity -- explosion rate at each horizon
  3. Spatial structure -- 5x5 grid validity at h=252
  4. Path stationarity -- rolling std of daily changes

Usage:
    PYTHONPATH=. python results/validations/2026-04-06/scripts/183c_long_horizon.py \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel,
)

# ── Constants ──
MODEL_PATH = "models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
TEST_START = 4540
HISTORY_LEN = 30
FUTURE_LEN = 30
N_WINDOWS = 20
N_SAMPLES = 50
N_BLOCKS = 9  # 9 * 30 = 270 >= 252
TOTAL_FRAMES = 252
HORIZONS = [30, 60, 90, 180, 252]

LABELS_K = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
LABELS_T = ["1M", "3M", "6M", "1Y", "2Y"]
CELL_NAMES = [f"{t}/{k}" for t in LABELS_T for k in LABELS_K]


def normalize_iv(x: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1]"""
    return x * 2.0 - 1.0


def denormalize_iv(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return (x + 1.0) / 2.0


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def load_model(model_path: str, device: str) -> StateMetricTransportModel:
    """Load 183c model from checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc_cfg = EncoderConfig(**cfg["encoder"])
    dec_cfg = cfg["decoder"]

    model = StateMetricTransportModel(
        encoder_config=enc_cfg,
        decoder_config=dec_cfg,
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"Model loaded: {cfg['type']}")
    print(f"  Decoder n_frames={model.decoder.n_frames}, n_cells={model.decoder.n_cells}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, selection: {ckpt.get('selection_key', '?')}")
    return model


def load_data(data_path: str, device: str):
    """Load vol surface data and prepare test windows."""
    data = np.load(data_path)
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    returns = data["ret"]       # (N,)
    N = len(surfaces)
    print(f"Data loaded: {N} surfaces, test starts at {TEST_START}")

    # Build test windows: each window needs HISTORY_LEN history
    # We want N_WINDOWS windows starting from TEST_START
    windows = []
    max_idx = N - HISTORY_LEN - 1  # need at least 1 future frame for GT
    for i in range(N_WINDOWS):
        start = TEST_START + i
        if start + HISTORY_LEN > N:
            break
        hist = surfaces[start:start + HISTORY_LEN]  # (30, 5, 5) in [0,1]
        windows.append(hist)

    history = np.stack(windows)  # (N_win, 30, 5, 5) in [0,1]
    history_tensor = torch.tensor(history, dtype=torch.float32).to(device)
    # Convert to [-1, 1] for model input
    history_norm = normalize_iv(history_tensor)

    # Also load GT future for as many frames as available (for comparison)
    gt_futures = []
    for i in range(len(windows)):
        start = TEST_START + i + HISTORY_LEN
        end = min(start + TOTAL_FRAMES, N)
        gt = surfaces[start:end]  # (<=252, 5, 5)
        gt_futures.append(gt)

    return history_norm, history, returns, gt_futures


@torch.no_grad()
def generate_long_horizon(
    model: StateMetricTransportModel,
    history_norm: torch.Tensor,
    n_samples: int,
    n_blocks: int,
    device: str,
) -> np.ndarray:
    """Chain 30-frame blocks to generate 252-day paths.

    Args:
        history_norm: (B, 30, 5, 5) in [-1, 1]
        n_samples: number of ensemble members per window
        n_blocks: number of 30-frame blocks to chain

    Returns:
        all_samples: (B, n_samples, 252, 5, 5) in [0, 1]
    """
    B = history_norm.shape[0]
    chunk_size = 5  # samples per chunk to avoid OOM

    all_sample_blocks = []  # will collect (B, n_samples, 30, 5, 5) per block

    # For each sample, we need to track the "running history" independently.
    # Strategy: process samples in chunks.
    # After block 0: take each sample's output, use as next history for that sample.
    # This requires per-sample state tracking.

    print(f"\nGenerating {n_blocks} blocks of 30 frames ({n_blocks * 30} total, trimming to {TOTAL_FRAMES})")
    print(f"  {B} windows x {n_samples} samples x {n_blocks} blocks")

    # Block 0: all samples share the same history
    print(f"  Block 0: initial generation from real history...")
    t0 = time.time()
    # sample_batched expects history in [-1,1], returns [0,1]
    block0_01 = model.sample_batched(history_norm, n_samples=n_samples, chunk_size=chunk_size)
    # block0_01: (B, n_samples, 30, 5, 5) in [0, 1]
    all_sample_blocks.append(block0_01.cpu())
    print(f"    Done in {time.time() - t0:.1f}s")

    # Blocks 1..n_blocks-1: each sample gets its own history from previous block
    for block_idx in range(1, n_blocks):
        t_block = time.time()
        # Previous block output: (B, n_samples, 30, 5, 5) in [0,1]
        prev_block = all_sample_blocks[-1]  # on CPU

        # Fold samples into batch dim: (B*n_samples, 30, 5, 5)
        prev_flat = prev_block.reshape(B * n_samples, 30, 5, 5).to(device)

        # Convert to [-1, 1] for model input
        prev_norm = normalize_iv(prev_flat)

        # Generate next block: each "window" is now a sample
        # Use n_samples=1 since each row is already one sample path
        # Process in sub-chunks to avoid OOM
        sub_chunks = []
        sub_chunk_size = max(1, min(32, B * n_samples))  # batch processing limit
        for sub_start in range(0, B * n_samples, sub_chunk_size):
            sub_end = min(sub_start + sub_chunk_size, B * n_samples)
            sub_hist = prev_norm[sub_start:sub_end]
            sub_out = model.sample_batched(sub_hist, n_samples=1, chunk_size=1)
            # sub_out: (sub_batch, 1, 30, 5, 5)
            sub_chunks.append(sub_out.cpu())

        next_block = torch.cat(sub_chunks, dim=0)  # (B*n_samples, 1, 30, 5, 5)
        next_block = next_block.squeeze(1)  # (B*n_samples, 30, 5, 5)
        next_block = next_block.reshape(B, n_samples, 30, 5, 5)
        all_sample_blocks.append(next_block)

        dt = time.time() - t_block
        print(f"  Block {block_idx}: {dt:.1f}s")

    # Concatenate all blocks: (B, n_samples, n_blocks*30, 5, 5)
    full_paths = torch.cat(all_sample_blocks, dim=2)
    # Trim to TOTAL_FRAMES
    full_paths = full_paths[:, :, :TOTAL_FRAMES, :, :]
    print(f"\nFinal shape: {full_paths.shape} (expect B={B}, S={n_samples}, T={TOTAL_FRAMES})")

    return full_paths.numpy()


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════


def eval_ensemble_spread(samples: np.ndarray, horizons: list[int]) -> dict:
    """Ensemble spread (per-cell std) at each horizon + monotonicity check.

    samples: (B, S, T, 5, 5) in [0, 1]
    """
    # std across samples at each time step: (B, T, 5, 5)
    std_per_t = samples.std(axis=1)
    # Average across windows: (T, 5, 5)
    mean_std = std_per_t.mean(axis=0)
    # Global mean across cells: (T,)
    global_std = mean_std.reshape(mean_std.shape[0], -1).mean(axis=1)

    # Check monotonicity: std at each horizon should increase
    horizon_spreads = {}
    for h in horizons:
        if h <= samples.shape[2]:
            h_std = mean_std[h - 1]  # (5, 5)
            horizon_spreads[f"h{h}"] = {
                "mean_std": float(h_std.mean()),
                "min_std": float(h_std.min()),
                "max_std": float(h_std.max()),
                "atm_6m_std": float(h_std[2, 2]),
                "per_cell": h_std.tolist(),
            }

    # Monotonicity: check if spread at each horizon is >= previous
    sorted_h = sorted([h for h in horizons if h <= samples.shape[2]])
    monotonic_pairs = []
    for i in range(1, len(sorted_h)):
        prev_h, curr_h = sorted_h[i - 1], sorted_h[i]
        prev_val = horizon_spreads[f"h{prev_h}"]["mean_std"]
        curr_val = horizon_spreads[f"h{curr_h}"]["mean_std"]
        monotonic_pairs.append({
            "from": prev_h,
            "to": curr_h,
            "prev_std": prev_val,
            "curr_std": curr_val,
            "growing": curr_val > prev_val,
            "ratio": curr_val / max(prev_val, 1e-8),
        })

    all_growing = all(p["growing"] for p in monotonic_pairs)

    return {
        "horizon_spreads": horizon_spreads,
        "monotonicity_pairs": monotonic_pairs,
        "all_monotonic": all_growing,
        "global_std_trajectory": global_std.tolist(),
    }


def eval_surface_validity(samples: np.ndarray, horizons: list[int]) -> dict:
    """Check for explosions and arbitrage at each horizon.

    samples: (B, S, T, 5, 5) in [0, 1]
    """
    B, S, T, H, W = samples.shape

    results = {}
    for h in horizons:
        if h > T:
            continue
        # Surfaces at horizon h: (B, S, 5, 5)
        surfaces_h = samples[:, :, h - 1, :, :]

        # Explosion: any cell > 1.0 or < 0.0 (outside valid IV range)
        explosions = (surfaces_h > 1.0) | (surfaces_h < 0.0)
        explosion_rate = float(explosions.any(axis=(2, 3)).mean())

        # Extreme values: > 0.9 or < 0.01
        extreme_high = float((surfaces_h > 0.9).mean())
        extreme_low = float((surfaces_h < 0.01).mean())

        # NaN/Inf check
        nan_rate = float(np.isnan(surfaces_h).any(axis=(2, 3)).mean())
        inf_rate = float(np.isinf(surfaces_h).any(axis=(2, 3)).mean())

        # Calendar arbitrage: tenor should be monotonically decreasing variance
        # IV should generally increase with tenor (for fixed strike)
        # Check: IV(longer tenor) >= IV(shorter tenor) for ATM
        atm_col = 2  # K=1.00
        atm_tenor_slice = surfaces_h[:, :, :, atm_col]  # (B, S, 5) across tenors
        # Calendar arb: check if any pair is inverted
        cal_arb_count = 0
        cal_arb_total = 0
        for t_idx in range(4):
            short = atm_tenor_slice[:, :, t_idx]
            long = atm_tenor_slice[:, :, t_idx + 1]
            cal_arb_count += int((short > long * 1.05).sum())  # 5% tolerance
            cal_arb_total += short.size

        cal_arb_rate = cal_arb_count / max(cal_arb_total, 1)

        # Butterfly arbitrage: convexity check (smile should be convex)
        # For each tenor row, check: IV(K-1) + IV(K+1) >= 2*IV(K)
        butterfly_violations = 0
        butterfly_total = 0
        for row in range(5):  # each tenor
            for col in range(1, 4):  # middle strikes
                left = surfaces_h[:, :, row, col - 1]
                mid = surfaces_h[:, :, row, col]
                right = surfaces_h[:, :, row, col + 1]
                butterfly_violations += int((left + right < 2 * mid - 0.005).sum())
                butterfly_total += left.size

        butterfly_arb_rate = butterfly_violations / max(butterfly_total, 1)

        # Summary statistics
        results[f"h{h}"] = {
            "explosion_rate": explosion_rate,
            "extreme_high_rate": extreme_high,
            "extreme_low_rate": extreme_low,
            "nan_rate": nan_rate,
            "inf_rate": inf_rate,
            "calendar_arb_rate": cal_arb_rate,
            "butterfly_arb_rate": butterfly_arb_rate,
            "mean_iv": float(surfaces_h.mean()),
            "std_iv": float(surfaces_h.std()),
            "min_iv": float(np.nanmin(surfaces_h)),
            "max_iv": float(np.nanmax(surfaces_h)),
        }

    return results


def eval_spatial_structure(samples: np.ndarray, history: np.ndarray, horizons: list[int]) -> dict:
    """Check if 5x5 grid structure is preserved at each horizon.

    Measures:
    - Term structure slope (ATM: 2Y - 1M)
    - Smile convexity (6M row: wings - ATM)
    - Cross-cell correlation structure vs history
    """
    B, S, T, H, W = samples.shape

    # History spatial stats (ground truth reference)
    hist_median = np.median(history, axis=1)  # (B, 5, 5)  # note: history is (B,30,5,5)
    hist_term_slope = hist_median[:, 4, 2] - hist_median[:, 0, 2]  # 2Y - 1M ATM
    hist_smile = 0.5 * (hist_median[:, 2, 0] + hist_median[:, 2, 4]) - hist_median[:, 2, 2]  # wings - ATM for 6M

    results = {}
    for h in horizons:
        if h > T:
            continue
        # Median sample at horizon h: (B, S, 5, 5) -> median over S -> (B, 5, 5)
        surf_h = np.median(samples[:, :, h - 1, :, :], axis=1)

        # Term structure slope
        term_slope = surf_h[:, 4, 2] - surf_h[:, 0, 2]

        # Smile convexity for 6M row
        smile_conv = 0.5 * (surf_h[:, 2, 0] + surf_h[:, 2, 4]) - surf_h[:, 2, 2]

        # Cross-cell correlation of generated surface vs history pattern
        gen_flat = surf_h.reshape(B, 25)
        hist_flat = hist_median.reshape(B, 25)
        # Per-window correlation between generated and historical spatial pattern
        corrs = []
        for b in range(B):
            c = np.corrcoef(gen_flat[b], hist_flat[b])[0, 1]
            if np.isfinite(c):
                corrs.append(c)

        results[f"h{h}"] = {
            "term_slope_mean": float(term_slope.mean()),
            "term_slope_std": float(term_slope.std()),
            "hist_term_slope_mean": float(hist_term_slope.mean()),
            "smile_convexity_mean": float(smile_conv.mean()),
            "smile_convexity_std": float(smile_conv.std()),
            "hist_smile_convexity_mean": float(hist_smile.mean()),
            "spatial_corr_with_history": float(np.mean(corrs)) if corrs else 0.0,
            "term_slope_preserved": abs(float(term_slope.mean())) > 0.001,
            "smile_preserved": float(smile_conv.mean()) > -0.01,  # should be positive
        }

    return results


def eval_path_stationarity(samples: np.ndarray, horizons: list[int], window_size: int = 20) -> dict:
    """Rolling std of daily changes -- detect drift/explosion in path dynamics.

    samples: (B, S, T, 5, 5) in [0, 1]
    """
    B, S, T, H, W = samples.shape

    # Daily changes: (B, S, T-1, 5, 5)
    daily_changes = np.diff(samples, axis=2)

    # Rolling std across time for ATM 6M cell
    atm_changes = daily_changes[:, :, :, 2, 2]  # (B, S, T-1)
    # Average across windows and samples: (T-1,)
    avg_abs_change = np.abs(atm_changes).mean(axis=(0, 1))

    # Rolling window std
    rolling_stds = []
    for t in range(window_size, len(avg_abs_change)):
        window = avg_abs_change[t - window_size:t]
        rolling_stds.append(float(window.std()))

    # Per-cell analysis across all 25 cells
    cell_daily_std = daily_changes.std(axis=(0, 1, 2))  # (5, 5) -- overall std of daily changes

    # Stationarity ratio: std of last quarter vs first quarter of daily changes
    quarter = max(1, (T - 1) // 4)
    first_q = daily_changes[:, :, :quarter, :, :]
    last_q = daily_changes[:, :, -(quarter):, :, :]
    first_std = first_q.std()
    last_std = last_q.std()
    stationarity_ratio = float(last_std / max(first_std, 1e-8))

    # Per-horizon daily change statistics
    horizon_stats = {}
    for h in horizons:
        if h > T:
            continue
        # Daily changes leading up to horizon h
        changes_to_h = daily_changes[:, :, :h - 1, :, :]  if h > 1 else daily_changes[:, :, :1, :, :]
        horizon_stats[f"h{h}"] = {
            "daily_change_mean": float(changes_to_h.mean()),
            "daily_change_std": float(changes_to_h.std()),
            "daily_change_abs_mean": float(np.abs(changes_to_h).mean()),
            "daily_change_kurtosis": float(
                _safe_kurtosis(changes_to_h.reshape(-1))
            ),
        }

    return {
        "stationarity_ratio_last_vs_first": stationarity_ratio,
        "stationary": 0.5 < stationarity_ratio < 2.0,
        "cell_daily_std": cell_daily_std.tolist(),
        "rolling_std_atm6m": rolling_stds if len(rolling_stds) <= 300 else rolling_stds[::2],
        "avg_abs_daily_change_atm6m": avg_abs_change.tolist() if len(avg_abs_change) <= 300 else avg_abs_change[::2].tolist(),
        "horizon_stats": horizon_stats,
    }


def _safe_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis, handling edge cases."""
    from scipy import stats as sp_stats
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return 0.0
    return float(sp_stats.kurtosis(arr, fisher=True))


def eval_gt_comparison(samples: np.ndarray, gt_futures: list[np.ndarray], horizons: list[int]) -> dict:
    """Compare generated paths with ground truth where available.

    samples: (B, S, T, 5, 5) in [0, 1]
    gt_futures: list of (T_gt, 5, 5) arrays (variable length, may be < 252)
    """
    B = len(gt_futures)
    results = {}

    for h in horizons:
        # Check which windows have GT at this horizon
        valid = []
        for i in range(min(B, samples.shape[0])):
            if gt_futures[i].shape[0] >= h:
                valid.append(i)

        if not valid:
            results[f"h{h}"] = {"n_valid": 0, "note": "No GT available at this horizon"}
            continue

        # GT at horizon h
        gt_at_h = np.stack([gt_futures[i][h - 1] for i in valid])  # (n_valid, 5, 5)

        # Generated at horizon h
        gen_at_h = samples[valid, :, h - 1, :, :]  # (n_valid, S, 5, 5)
        gen_median = np.median(gen_at_h, axis=1)  # (n_valid, 5, 5)

        # CI coverage: does GT fall within 5th-95th percentile?
        q05 = np.percentile(gen_at_h, 5, axis=1)
        q95 = np.percentile(gen_at_h, 95, axis=1)
        covered = (gt_at_h >= q05) & (gt_at_h <= q95)
        ci_coverage = float(covered.mean())

        # MAE
        mae = float(np.abs(gen_median - gt_at_h).mean())

        # Bias
        bias = float((gen_median - gt_at_h).mean())

        results[f"h{h}"] = {
            "n_valid": len(valid),
            "ci_90_coverage": ci_coverage,
            "mae": mae,
            "bias": bias,
            "gen_mean_iv": float(gen_median.mean()),
            "gt_mean_iv": float(gt_at_h.mean()),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="183c 252-day long-horizon test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--n_windows", type=int, default=N_WINDOWS)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n_blocks", type=int, default=N_BLOCKS)
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("183c Long-Horizon Test (252 days)")
    print("=" * 70)

    # 1. Load model
    print("\n[1/4] Loading model...")
    model = load_model(args.model_path, args.device)

    # 2. Load data
    print("\n[2/4] Loading data...")
    history_norm, history_raw, returns, gt_futures = load_data(args.data_path, args.device)
    # Use only n_windows
    history_norm = history_norm[:args.n_windows]
    history_raw = history_raw[:args.n_windows]
    gt_futures = gt_futures[:args.n_windows]

    print(f"  Using {len(gt_futures)} test windows")
    gt_lengths = [gf.shape[0] for gf in gt_futures]
    print(f"  GT future lengths: min={min(gt_lengths)}, max={max(gt_lengths)}, mean={np.mean(gt_lengths):.0f}")

    # 3. Generate 252-day samples
    print("\n[3/4] Generating 252-day paths...")
    t_gen = time.time()
    samples = generate_long_horizon(
        model, history_norm, args.n_samples, args.n_blocks, args.device
    )
    gen_time = time.time() - t_gen
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Sample range: [{samples.min():.4f}, {samples.max():.4f}]")

    # 4. Evaluate
    print("\n[4/4] Running evaluations...")

    print("  Ensemble spread...")
    spread_results = eval_ensemble_spread(samples, HORIZONS)

    print("  Surface validity...")
    validity_results = eval_surface_validity(samples, HORIZONS)

    print("  Spatial structure...")
    spatial_results = eval_spatial_structure(samples, history_raw, HORIZONS)

    print("  Path stationarity...")
    stationarity_results = eval_path_stationarity(samples, HORIZONS)

    print("  GT comparison...")
    gt_results = eval_gt_comparison(samples, gt_futures, HORIZONS)

    # Compile all results
    total_time = time.time() - t_start

    results = {
        "experiment": "183c",
        "model_type": "state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c",
        "model_path": args.model_path,
        "test_config": {
            "n_windows": args.n_windows,
            "n_samples": args.n_samples,
            "n_blocks": args.n_blocks,
            "total_frames": TOTAL_FRAMES,
            "horizons": HORIZONS,
            "test_start": TEST_START,
            "history_len": HISTORY_LEN,
            "chaining_method": "autoregressive_30frame_blocks",
        },
        "timing": {
            "generation_seconds": gen_time,
            "total_seconds": total_time,
        },
        "sample_stats": {
            "shape": list(samples.shape),
            "min": float(samples.min()),
            "max": float(samples.max()),
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "nan_count": int(np.isnan(samples).sum()),
            "inf_count": int(np.isinf(samples).sum()),
        },
        "ensemble_spread": spread_results,
        "surface_validity": validity_results,
        "spatial_structure": spatial_results,
        "path_stationarity": stationarity_results,
        "gt_comparison": gt_results,
    }

    # Verdict
    verdict = {
        "spread_monotonic": spread_results["all_monotonic"],
        "no_explosions_h30": validity_results.get("h30", {}).get("explosion_rate", 1.0) < 0.05,
        "no_explosions_h252": validity_results.get("h252", {}).get("explosion_rate", 1.0) < 0.20,
        "stationary_paths": stationarity_results["stationary"],
        "spatial_preserved_h252": spatial_results.get("h252", {}).get("spatial_corr_with_history", 0) > 0.5,
    }
    verdict["overall_pass"] = all(verdict.values())
    results["verdict"] = verdict

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n-- Ensemble Spread (monotonic growth) --")
    for p in spread_results["monotonicity_pairs"]:
        status = "OK" if p["growing"] else "FAIL"
        print(f"  h{p['from']}->h{p['to']}: {p['prev_std']:.5f} -> {p['curr_std']:.5f} (x{p['ratio']:.2f}) [{status}]")
    print(f"  All monotonic: {spread_results['all_monotonic']}")

    print("\n-- Surface Validity --")
    for h in HORIZONS:
        key = f"h{h}"
        if key in validity_results:
            v = validity_results[key]
            print(f"  h={h}: explosion={v['explosion_rate']:.3f}, cal_arb={v['calendar_arb_rate']:.3f}, "
                  f"butterfly_arb={v['butterfly_arb_rate']:.3f}, IV=[{v['min_iv']:.3f}, {v['max_iv']:.3f}]")

    print("\n-- Spatial Structure --")
    for h in HORIZONS:
        key = f"h{h}"
        if key in spatial_results:
            s = spatial_results[key]
            print(f"  h={h}: term_slope={s['term_slope_mean']:.4f} (hist={s['hist_term_slope_mean']:.4f}), "
                  f"smile={s['smile_convexity_mean']:.4f}, spatial_corr={s['spatial_corr_with_history']:.3f}")

    print("\n-- Path Stationarity --")
    print(f"  Ratio last/first quarter: {stationarity_results['stationarity_ratio_last_vs_first']:.3f}")
    print(f"  Stationary: {stationarity_results['stationary']}")

    print("\n-- GT Comparison --")
    for h in HORIZONS:
        key = f"h{h}"
        if key in gt_results:
            g = gt_results[key]
            if g.get("n_valid", 0) > 0:
                print(f"  h={h}: CI90={g['ci_90_coverage']:.3f}, MAE={g['mae']:.5f}, "
                      f"bias={g['bias']:.5f}, n_valid={g['n_valid']}")
            else:
                print(f"  h={h}: {g.get('note', 'no data')}")

    print("\n-- Verdict --")
    for k, v in verdict.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")
    print(f"\n  OVERALL: {'PASS' if verdict['overall_pass'] else 'FAIL'}")
    print(f"\n  Total time: {total_time:.1f}s")

    # Save results
    base_dir = Path("results/validations/2026-04-06")

    # Main results
    analysis_path = base_dir / "analysis" / "validation_audit" / "183c_long_horizon.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to: {analysis_path}")

    # Verification results (compact verdict + key numbers)
    verification = {
        "experiment": "183c",
        "test": "long_horizon_252d",
        "date": "2026-04-06",
        "verdict": verdict,
        "key_metrics": {
            "spread_h30": spread_results["horizon_spreads"].get("h30", {}).get("mean_std"),
            "spread_h252": spread_results["horizon_spreads"].get("h252", {}).get("mean_std"),
            "spread_growth_ratio": (
                spread_results["horizon_spreads"].get("h252", {}).get("mean_std", 0)
                / max(spread_results["horizon_spreads"].get("h30", {}).get("mean_std", 1e-8), 1e-8)
            ),
            "explosion_rate_h252": validity_results.get("h252", {}).get("explosion_rate"),
            "stationarity_ratio": stationarity_results["stationarity_ratio_last_vs_first"],
            "spatial_corr_h252": spatial_results.get("h252", {}).get("spatial_corr_with_history"),
            "gt_ci90_h30": gt_results.get("h30", {}).get("ci_90_coverage"),
            "gt_ci90_h252": gt_results.get("h252", {}).get("ci_90_coverage"),
        },
        "timing_seconds": total_time,
    }
    verif_path = base_dir / "verification_results" / "183c_long_horizon.json"
    verif_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verif_path, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"Verification saved to: {verif_path}")

    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
