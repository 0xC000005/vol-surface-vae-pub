#!/usr/bin/env python
"""
Long-horizon (252-day) test for 176b SharedLocalTemplateMixture model.

The 176b model generates 30 frames at once (one-shot structured joint distribution).
For 252-day paths, we chain 9 blocks of 30 frames (9*30=270 >= 252), using the last
frame of each block as the conditioning "history" for the next block. The real history
(30 days) conditions the first block; subsequent blocks are conditioned on a synthetic
history built from the last 30 generated frames.

Evaluates at horizons h=30, 60, 90, 180, 252:
  1. Ensemble spread (per-cell std) -- monotonic growth?
  2. Surface validity -- explosion rate
  3. Spatial structure -- 5x5 grid validity at h=252
  4. Path stationarity -- rolling std of daily changes (detect drift/explosion)

Usage:
    PYTHONPATH=. python results/validations/2026-04-06/scripts/176b_long_horizon.py \
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
from experiments.backfill.block_ar.train_176b_shared_local_template_mixture import (
    SharedLocalTemplateMixtureStudentTModel,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)

# ── Constants ──
MODEL_PATH = "models/backfill/shared_local_template_mixture_residual_flow_structured_joint_student_t_176b/best_model.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
TEST_START = 4540
HISTORY_LEN = 30
BLOCK_LEN = 30
N_BLOCKS = 9  # 9 * 30 = 270 >= 252
TOTAL_FRAMES = N_BLOCKS * BLOCK_LEN  # 270
N_WINDOWS = 20
N_SAMPLES = 50
CHUNK_SIZE = 10  # samples per chunk to avoid OOM
HORIZONS = [30, 60, 90, 180, 252]

LABELS_K = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
LABELS_T = ["1M", "3M", "6M", "1Y", "2Y"]
CELL_NAMES = [f"{t}/{k}" for t in LABELS_T for k in LABELS_K]

OUTPUT_DIR = Path("results/validations/2026-04-06/analysis/validation_audit")
VERIFICATION_DIR = Path("results/validations/2026-04-06/verification_results")


def load_model(model_path: str, device: str) -> SharedLocalTemplateMixtureStudentTModel:
    """Load 176b model from checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    raw_config = ckpt["config"]

    enc_cfg = EncoderConfig(**raw_config["encoder"])
    dec_cfg = raw_config["decoder"]

    model = SharedLocalTemplateMixtureStudentTModel(
        encoder_config=enc_cfg,
        decoder_config=dec_cfg,
        flow_config=raw_config["flow"],
        base_nu=raw_config.get("base_nu", 8.0),
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"  Loaded 176b model from {model_path}")
    print(f"  Config: n_frames={dec_cfg['n_frames']}, n_components={dec_cfg.get('n_components', 3)}, "
          f"base_nu={raw_config.get('base_nu', 8.0)}")
    return model


def load_data(data_path: str, device: str):
    """Load vol surface data, return surfaces tensor and returns array."""
    data = np.load(data_path)
    surfaces = torch.tensor(data["surface"], dtype=torch.float32, device=device)  # (N, 5, 5)
    returns = data["ret"].astype(np.float32)  # (N,)
    print(f"  Data: {surfaces.shape[0]} days, test start={TEST_START}")
    return surfaces, returns


@torch.no_grad()
def generate_long_horizon_chained(
    model: SharedLocalTemplateMixtureStudentTModel,
    history_01: torch.Tensor,
    n_samples: int,
    n_blocks: int,
    chunk_size: int = 10,
) -> torch.Tensor:
    """
    Generate long-horizon paths by chaining 30-frame blocks.

    The 176b model generates 30 frames conditioned on a 30-day history.
    For longer horizons, we chain blocks:
      Block 0: condition on real history -> generate frames 0-29
      Block 1: condition on generated frames 0-29 -> generate frames 30-59
      ...
      Block k: condition on generated frames [(k-1)*30 : k*30] -> generate frames [k*30 : (k+1)*30]

    Each sample path chains independently (no cross-sample mixing).

    Args:
        model: The 176b model
        history_01: (B, 30, 5, 5) history in [0, 1] scale
        n_samples: number of ensemble members
        n_blocks: number of 30-frame blocks to chain
        chunk_size: samples per GPU chunk

    Returns:
        (B, n_samples, n_blocks*30, 5, 5) in [0, 1] scale
    """
    B = history_01.shape[0]
    device = history_01.device
    block_len = model.decoder.n_frames  # 30

    # Normalize history to [-1, 1] for the encoder
    history_norm = normalize_iv(history_01)

    all_blocks = []  # will hold (B, n_samples, block_len, 5, 5) per block

    for block_idx in range(n_blocks):
        block_chunks = []
        for s_start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - s_start)
            # sample_batched expects [-1, 1] input for the encoder
            # but internally calls denormalize_iv to convert to [0, 1]
            # So we pass the normalized history
            samples_01 = model.sample_batched(
                history_norm, n_samples=k, chunk_size=k,
            )  # (B, k, 30, 5, 5) in [0, 1]
            block_chunks.append(samples_01)

        block_samples = torch.cat(block_chunks, dim=1)  # (B, n_samples, 30, 5, 5)
        all_blocks.append(block_samples)

        # For next block: each sample path's last 30 frames become its new history
        # We need to normalize back to [-1, 1] for the encoder
        # But we need to handle each sample independently
        # Strategy: use the median path as the conditioning history for the next block
        # This is a simplification -- ideally each sample continues its own path,
        # but the model expects (B, 30, 5, 5) not (B*K, 30, 5, 5)
        # We use per-sample conditioning by expanding batch dim
        if block_idx < n_blocks - 1:
            # Use the generated block as next history
            # Take each sample's own path as its conditioning
            # Reshape: (B, n_samples, 30, 5, 5) -> (B*n_samples, 30, 5, 5)
            next_hist_01 = block_samples.reshape(B * n_samples, block_len, 5, 5)
            history_norm = normalize_iv(next_hist_01)

            # Now we generate the next block with B*n_samples as batch
            # and 1 sample each, then reshape back
            # This preserves per-path continuity
            next_block_samples = model.sample_batched(
                history_norm, n_samples=1, chunk_size=1,
            )  # (B*n_samples, 1, 30, 5, 5)
            next_block_samples = next_block_samples.reshape(B, n_samples, block_len, 5, 5)

            # Override the simple approach -- use per-path chaining directly
            all_blocks_chain = all_blocks[:-1] if block_idx > 0 else []
            # Actually, let me restructure the loop. We need per-path chaining.
            # Break and restart with correct approach.
            pass

    # The above approach has a flaw with per-path chaining.
    # Let me use the correct approach: expand batch dim for per-path independence.
    # Restart generation with proper per-path chaining.
    return _generate_chained_perpatch(model, history_01, n_samples, n_blocks, chunk_size)


@torch.no_grad()
def _generate_chained_perpatch(
    model: SharedLocalTemplateMixtureStudentTModel,
    history_01: torch.Tensor,
    n_samples: int,
    n_blocks: int,
    chunk_size: int = 10,
) -> torch.Tensor:
    """
    Per-path chained generation: each sample follows its own trajectory.

    Strategy: fold n_samples into batch dimension so each sample gets
    its own conditioning history.

    Block 0: (B, 30, 5, 5) history -> generate (B, K, 30, 5, 5) with K samples
    Block 1+: (B*K, 30, 5, 5) from previous block -> generate (B*K, 1, 30, 5, 5)

    This ensures each path is conditioned on its own generated history.
    """
    B = history_01.shape[0]
    device = history_01.device
    block_len = model.decoder.n_frames  # 30

    all_blocks = []

    # Block 0: generate K samples from real history
    history_norm = normalize_iv(history_01)  # (B, 30, 5, 5)
    block0_chunks = []
    for s_start in range(0, n_samples, chunk_size):
        k = min(chunk_size, n_samples - s_start)
        samples_01 = model.sample_batched(
            history_norm, n_samples=k, chunk_size=k,
        )  # (B, k, 30, 5, 5) in [0, 1]
        block0_chunks.append(samples_01)
    block0 = torch.cat(block0_chunks, dim=1)  # (B, n_samples, 30, 5, 5)
    all_blocks.append(block0)
    print(f"    Block 0: generated {block0.shape}")

    # Blocks 1+: each path generates 1 sample from its own last 30 frames
    for block_idx in range(1, n_blocks):
        # Previous block's output: (B, n_samples, 30, 5, 5)
        prev_block = all_blocks[-1]
        # Fold samples into batch: (B*n_samples, 30, 5, 5)
        prev_flat = prev_block.reshape(B * n_samples, block_len, 5, 5)
        hist_norm = normalize_iv(prev_flat)  # (B*n_samples, 30, 5, 5)

        # Generate 1 sample per path, in chunks to avoid OOM
        # B*n_samples could be large (20*50=1000), chunk over this
        batch_chunk = 32  # process 32 paths at a time
        next_blocks = []
        for b_start in range(0, B * n_samples, batch_chunk):
            b_end = min(b_start + batch_chunk, B * n_samples)
            chunk_hist = hist_norm[b_start:b_end]
            chunk_samples = model.sample_batched(
                chunk_hist, n_samples=1, chunk_size=1,
            )  # (chunk_B, 1, 30, 5, 5)
            next_blocks.append(chunk_samples.squeeze(1))  # (chunk_B, 30, 5, 5)

        next_flat = torch.cat(next_blocks, dim=0)  # (B*n_samples, 30, 5, 5)
        next_block = next_flat.reshape(B, n_samples, block_len, 5, 5)
        all_blocks.append(next_block)
        print(f"    Block {block_idx}: generated {next_block.shape}")

    # Concatenate all blocks along time axis
    # Each block: (B, n_samples, 30, 5, 5)
    full_paths = torch.cat(all_blocks, dim=2)  # (B, n_samples, 270, 5, 5)
    return full_paths


def compute_explosion_rate(samples_01: np.ndarray, threshold: float = 1.5) -> float:
    """Fraction of samples with any cell > threshold."""
    max_per_sample = samples_01.max(axis=(-1, -2))  # (N, K, T)
    exploded = (max_per_sample > threshold).any(axis=-1)  # (N, K)
    return float(exploded.mean())


def compute_ensemble_spread(samples_01: np.ndarray, horizons: list[int]) -> dict:
    """Per-cell std across ensemble members at each horizon."""
    result = {}
    for h in horizons:
        if h > samples_01.shape[2]:
            continue
        # samples at horizon h: (N, K, 5, 5)
        s_h = samples_01[:, :, h - 1, :, :]
        cell_std = s_h.std(axis=1)  # (N, 5, 5)
        mean_std = cell_std.mean(axis=0)  # (5, 5)
        result[str(h)] = {
            "mean_cell_std": float(mean_std.mean()),
            "per_cell_std": mean_std.tolist(),
            "min_cell_std": float(mean_std.min()),
            "max_cell_std": float(mean_std.max()),
        }
    return result


def check_monotonic_spread(spread_data: dict) -> dict:
    """Check if ensemble spread grows monotonically with horizon."""
    horizons = sorted([int(h) for h in spread_data.keys()])
    means = [spread_data[str(h)]["mean_cell_std"] for h in horizons]
    monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    return {
        "horizons": horizons,
        "mean_stds": means,
        "monotonic": monotonic,
        "ratio_252_30": means[-1] / means[0] if means[0] > 0 else float("inf"),
    }


def compute_surface_validity(samples_01: np.ndarray, horizons: list[int]) -> dict:
    """Check surface validity at each horizon: explosion, range."""
    result = {}
    for h in horizons:
        if h > samples_01.shape[2]:
            continue
        s_h = samples_01[:, :, h - 1, :, :]  # (N, K, 5, 5)
        result[str(h)] = {
            "explosion_rate": float((s_h > 1.5).any(axis=(-1, -2)).mean()),
            "negative_rate": float((s_h < 0).any(axis=(-1, -2)).mean()),
            "mean_iv": float(s_h.mean()),
            "std_iv": float(s_h.std()),
            "min_iv": float(s_h.min()),
            "max_iv": float(s_h.max()),
            "median_iv": float(np.median(s_h)),
        }
    return result


def compute_spatial_structure(samples_01: np.ndarray, horizon: int = 252) -> dict:
    """Check spatial structure at a given horizon."""
    if horizon > samples_01.shape[2]:
        horizon = samples_01.shape[2]
    s_h = samples_01[:, :, horizon - 1, :, :]  # (N, K, 5, 5)
    median_surface = np.median(s_h, axis=(0, 1))  # (5, 5) median across windows and samples

    # Term structure slope: tenor increasing should show specific pattern
    # Mean across moneyness for each tenor
    tenor_means = median_surface.mean(axis=1)  # (5,) across tenors
    term_slope = float(tenor_means[-1] - tenor_means[0])

    # Smile convexity: for each tenor, check U-shape across moneyness
    moneyness_means = median_surface.mean(axis=0)  # (5,) across moneyness
    # Convexity = wings vs ATM: (left + right) / 2 - center
    smile_convexity = float((moneyness_means[0] + moneyness_means[-1]) / 2 - moneyness_means[2])

    # Cross-cell correlation of levels
    flat = s_h.reshape(-1, 25)
    if flat.shape[0] > 1:
        corr = np.corrcoef(flat, rowvar=False)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[np.isfinite(upper)]
        mean_corr = float(upper.mean()) if len(upper) > 0 else 0.0
    else:
        mean_corr = 0.0

    return {
        "horizon": horizon,
        "median_surface": median_surface.tolist(),
        "tenor_means": tenor_means.tolist(),
        "term_structure_slope": term_slope,
        "moneyness_means": moneyness_means.tolist(),
        "smile_convexity": smile_convexity,
        "mean_cross_cell_correlation": mean_corr,
    }


def compute_path_stationarity(samples_01: np.ndarray, window_size: int = 30) -> dict:
    """
    Rolling std of daily changes to detect drift/explosion.
    Compute for ATM 3M cell (row=1, col=2) as representative.
    """
    # ATM 3M cell: tenor=3M (idx 1), moneyness=K=1.00 (idx 2)
    atm_paths = samples_01[:, :, :, 1, 2]  # (N, K, T)
    daily_changes = np.diff(atm_paths, axis=2)  # (N, K, T-1)

    # Rolling std across all paths
    T = daily_changes.shape[2]
    rolling_stds = []
    centers = []
    for start in range(0, T - window_size + 1, window_size // 2):
        end = start + window_size
        if end > T:
            break
        window = daily_changes[:, :, start:end]
        rolling_stds.append(float(window.std()))
        centers.append(start + window_size // 2)

    # Stationarity check: is the last window's std within 3x of the first?
    if len(rolling_stds) >= 2:
        ratio_last_first = rolling_stds[-1] / rolling_stds[0] if rolling_stds[0] > 0 else float("inf")
        stationary = ratio_last_first < 3.0
    else:
        ratio_last_first = 1.0
        stationary = True

    return {
        "cell": "3M/K=1.00",
        "window_size": window_size,
        "rolling_std_centers": centers,
        "rolling_stds": rolling_stds,
        "ratio_last_first": ratio_last_first,
        "stationary": stationary,
    }


def compute_gt_comparison(
    samples_01: np.ndarray,
    surfaces: np.ndarray,
    test_indices: np.ndarray,
    horizons: list[int],
) -> dict:
    """Compare generated paths to ground truth at available horizons."""
    result = {}
    for h in horizons:
        gt_indices = test_indices + HISTORY_LEN + h - 1
        valid = gt_indices < len(surfaces)
        if valid.sum() == 0:
            continue
        gt_surfaces = surfaces[gt_indices[valid]]  # (N_valid, 5, 5)
        gen_at_h = samples_01[valid, :, h - 1, :, :]  # (N_valid, K, 5, 5)

        # CI coverage: what fraction of GT cells fall within 5th-95th percentile?
        lo = np.percentile(gen_at_h, 5, axis=1)  # (N_valid, 5, 5)
        hi = np.percentile(gen_at_h, 95, axis=1)
        covered = (gt_surfaces >= lo) & (gt_surfaces <= hi)
        coverage = float(covered.mean())

        # Median prediction error
        gen_median = np.median(gen_at_h, axis=1)  # (N_valid, 5, 5)
        mae = float(np.abs(gen_median - gt_surfaces).mean())
        bias = float((gen_median - gt_surfaces).mean())

        result[str(h)] = {
            "n_valid": int(valid.sum()),
            "ci_90_coverage": coverage,
            "median_mae": mae,
            "median_bias": bias,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="176b long-horizon (252-day) test")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_windows", type=int, default=N_WINDOWS)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n_blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    print("=" * 70)
    print("176b Long-Horizon Test (252 days)")
    print("=" * 70)

    # Load model and data
    model = load_model(args.model_path, args.device)
    surfaces, returns = load_data(args.data_path, args.device)

    # Build test windows
    max_future = args.n_blocks * BLOCK_LEN  # 270
    valid_end = len(surfaces) - HISTORY_LEN - max_future
    test_indices = np.arange(TEST_START, min(TEST_START + args.n_windows, valid_end))
    actual_windows = len(test_indices)
    print(f"  Test windows: {actual_windows} (requested {args.n_windows})")

    if actual_windows == 0:
        print("ERROR: No valid test windows. Data too short for 252-day horizon.")
        return

    # Build history tensors
    history_01 = torch.stack([
        surfaces[idx:idx + HISTORY_LEN] for idx in test_indices
    ])  # (N, 30, 5, 5)
    print(f"  History shape: {history_01.shape}")

    # Generate long-horizon samples
    print(f"\nGenerating {args.n_samples} samples x {args.n_blocks} blocks x {actual_windows} windows...")
    t0 = time.time()
    samples_01 = _generate_chained_perpatch(
        model, history_01, args.n_samples, args.n_blocks, args.chunk_size,
    )  # (N, K, 270, 5, 5)
    elapsed = time.time() - t0
    print(f"  Generation time: {elapsed:.1f}s")
    print(f"  Output shape: {samples_01.shape}")

    # Move to numpy for analysis
    samples_np = samples_01.cpu().numpy()
    surfaces_np = surfaces.cpu().numpy()

    # Trim to 252 frames for analysis
    samples_252 = samples_np[:, :, :252, :, :]
    print(f"  Analysis shape (trimmed to 252): {samples_252.shape}")

    # ── Evaluation ──
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # 1. Ensemble spread
    print("\n1. Ensemble Spread")
    spread = compute_ensemble_spread(samples_252, HORIZONS)
    mono = check_monotonic_spread(spread)
    print(f"   Monotonic growth: {mono['monotonic']}")
    print(f"   Spread ratio (h=252/h=30): {mono['ratio_252_30']:.2f}")
    for h in HORIZONS:
        hs = str(h)
        if hs in spread:
            print(f"   h={h:3d}: mean_std={spread[hs]['mean_cell_std']:.4f}, "
                  f"min={spread[hs]['min_cell_std']:.4f}, max={spread[hs]['max_cell_std']:.4f}")

    # 2. Surface validity
    print("\n2. Surface Validity")
    validity = compute_surface_validity(samples_252, HORIZONS)
    for h in HORIZONS:
        hs = str(h)
        if hs in validity:
            v = validity[hs]
            print(f"   h={h:3d}: explosion={v['explosion_rate']:.3f}, "
                  f"negative={v['negative_rate']:.3f}, "
                  f"mean={v['mean_iv']:.4f}, range=[{v['min_iv']:.4f}, {v['max_iv']:.4f}]")

    # 3. Spatial structure at h=252
    print("\n3. Spatial Structure at h=252")
    spatial = compute_spatial_structure(samples_252, horizon=252)
    print(f"   Term structure slope: {spatial['term_structure_slope']:.4f}")
    print(f"   Smile convexity: {spatial['smile_convexity']:.4f}")
    print(f"   Cross-cell correlation: {spatial['mean_cross_cell_correlation']:.4f}")
    print(f"   Median surface at h=252:")
    for i, t_label in enumerate(LABELS_T):
        row_str = "   " + f"  {t_label:>4s}: " + "  ".join(f"{spatial['median_surface'][i][j]:.4f}"
                                                             for j in range(5))
        print(row_str)

    # 4. Path stationarity
    print("\n4. Path Stationarity")
    stationarity = compute_path_stationarity(samples_252)
    print(f"   Cell: {stationarity['cell']}")
    print(f"   Rolling std ratio (last/first): {stationarity['ratio_last_first']:.2f}")
    print(f"   Stationary: {stationarity['stationary']}")
    print(f"   Rolling stds: {[f'{s:.5f}' for s in stationarity['rolling_stds']]}")

    # 5. GT comparison (where available)
    print("\n5. Ground Truth Comparison")
    gt_comparison = compute_gt_comparison(
        samples_252, surfaces_np, test_indices, HORIZONS,
    )
    for h in HORIZONS:
        hs = str(h)
        if hs in gt_comparison:
            gc = gt_comparison[hs]
            print(f"   h={h:3d}: CI90={gc['ci_90_coverage']:.3f}, "
                  f"MAE={gc['median_mae']:.4f}, bias={gc['median_bias']:+.4f}, "
                  f"n_valid={gc['n_valid']}")

    # 6. Overall explosion rate
    overall_explosion = compute_explosion_rate(samples_252, threshold=1.5)
    print(f"\n6. Overall explosion rate (>1.5): {overall_explosion:.4f}")

    # ── Summary ──
    spread_pass = mono["monotonic"]
    explosion_pass = all(
        validity.get(str(h), {}).get("explosion_rate", 1.0) < 0.10
        for h in HORIZONS
    )
    stationarity_pass = stationarity["stationary"]
    spatial_pass = spatial["smile_convexity"] > 0  # basic smile present

    overall_pass = spread_pass and explosion_pass and stationarity_pass and spatial_pass

    summary = {
        "model": "176b",
        "model_path": args.model_path,
        "n_windows": actual_windows,
        "n_samples": args.n_samples,
        "n_blocks": args.n_blocks,
        "total_frames": args.n_blocks * BLOCK_LEN,
        "horizons_tested": HORIZONS,
        "generation_time_s": elapsed,
        "results": {
            "ensemble_spread": spread,
            "monotonic_spread": mono,
            "surface_validity": validity,
            "spatial_structure_252": spatial,
            "path_stationarity": stationarity,
            "gt_comparison": gt_comparison,
            "overall_explosion_rate": overall_explosion,
        },
        "pass_fail": {
            "monotonic_spread": spread_pass,
            "explosion_rate": explosion_pass,
            "stationarity": stationarity_pass,
            "spatial_structure": spatial_pass,
            "overall": overall_pass,
        },
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, passed in summary["pass_fail"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:25s}: {status}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    # Helper for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        return obj

    results_path = OUTPUT_DIR / "176b_long_horizon.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    verification = {
        "test": "176b_long_horizon",
        "model": "176b",
        "pass": overall_pass,
        "tests": summary["pass_fail"],
        "key_metrics": {
            "spread_ratio_252_30": mono["ratio_252_30"],
            "overall_explosion_rate": overall_explosion,
            "stationarity_ratio": stationarity["ratio_last_first"],
            "smile_convexity_252": spatial["smile_convexity"],
            "cross_cell_corr_252": spatial["mean_cross_cell_correlation"],
        },
    }
    verif_path = VERIFICATION_DIR / "176b_long_horizon.json"
    with open(verif_path, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"  Verification saved to: {verif_path}")


if __name__ == "__main__":
    main()
