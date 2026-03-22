#!/usr/bin/env python
"""
Long-horizon spread diagnostic for 144b.

Tests FOUR hypotheses for why ensemble spread stabilizes instead of growing:

  A) AR(1) noise has stationary variance
  B) Transformer shrinks deltas at later horizons
  C) Log-space floor clamp absorbs spread
  D) Member re-convergence via attention consensus

Usage:
    PYTHONPATH=. python results/validations/2026-03-22/scripts/long_horizon_spread_diagnostic.py \
        --model_path models/backfill/afcrps_144b/best_model.pt \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Project imports ──
from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
)

RESULTS_DIR = Path("results/validations/2026-03-22/analysis/long_horizon_root_cause")
VERIFICATION_DIR = Path("results/validations/2026-03-22/verification_results")


# ======================================================================
# Data loading
# ======================================================================

def load_data(n_batches: int = 10, batch_size: int = 16, history_len: int = 30):
    """Load test windows from vol_surface_with_ret.npz."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    N = len(surfaces)

    # Normalize to [-1, 1]
    surfaces_norm = surfaces * 2.0 - 1.0

    # Build windows from the last 20% of data (test set)
    test_start = int(N * 0.8)
    windows = []
    for i in range(test_start, N - history_len):
        windows.append(surfaces_norm[i : i + history_len])
        if len(windows) >= n_batches * batch_size:
            break

    windows = np.array(windows[: n_batches * batch_size])  # (n_batches*batch_size, 30, 5, 5)
    # Reshape into batches
    batches = windows.reshape(n_batches, batch_size, history_len, 5, 5)
    return batches


# ======================================================================
# Model loading
# ======================================================================

def load_model(model_path: str, device: str = "cuda"):
    """Load 144b model."""
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    config = SinglePassConfig(**checkpoint["config"])
    model = SinglePassBlockAR(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model, config


# ======================================================================
# Hypothesis A: AR(1) noise stationary variance
# ======================================================================

def test_hypothesis_A(config: SinglePassConfig, max_steps: int = 252):
    """Compare theoretical AR(1) variance trajectory with the stationary limit.

    AR(1) process: z_{t+1} = rho * z_t + sqrt(1-rho^2) * eps
    Starting from z_0 ~ N(0,1), the variance at step t is:
        Var(z_t) = rho^{2t} * Var(z_0) + (1 - rho^{2t}) * sigma_stationary^2
    Since z_0 ~ N(0,1) and sigma_stationary^2 = 1, Var(z_t) = 1 for all t.

    But the CUMULATIVE effect on the trajectory (sum of deltas driven by noise)
    is what matters for spread. If deltas are proportional to z_t, the cumulative
    variance of the trajectory sums:
        Var(sum_{i=0}^{t-1} z_i) = t + 2*rho*(t-1)/(1-rho) - 2*rho^2*(1-rho^{2(t-1)})/((1-rho)^2*(1-rho^2))

    In log-space: log(IV_t) = log(IV_0) + sum_{i=0}^{t-1} vs*delta_i
    If delta_i ~ f(z_i), spread depends on Var(sum of z_i), not Var(z_t).
    """
    rho = config.ar_frame_rho  # 0.8

    results = {}

    # 1. Marginal variance of z_t at each step (should be ~1 for unit-variance start)
    marginal_var = []
    for t in range(max_steps):
        # Var(z_t) = rho^{2t} * 1 + (1 - rho^{2t}) * 1 = 1 (always 1)
        v = rho ** (2 * t) * 1.0 + (1 - rho ** (2 * t)) * 1.0
        marginal_var.append(v)

    # 2. Cumulative variance: Var(S_t) where S_t = sum_{i=0}^{t-1} z_i
    # For AR(1) with stationary variance sigma^2=1:
    # Cov(z_i, z_j) = rho^|i-j|
    # Var(S_t) = sum_{i,j=0}^{t-1} rho^|i-j| = t + 2*sum_{k=1}^{t-1} (t-k)*rho^k
    cumulative_var = []
    for t in range(1, max_steps + 1):
        # Var(S_t) = t + 2 * sum_{k=1}^{t-1} (t-k) * rho^k
        cross_sum = 0.0
        for k in range(1, t):
            cross_sum += (t - k) * rho ** k
        var_st = t + 2 * cross_sum
        cumulative_var.append(var_st)

    # 3. Cumulative std (what drives ensemble spread)
    cumulative_std = [math.sqrt(v) for v in cumulative_var]

    # 4. Spread growth rate: ratio of cumulative_std[t] / cumulative_std[0]
    # For i.i.d. noise, cumulative_std grows as sqrt(t)
    # For AR(1), it grows roughly as sqrt(t * (1+rho)/(1-rho)) for large t
    # The "effective number of independent innovations" per step is (1-rho)/(1+rho)

    # Stationary cumulative variance approximation for large t:
    # Var(S_t) ~ t * (1+rho)/(1-rho) for large t
    effective_var_per_step = (1 + rho) / (1 - rho)

    # Compute relative growth rates
    horizon_points = [1, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200, 252]
    horizon_points = [h for h in horizon_points if h <= max_steps]

    growth_table = {}
    for h in horizon_points:
        cum_std_h = cumulative_std[h - 1]
        sqrt_h = math.sqrt(h)
        growth_table[str(h)] = {
            "cumulative_std": round(cum_std_h, 4),
            "sqrt_t_reference": round(sqrt_h, 4),
            "ratio_to_iid": round(cum_std_h / sqrt_h, 4),
            "growth_from_h1": round(cum_std_h / cumulative_std[0], 4),
        }

    # Key insight: AR(1) noise with rho=0.8 makes cumulative std grow as
    # sqrt(t * 9) = 3*sqrt(t) initially, then transitions to sqrt(t * 9) for large t.
    # The marginal variance is always 1 (stationary), so the noise z_t itself
    # does NOT plateau. But each z_t is highly correlated with z_{t-1}, so
    # the "new information" per step decreases.

    # However, the cumulative effect DOES still grow (as sqrt(t * effective_var_per_step)).
    # So pure AR(1) noise does NOT explain plateau — it still grows.
    # The question is whether the RATE of growth slows enough.

    # Compare growth between h=30 and h=252:
    # If spread at h=30 is S_30 and at h=252 is S_252,
    # the ratio S_252/S_30 tells us how much more spread we should expect.
    h30_std = cumulative_std[29]
    h252_std = cumulative_std[min(251, max_steps - 1)]
    expected_growth_252_vs_30 = h252_std / h30_std

    results = {
        "rho": rho,
        "effective_var_per_step": round(effective_var_per_step, 4),
        "marginal_variance_is_constant": True,  # Always 1 for unit-variance start
        "cumulative_std_h30": round(h30_std, 4),
        "cumulative_std_h252": round(h252_std, 4),
        "expected_growth_252_vs_30": round(expected_growth_252_vs_30, 4),
        "growth_table": growth_table,
        "conclusion": (
            f"AR(1) with rho={rho} has stationary MARGINAL variance (always 1), but "
            f"CUMULATIVE std still grows as sqrt(t * {effective_var_per_step:.1f}). "
            f"Expected spread growth from h=30 to h=252 is {expected_growth_252_vs_30:.2f}x. "
            f"The noise process alone does NOT explain spread plateau — cumulative effect "
            f"keeps growing. However, growth is sub-linear (sqrt), so spread growth DECELERATES."
        ),
    }

    return results


# ======================================================================
# Hypothesis B: Transformer shrinks deltas at later horizons
# ======================================================================

@torch.no_grad()
def test_hypothesis_B(model: SinglePassBlockAR, config: SinglePassConfig,
                      batches: np.ndarray, device: str = "cuda",
                      n_samples: int = 8):
    """Measure mean absolute delta at each of the 30 horizon steps.

    If the causal transformer produces smaller deltas at later steps,
    it's actively compressing spread growth.
    """
    H, W = config.surface_h, config.surface_w
    n_frames = config.future_len  # 30
    rho = config.ar_frame_rho
    floor = config.ar_frame_floor_clamp

    # Collect deltas at each horizon step
    all_deltas = {h: [] for h in range(n_frames)}
    all_raw_deltas = {h: [] for h in range(n_frames)}  # Before vol_scale
    all_vol_scales = []

    for batch_np in batches:
        history = torch.tensor(batch_np, dtype=torch.float32, device=device)
        B = history.shape[0]

        # Repeat for n_samples
        history_k = history.repeat_interleave(n_samples, dim=0)
        Bk = history_k.shape[0]

        # Manually run the AR loop to capture per-step deltas
        _, vol_scale = model._compute_vol_scale(history_k)
        z = model._sample_noise(Bk, history_k.device)
        z_t = z
        condition = model.encoder(history_k, mask=None)
        prev_frame = denormalize_iv(history_k[:, -1])  # (Bk, H, W)

        # Init transformer context
        from diffusion.block_ar.single_pass_ar import CausalARTransformerDecoder
        if isinstance(model.frame_decoder, CausalARTransformerDecoder):
            hist_flat = denormalize_iv(history_k).reshape(Bk, history_k.shape[1], H * W)
            model.frame_decoder.init_context(condition, hist_flat)

        gru_outputs, h_last = model._init_gru_state(history_k)

        for step_idx in range(n_frames):
            if step_idx > 0:
                eps_t = torch.randn_like(z_t)
                z_t = rho * z_t + math.sqrt(1 - rho**2) * eps_t

            local_pos, horizon_bucket = model._get_ar_frame_positions(
                step_idx=step_idx, batch_size=Bk, device=device,
                position_mode="native",
            )

            prev_flat = prev_frame.reshape(Bk, H * W)
            noise_input = model._get_noise_for_decoder(z_t)

            delta = model.frame_decoder(
                prev_flat, condition, noise_input, local_pos, horizon_bucket,
            ).reshape(Bk, H, W)

            # Record raw delta (before vol_scale)
            raw_delta_abs = delta.abs().mean().item()
            all_raw_deltas[step_idx].append(raw_delta_abs)

            # Apply vol_scale and compute actual change
            vs = model._get_ar_frame_vol_scale(condition, vol_scale, None)
            # In log-space: iv_t = prev * exp(vs * delta)
            # The "effective delta" in IV space is prev * (exp(vs*delta) - 1)
            # But for spread, what matters is vs * delta (the log-space shift)
            effective_delta = (vs * delta).abs().mean().item()
            all_deltas[step_idx].append(effective_delta)

            # Step forward
            iv_t = (prev_frame * torch.exp(vs * delta)).clamp(floor, 1.0)
            prev_frame = iv_t

            if not config.ar_freeze_gru_state:
                condition, gru_outputs, h_last = model._gru_step(
                    iv_t, gru_outputs, h_last,
                )

    # Aggregate
    delta_by_horizon = {}
    for h in range(n_frames):
        delta_by_horizon[str(h)] = {
            "mean_abs_raw_delta": round(float(np.mean(all_raw_deltas[h])), 6),
            "mean_abs_effective_delta": round(float(np.mean(all_deltas[h])), 6),
        }

    # Check trend: is delta decreasing over steps?
    raw_deltas_seq = [np.mean(all_raw_deltas[h]) for h in range(n_frames)]
    eff_deltas_seq = [np.mean(all_deltas[h]) for h in range(n_frames)]

    # Compare first 5 vs last 5 steps
    first5_raw = np.mean(raw_deltas_seq[:5])
    last5_raw = np.mean(raw_deltas_seq[-5:])
    first5_eff = np.mean(eff_deltas_seq[:5])
    last5_eff = np.mean(eff_deltas_seq[-5:])

    raw_ratio = last5_raw / (first5_raw + 1e-12)
    eff_ratio = last5_eff / (first5_eff + 1e-12)

    # Linear regression slope for trend
    x = np.arange(n_frames)
    raw_slope = np.polyfit(x, raw_deltas_seq, 1)[0]
    eff_slope = np.polyfit(x, eff_deltas_seq, 1)[0]

    shrinking = raw_ratio < 0.8  # >20% decrease from first5 to last5

    results = {
        "delta_by_horizon": delta_by_horizon,
        "summary": {
            "first5_mean_raw_delta": round(float(first5_raw), 6),
            "last5_mean_raw_delta": round(float(last5_raw), 6),
            "raw_delta_ratio_last5_over_first5": round(float(raw_ratio), 4),
            "first5_mean_effective_delta": round(float(first5_eff), 6),
            "last5_mean_effective_delta": round(float(last5_eff), 6),
            "effective_delta_ratio_last5_over_first5": round(float(eff_ratio), 4),
            "raw_delta_linear_slope": round(float(raw_slope), 8),
            "effective_delta_linear_slope": round(float(eff_slope), 8),
        },
        "hypothesis_confirmed": bool(shrinking),
        "conclusion": (
            f"Raw delta ratio (last5/first5) = {raw_ratio:.4f}. "
            f"{'CONFIRMED: Transformer shrinks deltas at later horizons (>20% decrease).' if shrinking else 'FALSIFIED: Delta magnitude does NOT decrease significantly over horizon.'}"
        ),
    }

    return results


# ======================================================================
# Hypothesis C: Log-space floor clamp absorbs spread
# ======================================================================

@torch.no_grad()
def test_hypothesis_C(model: SinglePassBlockAR, config: SinglePassConfig,
                      batches: np.ndarray, device: str = "cuda",
                      n_samples: int = 20):
    """Measure fraction of generated values at/near the floor at each horizon."""
    H, W = config.surface_h, config.surface_w
    n_frames = config.future_len  # 30
    floor = config.ar_frame_floor_clamp  # 0.01
    near_floor_threshold = 0.02  # "near floor" = IV < 2%

    floor_fractions = {h: [] for h in range(n_frames)}
    near_floor_fractions = {h: [] for h in range(n_frames)}

    for batch_np in batches:
        history = torch.tensor(batch_np, dtype=torch.float32, device=device)
        B = history.shape[0]

        # Generate samples
        samples = model.sample(history, n_samples=n_samples)  # (B, n_samples, 30, 5, 5)

        for h in range(n_frames):
            frame_samples = samples[:, :, h]  # (B, n_samples, 5, 5)
            total = frame_samples.numel()

            at_floor = (frame_samples <= floor + 1e-6).sum().item() / total
            near_floor = (frame_samples < near_floor_threshold).sum().item() / total

            floor_fractions[h].append(at_floor)
            near_floor_fractions[h].append(near_floor)

    # Aggregate
    floor_by_horizon = {}
    for h in range(n_frames):
        floor_by_horizon[str(h)] = {
            "fraction_at_floor": round(float(np.mean(floor_fractions[h])), 6),
            "fraction_near_floor": round(float(np.mean(near_floor_fractions[h])), 6),
        }

    # Check trend
    at_floor_seq = [np.mean(floor_fractions[h]) for h in range(n_frames)]
    near_floor_seq = [np.mean(near_floor_fractions[h]) for h in range(n_frames)]

    floor_max = max(at_floor_seq)
    near_floor_max = max(near_floor_seq)
    floor_h30 = at_floor_seq[-1]
    near_floor_h30 = near_floor_seq[-1]

    # Significant floor impact = >5% of values near floor at any horizon
    significant = near_floor_max > 0.05

    results = {
        "floor_clamp": floor,
        "near_floor_threshold": near_floor_threshold,
        "floor_by_horizon": floor_by_horizon,
        "summary": {
            "max_fraction_at_floor": round(float(floor_max), 6),
            "max_fraction_near_floor": round(float(near_floor_max), 6),
            "h30_fraction_at_floor": round(float(floor_h30), 6),
            "h30_fraction_near_floor": round(float(near_floor_h30), 6),
        },
        "hypothesis_confirmed": bool(significant),
        "conclusion": (
            f"Max near-floor fraction = {near_floor_max:.4%}, h30 = {near_floor_h30:.4%}. "
            f"{'CONFIRMED: Floor clamp absorbs significant spread (>5% near floor).' if significant else 'FALSIFIED: Very few values hit the floor — clamp is not compressing spread.'}"
        ),
    }

    return results


# ======================================================================
# Hypothesis D: Member re-convergence via attention consensus
# ======================================================================

@torch.no_grad()
def test_hypothesis_D(model: SinglePassBlockAR, config: SinglePassConfig,
                      batches: np.ndarray, device: str = "cuda",
                      n_samples: int = 20):
    """Compute pairwise cosine similarity between ensemble members at each horizon.

    If attention pulls members toward consensus, cosine similarity will INCREASE
    over later horizons (members converge).
    """
    H, W = config.surface_h, config.surface_w
    n_frames = config.future_len  # 30

    horizon_checkpoints = [0, 4, 9, 14, 19, 24, 29]  # h=1,5,10,15,20,25,30 (0-indexed)

    cosine_sims = {h: [] for h in horizon_checkpoints}
    spread_by_horizon = {h: [] for h in range(n_frames)}

    for batch_np in batches:
        history = torch.tensor(batch_np, dtype=torch.float32, device=device)
        B = history.shape[0]

        # Generate samples
        samples = model.sample(history, n_samples=n_samples)  # (B, n_samples, 30, 5, 5)

        # Compute spread at each horizon
        for h in range(n_frames):
            frame = samples[:, :, h]  # (B, K, 5, 5)
            spread = frame.std(dim=1).mean().item()  # Ensemble std, averaged
            spread_by_horizon[h].append(spread)

        # Pairwise cosine similarity at checkpoint horizons
        for h in horizon_checkpoints:
            frame = samples[:, :, h].reshape(B, n_samples, H * W)  # (B, K, 25)

            # Subtract per-batch mean to measure MEMBER DIFFERENCES, not absolute level
            frame_centered = frame - frame.mean(dim=1, keepdim=True)  # (B, K, 25)

            # For each batch element, compute pairwise cosine sim between all K members
            batch_cosines = []
            for b in range(B):
                members = frame_centered[b]  # (K, 25)
                # Normalize
                norms = members.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                members_normed = members / norms
                # Pairwise cosine: K x K matrix
                cos_mat = members_normed @ members_normed.T  # (K, K)
                # Upper triangle (exclude diagonal)
                mask = torch.triu(torch.ones(n_samples, n_samples, device=device), diagonal=1).bool()
                pairwise = cos_mat[mask]
                batch_cosines.append(pairwise.mean().item())

            cosine_sims[h].append(float(np.mean(batch_cosines)))

    # Also compute cosine similarity on raw (uncentered) samples
    # This measures whether members produce SIMILAR surfaces (high = consensus)
    raw_cosine_sims = {h: [] for h in horizon_checkpoints}
    for batch_np in batches:
        history = torch.tensor(batch_np, dtype=torch.float32, device=device)
        B = history.shape[0]
        samples = model.sample(history, n_samples=n_samples)

        for h in horizon_checkpoints:
            frame = samples[:, :, h].reshape(B, n_samples, H * W)
            batch_cosines = []
            for b in range(B):
                members = frame[b]  # (K, 25)
                norms = members.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                members_normed = members / norms
                cos_mat = members_normed @ members_normed.T
                mask = torch.triu(torch.ones(n_samples, n_samples, device=device), diagonal=1).bool()
                pairwise = cos_mat[mask]
                batch_cosines.append(pairwise.mean().item())
            raw_cosine_sims[h].append(float(np.mean(batch_cosines)))

    # Aggregate
    cosine_by_horizon = {}
    for h in horizon_checkpoints:
        h_label = str(h + 1)  # Convert to 1-indexed
        cosine_by_horizon[h_label] = {
            "centered_cosine_sim": round(float(np.mean(cosine_sims[h])), 4),
            "raw_cosine_sim": round(float(np.mean(raw_cosine_sims[h])), 4),
        }

    spread_trajectory = {}
    for h in range(n_frames):
        spread_trajectory[str(h + 1)] = round(float(np.mean(spread_by_horizon[h])), 6)

    # Check for re-convergence: does cosine increase after some horizon?
    raw_cos_values = [np.mean(raw_cosine_sims[h]) for h in horizon_checkpoints]
    centered_cos_values = [np.mean(cosine_sims[h]) for h in horizon_checkpoints]

    # Compare h=1 vs h=30
    cos_h1_raw = raw_cos_values[0]
    cos_h30_raw = raw_cos_values[-1]
    cos_h1_centered = centered_cos_values[0]
    cos_h30_centered = centered_cos_values[-1]

    # Check if cosine INCREASES from some midpoint to h=30
    midpoint_raw = raw_cos_values[3]  # h=15
    reconverges = cos_h30_raw > midpoint_raw and (cos_h30_raw - midpoint_raw) > 0.005

    # Spread plateau check
    spread_values = [np.mean(spread_by_horizon[h]) for h in range(n_frames)]
    spread_h10 = spread_values[9]
    spread_h30 = spread_values[29]
    spread_growth_factor = spread_h30 / (spread_h10 + 1e-12)

    results = {
        "cosine_similarity_by_horizon": cosine_by_horizon,
        "spread_trajectory": spread_trajectory,
        "summary": {
            "raw_cosine_h1": round(float(cos_h1_raw), 4),
            "raw_cosine_h15": round(float(midpoint_raw), 4),
            "raw_cosine_h30": round(float(cos_h30_raw), 4),
            "centered_cosine_h1": round(float(cos_h1_centered), 4),
            "centered_cosine_h30": round(float(cos_h30_centered), 4),
            "spread_h1": round(float(spread_values[0]), 6),
            "spread_h10": round(float(spread_h10), 6),
            "spread_h30": round(float(spread_values[29]), 6),
            "spread_growth_h30_over_h10": round(float(spread_growth_factor), 4),
        },
        "hypothesis_confirmed": bool(reconverges),
        "conclusion": (
            f"Raw cosine sim: h1={cos_h1_raw:.4f}, h15={midpoint_raw:.4f}, h30={cos_h30_raw:.4f}. "
            f"Spread growth h30/h10 = {spread_growth_factor:.4f}. "
            f"{'CONFIRMED: Members re-converge after midpoint (cosine increases).' if reconverges else 'FALSIFIED: No re-convergence pattern detected — cosine does not increase toward h=30.'}"
        ),
    }

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Long-horizon spread diagnostic for 144b")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/afcrps_144b/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Ensemble members per sample call")
    args = parser.parse_args()

    print("=" * 70)
    print("LONG-HORIZON SPREAD DIAGNOSTIC: 144b")
    print("=" * 70)

    t0 = time.time()

    # Load model and data
    print("\n[1/6] Loading model...")
    model, config = load_model(args.model_path, args.device)
    print(f"  Config: ar_frame_rho={config.ar_frame_rho}, "
          f"log_space={config.ar_frame_log_space}, "
          f"floor={config.ar_frame_floor_clamp}, "
          f"causal_transformer={config.ar_causal_transformer}, "
          f"cln={config.ar_causal_cln}")

    print("\n[2/6] Loading data...")
    batches = load_data(n_batches=args.n_batches, batch_size=args.batch_size,
                        history_len=config.history_len)
    print(f"  {batches.shape[0]} batches x {batches.shape[1]} windows")

    # Hypothesis A: AR(1) noise theory
    print("\n[3/6] Hypothesis A: AR(1) noise stationary variance...")
    result_A = test_hypothesis_A(config, max_steps=252)
    print(f"  rho = {result_A['rho']}")
    print(f"  Effective variance per step = {result_A['effective_var_per_step']}")
    print(f"  Cumulative std at h=30: {result_A['cumulative_std_h30']:.4f}")
    print(f"  Cumulative std at h=252: {result_A['cumulative_std_h252']:.4f}")
    print(f"  Expected growth 252/30: {result_A['expected_growth_252_vs_30']:.2f}x")
    print(f"  -> {result_A['conclusion'][:120]}...")

    # Hypothesis B: Transformer shrinks deltas
    print("\n[4/6] Hypothesis B: Transformer delta shrinkage...")
    result_B = test_hypothesis_B(model, config, batches, args.device,
                                  n_samples=min(args.n_samples, 8))
    print(f"  First 5 steps mean raw delta: {result_B['summary']['first5_mean_raw_delta']:.6f}")
    print(f"  Last 5 steps mean raw delta: {result_B['summary']['last5_mean_raw_delta']:.6f}")
    print(f"  Ratio (last5/first5): {result_B['summary']['raw_delta_ratio_last5_over_first5']:.4f}")
    print(f"  -> {result_B['conclusion']}")

    # Hypothesis C: Floor clamp
    print("\n[5/6] Hypothesis C: Log-space floor clamp absorption...")
    result_C = test_hypothesis_C(model, config, batches, args.device,
                                  n_samples=args.n_samples)
    print(f"  Max near-floor fraction: {result_C['summary']['max_fraction_near_floor']:.4%}")
    print(f"  h30 near-floor fraction: {result_C['summary']['h30_fraction_near_floor']:.4%}")
    print(f"  -> {result_C['conclusion']}")

    # Hypothesis D: Attention consensus
    print("\n[6/6] Hypothesis D: Member re-convergence via attention consensus...")
    result_D = test_hypothesis_D(model, config, batches, args.device,
                                  n_samples=args.n_samples)
    print(f"  Raw cosine: h1={result_D['summary']['raw_cosine_h1']:.4f}, "
          f"h15={result_D['summary']['raw_cosine_h15']:.4f}, "
          f"h30={result_D['summary']['raw_cosine_h30']:.4f}")
    print(f"  Spread: h1={result_D['summary']['spread_h1']:.6f}, "
          f"h10={result_D['summary']['spread_h10']:.6f}, "
          f"h30={result_D['summary']['spread_h30']:.6f}")
    print(f"  -> {result_D['conclusion']}")

    elapsed = time.time() - t0

    # ── Combine results ──
    all_results = {
        "model": args.model_path,
        "config_summary": {
            "ar_frame_rho": config.ar_frame_rho,
            "ar_frame_log_space": config.ar_frame_log_space,
            "ar_frame_floor_clamp": config.ar_frame_floor_clamp,
            "ar_causal_transformer": config.ar_causal_transformer,
            "ar_causal_cln": config.ar_causal_cln,
            "noise_dim": config.noise_dim,
            "ar_causal_d_model": config.ar_causal_d_model,
            "ar_freeze_gru_state": config.ar_freeze_gru_state,
        },
        "hypothesis_A_ar1_noise": result_A,
        "hypothesis_B_delta_shrinkage": result_B,
        "hypothesis_C_floor_clamp": result_C,
        "hypothesis_D_attention_consensus": result_D,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Summary of findings
    confirmed = []
    falsified = []
    for label, result in [("A", result_A), ("B", result_B), ("C", result_C), ("D", result_D)]:
        if label == "A":
            # A is a theoretical analysis, not confirmed/falsified in the same way
            confirmed.append(f"A: AR(1) noise growth decelerates (sub-linear) but does NOT plateau")
        elif result.get("hypothesis_confirmed", False):
            confirmed.append(f"{label}: {result['conclusion'].split('.')[0]}")
        else:
            falsified.append(f"{label}: {result['conclusion'].split('.')[0]}")

    all_results["summary"] = {
        "confirmed_hypotheses": confirmed,
        "falsified_hypotheses": falsified,
        "primary_root_cause": "See individual hypothesis results for details",
    }

    # Save
    output_path = RESULTS_DIR / "spread_diagnostic_144b.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save verification result
    verification_result = {
        "task": "long_horizon_spread_diagnostic",
        "model": args.model_path,
        "status": "complete",
        "hypotheses_tested": 4,
        "confirmed": [h.split(":")[0].strip() for h in confirmed],
        "falsified": [h.split(":")[0].strip() for h in falsified],
        "key_numbers": {
            "ar1_expected_growth_252_vs_30": result_A["expected_growth_252_vs_30"],
            "delta_ratio_last5_over_first5": result_B["summary"]["raw_delta_ratio_last5_over_first5"],
            "max_near_floor_fraction": result_C["summary"]["max_fraction_near_floor"],
            "raw_cosine_h30": result_D["summary"]["raw_cosine_h30"],
            "spread_h30": result_D["summary"]["spread_h30"],
            "spread_growth_h30_over_h10": result_D["summary"]["spread_growth_h30_over_h10"],
        },
        "results_path": str(output_path),
        "elapsed_seconds": round(elapsed, 1),
    }

    verify_path = VERIFICATION_DIR / "long_horizon_spread.json"
    verify_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verify_path, "w") as f:
        json.dump(verification_result, f, indent=2)
    print(f"Verification result saved to {verify_path}")

    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    for line in confirmed:
        print(f"  [CONFIRMED] {line}")
    for line in falsified:
        print(f"  [FALSIFIED] {line}")
    print("=" * 70)


if __name__ == "__main__":
    main()
