#!/usr/bin/env python
"""
Definitive test: Is global t_min residual STRUCTURED or just random noise?

Compares two sampling strategies with matched total variance:
  A) Global t_min (mgr=5): stop denoising at t_min(h), residual comes from reverse process
  B) Post-hoc Gaussian: fully denoise (mgr=0), then add i.i.d. N(0, sigma) noise to match A's variance

If the residual from A is structured (learned), then A should:
  1. Have LOWER arb violation rates than B (residual respects surface constraints)
  2. Have NON-UNIFORM per-cell variance (some cells more uncertain than others)
  3. Have POSITIVE inter-cell correlation (surface moves as a whole, not cell-by-cell)
  4. Have CONDITIONING-DEPENDENT residual patterns (different histories → different residual)

If the residual is just random noise, A and B should be statistically indistinguishable.
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def test_calendar_arbitrage(samples):
    """Calendar arb violation rate. samples: (N, T, 5, 5)."""
    tenors = np.array([1, 2, 4, 8, 12])
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]
        total_var = surf ** 2 * tenors[:, None]
        for i in range(4):
            violation = (total_var[:, i, :] > total_var[:, i + 1, :] * 1.001)
            violations.append(violation.mean())
    return float(np.mean(violations))


def test_butterfly_arbitrage(samples):
    """Butterfly arb violation rate. samples: (N, T, 5, 5)."""
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]
        d2_dk2 = surf[:, :, :-2] - 2 * surf[:, :, 1:-1] + surf[:, :, 2:]
        violation = (d2_dk2 < -0.005).mean()
        violations.append(float(violation))
    return float(np.mean(violations))


def main():
    parser = argparse.ArgumentParser(description="Structured vs Random Residual Test")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/block_ar_dual_path/best_coverage_model.pt")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--mgr", type=int, default=5, help="max_global_residual for method A")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_ema", action="store_true")
    args = parser.parse_args()

    config = get_default_config()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)
    model = ConditionalBlockARDDPM(model_config)
    if "ema_params" in checkpoint and not args.no_ema:
        state_dict = model.state_dict()
        for name in state_dict:
            if name in checkpoint["ema_params"]:
                state_dict[name] = checkpoint["ema_params"][name]
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    data = np.load(config.data_path)
    surfaces = data["surface"]
    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len, start_idx=config.test_start,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print("=" * 70)
    print(f"STRUCTURED vs RANDOM RESIDUAL TEST (mgr={args.mgr})")
    print("=" * 70)

    # =========================================================================
    # Generate samples: Method A (global t_min) and Method B (mgr=0 baseline)
    # =========================================================================
    all_A = []  # global t_min samples
    all_B0 = []  # fully denoised (mgr=0) samples
    all_gt = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Generating", total=args.max_batches)
        ):
            if batch_idx >= args.max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            # Method A: global t_min
            samples_A = model.sample_batched(
                history, n_samples=args.n_samples, max_global_residual=args.mgr
            )
            # Method B baseline: fully denoise
            samples_B0 = model.sample_batched(
                history, n_samples=args.n_samples, max_global_residual=0
            )

            all_A.append(samples_A.cpu().numpy())
            all_B0.append(samples_B0.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())

    A = np.concatenate(all_A, axis=0)    # (N, S, T, 5, 5)
    B0 = np.concatenate(all_B0, axis=0)  # (N, S, T, 5, 5)
    gt = np.concatenate(all_gt, axis=0)  # (N, T, 5, 5)
    N, S, T, H, W = A.shape
    print(f"\nSamples: N={N}, S={S}, T={T}, grid={H}x{W}")

    # =========================================================================
    # Match variance: compute per-horizon variance of A, inject equivalent noise into B0
    # =========================================================================
    print("\n--- Variance Matching ---")
    var_A_per_h = np.var(A, axis=1).mean(axis=(0, 2, 3))  # (T,) avg var per horizon
    var_B0_per_h = np.var(B0, axis=1).mean(axis=(0, 2, 3))

    # Additional noise variance needed: var_A - var_B0
    # B_posthoc will have same total variance as A
    rng = np.random.default_rng(42)
    B_posthoc = B0.copy()
    for h in range(T):
        needed_var = max(0, var_A_per_h[h] - var_B0_per_h[h])
        sigma = np.sqrt(needed_var)
        noise = rng.normal(0, sigma, size=(N, S, H, W)).astype(np.float32)
        B_posthoc[:, :, h, :, :] += noise
    B_posthoc = np.clip(B_posthoc, 0, 1)

    var_Bp_per_h = np.var(B_posthoc, axis=1).mean(axis=(0, 2, 3))

    print(f"  {'Horizon':<10} {'Var(A) t_min':<15} {'Var(B0) clean':<15} {'Var(Bp) posthoc':<15}")
    for h_idx in [0, 9, 19, 29]:
        print(f"  h={h_idx+1:<7} {var_A_per_h[h_idx]:<15.6f} {var_B0_per_h[h_idx]:<15.6f} {var_Bp_per_h[h_idx]:<15.6f}")

    # =========================================================================
    # TEST 1: Arb violation rates (A vs B_posthoc, matched variance)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: ARBITRAGE VIOLATIONS (same variance, structured vs random)")
    print("=" * 70)

    # Flatten samples for arb tests: (N*S, T, 5, 5)
    A_flat = A.reshape(N * S, T, H, W)
    Bp_flat = B_posthoc.reshape(N * S, T, H, W)
    B0_flat = B0.reshape(N * S, T, H, W)

    cal_A = test_calendar_arbitrage(A_flat)
    cal_Bp = test_calendar_arbitrage(Bp_flat)
    cal_B0 = test_calendar_arbitrage(B0_flat)

    but_A = test_butterfly_arbitrage(A_flat)
    but_Bp = test_butterfly_arbitrage(Bp_flat)
    but_B0 = test_butterfly_arbitrage(B0_flat)

    print(f"\n  {'Method':<25} {'Calendar Arb':<15} {'Butterfly Arb':<15}")
    print(f"  {'-'*55}")
    print(f"  {'A: global t_min':<25} {cal_A:<15.1%} {but_A:<15.1%}")
    print(f"  {'B: post-hoc Gaussian':<25} {cal_Bp:<15.1%} {but_Bp:<15.1%}")
    print(f"  {'B0: clean (mgr=0)':<25} {cal_B0:<15.1%} {but_B0:<15.1%}")

    cal_delta = cal_Bp - cal_A
    but_delta = but_Bp - but_A
    print(f"\n  Post-hoc EXCESS over t_min:  cal +{cal_delta:.1%},  butterfly +{but_delta:.1%}")
    if cal_delta > 0 and but_delta > 0:
        print("  → STRUCTURED: t_min has FEWER arb violations than random noise of same variance")
    elif cal_delta > 0 or but_delta > 0:
        print("  → PARTIALLY STRUCTURED: t_min better on one arb metric")
    else:
        print("  → NOT STRUCTURED: post-hoc noise is equally good or better")

    # =========================================================================
    # TEST 2: Per-cell variance uniformity
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: PER-CELL VARIANCE UNIFORMITY (structured = non-uniform)")
    print("=" * 70)

    # Compute per-cell variance at h=29 (max residual)
    # A: (N, S, 5, 5) at h=29
    var_A_cells = np.var(A[:, :, -1, :, :], axis=1)  # (N, 5, 5) per-example
    var_Bp_cells = np.var(B_posthoc[:, :, -1, :, :], axis=1)

    # Average across examples
    avg_var_A = var_A_cells.mean(axis=0)   # (5, 5)
    avg_var_Bp = var_Bp_cells.mean(axis=0)  # (5, 5)

    # Coefficient of variation across cells
    cv_A = np.std(avg_var_A) / np.mean(avg_var_A)
    cv_Bp = np.std(avg_var_Bp) / np.mean(avg_var_Bp)

    print(f"\n  Per-cell variance at h=30 (5x5 grid):")
    print(f"\n  Method A (global t_min) — CV across cells: {cv_A:.3f}")
    print(f"  {avg_var_A.round(6)}")
    print(f"\n  Method B (post-hoc) — CV across cells: {cv_Bp:.3f}")
    print(f"  {avg_var_Bp.round(6)}")

    ratio = cv_A / cv_Bp if cv_Bp > 0 else float('inf')
    print(f"\n  CV ratio (A/B): {ratio:.2f}x")
    if cv_A > cv_Bp * 1.2:
        print("  → STRUCTURED: t_min has significantly non-uniform variance across cells")
    else:
        print("  → INCONCLUSIVE: variance uniformity similar")

    # =========================================================================
    # TEST 3: Inter-cell correlation of residual
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: INTER-CELL CORRELATION (structured = correlated residual)")
    print("=" * 70)

    # Residual = deviation from median across samples
    # At h=29: (N, S, 5, 5)
    A_h29 = A[:, :, -1, :, :]            # (N, S, 5, 5)
    Bp_h29 = B_posthoc[:, :, -1, :, :]   # (N, S, 5, 5)

    A_median = np.median(A_h29, axis=1, keepdims=True)   # (N, 1, 5, 5)
    Bp_median = np.median(Bp_h29, axis=1, keepdims=True)

    resid_A = (A_h29 - A_median).reshape(N * S, H * W)     # (N*S, 25)
    resid_Bp = (Bp_h29 - Bp_median).reshape(N * S, H * W)  # (N*S, 25)

    # Average pairwise correlation across all cell pairs
    corr_A = np.corrcoef(resid_A.T)   # (25, 25)
    corr_Bp = np.corrcoef(resid_Bp.T)  # (25, 25)

    # Off-diagonal mean
    mask = ~np.eye(25, dtype=bool)
    avg_corr_A = corr_A[mask].mean()
    avg_corr_Bp = corr_Bp[mask].mean()

    print(f"\n  Average off-diagonal inter-cell correlation at h=30:")
    print(f"  Method A (global t_min): {avg_corr_A:.4f}")
    print(f"  Method B (post-hoc):     {avg_corr_Bp:.4f}")
    print(f"  Ratio (A/B):             {avg_corr_A / avg_corr_Bp if avg_corr_Bp != 0 else float('inf'):.2f}x")

    if avg_corr_A > avg_corr_Bp * 1.5:
        print("  → STRUCTURED: t_min residual is spatially correlated (surface moves together)")
    elif avg_corr_A > avg_corr_Bp * 1.1:
        print("  → PARTIALLY STRUCTURED: some spatial correlation in residual")
    else:
        print("  → NOT STRUCTURED: residual correlation similar to random noise")

    # =========================================================================
    # TEST 4: Conditioning dependence of residual
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: CONDITIONING DEPENDENCE (structured = history-dependent residual)")
    print("=" * 70)

    # For each example, compute the variance pattern (5x5) at h=29
    # Then measure: does the variance PATTERN change across conditioning histories?
    # If structured: different histories → different variance patterns
    # If random: all histories → same variance pattern (just uniform noise)

    # Compute per-example variance patterns at h=29: (N, 5, 5)
    var_patterns_A = np.var(A[:, :, -1, :, :], axis=1)   # (N, 5, 5)
    var_patterns_Bp = np.var(B_posthoc[:, :, -1, :, :], axis=1)

    # Measure how much the pattern varies across examples
    # Flatten to (N, 25), compute variance of each cell's variance across examples
    vp_A = var_patterns_A.reshape(N, 25)
    vp_Bp = var_patterns_Bp.reshape(N, 25)

    # For each cell, how much does its variance change across conditioning histories?
    cross_example_var_A = np.var(vp_A, axis=0).mean()   # scalar
    cross_example_var_Bp = np.var(vp_Bp, axis=0).mean()

    print(f"\n  Cross-example variance of per-cell variance patterns at h=30:")
    print(f"  Method A (global t_min): {cross_example_var_A:.2e}")
    print(f"  Method B (post-hoc):     {cross_example_var_Bp:.2e}")
    ratio_cond = cross_example_var_A / cross_example_var_Bp if cross_example_var_Bp > 0 else float('inf')
    print(f"  Ratio (A/B):             {ratio_cond:.2f}x")

    if ratio_cond > 1.5:
        print("  → STRUCTURED: t_min variance pattern depends on conditioning history")
    else:
        print("  → INCONCLUSIVE: similar conditioning dependence")

    # Also: correlation between per-cell variance and GT surface values
    # If structured, cells with higher IV might have different uncertainty
    gt_h29 = gt[:, -1, :, :].reshape(N, 25)  # (N, 25)
    corr_var_gt_A = np.array([np.corrcoef(vp_A[:, c], gt_h29[:, c])[0, 1] for c in range(25)])
    corr_var_gt_Bp = np.array([np.corrcoef(vp_Bp[:, c], gt_h29[:, c])[0, 1] for c in range(25)])

    print(f"\n  Correlation between per-cell variance and GT IV level:")
    print(f"  Method A (global t_min): mean |corr| = {np.abs(corr_var_gt_A).mean():.4f}")
    print(f"  Method B (post-hoc):     mean |corr| = {np.abs(corr_var_gt_Bp).mean():.4f}")

    if np.abs(corr_var_gt_A).mean() > np.abs(corr_var_gt_Bp).mean() * 1.3:
        print("  → STRUCTURED: t_min uncertainty is linked to IV level (heteroscedastic)")
    else:
        print("  → INCONCLUSIVE: similar relationship to IV level")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    tests = {
        "Arb violations (cal)": cal_A < cal_Bp,
        "Arb violations (but)": but_A < but_Bp,
        "Per-cell variance non-uniformity": cv_A > cv_Bp * 1.2,
        "Inter-cell correlation": avg_corr_A > avg_corr_Bp * 1.5,
        "Conditioning dependence": ratio_cond > 1.5,
    }

    structured_count = sum(tests.values())
    print(f"\n  Evidence for STRUCTURED residual: {structured_count}/5 tests")
    for name, passed in tests.items():
        status = "STRUCTURED" if passed else "inconclusive"
        print(f"    {name:<40} {status}")

    if structured_count >= 4:
        print(f"\n  CONCLUSION: Global t_min residual is DEFINITIVELY structured,")
        print(f"  not random noise. The model's learned uncertainty is being expressed.")
    elif structured_count >= 2:
        print(f"\n  CONCLUSION: Global t_min residual shows PARTIAL structure.")
        print(f"  Some evidence of learned uncertainty, but not fully distinguishable from noise.")
    else:
        print(f"\n  CONCLUSION: Global t_min residual is NOT clearly distinguishable from random noise.")


if __name__ == "__main__":
    main()
