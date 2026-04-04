#!/usr/bin/env python
"""
167d Mean Reversion Analysis
============================
Measures mean reversion behavior of 167d (CLN active) and compares to:
- Baseline (mean slope -0.22, 50% of GT)
- 167b (CLN frozen, mean slope -0.053, 12% of GT)
- GT slope (-0.44)

Computes:
1. Aggregate and per-cell delta_base slopes (regression of delta vs prev_frame)
2. Total delta (delta_base + L@eps) slopes averaged over 50 eps samples
3. L contribution analysis (does L@eps help or hurt reversion?)
4. Spread vs centering decomposition

Usage:
    PYTHONPATH=. python results/validations/2026-04-04/scripts/167d_mean_reversion.py \
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_167d_e2e_factorized import (
    ARFactorizedCleanModel,
    normalize_iv,
    denormalize_iv,
    reflecting_boundary,
    compute_cond_ref,
)


def make_serializable(obj):
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


MONEYNESS = ["80%", "90%", "100%", "110%", "120%"]
TENORS = ["1M", "3M", "6M", "9M", "12M"]


def cell_label(r, c):
    return f"[{MONEYNESS[c]},{TENORS[r]}]"


def compute_reversion_slopes(delta, prev_frame):
    """
    Regress delta vs prev_frame per cell.
    delta: (N, 25), prev_frame: (N, 25)
    Returns: slopes (25,), r_squared (25,)
    """
    slopes = np.zeros(25)
    r_squared = np.zeros(25)
    for c in range(25):
        x = prev_frame[:, c]
        y = delta[:, c]
        # Center for numerical stability
        x_c = x - x.mean()
        y_c = y - y.mean()
        ss_xx = (x_c ** 2).sum()
        if ss_xx < 1e-12:
            slopes[c] = 0.0
            r_squared[c] = 0.0
            continue
        slope = (x_c * y_c).sum() / ss_xx
        y_pred = slope * x_c
        ss_res = ((y_c - y_pred) ** 2).sum()
        ss_tot = (y_c ** 2).sum()
        slopes[c] = slope
        r_squared[c] = 1.0 - ss_res / (ss_tot + 1e-12)
    return slopes, r_squared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/afcrps_167d/best_model.pt")
    parser.add_argument("--data_path", type=str,
                        default="data/vol_surface_with_ret.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_windows", type=int, default=50)
    parser.add_argument("--n_eps_samples", type=int, default=50,
                        help="Number of eps samples for total delta averaging")
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data = np.load(args.data_path)
    surfaces = data["surface"]  # (N, 5, 5)
    N = len(surfaces)
    surf_tensor = torch.tensor(surfaces, dtype=torch.float32, device=device)

    # Load model
    print(f"Loading model from {args.model_path}")
    ckpt = torch.load(args.model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = ARFactorizedCleanModel(
        EncoderConfig(**cfg["encoder"]),
        cfg["decoder"],
        n_factors=cfg.get("n_factors", 5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    n_factors = cfg.get("n_factors", 5)
    noise_dim = cfg["decoder"]["noise_dim"]
    H = 30
    C = 25

    # Build test windows
    test_indices = []
    for i in range(args.test_start, N - 2 * H):
        test_indices.append(i)
        if len(test_indices) >= args.n_windows:
            break

    print(f"Using {len(test_indices)} test windows starting at index {args.test_start}")

    # Compute cond_ref from training data for FiLM
    train_indices = np.arange(0, min(args.test_start, N - 2 * H))
    cond_ref = compute_cond_ref(model, surf_tensor, train_indices, device, H=H, C=C)
    model.decoder.cond_ref.copy_(cond_ref)
    print(f"cond_ref computed, norm={cond_ref.norm().item():.4f}")

    # ===== PART 1 & 2: Per-cell mean reversion slopes =====
    print("\n=== Part 1 & 2: Mean Reversion Slopes ===")

    all_delta_base = []
    all_delta_total = []  # averaged over eps samples
    all_prev_frame = []
    all_gt_delta = []

    # For Part 4: ensemble statistics
    all_ensemble_mean_delta = []
    all_ensemble_std_delta = []

    n_ensemble = 16  # match training K

    t0 = time.time()
    with torch.no_grad():
        for wi, idx in enumerate(test_indices):
            if wi % 10 == 0:
                print(f"  Window {wi}/{len(test_indices)}...")

            # Build history and GT
            hist = surf_tensor[idx:idx + H].unsqueeze(0)  # (1, 30, 5, 5)
            gt_future = surf_tensor[idx + H:idx + 2 * H]  # (30, 5, 5)

            hist_norm = normalize_iv(hist)
            hist_flat = hist_norm.reshape(1, H, C)

            # Encoder forward
            gru_outputs, h_last = model.encoder.gru(hist_flat)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)  # (1, 128)

            # Previous frame (denormalized, as model sees it)
            prev = denormalize_iv(hist_norm[:, -1]).reshape(1, C)  # (1, 25)

            # Single noise sample for delta_base
            z = torch.randn(1, noise_dim, device=device)
            delta_base, L = model.decoder(cond, prev, z)
            # delta_base: (1, 25), L: (1, 25, n_factors)

            all_delta_base.append(delta_base.cpu().numpy().squeeze())
            all_prev_frame.append(prev.cpu().numpy().squeeze())

            # GT delta for first step
            gt_next = gt_future[0].reshape(25)  # (25,)
            gt_delta = (gt_next - prev.squeeze()).cpu().numpy()
            all_gt_delta.append(gt_delta)

            # Total delta averaged over n_eps_samples
            delta_totals = []
            for _ in range(args.n_eps_samples):
                eps = torch.randn(1, n_factors, device=device)
                delta_total = delta_base + torch.einsum("bcr,br->bc", L, eps)
                delta_totals.append(delta_total.cpu().numpy().squeeze())
            delta_total_mean = np.mean(delta_totals, axis=0)
            all_delta_total.append(delta_total_mean)

            # Ensemble statistics (K members)
            ensemble_deltas = []
            for _ in range(n_ensemble):
                eps_k = torch.randn(1, n_factors, device=device)
                delta_k = delta_base + torch.einsum("bcr,br->bc", L, eps_k)
                ensemble_deltas.append(delta_k.cpu().numpy().squeeze())
            ensemble_deltas = np.array(ensemble_deltas)  # (K, 25)
            all_ensemble_mean_delta.append(ensemble_deltas.mean(axis=0))
            all_ensemble_std_delta.append(ensemble_deltas.std(axis=0))

    elapsed = time.time() - t0
    print(f"  Forward passes done in {elapsed:.1f}s")

    # Convert to arrays
    delta_base_arr = np.array(all_delta_base)      # (N_win, 25)
    delta_total_arr = np.array(all_delta_total)     # (N_win, 25)
    prev_frame_arr = np.array(all_prev_frame)       # (N_win, 25)
    gt_delta_arr = np.array(all_gt_delta)           # (N_win, 25)
    ens_mean_arr = np.array(all_ensemble_mean_delta)  # (N_win, 25)
    ens_std_arr = np.array(all_ensemble_std_delta)    # (N_win, 25)

    # Per-cell slopes: delta_base vs prev_frame
    slopes_base, r2_base = compute_reversion_slopes(delta_base_arr, prev_frame_arr)

    # Per-cell slopes: total delta vs prev_frame
    slopes_total, r2_total = compute_reversion_slopes(delta_total_arr, prev_frame_arr)

    # Per-cell slopes: GT delta vs prev_frame
    slopes_gt, r2_gt = compute_reversion_slopes(gt_delta_arr, prev_frame_arr)

    # Reshape for 5x5 grid
    slopes_base_grid = slopes_base.reshape(5, 5)
    slopes_total_grid = slopes_total.reshape(5, 5)
    slopes_gt_grid = slopes_gt.reshape(5, 5)
    r2_base_grid = r2_base.reshape(5, 5)
    r2_total_grid = r2_total.reshape(5, 5)

    # Reversion ratios
    reversion_ratio_base = np.abs(slopes_base) / (np.abs(slopes_gt) + 1e-12)
    reversion_ratio_total = np.abs(slopes_total) / (np.abs(slopes_gt) + 1e-12)

    mean_base_slope = slopes_base.mean()
    mean_total_slope = slopes_total.mean()
    mean_gt_slope = slopes_gt.mean()
    mean_reversion_base = reversion_ratio_base.mean()
    mean_reversion_total = reversion_ratio_total.mean()

    print(f"\n  Aggregate mean reversion slopes:")
    print(f"    GT:          {mean_gt_slope:.4f}")
    print(f"    delta_base:  {mean_base_slope:.4f} ({mean_reversion_base*100:.1f}% of GT)")
    print(f"    delta_total: {mean_total_slope:.4f} ({mean_reversion_total*100:.1f}% of GT)")
    print(f"    Baseline reference: -0.22 (50% of GT)")
    print(f"    167b reference:     -0.053 (12% of GT)")

    # ===== PART 3: L contribution to reversion =====
    print("\n=== Part 3: L Contribution to Reversion ===")

    L_delta_slopes = slopes_total - slopes_base  # negative = helps reversion, positive = opposes
    L_helps_reversion = (L_delta_slopes < 0).sum()
    L_opposes_reversion = (L_delta_slopes > 0).sum()
    L_delta_mean = L_delta_slopes.mean()

    print(f"  L@eps effect on slopes: {L_delta_mean:.6f}")
    print(f"  Cells where L helps reversion: {L_helps_reversion}/25")
    print(f"  Cells where L opposes reversion: {L_opposes_reversion}/25")

    # Per-cell L norm
    L_norms = []
    with torch.no_grad():
        for wi, idx in enumerate(test_indices):
            hist = surf_tensor[idx:idx + H].unsqueeze(0)
            hist_norm = normalize_iv(hist)
            hist_flat = hist_norm.reshape(1, H, C)
            gru_outputs, _ = model.encoder.gru(hist_flat)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)
            prev = denormalize_iv(hist_norm[:, -1]).reshape(1, C)
            z = torch.randn(1, noise_dim, device=device)
            _, L = model.decoder(cond, prev, z)
            # L: (1, 25, n_factors) - Frobenius norm per cell
            L_norm = L.squeeze(0).norm(dim=-1).cpu().numpy()  # (25,)
            L_norms.append(L_norm)
    L_norms_arr = np.array(L_norms).mean(axis=0)  # (25,) mean over windows
    L_norms_grid = L_norms_arr.reshape(5, 5)

    print(f"  Mean L norm per cell: {L_norms_arr.mean():.6f}")
    print(f"  Max L norm cell: {cell_label(*divmod(L_norms_arr.argmax(), 5))} = {L_norms_arr.max():.6f}")
    print(f"  Min L norm cell: {cell_label(*divmod(L_norms_arr.argmin(), 5))} = {L_norms_arr.min():.6f}")

    # ===== PART 4: Spread vs centering decomposition =====
    print("\n=== Part 4: Spread vs Centering Decomposition ===")

    # Centering = ensemble mean of delta, Spread = ensemble std of delta
    centering_mean = np.abs(ens_mean_arr).mean(axis=0)  # (25,)
    spread_mean = ens_std_arr.mean(axis=0)               # (25,)
    ratio = centering_mean / (spread_mean + 1e-12)

    mean_centering = centering_mean.mean()
    mean_spread = spread_mean.mean()
    mean_ratio = ratio.mean()

    print(f"  Mean |centering| (|E[delta]|): {mean_centering:.6f}")
    print(f"  Mean spread (std[delta]):      {mean_spread:.6f}")
    print(f"  Mean centering/spread ratio:   {mean_ratio:.4f}")

    # ===== Load 167b reference data =====
    print("\n=== Comparison with 167b ===")
    ref_167b_path = Path("results/validations/2026-04-04/analysis/167b_followup/s2_coverage_analysis.json")
    ref_167b = None
    if ref_167b_path.exists():
        with open(ref_167b_path) as f:
            ref_167b = json.load(f)
        ref_slopes_base_167b = np.array(ref_167b["part4_reversion"]["slopes_base"])
        ref_slopes_total_167b = np.array(ref_167b["part4_reversion"]["slopes_total"])
        ref_slopes_gt_167b = np.array(ref_167b["part4_reversion"]["slopes_gt"])
        ref_agg_167b = ref_167b["part4_reversion"]["aggregate"]

        print(f"  167b aggregate: base={ref_agg_167b['mean_base_slope']:.4f}, "
              f"total={ref_agg_167b['mean_total_slope']:.4f}")

        # Per-cell comparison: which cells improved most?
        improvement_base = np.abs(slopes_base_grid) - np.abs(ref_slopes_base_167b)
        improvement_total = np.abs(slopes_total_grid) - np.abs(ref_slopes_total_167b)

        print(f"\n  Per-cell improvement (167d - 167b, positive = 167d reverts more):")
        print(f"  delta_base improvement: min={improvement_base.min():.4f}, "
              f"max={improvement_base.max():.4f}, mean={improvement_base.mean():.4f}")
        print(f"  delta_total improvement: min={improvement_total.min():.4f}, "
              f"max={improvement_total.max():.4f}, mean={improvement_total.mean():.4f}")

        # Identify worst 167b cells and check if they improved
        ref_slopes_flat = np.abs(ref_slopes_base_167b).flatten()
        worst_167b_cells = ref_slopes_flat.argsort()[:5]  # 5 weakest reversion cells
        print(f"\n  Worst 5 cells in 167b (weakest reversion) — did 167d fix them?")
        for ci in worst_167b_cells:
            r, c = divmod(ci, 5)
            lbl = cell_label(r, c)
            s167b = ref_slopes_base_167b[r][c]
            s167d = slopes_base_grid[r, c]
            sgt = slopes_gt_grid[r, c]
            print(f"    {lbl}: 167b={s167b:.4f}, 167d={s167d:.4f}, GT={sgt:.4f}, "
                  f"improvement={abs(s167d)-abs(s167b):.4f}")
    else:
        print("  167b reference not found, skipping comparison.")

    # ===== Build results =====
    results = {
        "metadata": {
            "model_path": args.model_path,
            "n_windows": len(test_indices),
            "test_start": args.test_start,
            "n_eps_samples": args.n_eps_samples,
            "n_ensemble": n_ensemble,
            "seed": args.seed,
            "elapsed_s": round(elapsed, 1),
        },
        "aggregate": {
            "mean_gt_slope": float(mean_gt_slope),
            "mean_base_slope": float(mean_base_slope),
            "mean_total_slope": float(mean_total_slope),
            "mean_reversion_ratio_base": float(mean_reversion_base),
            "mean_reversion_ratio_total": float(mean_reversion_total),
            "baseline_reference_slope": -0.22,
            "baseline_reference_ratio": 0.50,
            "ref_167b_base_slope": float(ref_agg_167b["mean_base_slope"]) if ref_167b else None,
            "ref_167b_ratio": float(ref_agg_167b["mean_reversion_ratio"]) if ref_167b else None,
        },
        "per_cell": {
            "slopes_base": slopes_base_grid.tolist(),
            "slopes_total": slopes_total_grid.tolist(),
            "slopes_gt": slopes_gt_grid.tolist(),
            "r_squared_base": r2_base_grid.tolist(),
            "r_squared_total": r2_total_grid.tolist(),
            "reversion_ratio_base": reversion_ratio_base.reshape(5, 5).tolist(),
            "reversion_ratio_total": reversion_ratio_total.reshape(5, 5).tolist(),
        },
        "L_contribution": {
            "L_delta_slopes": L_delta_slopes.reshape(5, 5).tolist(),
            "L_helps_reversion_count": int(L_helps_reversion),
            "L_opposes_reversion_count": int(L_opposes_reversion),
            "L_delta_mean": float(L_delta_mean),
            "L_mean_norms": L_norms_grid.tolist(),
            "interpretation": (
                "negative L_delta_slopes means L@eps pushes slope more negative (helps reversion); "
                "positive means L@eps weakens reversion. "
                "If |L_delta_mean| << |base slope|, L is orthogonal to reversion."
            ),
        },
        "spread_centering": {
            "centering_abs_mean": centering_mean.reshape(5, 5).tolist(),
            "spread_mean": spread_mean.reshape(5, 5).tolist(),
            "centering_spread_ratio": ratio.reshape(5, 5).tolist(),
            "aggregate_centering": float(mean_centering),
            "aggregate_spread": float(mean_spread),
            "aggregate_ratio": float(mean_ratio),
        },
    }

    if ref_167b:
        results["comparison_167b"] = {
            "improvement_base_grid": improvement_base.tolist(),
            "improvement_total_grid": improvement_total.tolist(),
            "improvement_base_mean": float(improvement_base.mean()),
            "improvement_total_mean": float(improvement_total.mean()),
            "worst_167b_cells_improved": [],
        }
        ref_slopes_flat = np.abs(ref_slopes_base_167b).flatten()
        worst_167b_cells = ref_slopes_flat.argsort()[:5]
        for ci in worst_167b_cells:
            r, c = divmod(ci, 5)
            results["comparison_167b"]["worst_167b_cells_improved"].append({
                "cell": cell_label(r, c),
                "slope_167b": float(ref_slopes_base_167b[r][c]),
                "slope_167d": float(slopes_base_grid[r, c]),
                "slope_gt": float(slopes_gt_grid[r, c]),
                "improvement": float(abs(slopes_base_grid[r, c]) - abs(ref_slopes_base_167b[r][c])),
            })

    # Save results
    out_dir = Path("results/validations/2026-04-04/analysis/167d_followup")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mean_reversion.json"
    with open(out_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save verification results
    verify_dir = Path("results/validations/2026-04-04/verification_results")
    verify_dir.mkdir(parents=True, exist_ok=True)

    # Build verification summary
    cln_restored = abs(mean_base_slope) > abs(-0.053)  # better than 167b
    cln_near_baseline = abs(mean_base_slope) > 0.15  # within range of baseline -0.22
    L_orthogonal = abs(L_delta_mean) < 0.01 * abs(mean_base_slope)

    verification = {
        "test": "167d_mean_reversion",
        "hypothesis": "CLN being active restores centering behavior (mean reversion) toward baseline",
        "key_numbers": {
            "167d_base_slope": float(mean_base_slope),
            "167d_total_slope": float(mean_total_slope),
            "167d_reversion_pct_gt": float(mean_reversion_base * 100),
            "baseline_slope": -0.22,
            "baseline_pct_gt": 50.0,
            "167b_base_slope": float(ref_agg_167b["mean_base_slope"]) if ref_167b else None,
            "167b_pct_gt": float(ref_agg_167b["mean_reversion_ratio"] * 100) if ref_167b else None,
            "gt_slope": float(mean_gt_slope),
        },
        "conclusions": {
            "cln_restored_vs_167b": cln_restored,
            "cln_near_baseline": cln_near_baseline,
            "L_orthogonal_to_reversion": L_orthogonal,
            "L_helps_cells": int(L_helps_reversion),
            "L_opposes_cells": int(L_opposes_reversion),
        },
        "interpretation": "",
    }

    if cln_restored and cln_near_baseline:
        verification["interpretation"] = (
            f"CLN active FULLY RESTORED mean reversion: "
            f"{mean_base_slope:.4f} ({mean_reversion_base*100:.1f}% of GT), "
            f"recovering from 167b's {ref_agg_167b['mean_reversion_ratio']*100:.1f}% "
            f"back to near baseline 50%."
        )
    elif cln_restored:
        verification["interpretation"] = (
            f"CLN active PARTIALLY restored mean reversion: "
            f"{mean_base_slope:.4f} ({mean_reversion_base*100:.1f}% of GT), "
            f"better than 167b ({ref_agg_167b['mean_reversion_ratio']*100:.1f}%) "
            f"but below baseline (50%)."
        )
    else:
        verification["interpretation"] = (
            f"CLN active did NOT restore mean reversion: "
            f"{mean_base_slope:.4f} ({mean_reversion_base*100:.1f}% of GT), "
            f"similar to or worse than 167b ({ref_agg_167b['mean_reversion_ratio']*100:.1f}%)."
        )

    verify_path = verify_dir / "167d_mean_reversion.json"
    with open(verify_path, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"Verification saved to {verify_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean Reversion Comparison")
    print("=" * 70)
    print(f"{'Model':<15} {'Mean Slope':<15} {'% of GT':<12} {'Status':<20}")
    print("-" * 70)
    print(f"{'GT':<15} {mean_gt_slope:<15.4f} {'100%':<12} {'Reference':<20}")
    print(f"{'Baseline':<15} {-0.22:<15.4f} {'50%':<12} {'Old reference':<20}")
    if ref_167b:
        print(f"{'167b (frozen)':<15} "
              f"{ref_agg_167b['mean_base_slope']:<15.4f} "
              f"{ref_agg_167b['mean_reversion_ratio']*100:<12.1f}% "
              f"{'CLN frozen':<20}")
    print(f"{'167d (active)':<15} {mean_base_slope:<15.4f} "
          f"{mean_reversion_base*100:<12.1f}% "
          f"{'CLN active':<20}")
    print(f"{'167d total':<15} {mean_total_slope:<15.4f} "
          f"{mean_reversion_total*100:<12.1f}% "
          f"{'base + L@eps':<20}")
    print("=" * 70)

    # Per-cell grid
    print("\nPer-cell reversion ratio (% of GT), delta_base:")
    print(f"{'':>10}", end="")
    for c in range(5):
        print(f"  {MONEYNESS[c]:>6}", end="")
    print()
    for r in range(5):
        print(f"{TENORS[r]:>10}", end="")
        for c in range(5):
            val = reversion_ratio_base.reshape(5, 5)[r, c] * 100
            print(f"  {val:6.1f}%", end="")
        print()

    print(f"\nL contribution: helps reversion in {L_helps_reversion}/25 cells, "
          f"opposes in {L_opposes_reversion}/25")
    print(f"Centering/spread ratio: {mean_ratio:.4f}")


if __name__ == "__main__":
    main()
