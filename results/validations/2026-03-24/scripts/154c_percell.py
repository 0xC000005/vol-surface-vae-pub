#!/usr/bin/env python
"""
154c Per-Cell CI Breakdown + Save Test Eval Results

Comprehensive per-cell analysis of base (153a) + residual (154c) flow matching model.
Computes:
1. Per-cell CI coverage at h=1, h=15, h=30 (5x5 grid for each)
2. Per-cell spread at h=1, h=15, h=30
3. Per-cell bias (gen mean - GT mean) at h=30
4. OTM moneyness gradient check
5. Per-horizon CI coverage (all 30 horizons)
6. Worst 5 / best 5 cells
7. CI failure from BIAS or SPREAD? (correlation analysis)
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv
)

ROOT = Path("/home/max/Documents/vol-surface-vae-pub")
OUT_DIR = ROOT / "results/validations/2026-03-24"
ANALYSIS_DIR = OUT_DIR / "analysis/154c_percell"
RESULT_FILE = OUT_DIR / "verification_results/154c_percell.json"


def make_serializable(obj):
    """Convert numpy/torch types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Load models ──────────────────────────────────────────────────
    print("Loading models...")
    bc = torch.load(str(ROOT / "models/backfill/flow_153a/final_model.pt"),
                    weights_only=False, map_location=device)
    base = ConditionalFactoredVelocityTransformer(
        n_frames=30, n_cells=25, d_model=128, n_heads=4, n_layers=4, cond_dim=128)
    base.load_state_dict(bc["model_state_dict"])
    base.to(device).eval()

    enc, cd = load_encoder(
        str(ROOT / "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"), device)
    bm = torch.from_numpy(bc["train_mean"]).float().to(device)
    bs = torch.from_numpy(bc["train_std"]).float().to(device)

    rc = torch.load(str(ROOT / "models/backfill/flow_154c/final_model.pt"),
                    weights_only=False, map_location=device)
    rcfg = rc["config"]
    res = ConditionalFactoredVelocityTransformer(
        n_frames=rcfg["n_frames"], n_cells=rcfg["n_cells"],
        d_model=rcfg["d_model"], n_heads=rcfg["n_heads"],
        n_layers=rcfg["n_layers"], cond_dim=rcfg["cond_dim"])
    res.load_state_dict(rc["model_state_dict"])
    res.to(device).eval()
    rm = torch.from_numpy(rc["res_mean"]).float().to(device)
    rs_stat = torch.from_numpy(rc["res_std"]).float().to(device)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading data...")
    data = np.load(str(ROOT / "data/vol_surface_with_ret.npz"))
    surfaces = data["surface"]
    rets = data["ret"]
    H, T, DIM = 30, 30, 750
    ts = 4540
    ns = 50
    nst = 8
    dt = 1.0 / nst

    tw = [(surfaces[i:i+H], surfaces[i+H:i+H+T])
          for i in range(ts, min(ts + 160, len(surfaces) - H - T + 1))]
    N = len(tw)
    print(f"Eval 154c per-cell: {N} windows, {ns} samples each")

    # ── Generate samples ─────────────────────────────────────────────
    all_samples = []  # (N, ns, T, 5, 5)
    all_gt = []       # (N, T, 5, 5)

    with torch.no_grad():
        for w in range(N):
            hist_raw = torch.from_numpy(tw[w][0][None].astype(np.float32)).to(device)
            cond = enc(normalize_iv(hist_raw))

            # Base prediction: average of 3 ODE runs
            base_preds = []
            for _ in range(3):
                x = torch.randn(1, DIM, device=device)
                for s in range(nst):
                    t_val = torch.full((1,), s * dt, device=device)
                    x = x + base(x, t_val, cond=cond) * dt
                base_preds.append((x * bs + bm).clamp(0, 1))
            bp = torch.stack(base_preds).mean(0)  # (1, 750)

            # Residual samples
            samps = []
            for _ in range(ns):
                x = torch.randn(1, DIM, device=device)
                for s in range(nst):
                    t_val = torch.full((1,), s * dt, device=device)
                    x = x + res(x, t_val, cond=cond) * dt
                combined = (bp + x * rs_stat + rm).clamp(0, 1)
                samps.append(combined.cpu().numpy().reshape(T, 5, 5))

            all_samples.append(np.array(samps))  # (ns, T, 5, 5)
            all_gt.append(tw[w][1])               # (T, 5, 5)

            if (w + 1) % 40 == 0:
                elapsed = time.time() - t0
                print(f"  {w+1}/{N} windows done ({elapsed:.0f}s)")

    cs = np.array(all_samples)  # (N, ns, T, 5, 5)
    gt = np.array(all_gt)       # (N, T, 5, 5)

    elapsed_gen = time.time() - t0
    print(f"Generation done in {elapsed_gen:.0f}s")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    moneyness_labels = ["0.90 (ITM)", "0.95", "1.00 (ATM)", "1.05", "1.10 (OTM)"]
    tenor_labels = ["1M", "3M", "6M", "9M", "12M"]

    # ── 1. Per-cell CI coverage at h=1, h=15, h=30 ──────────────────
    print("\n=== Per-Cell CI Coverage ===")
    ci_grids = {}
    horizons_check = {"h1": 0, "h15": 14, "h30": 29}

    for hname, hidx in horizons_check.items():
        grid = np.zeros((5, 5))
        for r in range(5):
            for c in range(5):
                lo = np.percentile(cs[:, :, hidx, r, c], 5, axis=1)  # (N,)
                hi = np.percentile(cs[:, :, hidx, r, c], 95, axis=1)  # (N,)
                cov = ((gt[:, hidx, r, c] >= lo) & (gt[:, hidx, r, c] <= hi)).mean()
                grid[r, c] = cov
        ci_grids[hname] = grid
        print(f"\n{hname} CI coverage (90% target, 85% pass):")
        print("          " + "  ".join(f"{t:>6s}" for t in tenor_labels))
        for r in range(5):
            row_str = f"  {moneyness_labels[r]:>11s}  "
            for c in range(5):
                v = grid[r, c]
                marker = "*" if v < 0.85 else " "
                row_str += f"{v:.3f}{marker} "
            print(row_str)

    # ── 2. Per-cell spread at h=1, h=15, h=30 ───────────────────────
    print("\n=== Per-Cell Spread (Std of Samples) ===")
    spread_grids = {}
    for hname, hidx in horizons_check.items():
        grid = np.zeros((5, 5))
        for r in range(5):
            for c in range(5):
                grid[r, c] = cs[:, :, hidx, r, c].std(axis=1).mean()
        spread_grids[hname] = grid
        print(f"\n{hname} spread:")
        print("          " + "  ".join(f"{t:>8s}" for t in tenor_labels))
        for r in range(5):
            row_str = f"  {moneyness_labels[r]:>11s}  "
            for c in range(5):
                row_str += f"{grid[r, c]:.5f}  "
            print(row_str)

    # ── 3. Per-cell bias at h=30 ─────────────────────────────────────
    print("\n=== Per-Cell Bias at h=30 (gen_mean - gt_mean) ===")
    bias_grid = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            gen_mean = cs[:, :, 29, r, c].mean()
            gt_mean = gt[:, 29, r, c].mean()
            bias_grid[r, c] = gen_mean - gt_mean
    print("          " + "  ".join(f"{t:>8s}" for t in tenor_labels))
    for r in range(5):
        row_str = f"  {moneyness_labels[r]:>11s}  "
        for c in range(5):
            v = bias_grid[r, c]
            row_str += f"{v:+.5f}  "
        print(row_str)

    # ── 4. OTM moneyness gradient ────────────────────────────────────
    print("\n=== OTM Moneyness Gradient (mean IV by moneyness row) ===")
    gt_mean_by_row = gt.mean(axis=(0, 1))  # (5, 5) → mean over windows and horizons
    gen_mean_by_row = cs.mean(axis=(0, 1, 2))  # (5, 5)

    print("  Row   GT_mean    Gen_mean   Diff")
    for r in range(5):
        gt_r = gt_mean_by_row[r].mean()
        gen_r = gen_mean_by_row[r].mean()
        print(f"  {moneyness_labels[r]:>11s}  {gt_r:.5f}   {gen_r:.5f}   {gen_r - gt_r:+.5f}")

    # Check if gradient monotonicity is preserved (U-shape: ITM high, ATM low, OTM high)
    gt_smile = gt_mean_by_row.mean(axis=1)  # avg across tenors
    gen_smile = gen_mean_by_row.mean(axis=1)
    gt_gradient_preserved = (
        gt_smile[0] > gt_smile[2] and gt_smile[4] > gt_smile[2]  # U-shape in GT
    )
    gen_gradient_preserved = (
        gen_smile[0] > gen_smile[2] and gen_smile[4] > gen_smile[2]  # U-shape in gen
    )
    print(f"\n  GT smile shape (ITM>ATM and OTM>ATM): {gt_gradient_preserved}")
    print(f"  Gen smile shape (ITM>ATM and OTM>ATM): {gen_gradient_preserved}")
    print(f"  GT row means: {[f'{v:.5f}' for v in gt_smile.tolist()]}")
    print(f"  Gen row means: {[f'{v:.5f}' for v in gen_smile.tolist()]}")

    # ── 5. Per-horizon CI coverage (all 30 horizons) ─────────────────
    print("\n=== Per-Horizon CI Coverage ===")
    horizon_ci = np.zeros(T)
    for h_idx in range(T):
        lo = np.percentile(cs[:, :, h_idx], 5, axis=1)  # (N, 5, 5)
        hi = np.percentile(cs[:, :, h_idx], 95, axis=1)
        cov = ((gt[:, h_idx] >= lo) & (gt[:, h_idx] <= hi)).mean()
        horizon_ci[h_idx] = cov

    horizon_pass = sum(1 for h_idx in range(T) if horizon_ci[h_idx] >= 0.85)
    print(f"  Horizons passing (>=0.85): {horizon_pass}/30")
    for h_idx in range(T):
        marker = "PASS" if horizon_ci[h_idx] >= 0.85 else "FAIL"
        print(f"  h{h_idx+1:2d}: {horizon_ci[h_idx]:.4f}  [{marker}]")

    # ── 6. Aggregate per-cell CI (across all horizons) ───────────────
    print("\n=== Aggregate Per-Cell CI (all horizons) ===")
    cell_ci_all = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            lo = np.percentile(cs[:, :, :, r, c], 5, axis=1)  # (N, T)
            hi = np.percentile(cs[:, :, :, r, c], 95, axis=1)  # (N, T)
            cov = ((gt[:, :, r, c] >= lo) & (gt[:, :, r, c] <= hi)).mean()
            cell_ci_all[r, c] = cov

    # Sort cells by CI
    cell_list = []
    for r in range(5):
        for c in range(5):
            cell_list.append({
                "cell": f"({r},{c})",
                "moneyness": moneyness_labels[r],
                "tenor": tenor_labels[c],
                "ci_coverage": float(cell_ci_all[r, c]),
                "h30_ci": float(ci_grids["h30"][r, c]),
                "h30_bias": float(bias_grid[r, c]),
                "h30_spread": float(spread_grids["h30"][r, c]),
            })

    cell_list_sorted = sorted(cell_list, key=lambda x: x["ci_coverage"])
    worst5 = cell_list_sorted[:5]
    best5 = cell_list_sorted[-5:][::-1]

    print("\nWorst 5 cells:")
    for item in worst5:
        print(f"  {item['cell']} {item['moneyness']:>11s} x {item['tenor']:>3s}  "
              f"CI={item['ci_coverage']:.3f}  h30_ci={item['h30_ci']:.3f}  "
              f"bias={item['h30_bias']:+.5f}  spread={item['h30_spread']:.5f}")

    print("\nBest 5 cells:")
    for item in best5:
        print(f"  {item['cell']} {item['moneyness']:>11s} x {item['tenor']:>3s}  "
              f"CI={item['ci_coverage']:.3f}  h30_ci={item['h30_ci']:.3f}  "
              f"bias={item['h30_bias']:+.5f}  spread={item['h30_spread']:.5f}")

    # ── 7. CI failure: BIAS vs SPREAD correlation ────────────────────
    print("\n=== CI Failure Diagnosis: BIAS vs SPREAD ===")

    # For each cell, compute absolute bias and spread at h=30
    abs_biases = []
    spreads = []
    coverages = []
    for r in range(5):
        for c in range(5):
            abs_biases.append(abs(bias_grid[r, c]))
            spreads.append(spread_grids["h30"][r, c])
            coverages.append(ci_grids["h30"][r, c])

    abs_biases = np.array(abs_biases)
    spreads = np.array(spreads)
    coverages = np.array(coverages)

    # Correlation between CI failure and bias/spread
    ci_deficit = 0.90 - coverages  # positive = undercoverage
    r_bias_ci, p_bias_ci = pearsonr(abs_biases, ci_deficit)
    r_spread_ci, p_spread_ci = pearsonr(spreads, ci_deficit)

    # Also check bias/spread ratio
    bias_spread_ratio = abs_biases / (spreads + 1e-8)
    r_ratio_ci, p_ratio_ci = pearsonr(bias_spread_ratio, ci_deficit)

    print(f"  Corr(|bias|, CI_deficit): r={r_bias_ci:.3f}, p={p_bias_ci:.4f}")
    print(f"  Corr(spread, CI_deficit): r={r_spread_ci:.3f}, p={p_spread_ci:.4f}")
    print(f"  Corr(|bias|/spread, CI_deficit): r={r_ratio_ci:.3f}, p={p_ratio_ci:.4f}")

    if abs(r_bias_ci) > abs(r_spread_ci):
        diagnosis = "BIAS-dominated"
        print(f"  Diagnosis: {diagnosis} — bias correlates more with CI failure")
    else:
        diagnosis = "SPREAD-dominated"
        print(f"  Diagnosis: {diagnosis} — spread correlates more with CI failure")

    # ── Additional: Per-cell bias/spread breakdown for failing cells ──
    print("\n=== Failing Cells (h30 CI < 0.85) Detail ===")
    failing_cells = [item for item in cell_list if item["h30_ci"] < 0.85]
    print(f"  {len(failing_cells)} cells fail at h=30")
    for item in sorted(failing_cells, key=lambda x: x["h30_ci"]):
        bsr = abs(item["h30_bias"]) / (item["h30_spread"] + 1e-8)
        print(f"  {item['cell']} {item['moneyness']:>11s} x {item['tenor']:>3s}  "
              f"CI={item['h30_ci']:.3f}  bias={item['h30_bias']:+.5f}  "
              f"spread={item['h30_spread']:.5f}  |bias|/spread={bsr:.2f}")

    # ── Summary statistics ───────────────────────────────────────────
    worst_cell_ci = cell_ci_all.min()
    mean_cell_ci = cell_ci_all.mean()
    cells_passing_85 = (cell_ci_all >= 0.85).sum()
    cells_passing_90 = (cell_ci_all >= 0.90).sum()

    print(f"\n=== Summary ===")
    print(f"  Worst cell CI (all horizons): {worst_cell_ci:.4f}")
    print(f"  Mean cell CI (all horizons): {mean_cell_ci:.4f}")
    print(f"  Cells >= 0.85: {cells_passing_85}/25")
    print(f"  Cells >= 0.90: {cells_passing_90}/25")
    print(f"  Horizon CI pass (>=0.85): {horizon_pass}/30")
    print(f"  Mean |bias| at h30: {abs_biases.mean():.5f}")
    print(f"  Mean spread at h30: {spreads.mean():.5f}")
    print(f"  Mean |bias|/spread at h30: {bias_spread_ratio.mean():.2f}")

    elapsed_total = time.time() - t0
    print(f"\nTotal time: {elapsed_total:.0f}s")

    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════

    # -- ci_grid.json --
    ci_grid_result = {
        "description": "Per-cell CI coverage at selected horizons",
        "moneyness_labels": moneyness_labels,
        "tenor_labels": tenor_labels,
        "ci_grids": {k: v.tolist() for k, v in ci_grids.items()},
        "ci_all_horizons": cell_ci_all.tolist(),
        "per_horizon_ci": horizon_ci.tolist(),
        "horizon_pass_count": int(horizon_pass),
    }
    with open(str(ANALYSIS_DIR / "ci_grid.json"), "w") as f:
        json.dump(make_serializable(ci_grid_result), f, indent=2)
    print(f"Saved: {ANALYSIS_DIR / 'ci_grid.json'}")

    # -- bias_spread.json --
    bias_spread_result = {
        "description": "Per-cell bias and spread analysis",
        "spread_grids": {k: v.tolist() for k, v in spread_grids.items()},
        "bias_grid_h30": bias_grid.tolist(),
        "abs_bias_mean": float(abs_biases.mean()),
        "spread_mean": float(spreads.mean()),
        "bias_spread_ratio_mean": float(bias_spread_ratio.mean()),
        "ci_failure_diagnosis": diagnosis,
        "corr_abs_bias_vs_ci_deficit": {"r": float(r_bias_ci), "p": float(p_bias_ci)},
        "corr_spread_vs_ci_deficit": {"r": float(r_spread_ci), "p": float(p_spread_ci)},
        "corr_bias_spread_ratio_vs_ci_deficit": {"r": float(r_ratio_ci), "p": float(p_ratio_ci)},
        "gt_smile_means": gt_smile.tolist(),
        "gen_smile_means": gen_smile.tolist(),
        "gt_smile_preserved": bool(gt_gradient_preserved),
        "gen_smile_preserved": bool(gen_gradient_preserved),
        "worst_5_cells": worst5,
        "best_5_cells": best5,
        "failing_cells_h30": [
            {**item, "abs_bias_spread_ratio": abs(item["h30_bias"]) / (item["h30_spread"] + 1e-8)}
            for item in sorted(failing_cells, key=lambda x: x["h30_ci"])
        ],
    }
    with open(str(ANALYSIS_DIR / "bias_spread.json"), "w") as f:
        json.dump(make_serializable(bias_spread_result), f, indent=2)
    print(f"Saved: {ANALYSIS_DIR / 'bias_spread.json'}")

    # -- Main verification result --
    verification_result = {
        "experiment": "154c",
        "description": "Per-cell CI breakdown for base (153a) + residual (154c) flow matching",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_windows": N,
            "n_samples": ns,
            "ode_steps": nst,
            "base_ode_runs": 3,
            "test_start": ts,
        },
        "summary": {
            "worst_cell_ci_all_horizons": float(worst_cell_ci),
            "mean_cell_ci_all_horizons": float(mean_cell_ci),
            "cells_passing_85pct": int(cells_passing_85),
            "cells_passing_90pct": int(cells_passing_90),
            "horizon_pass_count": int(horizon_pass),
            "mean_abs_bias_h30": float(abs_biases.mean()),
            "mean_spread_h30": float(spreads.mean()),
            "mean_bias_spread_ratio_h30": float(bias_spread_ratio.mean()),
            "ci_failure_diagnosis": diagnosis,
            "gt_smile_preserved": bool(gt_gradient_preserved),
            "gen_smile_preserved": bool(gen_gradient_preserved),
        },
        "per_cell_ci_grid_h1": ci_grids["h1"].tolist(),
        "per_cell_ci_grid_h15": ci_grids["h15"].tolist(),
        "per_cell_ci_grid_h30": ci_grids["h30"].tolist(),
        "per_cell_ci_all_horizons": cell_ci_all.tolist(),
        "per_horizon_ci": horizon_ci.tolist(),
        "per_cell_spread_h30": spread_grids["h30"].tolist(),
        "per_cell_bias_h30": bias_grid.tolist(),
        "worst_5_cells": worst5,
        "best_5_cells": best5,
        "correlation_analysis": {
            "abs_bias_vs_ci_deficit": {"r": float(r_bias_ci), "p": float(p_bias_ci)},
            "spread_vs_ci_deficit": {"r": float(r_spread_ci), "p": float(p_spread_ci)},
            "bias_spread_ratio_vs_ci_deficit": {"r": float(r_ratio_ci), "p": float(p_ratio_ci)},
        },
        "runtime_seconds": round(elapsed_total, 1),
    }

    with open(str(RESULT_FILE), "w") as f:
        json.dump(make_serializable(verification_result), f, indent=2)
    print(f"Saved: {RESULT_FILE}")

    return verification_result


if __name__ == "__main__":
    result = main()
