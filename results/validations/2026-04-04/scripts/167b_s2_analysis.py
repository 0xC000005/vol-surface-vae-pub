#!/usr/bin/env python
"""
167b S2 (CI Coverage) Deep-Dive Analysis

Tasks:
1. Per-cell CI coverage map across horizons, identify worst cells
2. Coverage vs horizon profile
3. Median bias analysis (spatial pattern)
4. Mean reversion check (delta_base slope vs prev_frame)
5. Gap-to-pass analysis: what would it take?

Compares 167b to 164a softplus baseline.
"""

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig

# Import model class
from experiments.backfill.block_ar.train_167b_clean_isolation import (
    ARFactorizedCleanModel,
    normalize_iv,
    denormalize_iv,
    compute_cond_ref,
)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    cfg = checkpoint["config"]
    encoder_config = EncoderConfig(**cfg["encoder"])
    decoder_config = cfg["decoder"]
    n_factors = cfg.get("n_factors", 5)
    model = ARFactorizedCleanModel(encoder_config, decoder_config, n_factors=n_factors)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_windows", type=int, default=200)
    args = parser.parse_args()

    device = args.device
    OUT_DIR = Path("results/validations/2026-04-04/analysis/167b_followup")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VER_DIR = Path("results/validations/2026-04-04/verification_results")
    VER_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Part 0: Load data and summaries
    # =========================================================================
    print("=" * 70)
    print("167b S2 (CI Coverage) Deep-Dive Analysis")
    print("=" * 70)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)
    H, T = 30, 30

    # Load summaries
    with open("results/block_ar/167b_best_30d/summary.json") as f:
        summary_167b = json.load(f)
    with open(
        "results/block_ar/164a_v3_percell_bptt_softplus_best_30d/summary.json"
    ) as f:
        summary_164a = json.load(f)

    # Cell labels for readability
    MONEYNESS = ["80%", "90%", "ATM", "110%", "120%"]
    TENORS = ["1M", "3M", "6M", "12M", "24M"]

    # =========================================================================
    # Part 1: Per-cell CI coverage map
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Per-Cell CI Coverage Map (90% CI)")
    print("=" * 70)

    HORIZONS = ["1", "7", "14", "30"]
    CELL_COV_LOW = 0.70  # gate lower bound
    CELL_COV_HIGH = 0.95  # gate upper bound

    cov_167b = summary_167b["coverage"]["per_cell_coverage"]
    cov_164a = summary_164a["coverage"]["per_cell_coverage"]

    # Track worst cells across both models
    worst_cells_167b = {}
    worst_cells_164a = {}

    for h in HORIZONS:
        grid_167b = np.array(cov_167b[h])
        grid_164a = np.array(cov_164a[h])
        diff = grid_167b - grid_164a

        worst_idx_167b = np.unravel_index(grid_167b.argmin(), (5, 5))
        worst_idx_164a = np.unravel_index(grid_164a.argmin(), (5, 5))
        worst_cells_167b[h] = worst_idx_167b
        worst_cells_164a[h] = worst_idx_164a

        print(f"\n--- Horizon h={h} ---")
        print(f"  167b worst: cell ({worst_idx_167b[0]},{worst_idx_167b[1]}) "
              f"= {grid_167b[worst_idx_167b]:.1%}"
              f" [{MONEYNESS[worst_idx_167b[1]]}, {TENORS[worst_idx_167b[0]]}]")
        print(f"  164a worst: cell ({worst_idx_164a[0]},{worst_idx_164a[1]}) "
              f"= {grid_164a[worst_idx_164a]:.1%}"
              f" [{MONEYNESS[worst_idx_164a[1]]}, {TENORS[worst_idx_164a[0]]}]")

        # Print full grid
        print(f"\n  167b coverage grid:")
        print(f"  {'':>6}", end="")
        for m in MONEYNESS:
            print(f"  {m:>6}", end="")
        print()
        for r in range(5):
            print(f"  {TENORS[r]:>6}", end="")
            for c in range(5):
                val = grid_167b[r, c]
                marker = "*" if val < CELL_COV_LOW else (" " if val <= CELL_COV_HIGH else "!")
                print(f"  {val:5.1%}{marker}", end="")
            print()

        # Show cells failing the gate
        failing = np.argwhere(grid_167b < CELL_COV_LOW)
        if len(failing) > 0:
            print(f"\n  FAILING cells (< {CELL_COV_LOW:.0%}):")
            for r, c in failing:
                print(f"    ({r},{c}) [{MONEYNESS[c]}, {TENORS[r]}] "
                      f"= {grid_167b[r, c]:.1%}  "
                      f"(164a: {grid_164a[r, c]:.1%}, diff: {diff[r, c]:+.1%})")

    # Check if same cells fail in both models
    print("\n\n--- Same-Cell Failure Analysis ---")
    for h in HORIZONS:
        same = worst_cells_167b[h] == worst_cells_164a[h]
        r167, c167 = worst_cells_167b[h]
        r164, c164 = worst_cells_164a[h]
        print(f"  h={h}: 167b worst=({r167},{c167}) "
              f"[{MONEYNESS[c167]},{TENORS[r167]}], "
              f"164a worst=({r164},{c164}) "
              f"[{MONEYNESS[c164]},{TENORS[r164]}]  "
              f"{'SAME' if np.all(same) else 'DIFFERENT'}")

    # =========================================================================
    # Part 2: Coverage vs Horizon
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("PART 2: Coverage vs Horizon")
    print("=" * 70)

    horizon_data = summary_167b["coverage"]["per_horizon"]
    worst_cell = summary_167b["coverage"]["worst_cell_per_horizon"]
    best_cell = summary_167b["coverage"]["best_cell_per_horizon"]

    print(f"\n  {'Horizon':>8}  {'Overall':>8}  {'Worst':>8}  {'Best':>8}  {'HPass':>6}  {'Gap to 70%':>10}")
    print(f"  {'':>8}  {'(90%CI)':>8}  {'Cell':>8}  {'Cell':>8}  {'':>6}  {'':>10}")
    print("  " + "-" * 60)

    for h in HORIZONS:
        overall = horizon_data[h]["0.9"]
        w = worst_cell[h]
        b = best_cell[h]
        hpass = summary_167b["coverage"]["horizon_pass"][h]
        gap = max(0, CELL_COV_LOW - w)
        print(f"  h={h:>4}  {overall:>7.1%}  {w:>7.1%}  {b:>7.1%}  {'PASS' if hpass else 'FAIL':>6}  {gap:>9.1%}")

    # Compare with 164a
    worst_164a = summary_164a["coverage"]["worst_cell_per_horizon"]
    print(f"\n  Comparison with 164a baseline:")
    print(f"  {'Horizon':>8}  {'167b worst':>10}  {'164a worst':>10}  {'Diff':>8}")
    print("  " + "-" * 40)
    for h in HORIZONS:
        w167 = worst_cell[h]
        w164 = worst_164a[h]
        print(f"  h={h:>4}  {w167:>9.1%}  {w164:>9.1%}  {w167 - w164:>+7.1%}")

    # =========================================================================
    # Part 3: Median Bias Analysis
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("PART 3: Median Bias Analysis")
    print("=" * 70)

    bias_167b = summary_167b["distributional"]["median_bias"]
    bias_164a = summary_164a["distributional"]["median_bias"]

    above_frac_167b = np.array(bias_167b["above_frac"])
    above_frac_164a = np.array(bias_164a["above_frac"])
    mean_bias_167b = np.array(bias_167b["mean_bias"])
    mean_bias_164a = np.array(bias_164a["mean_bias"])

    print(f"\n  Median Above Fraction (ideal: 0.50):")
    print(f"  {'':>6}", end="")
    for m in MONEYNESS:
        print(f"  {m:>8}", end="")
    print()
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            val = above_frac_167b[r, c]
            # Flag cells with strong bias (>0.15 from 0.5)
            marker = "!" if abs(val - 0.5) > 0.15 else " "
            print(f"  {val:7.3f}{marker}", end="")
        print()

    print(f"\n  Mean Bias (IV points, across all horizons):")
    print(f"  {'':>6}", end="")
    for m in MONEYNESS:
        print(f"  {m:>8}", end="")
    print()
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            val = mean_bias_167b[r, c]
            # Convert to IV points (* 100)
            ivpts = val * 100
            marker = "!" if abs(ivpts) > 3.0 else " "
            print(f"  {ivpts:+6.2f}{marker}", end="")
        print()

    # Identify worst bias cells
    print(f"\n  Worst bias cells (|mean_bias| > 2 IV pts):")
    for r in range(5):
        for c in range(5):
            ivpts = mean_bias_167b[r, c] * 100
            if abs(ivpts) > 2.0:
                print(f"    ({r},{c}) [{MONEYNESS[c]}, {TENORS[r]}]: "
                      f"bias = {ivpts:+.2f} IV pts, "
                      f"above_frac = {above_frac_167b[r, c]:.3f}")

    # Cross-reference worst bias with worst coverage
    print(f"\n  Cross-reference: worst bias vs worst coverage:")
    for h in HORIZONS:
        grid_cov = np.array(cov_167b[h])
        for r in range(5):
            for c in range(5):
                if grid_cov[r, c] < CELL_COV_LOW:
                    bias = mean_bias_167b[r, c] * 100
                    above = above_frac_167b[r, c]
                    print(f"    h={h} ({r},{c}) [{MONEYNESS[c]},{TENORS[r]}]: "
                          f"cov={grid_cov[r, c]:.1%}, bias={bias:+.2f}ivpts, "
                          f"above={above:.3f}")

    # Comparison with 164a bias
    print(f"\n  167b vs 164a mean bias comparison (IV pts):")
    print(f"  {'':>6}", end="")
    for m in MONEYNESS:
        print(f"  {m:>8}", end="")
    print()
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            b167 = mean_bias_167b[r, c] * 100
            b164 = mean_bias_164a[r, c] * 100
            diff = b167 - b164
            print(f"  {diff:+6.2f} ", end="")
        print()

    # =========================================================================
    # Part 4: Mean Reversion Check
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("PART 4: Mean Reversion Check (delta_base slope vs prev_frame)")
    print("=" * 70)

    model_path = "models/backfill/afcrps_167b/best_model.pt"
    model, ckpt = load_model(model_path, device)
    print(f"  Loaded model from epoch {ckpt['epoch']}")

    # Compute cond_ref from train set
    TEST_START = 4511
    surf_tensor = torch.tensor(surfaces, dtype=torch.float32, device=device)
    train_indices = np.arange(0, TEST_START - H - T - 441)
    cond_ref = compute_cond_ref(model, surf_tensor, train_indices, device)
    model.decoder.cond_ref.copy_(cond_ref)
    print(f"  cond_ref computed from {len(train_indices)} train windows")

    # Prepare test windows
    test_start = 4540  # test split
    N_total = surfaces.shape[0]
    max_test_windows = min(args.max_windows, N_total - test_start - H - T + 1)
    test_indices = np.arange(test_start, test_start + max_test_windows)

    # Collect delta_base, prev_frame pairs from multiple windows
    print(f"  Generating samples from {len(test_indices)} test windows...")

    all_delta_base = []
    all_prev_frame = []
    all_L_norms = []
    all_delta_total = []
    all_gt_delta = []

    with torch.no_grad():
        batch_size = 16
        for batch_start in range(0, len(test_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(test_indices))
            idx = test_indices[batch_start:batch_end]
            B = len(idx)

            # Prepare history and future
            hist_raw = torch.stack(
                [surf_tensor[i : i + H] for i in idx]
            )  # (B, H, 5, 5)
            fut_raw = torch.stack(
                [surf_tensor[i + H : i + H + T] for i in idx]
            )  # (B, T, 5, 5)

            hist_norm = normalize_iv(hist_raw)
            hist_flat = hist_norm.reshape(B, H, 25)

            # Encode
            gru_outputs, h_last = model.encoder.gru(hist_flat)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)

            # Step through frames, collect delta_base at each step
            last_frame = denormalize_iv(hist_raw[:, -1]).reshape(B, 25)
            prev = last_frame
            gru_state = h_last.contiguous()
            gru_outs = gru_outputs

            for t in range(T):
                z_t = torch.randn(B, model.decoder.noise_dim, device=device)
                if t > 0:
                    al = model.encoder.attn_proj(gru_outs).squeeze(-1)
                    aw = F.softmax(al, dim=1)
                    hp = (aw.unsqueeze(-1) * gru_outs).sum(dim=1)
                    cond_t = model.encoder.bottleneck(hp)
                else:
                    cond_t = cond

                delta_base, L = model.decoder(cond_t, prev, z_t)

                eps = torch.randn(B, model.n_factors, device=device)
                delta_total = delta_base + torch.einsum("bcr,br->bc", L, eps)

                # GT delta
                gt_frame = fut_raw[:, t].reshape(B, 25)
                gt_delta = gt_frame - prev

                all_delta_base.append(delta_base.cpu().numpy())
                all_prev_frame.append(prev.cpu().numpy())
                all_L_norms.append(
                    L.norm(dim=-1).cpu().numpy()
                )  # (B, 25)
                all_delta_total.append(delta_total.cpu().numpy())
                all_gt_delta.append(gt_delta.cpu().numpy())

                # Step forward using GT (teacher forcing for reversion analysis)
                prev = gt_frame

                # Update GRU
                fn = normalize_iv(gt_frame.reshape(B, 5, 5)).reshape(B, 1, 25)
                go, gru_state = model.encoder.gru(fn, gru_state)
                gru_outs = torch.cat([gru_outs, go], dim=1)

            if batch_start % 64 == 0:
                print(f"    Processed {batch_end}/{len(test_indices)} windows", flush=True)

    # Stack all
    delta_base_all = np.concatenate(all_delta_base, axis=0)  # (N*T, 25)
    prev_frame_all = np.concatenate(all_prev_frame, axis=0)
    L_norms_all = np.concatenate(all_L_norms, axis=0)
    delta_total_all = np.concatenate(all_delta_total, axis=0)
    gt_delta_all = np.concatenate(all_gt_delta, axis=0)

    print(f"\n  Total (step, window) pairs: {delta_base_all.shape[0]}")

    # Compute mean reversion slope per cell
    # For each cell: regress delta vs prev_frame (demeaned)
    print(f"\n  Mean Reversion Slopes (delta_base vs prev_frame):")
    print(f"  Negative = mean reverting. GT benchmark: ~-0.44")
    print(f"  Previous 167a finding: model slope was -0.22 (50% under-reversion)")
    print()
    print(f"  {'':>6}", end="")
    for m in MONEYNESS:
        print(f"  {m:>8}", end="")
    print()

    slopes_base = np.zeros((5, 5))
    slopes_total = np.zeros((5, 5))
    slopes_gt = np.zeros((5, 5))
    r_squared_base = np.zeros((5, 5))

    for cell in range(25):
        r, c = divmod(cell, 5)
        x = prev_frame_all[:, cell]
        y_base = delta_base_all[:, cell]
        y_total = delta_total_all[:, cell]
        y_gt = gt_delta_all[:, cell]

        # Demean
        x_dm = x - x.mean()
        y_base_dm = y_base - y_base.mean()
        y_total_dm = y_total - y_total.mean()
        y_gt_dm = y_gt - y_gt.mean()

        denom = (x_dm ** 2).sum()
        if denom > 0:
            slopes_base[r, c] = (x_dm * y_base_dm).sum() / denom
            slopes_total[r, c] = (x_dm * y_total_dm).sum() / denom
            slopes_gt[r, c] = (x_dm * y_gt_dm).sum() / denom
            y_pred = slopes_base[r, c] * x_dm
            ss_res = ((y_base_dm - y_pred) ** 2).sum()
            ss_tot = (y_base_dm ** 2).sum()
            r_squared_base[r, c] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Print base slopes
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            print(f"  {slopes_base[r, c]:+7.4f}", end="")
        print()

    print(f"\n  GT slopes:")
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            print(f"  {slopes_gt[r, c]:+7.4f}", end="")
        print()

    print(f"\n  Total (base+L@eps) slopes:")
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            print(f"  {slopes_total[r, c]:+7.4f}", end="")
        print()

    print(f"\n  Reversion ratio (model_slope / GT_slope, ideal=1.0):")
    reversion_ratio = np.zeros((5, 5))
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            if abs(slopes_gt[r, c]) > 1e-6:
                reversion_ratio[r, c] = slopes_base[r, c] / slopes_gt[r, c]
            print(f"  {reversion_ratio[r, c]:7.3f}", end="")
        print()

    print(f"\n  Aggregate stats:")
    print(f"    Mean GT slope:    {slopes_gt.mean():+.4f}")
    print(f"    Mean base slope:  {slopes_base.mean():+.4f}")
    print(f"    Mean total slope: {slopes_total.mean():+.4f}")
    print(f"    Mean reversion ratio (base/GT): {np.mean(reversion_ratio[np.abs(slopes_gt)>1e-6]):.3f}")
    print(f"    Mean R^2 (base): {r_squared_base.mean():.4f}")

    # L norm analysis
    print(f"\n  Load (L) magnitude per cell:")
    L_mean = L_norms_all.mean(axis=0).reshape(5, 5)
    print(f"  {'':>6}", end="")
    for m in MONEYNESS:
        print(f"  {m:>8}", end="")
    print()
    for r in range(5):
        print(f"  {TENORS[r]:>6}", end="")
        for c in range(5):
            print(f"  {L_mean[r, c]:8.5f}", end="")
        print()

    # =========================================================================
    # Part 5: What Would It Take to Pass S2?
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("PART 5: What Would It Take to Pass S2?")
    print("=" * 70)

    # S2 passes when:
    # 1. All horizons pass (per-horizon 90% CI >= 80%) - CURRENTLY PASSING
    # 2. worst_cell_pass: worst cell across all horizons >= 70%
    # 3. No cell exceeds 95%

    print(f"\n  S2 pass conditions:")
    print(f"    Per-horizon overall (90% CI >= 80%): "
          f"{'PASS' if all(summary_167b['coverage']['horizon_pass'].values()) else 'FAIL'}")
    print(f"    Worst cell across all h >= 70%: "
          f"{'PASS' if summary_167b['coverage']['worst_cell_pass'] else 'FAIL'}")
    print(f"    Overall CI (90% level): {summary_167b['coverage']['overall']['0.9']:.1%}")

    # Per-horizon: what's the gap?
    print(f"\n  Per-Horizon Gap Analysis:")
    total_failing_cells = 0
    gap_details = {}

    for h in HORIZONS:
        grid = np.array(cov_167b[h])
        failing_mask = grid < CELL_COV_LOW
        n_failing = failing_mask.sum()
        total_failing_cells += n_failing

        if n_failing > 0:
            gaps = CELL_COV_LOW - grid[failing_mask]
            print(f"\n    h={h}: {n_failing} cells below 70%")
            for r, c in np.argwhere(failing_mask):
                gap = CELL_COV_LOW - grid[r, c]
                gap_details[(int(h), r, c)] = {
                    "coverage": float(grid[r, c]),
                    "gap": float(gap),
                    "cell": f"[{MONEYNESS[c]}, {TENORS[r]}]",
                    "bias": float(mean_bias_167b[r, c] * 100),
                    "reversion_ratio": float(reversion_ratio[r, c]),
                }
                print(f"      ({r},{c}) [{MONEYNESS[c]},{TENORS[r]}]: "
                      f"{grid[r, c]:.1%} → gap = {gap:.1%} "
                      f"(bias={mean_bias_167b[r, c]*100:+.1f}ivpts, "
                      f"reversion={reversion_ratio[r, c]:.2f}x)")
        else:
            print(f"\n    h={h}: ALL cells pass 70% gate")

    print(f"\n  Total failing cells across all horizons: {total_failing_cells}")
    print(f"  Total cells checked: {4 * 25} = 100")

    # Identify the ONE cell that's the worst across all horizons
    worst_overall = None
    worst_overall_cov = 1.0
    for h in HORIZONS:
        grid = np.array(cov_167b[h])
        mn = grid.min()
        if mn < worst_overall_cov:
            worst_overall_cov = mn
            idx = np.unravel_index(grid.argmin(), (5, 5))
            worst_overall = (h, idx[0], idx[1])

    print(f"\n  THE critical blocker:")
    h, r, c = worst_overall
    print(f"    Cell ({r},{c}) [{MONEYNESS[c]}, {TENORS[r]}] at h={h}")
    print(f"    Coverage: {worst_overall_cov:.1%} (need 70%)")
    print(f"    Gap: {CELL_COV_LOW - worst_overall_cov:.1%}")
    print(f"    Mean bias: {mean_bias_167b[r, c]*100:+.2f} IV pts")
    print(f"    Above fraction: {above_frac_167b[r, c]:.3f} (ideal=0.500)")
    print(f"    Reversion ratio: {reversion_ratio[r, c]:.3f}")

    # What if we could reduce bias by 50%?
    print(f"\n  Scenario analysis:")
    # The coverage failure is driven by bias (median not centered)
    # Coverage = P(GT in CI). If median is off, CI shifts, coverage drops.
    # Rough estimate: each 1 IV pt of bias costs ~5-10% coverage
    bias_gap = abs(mean_bias_167b[r, c]) * 100
    cov_gap = (CELL_COV_LOW - worst_overall_cov) * 100
    print(f"    Bias at worst cell: {bias_gap:.2f} IV pts")
    print(f"    Coverage gap: {cov_gap:.1f}%")
    if bias_gap > 0:
        print(f"    Implied coverage/bias sensitivity: ~{cov_gap/bias_gap:.1f}% per IV pt")

    # Which cells consistently fail across horizons?
    print(f"\n  Cells failing across multiple horizons:")
    fail_counts = np.zeros((5, 5), dtype=int)
    for h in HORIZONS:
        grid = np.array(cov_167b[h])
        fail_counts += (grid < CELL_COV_LOW).astype(int)
    for r in range(5):
        for c in range(5):
            if fail_counts[r, c] > 0:
                print(f"    ({r},{c}) [{MONEYNESS[c]},{TENORS[r]}]: "
                      f"fails {fail_counts[r, c]}/4 horizons")

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "part1_coverage_maps": {
            h: {
                "167b_grid": np.array(cov_167b[h]).tolist(),
                "164a_grid": np.array(cov_164a[h]).tolist(),
                "diff_grid": (np.array(cov_167b[h]) - np.array(cov_164a[h])).tolist(),
                "167b_worst_cell": list(worst_cells_167b[h]),
                "164a_worst_cell": list(worst_cells_164a[h]),
                "same_worst": bool(np.all(np.array(worst_cells_167b[h]) == np.array(worst_cells_164a[h]))),
                "n_failing_167b": int((np.array(cov_167b[h]) < CELL_COV_LOW).sum()),
                "n_failing_164a": int((np.array(cov_164a[h]) < CELL_COV_LOW).sum()),
            }
            for h in HORIZONS
        },
        "part2_coverage_vs_horizon": {
            h: {
                "overall_90ci": float(horizon_data[h]["0.9"]),
                "worst_cell": float(worst_cell[h]),
                "best_cell": float(best_cell[h]),
                "horizon_pass": bool(summary_167b["coverage"]["horizon_pass"][h]),
                "gap_to_70pct": float(max(0, CELL_COV_LOW - worst_cell[h])),
                "164a_worst_cell": float(worst_164a[h]),
                "diff_vs_164a": float(worst_cell[h] - worst_164a[h]),
            }
            for h in HORIZONS
        },
        "part3_bias": {
            "above_frac_167b": above_frac_167b.tolist(),
            "above_frac_164a": above_frac_164a.tolist(),
            "mean_bias_167b_ivpts": (mean_bias_167b * 100).tolist(),
            "mean_bias_164a_ivpts": (mean_bias_164a * 100).tolist(),
            "worst_bias_cells": [
                {"cell": (r, c), "label": f"[{MONEYNESS[c]},{TENORS[r]}]",
                 "bias_ivpts": float(mean_bias_167b[r, c] * 100),
                 "above_frac": float(above_frac_167b[r, c])}
                for r in range(5) for c in range(5)
                if abs(mean_bias_167b[r, c] * 100) > 2.0
            ],
        },
        "part4_reversion": {
            "slopes_base": slopes_base.tolist(),
            "slopes_total": slopes_total.tolist(),
            "slopes_gt": slopes_gt.tolist(),
            "reversion_ratio": reversion_ratio.tolist(),
            "r_squared_base": r_squared_base.tolist(),
            "L_mean_norms": L_mean.tolist(),
            "aggregate": {
                "mean_gt_slope": float(slopes_gt.mean()),
                "mean_base_slope": float(slopes_base.mean()),
                "mean_total_slope": float(slopes_total.mean()),
                "mean_reversion_ratio": float(
                    np.mean(reversion_ratio[np.abs(slopes_gt) > 1e-6])
                ),
                "mean_r_squared": float(r_squared_base.mean()),
            },
        },
        "part5_gap_to_pass": {
            "worst_cell_overall": {
                "horizon": worst_overall[0],
                "cell": [int(worst_overall[1]), int(worst_overall[2])],
                "label": f"[{MONEYNESS[worst_overall[2]]},{TENORS[worst_overall[1]]}]",
                "coverage": float(worst_overall_cov),
                "gap": float(CELL_COV_LOW - worst_overall_cov),
                "bias_ivpts": float(mean_bias_167b[worst_overall[1], worst_overall[2]] * 100),
                "above_frac": float(above_frac_167b[worst_overall[1], worst_overall[2]]),
                "reversion_ratio": float(reversion_ratio[worst_overall[1], worst_overall[2]]),
            },
            "total_failing_cells": int(total_failing_cells),
            "fail_counts_per_cell": fail_counts.tolist(),
            "gap_details": {
                f"h{k[0]}_r{k[1]}_c{k[2]}": v for k, v in gap_details.items()
            },
        },
    }

    # JSON encoder for numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    # Save analysis results
    with open(OUT_DIR / "s2_coverage_analysis.json", "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved analysis to {OUT_DIR / 's2_coverage_analysis.json'}")

    # Save verification results
    verification = {
        "experiment": "167b S2 deep-dive",
        "model": "models/backfill/afcrps_167b/best_model.pt",
        "model_epoch": int(ckpt["epoch"]),
        "n_test_windows": len(test_indices),
        "n_samples_per_window": "N/A (single-sample for reversion, summary.json for coverage)",
        "s2_status": "FAIL",
        "s2_reason": "worst_cell_pass=False",
        "overall_90ci": float(summary_167b["coverage"]["overall"]["0.9"]),
        "worst_cell_coverage": float(worst_overall_cov),
        "worst_cell_location": f"h={worst_overall[0]}, cell=({worst_overall[1]},{worst_overall[2]})",
        "gap_to_pass": float(CELL_COV_LOW - worst_overall_cov),
        "primary_mechanism": "systematic negative bias in short-tenor OTM put cells",
        "mean_reversion_ratio": float(
            np.mean(reversion_ratio[np.abs(slopes_gt) > 1e-6])
        ),
        "key_findings": [
            f"Worst cell ({worst_overall[1]},{worst_overall[2]}) has coverage {worst_overall_cov:.1%}, needs 70%",
            f"Gap = {(CELL_COV_LOW - worst_overall_cov)*100:.1f}%",
            f"Mean model reversion ratio: {np.mean(reversion_ratio[np.abs(slopes_gt)>1e-6]):.3f}x GT",
            f"Same worst cells as 164a: {any(np.all(np.array(worst_cells_167b[h]) == np.array(worst_cells_164a[h])) for h in HORIZONS)}",
        ],
    }
    with open(VER_DIR / "167b_s2_analysis.json", "w") as f:
        json.dump(verification, f, indent=2)
    print(f"  Saved verification to {VER_DIR / '167b_s2_analysis.json'}")


if __name__ == "__main__":
    main()
