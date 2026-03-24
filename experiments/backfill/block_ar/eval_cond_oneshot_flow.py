#!/usr/bin/env python
"""
Full evaluation for conditional one-shot flow matching models (153a etc).

Computes all 9-suite-equivalent metrics on test data:
1. Surface Validity (explosion, calendar arb, butterfly arb)
2. CI Coverage (per-horizon, per-cell 90% CI)
3. Conditionality (turb/calm width ratio, per-cell MAE reduction)
4. Time Series (ACF correlation, kurtosis ratio)
5. Block-AR Specific (boundary smoothness, growing uncertainty)
6. Cointegration (cell-cell pass rate)
7. Regime Coverage (per-regime per-cell CI)
8. Distributional (KS daily changes, KS levels, median bias)
9. Cross-Cell Correlation (correlation ratio, effective rank ratio)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/eval_cond_oneshot_flow.py \
        --model_path models/backfill/flow_153a/final_model.pt \
        --n_samples 50 --max_batches 20 \
        --output_dir results/block_ar/153a_30d --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import ks_2samp, kurtosis, spearmanr
from statsmodels.tsa.stattools import coint

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv,
)


def make_serial(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serial(v) for v in obj]
    return obj


def generate_samples(velocity_net, encoder, history_batch, n_samples, n_steps,
                     train_mean, train_std, device):
    """Generate samples for a batch of history windows.

    Args:
        history_batch: (B, 30, 5, 5) in [0, 1]

    Returns:
        (B, n_samples, 30, 5, 5) in [0, 1]
    """
    B = history_batch.shape[0]
    DIM = 750
    dt = 1.0 / n_steps

    # Normalize history to [-1, 1] for encoder
    hist_norm = normalize_iv(history_batch)
    cond = encoder(hist_norm)  # (B, cond_dim)

    # Expand for n_samples
    cond_exp = cond.repeat_interleave(n_samples, dim=0)  # (B*S, cond_dim)

    # ODE from N(0,I)
    x = torch.randn(B * n_samples, DIM, device=device)
    for step in range(n_steps):
        t = torch.full((B * n_samples,), step * dt, device=device)
        x = x + velocity_net(x, t, cond=cond_exp) * dt

    # Denormalize
    mean_t = torch.from_numpy(train_mean).float().to(device)
    std_t = torch.from_numpy(train_std).float().to(device)
    samples = x * std_t + mean_t
    samples = torch.clamp(samples, 0, 1)

    return samples.reshape(B, n_samples, 30, 5, 5)


def eff_rank(corr):
    ev = np.linalg.eigvalsh(corr)[::-1]
    ev = np.maximum(ev, 0)
    p = ev / (ev.sum() + 1e-10); p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    velocity_net = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    velocity_net.load_state_dict(ckpt["model_state_dict"])
    velocity_net.to(device).eval()

    encoder, cond_dim = load_encoder(args.encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False

    train_mean = ckpt["train_mean"]
    train_std = ckpt["train_std"]
    n_steps = cfg.get("n_steps", 8)

    # Test data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    rets = data["ret"]
    H, T = 30, 30
    test_start = 4540

    # Build test windows
    test_windows = []
    for i in range(test_start, len(surfaces) - H - T + 1):
        history = surfaces[i:i+H]
        future = surfaces[i+H:i+H+T]
        test_windows.append((history, future))

    n_windows = min(args.max_batches * 8, len(test_windows))
    print(f"Evaluating on {n_windows} test windows, {args.n_samples} samples each")

    # Generate all samples
    all_samples = []  # will be (N, n_samples, 30, 5, 5)
    all_gt = []       # (N, 30, 5, 5)
    all_hist = []     # (N, 30, 5, 5)
    batch_size = 8

    with torch.no_grad():
        for i in range(0, n_windows, batch_size):
            batch_end = min(i + batch_size, n_windows)
            batch_hist = np.array([test_windows[j][0] for j in range(i, batch_end)], dtype=np.float32)
            batch_gt = np.array([test_windows[j][1] for j in range(i, batch_end)], dtype=np.float32)

            hist_t = torch.from_numpy(batch_hist).to(device)
            samp = generate_samples(velocity_net, encoder, hist_t, args.n_samples,
                                    n_steps, train_mean, train_std, device)

            all_samples.append(samp.cpu().numpy())
            all_gt.append(batch_gt)
            all_hist.append(batch_hist)

            if (i // batch_size + 1) % 5 == 0:
                print(f"  Batch {i // batch_size + 1}/{(n_windows + batch_size - 1) // batch_size}")

    cond_samples = np.concatenate(all_samples)  # (N, S, 30, 5, 5)
    ground_truth = np.concatenate(all_gt)        # (N, 30, 5, 5)
    history_arr = np.concatenate(all_hist)        # (N, 30, 5, 5)

    N, S = cond_samples.shape[:2]
    print(f"\nGenerated: samples={cond_samples.shape}, GT={ground_truth.shape}")

    # ═══════════════════════════════════════════════════
    # Suite 1: Surface Validity
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 1: Surface Validity ---")
    explosion_count = 0
    cal_arb_count = 0
    but_arb_count = 0
    total_surfaces = N * S * T

    for i in range(min(N, 50)):
        for s in range(S):
            for t in range(T):
                surf = cond_samples[i, s, t]  # (5, 5)
                # Explosion
                if surf.max() > 0.99 or surf.min() < 0.001:
                    explosion_count += 1
                # Calendar arb: IV should decrease with tenor (columns)
                for row in range(5):
                    for col in range(4):
                        if surf[row, col+1] > surf[row, col] + 0.01:
                            cal_arb_count += 1
                # Butterfly arb: smile convexity (rows)
                for col in range(5):
                    for row in range(1, 4):
                        if surf[row, col] > (surf[row-1, col] + surf[row+1, col]) / 2 + 0.01:
                            but_arb_count += 1

    total_checked = min(N, 50) * S * T
    expl_rate = explosion_count / total_checked
    cal_arb_rate = cal_arb_count / (total_checked * 5 * 4)
    but_arb_rate = but_arb_count / (total_checked * 5 * 3)
    s1_pass = expl_rate < 0.05 and cal_arb_rate < 0.10 and but_arb_rate < 0.25
    print(f"  Explosion rate: {expl_rate:.3f}, Cal arb: {cal_arb_rate:.3f}, But arb: {but_arb_rate:.3f}")
    print(f"  PASS: {s1_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 2: CI Coverage
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 2: CI Coverage ---")
    ci_pass_horizon = 0
    ci_pass_cell = 0
    worst_cell_cov = 1.0

    for h in range(T):
        gen_h = cond_samples[:, :, h, :, :]  # (N, S, 5, 5)
        gt_h = ground_truth[:, h, :, :]       # (N, 5, 5)
        lo = np.percentile(gen_h, 5, axis=1)   # (N, 5, 5)
        hi = np.percentile(gen_h, 95, axis=1)  # (N, 5, 5)
        covered = (gt_h >= lo) & (gt_h <= hi)   # (N, 5, 5)
        cov_rate = covered.mean()
        if cov_rate >= 0.85:
            ci_pass_horizon += 1

    # Per-cell CI
    for r in range(5):
        for c in range(5):
            gen_cell = cond_samples[:, :, :, r, c]  # (N, S, T)
            gt_cell = ground_truth[:, :, r, c]       # (N, T)
            lo = np.percentile(gen_cell, 5, axis=1)  # (N, T)
            hi = np.percentile(gen_cell, 95, axis=1)
            covered = (gt_cell >= lo) & (gt_cell <= hi)
            cell_cov = covered.mean()
            worst_cell_cov = min(worst_cell_cov, cell_cov)
            if cell_cov >= 0.85:
                ci_pass_cell += 1

    s2_pass = ci_pass_horizon >= 25 and worst_cell_cov >= 0.80
    print(f"  Horizon CI pass: {ci_pass_horizon}/30, Per-cell pass: {ci_pass_cell}/25")
    print(f"  Worst cell coverage: {worst_cell_cov:.3f}")
    print(f"  PASS: {s2_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 3: Conditionality
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 3: Conditionality ---")
    # Compute rolling 30-day realized vol for test windows
    rv = np.array([np.std(rets[i:i+H]) for i in range(test_start, test_start + n_windows)])
    turb_thresh = np.percentile(rv, 80)
    calm_thresh = np.percentile(rv, 20)
    turb_mask = rv > turb_thresh
    calm_mask = rv < calm_thresh

    if turb_mask.sum() > 0 and calm_mask.sum() > 0:
        turb_spread = cond_samples[turb_mask].std(axis=1).mean()
        calm_spread = cond_samples[calm_mask].std(axis=1).mean()
        turb_calm_ratio = turb_spread / (calm_spread + 1e-8)

        # Per-cell MAE reduction
        ensemble_mean = cond_samples.mean(axis=1)  # (N, 30, 5, 5)
        mae_cond = np.abs(ensemble_mean - ground_truth).mean()
        # Unconditional baseline: overall mean as forecast
        overall_mean = ground_truth.mean(axis=0)  # (30, 5, 5)
        mae_uncond = np.abs(ground_truth - overall_mean[None]).mean()
        mae_reduction = 1 - mae_cond / mae_uncond
    else:
        turb_calm_ratio = 1.0
        mae_reduction = 0.0

    s3_pass = turb_calm_ratio > 1.15
    print(f"  Turb/calm ratio: {turb_calm_ratio:.3f}")
    print(f"  MAE reduction: {mae_reduction:.3f}")
    print(f"  PASS: {s3_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 4: Time Series Properties
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 4: Time Series Properties ---")
    # ACF on daily changes (flatten across cells)
    gen_changes = np.diff(cond_samples[:, 0, :, :, :], axis=1).reshape(-1)
    gt_changes = np.diff(ground_truth, axis=1).reshape(-1)

    def acf(series, max_lag=10):
        m = np.mean(series); v = np.var(series)
        if v == 0: return np.zeros(max_lag + 1)
        return [1.0] + [np.mean((series[:-l] - m) * (series[l:] - m)) / v for l in range(1, max_lag + 1)]

    gen_acf = acf(gen_changes); gt_acf = acf(gt_changes)
    acf_corr = np.corrcoef(gen_acf, gt_acf)[0, 1]

    # Kurtosis
    gen_ch_flat = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch_flat = np.diff(ground_truth, axis=1).reshape(-1, 25)
    gen_kurt = kurtosis(gen_ch_flat.flatten(), fisher=True)
    gt_kurt = kurtosis(gt_ch_flat.flatten(), fisher=True)
    kurt_ratio = gen_kurt / (gt_kurt + 1e-6)

    s4_pass = acf_corr > 0.7 and 0.5 <= kurt_ratio <= 2.0
    print(f"  ACF correlation: {acf_corr:.3f}")
    print(f"  Kurtosis ratio: {kurt_ratio:.3f}")
    print(f"  PASS: {s4_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 5: Growing Uncertainty
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 5: Growing Uncertainty ---")
    spreads = []
    for h in range(T):
        spread_h = cond_samples[:, :, h].std(axis=1).mean()
        spreads.append(spread_h)
    spreads = np.array(spreads)

    mono_count = sum(1 for i in range(len(spreads)-1) if spreads[i+1] >= spreads[i] * 0.97)
    growing = mono_count >= 20  # 20/29 transitions monotonic
    s5_pass = growing
    print(f"  Spread h1={spreads[0]:.5f}, h30={spreads[-1]:.5f}")
    print(f"  Monotonic transitions: {mono_count}/29")
    print(f"  PASS: {s5_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 6: Cointegration
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 6: Cointegration ---")
    coint_pairs = 0; coint_total = 0
    for i in range(min(N, 30)):
        sample_path = cond_samples[i, 0, :, :, :].reshape(T, 25)  # (T, 25)
        for c1 in range(0, 25, 5):
            for c2 in range(c1+1, min(c1+5, 25)):
                try:
                    _, pval, _ = coint(sample_path[:, c1], sample_path[:, c2])
                    if pval < 0.05:
                        coint_pairs += 1
                    coint_total += 1
                except:
                    coint_total += 1

    coint_rate = coint_pairs / max(coint_total, 1)
    s6_pass = coint_rate > 0.50
    print(f"  Cointegration pass rate: {coint_rate:.3f} ({coint_pairs}/{coint_total})")
    print(f"  PASS: {s6_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 7: Regime Coverage
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 7: Regime Coverage ---")
    # Split by realized vol regime
    if turb_mask.sum() > 5 and calm_mask.sum() > 5:
        regime_pass = 0
        for mask, name in [(turb_mask, "turb"), (calm_mask, "calm")]:
            regime_samp = cond_samples[mask]
            regime_gt = ground_truth[mask]
            cell_pass = 0
            for r in range(5):
                for c in range(5):
                    gen_cell = regime_samp[:, :, :, r, c]
                    gt_cell = regime_gt[:, :, r, c]
                    lo = np.percentile(gen_cell, 5, axis=1)
                    hi = np.percentile(gen_cell, 95, axis=1)
                    covered = (gt_cell >= lo) & (gt_cell <= hi)
                    if covered.mean() >= 0.80:
                        cell_pass += 1
            if cell_pass >= 20:
                regime_pass += 1
            print(f"  {name}: {cell_pass}/25 cells pass CI")
        s7_pass = regime_pass == 2
    else:
        s7_pass = False
    print(f"  PASS: {s7_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 8: Distributional
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 8: Distributional ---")
    # KS on daily changes
    gen_ch = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch = np.diff(ground_truth, axis=1).reshape(-1, 25)
    ks_daily_pass = sum(1 for c in range(25) if ks_2samp(gen_ch[:, c], gt_ch[:, c])[0] < 0.15)

    # KS on levels
    gen_levels = cond_samples[:, 0, -1].reshape(-1, 25)
    gt_levels = ground_truth[:, -1].reshape(-1, 25)
    ks_level_pass = sum(1 for c in range(25) if ks_2samp(gen_levels[:, c], gt_levels[:, c])[0] < 0.15)

    # Median bias
    gen_median = np.median(cond_samples, axis=1)  # (N, 30, 5, 5)
    bias = (gen_median - ground_truth).mean()

    s8_pass = ks_daily_pass >= 15 and ks_level_pass >= 15
    print(f"  KS daily: {ks_daily_pass}/25, KS levels: {ks_level_pass}/25")
    print(f"  Median bias: {bias:.5f}")
    print(f"  PASS: {s8_pass}")

    # ═══════════════════════════════════════════════════
    # Suite 9: Cross-Cell Correlation
    # ═══════════════════════════════════════════════════
    print("\n--- Suite 9: Cross-Cell Correlation ---")
    gen_ch = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch = np.diff(ground_truth, axis=1).reshape(-1, 25)
    gen_corr = np.corrcoef(gen_ch.T)
    gt_corr = np.corrcoef(gt_ch.T)

    er_gen = eff_rank(gen_corr)
    er_gt = eff_rank(gt_corr)
    rank_ratio = er_gen / er_gt
    corr_ratio = np.abs(gen_corr).mean() / (np.abs(gt_corr).mean() + 1e-6)
    frob = np.linalg.norm(gen_corr - gt_corr, 'fro')

    # PC alignment
    gt_vecs = np.linalg.eigh(gt_corr)[1][:, ::-1]
    gen_vecs = np.linalg.eigh(gen_corr)[1][:, ::-1]
    pc_aligns = [abs(float(np.dot(gt_vecs[:, i], gen_vecs[:, i]))) for i in range(5)]

    s9_pass = rank_ratio >= 0.50 and corr_ratio >= 0.60
    print(f"  Eff rank: gen={er_gen:.2f}, GT={er_gt:.2f}, ratio={rank_ratio:.3f}")
    print(f"  Corr ratio: {corr_ratio:.3f}")
    print(f"  Frobenius: {frob:.2f}")
    print(f"  PC alignments: {[f'{a:.3f}' for a in pc_aligns]}")
    print(f"  PASS: {s9_pass}")

    # ═══════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════
    suites = [s1_pass, s2_pass, s3_pass, s4_pass, s5_pass, s6_pass, s7_pass, s8_pass, s9_pass]
    suite_names = ["Surface Validity", "CI Coverage", "Conditionality", "Time Series",
                   "Growing Uncertainty", "Cointegration", "Regime Coverage",
                   "Distributional", "Cross-Cell Correlation"]
    pass_count = sum(suites)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {pass_count}/9 suites PASS")
    print(f"{'='*60}")
    for name, passed in zip(suite_names, suites):
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    summary = {
        "model_path": args.model_path,
        "n_samples": args.n_samples,
        "n_windows": N,
        "pass_count": pass_count,
        "total_suites": 9,
        "suites": {name: bool(passed) for name, passed in zip(suite_names, suites)},
        "metrics": {
            "explosion_rate": round(float(expl_rate), 4),
            "calendar_arb_rate": round(float(cal_arb_rate), 4),
            "butterfly_arb_rate": round(float(but_arb_rate), 4),
            "ci_horizon_pass": int(ci_pass_horizon),
            "ci_worst_cell": round(float(worst_cell_cov), 4),
            "ci_cell_pass": int(ci_pass_cell),
            "turb_calm_ratio": round(float(turb_calm_ratio), 4),
            "mae_reduction": round(float(mae_reduction), 4),
            "acf_correlation": round(float(acf_corr), 4),
            "kurt_ratio": round(float(kurt_ratio), 4),
            "spread_h1": round(float(spreads[0]), 5),
            "spread_h30": round(float(spreads[-1]), 5),
            "monotonic_transitions": int(mono_count),
            "coint_rate": round(float(coint_rate), 4),
            "ks_daily_pass": int(ks_daily_pass),
            "ks_level_pass": int(ks_level_pass),
            "median_bias": round(float(bias), 5),
            "eff_rank_gen": round(float(er_gen), 4),
            "eff_rank_gt": round(float(er_gt), 4),
            "rank_ratio": round(float(rank_ratio), 4),
            "corr_ratio": round(float(corr_ratio), 4),
            "frobenius": round(float(frob), 4),
            "pc_alignments": [round(float(a), 4) for a in pc_aligns],
        },
    }

    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(make_serial(summary), f, indent=2)
    print(f"\nSummary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
