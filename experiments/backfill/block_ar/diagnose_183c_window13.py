#!/usr/bin/env python
"""
Diagnostic: Investigate catastrophically bad coverage window (39.1%) in 183c evaluation.

Identifies window 13 characteristics, per-horizon/per-cell coverage breakdown,
and checks whether bad-coverage windows share a pattern.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import build_rollout_windows
from experiments.backfill.block_ar.analyze_183c_best_mechanism import load_model
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
    unconstrained_to_iv,
)


def compute_vov(history_01: np.ndarray) -> np.ndarray:
    """Volatility-of-volatility: std of daily changes in mean IV level."""
    mean_iv = history_01.mean(axis=(2, 3))  # (W, T)
    changes = np.diff(mean_iv, axis=1)       # (W, T-1)
    return changes.std(axis=1)               # (W,)


def per_window_coverage(samples: np.ndarray, gt: np.ndarray, alpha: float = 0.90) -> np.ndarray:
    """Compute per-window coverage (fraction of cells*horizons inside 90% CI)."""
    tail = (1.0 - alpha) / 2.0
    lo = np.quantile(samples, tail, axis=1)       # (W, H, 5, 5)
    hi = np.quantile(samples, 1 - tail, axis=1)   # (W, H, 5, 5)
    inside = (gt >= lo) & (gt <= hi)               # (W, H, 5, 5)
    return inside.mean(axis=(1, 2, 3))             # (W,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Parameters matching evaluate_220b defaults ---
    model_path = ("models/backfill/state_metric_transport_pathwise_residual_law_"
                  "mean_reverting_covariance_mixture_structured_joint_student_t_183c/"
                  "best_model.pt")
    data_path = "data/vol_surface_with_ret.npz"
    history_len = 30
    future_len = 30
    test_start = 4511   # evaluate_220b default (user said 4540 but script says 4511)
    val_size = 441
    max_windows = 192
    n_samples = args.n_samples

    print("=" * 80)
    print("DIAGNOSTIC: 183c Window 13 Coverage Investigation")
    print("=" * 80)
    print(f"test_start={test_start}, val_size={val_size}, max_windows={max_windows}")
    print(f"NOTE: evaluate_220b script default is test_start=4511, NOT 4540")
    print()

    # --- Compute raw indices to identify what window 13 corresponds to ---
    max_train_idx = test_start - history_len - future_len
    val_start = max_train_idx - val_size
    val_indices = np.arange(val_start, max_train_idx)[:max_windows]
    print(f"val_indices range: [{val_indices[0]}, {val_indices[-1]}]")
    print(f"Window 13: raw surface index = {val_indices[13]}")
    print(f"  History: surface[{val_indices[13]}:{val_indices[13] + history_len}]")
    print(f"  Future:  surface[{val_indices[13] + history_len}:{val_indices[13] + history_len + future_len}]")
    print()

    # --- Load data ---
    print("Loading data windows...")
    batch = build_rollout_windows(
        data_path=data_path,
        history_len=history_len,
        future_len=future_len,
        test_start=test_start,
        val_size=val_size,
        max_windows=max_windows,
        device=device,
        split="val",
    )
    n_windows = batch.history_01.shape[0]
    print(f"Loaded {n_windows} windows")

    # --- Load model ---
    print("Loading 183c model...")
    model, _ = load_model(model_path, device)
    print("Model loaded.")
    print()

    # --- Generate samples for ALL windows ---
    print(f"Generating {n_samples} samples for all {n_windows} windows...")
    all_samples = []
    bs = 16
    for start in range(0, n_windows, bs):
        end = min(start + bs, n_windows)
        hist_batch = normalize_iv(batch.history_01[start:end])
        with torch.no_grad():
            samp = model.sample_batched(
                hist_batch, n_samples=n_samples, n_steps=future_len, chunk_size=8,
            )
        all_samples.append(samp.cpu().numpy())
        print(f"  Generated windows {start}-{end-1}")
    samples = np.concatenate(all_samples, axis=0)  # (W, K, H, 5, 5)
    gt = batch.future_01.detach().cpu().numpy()       # (W, H, 5, 5)
    hist_01 = batch.history_01.detach().cpu().numpy()  # (W, 30, 5, 5)
    print(f"Samples shape: {samples.shape}, GT shape: {gt.shape}")
    print()

    # --- Per-window coverage ---
    cov_per_window = per_window_coverage(samples, gt)
    print("=" * 80)
    print("SECTION 1: PER-WINDOW COVERAGE DISTRIBUTION")
    print("=" * 80)
    print(f"Window 13 coverage: {cov_per_window[13]:.3f}")
    print(f"Overall mean coverage: {cov_per_window.mean():.3f}")
    print(f"Overall median coverage: {np.median(cov_per_window):.3f}")
    print(f"Min coverage: {cov_per_window.min():.3f} (window {cov_per_window.argmin()})")
    print(f"Max coverage: {cov_per_window.max():.3f} (window {cov_per_window.argmax()})")
    print()

    for threshold in [0.60, 0.70, 0.80, 0.85]:
        n_bad = (cov_per_window < threshold).sum()
        bad_indices = np.where(cov_per_window < threshold)[0]
        print(f"Windows with coverage < {threshold:.0%}: {n_bad}")
        if n_bad > 0 and n_bad <= 20:
            for idx in bad_indices:
                print(f"  Window {idx}: coverage={cov_per_window[idx]:.3f}, "
                      f"raw_idx={val_indices[idx]}")
    print()

    # --- VoV regime classification ---
    vov = compute_vov(hist_01)
    q20 = np.quantile(vov, 0.2)
    q80 = np.quantile(vov, 0.8)
    regime = np.where(vov <= q20, "calm", np.where(vov >= q80, "turb", "mid"))

    print("=" * 80)
    print("SECTION 2: WINDOW 13 CHARACTERISTICS")
    print("=" * 80)
    w13_hist = hist_01[13]  # (30, 5, 5)
    w13_future = gt[13]     # (30, 5, 5)
    w13_vov = vov[13]
    w13_regime = regime[13]

    print(f"VoV: {w13_vov:.6f} (q20={q20:.6f}, q80={q80:.6f})")
    print(f"Regime: {w13_regime}")
    print(f"History last-frame IV: mean={w13_hist[-1].mean():.4f}, "
          f"min={w13_hist[-1].min():.4f}, max={w13_hist[-1].max():.4f}")
    print(f"History mean IV level: {w13_hist.mean():.4f}")
    print(f"History mean IV range (max-min across time): "
          f"{(w13_hist.mean(axis=(1, 2)).max() - w13_hist.mean(axis=(1, 2)).min()):.4f}")

    # Recent changes (last 5 days of history)
    recent_changes = np.diff(w13_hist.mean(axis=(1, 2)))[-5:]
    print(f"Recent 5-day mean IV changes: {[f'{c:.5f}' for c in recent_changes]}")

    # Future characteristics
    print(f"Future frame 1 mean IV: {w13_future[0].mean():.4f}")
    print(f"Future frame 15 mean IV: {w13_future[14].mean():.4f}")
    print(f"Future frame 30 mean IV: {w13_future[29].mean():.4f}")
    print(f"Future IV drift (frame30 - last_hist): {w13_future[29].mean() - w13_hist[-1].mean():.4f}")

    # Tenor labels
    tenor_labels = ["1M", "3M", "6M", "9M", "12M"]
    money_labels = ["80%", "90%", "100%", "110%", "120%"]

    print("\nHistory last-frame IV grid:")
    for i, m in enumerate(money_labels):
        row = " ".join(f"{w13_hist[-1, i, j]:.4f}" for j in range(5))
        print(f"  {m:>4s}: {row}")
    print(f"  Tenors: {' '.join(f'{t:>6s}' for t in tenor_labels)}")
    print()

    # --- Per-horizon coverage for window 13 ---
    print("=" * 80)
    print("SECTION 3: WINDOW 13 PER-HORIZON COVERAGE")
    print("=" * 80)
    w13_samples = samples[13]  # (K, 30, 5, 5)
    w13_gt = gt[13]            # (30, 5, 5)

    for h in range(future_len):
        samp_h = w13_samples[:, h]  # (K, 5, 5)
        gt_h = w13_gt[h]            # (5, 5)
        lo = np.quantile(samp_h, 0.05, axis=0)
        hi = np.quantile(samp_h, 0.95, axis=0)
        inside = ((gt_h >= lo) & (gt_h <= hi))
        cov_h = inside.mean()
        width_h = (hi - lo).mean()
        bias_h = (samp_h.mean(axis=0) - gt_h).mean()
        print(f"  h={h+1:2d}: coverage={cov_h:.3f}, width={width_h:.4f}, "
              f"mean_bias={bias_h:+.4f}, cells_covered={inside.sum()}/25")
    print()

    # Find worst horizon
    horizon_covs = []
    for h in range(future_len):
        samp_h = w13_samples[:, h]
        gt_h = w13_gt[h]
        lo = np.quantile(samp_h, 0.05, axis=0)
        hi = np.quantile(samp_h, 0.95, axis=0)
        inside = ((gt_h >= lo) & (gt_h <= hi))
        horizon_covs.append(inside.mean())
    worst_h = np.argmin(horizon_covs)
    print(f"Worst horizon: h={worst_h+1} with coverage={horizon_covs[worst_h]:.3f}")
    print()

    # --- Per-cell coverage at worst horizon ---
    print("=" * 80)
    print(f"SECTION 4: WINDOW 13 PER-CELL COVERAGE AT WORST HORIZON (h={worst_h+1})")
    print("=" * 80)
    samp_worst = w13_samples[:, worst_h]  # (K, 5, 5)
    gt_worst = w13_gt[worst_h]            # (5, 5)
    lo_worst = np.quantile(samp_worst, 0.05, axis=0)
    hi_worst = np.quantile(samp_worst, 0.95, axis=0)
    inside_worst = ((gt_worst >= lo_worst) & (gt_worst <= hi_worst))

    print("Coverage (1=covered, 0=missed):")
    for i, m in enumerate(money_labels):
        row = " ".join(f"{'OK' if inside_worst[i, j] else 'MISS':>6s}" for j in range(5))
        print(f"  {m:>4s}: {row}")
    print(f"  Tenors: {' '.join(f'{t:>6s}' for t in tenor_labels)}")

    print("\nGT vs CI comparison at worst horizon:")
    for i, m in enumerate(money_labels):
        for j, t in enumerate(tenor_labels):
            gt_val = gt_worst[i, j]
            lo_val = lo_worst[i, j]
            hi_val = hi_worst[i, j]
            samp_mean = samp_worst[:, i, j].mean()
            samp_std = samp_worst[:, i, j].std()
            status = "OK" if inside_worst[i, j] else "MISS"
            print(f"  [{m:>4s},{t:>3s}]: GT={gt_val:.4f}, "
                  f"CI=[{lo_val:.4f}, {hi_val:.4f}], "
                  f"mean={samp_mean:.4f}, std={samp_std:.4f}, {status}")
    print()

    # --- Also check the "almost worst" horizons ---
    print("=" * 80)
    print("SECTION 5: GENERATED vs GT DISTRIBUTION AT WORST HORIZON (AGGREGATED)")
    print("=" * 80)
    # Flatten across cells for window 13 at worst horizon
    samp_flat = samp_worst.reshape(n_samples, -1)  # (K, 25)
    gt_flat = gt_worst.reshape(-1)                  # (25,)

    # Per-cell stats
    for c in range(25):
        i, j = divmod(c, 5)
        gt_v = gt_flat[c]
        samp_c = samp_flat[:, c]
        print(f"  Cell ({money_labels[i]:>4s},{tenor_labels[j]:>3s}): "
              f"GT={gt_v:.4f}, gen_mean={samp_c.mean():.4f}, "
              f"gen_std={samp_c.std():.4f}, "
              f"gen_q05={np.quantile(samp_c, 0.05):.4f}, "
              f"gen_q95={np.quantile(samp_c, 0.95):.4f}, "
              f"z_score={(gt_v - samp_c.mean()) / max(samp_c.std(), 1e-8):.2f}")
    print()

    # --- Characterize bad windows ---
    print("=" * 80)
    print("SECTION 6: BAD WINDOW PATTERN ANALYSIS")
    print("=" * 80)

    bad_mask_60 = cov_per_window < 0.60
    bad_mask_70 = cov_per_window < 0.70
    bad_mask_80 = cov_per_window < 0.80

    print(f"Total windows: {n_windows}")
    print(f"Windows < 60%: {bad_mask_60.sum()}")
    print(f"Windows < 70%: {bad_mask_70.sum()}")
    print(f"Windows < 80%: {bad_mask_80.sum()}")
    print()

    # Regime breakdown of bad windows
    for thresh, mask in [("< 60%", bad_mask_60), ("< 70%", bad_mask_70), ("< 80%", bad_mask_80)]:
        if mask.sum() == 0:
            continue
        bad_regimes = regime[mask]
        n_calm = (bad_regimes == "calm").sum()
        n_mid = (bad_regimes == "mid").sum()
        n_turb = (bad_regimes == "turb").sum()
        print(f"Coverage {thresh} ({mask.sum()} windows): "
              f"calm={n_calm}, mid={n_mid}, turb={n_turb}")

    print()

    # Mean IV level and VoV of bad windows vs all
    for thresh, mask in [("< 60%", bad_mask_60), ("< 70%", bad_mask_70), ("< 80%", bad_mask_80)]:
        if mask.sum() == 0:
            continue
        bad_mean_iv = hist_01[mask, -1].mean()
        all_mean_iv = hist_01[:, -1].mean()
        bad_vov = vov[mask].mean()
        all_vov = vov.mean()
        bad_future_drift = (gt[mask, -1].mean(axis=(1, 2)) - hist_01[mask, -1].mean(axis=(1, 2))).mean()
        all_future_drift = (gt[:, -1].mean(axis=(1, 2)) - hist_01[:, -1].mean(axis=(1, 2))).mean()
        print(f"Coverage {thresh} windows:")
        print(f"  Mean IV (last hist): {bad_mean_iv:.4f} (all: {all_mean_iv:.4f})")
        print(f"  Mean VoV: {bad_vov:.6f} (all: {all_vov:.6f})")
        print(f"  Mean future drift: {bad_future_drift:+.4f} (all: {all_future_drift:+.4f})")
    print()

    # Check if bad windows are clustered in time
    bad_70_indices = np.where(bad_mask_70)[0]
    if len(bad_70_indices) > 1:
        gaps = np.diff(bad_70_indices)
        print(f"Windows < 70% indices: {bad_70_indices.tolist()}")
        print(f"  Index gaps between consecutive bad windows: {gaps.tolist()}")
        print(f"  Raw surface indices: {val_indices[bad_70_indices].tolist()}")
    print()

    # --- Per-cell aggregate coverage across ALL windows ---
    print("=" * 80)
    print("SECTION 7: AGGREGATE PER-CELL COVERAGE (ALL WINDOWS)")
    print("=" * 80)
    lo_all = np.quantile(samples, 0.05, axis=1)  # (W, H, 5, 5)
    hi_all = np.quantile(samples, 0.95, axis=1)
    inside_all = (gt >= lo_all) & (gt <= hi_all)

    # Per cell at h=30
    print("Per-cell coverage at h=30:")
    cell_cov_h30 = inside_all[:, 29].mean(axis=0)  # (5, 5)
    for i, m in enumerate(money_labels):
        row = " ".join(f"{cell_cov_h30[i, j]:.3f}" for j in range(5))
        print(f"  {m:>4s}: {row}")
    print(f"  Tenors: {' '.join(f'{t:>6s}' for t in tenor_labels)}")
    print()

    # Per-horizon aggregate
    print("Per-horizon aggregate coverage (all windows):")
    for h in [0, 4, 9, 14, 19, 24, 29]:
        cov_h = inside_all[:, h].mean()
        print(f"  h={h+1:2d}: {cov_h:.3f}")
    print()

    # --- Window 13: is the bias systematic? ---
    print("=" * 80)
    print("SECTION 8: WINDOW 13 BIAS ANALYSIS")
    print("=" * 80)
    w13_samp_mean = w13_samples.mean(axis=0)  # (30, 5, 5)
    w13_bias = w13_samp_mean - w13_gt          # (30, 5, 5)

    print("Per-horizon mean bias (gen - GT), averaged over cells:")
    for h in range(future_len):
        print(f"  h={h+1:2d}: bias={w13_bias[h].mean():+.5f}, |bias|={np.abs(w13_bias[h]).mean():.5f}")
    print()

    # Direction of bias: is model overshooting or undershooting?
    print("Bias direction at h=30:")
    for i, m in enumerate(money_labels):
        for j, t in enumerate(tenor_labels):
            print(f"  [{m:>4s},{t:>3s}]: GT={w13_gt[29, i, j]:.4f}, "
                  f"gen_mean={w13_samp_mean[29, i, j]:.4f}, "
                  f"bias={w13_bias[29, i, j]:+.4f}")
    print()

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
