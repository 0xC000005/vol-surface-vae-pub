#!/usr/bin/env python
"""Window Floor Methodology Audit for 144b.

Identifies which windows are "bad" (coverage < 50%) and whether they cluster
in time or correlate with market regimes.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig, denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.block_ar.config_block_ar import get_default_config
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/backfill/afcrps_144b/best_model.pt"
output_dir = Path("results/validations/2026-03-22/analysis/window_floor")
output_dir.mkdir(parents=True, exist_ok=True)

# Load model
print("Loading model...")
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
if isinstance(cfg, dict):
    sp_cfg = {k: v for k, v in cfg.items()
              if k in SinglePassConfig.__dataclass_fields__}
    sp_config = SinglePassConfig(**sp_cfg)
else:
    sp_config = cfg
model = SinglePassBlockAR(sp_config)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval().to(device)

# Load data using the same config as the test suite
config = get_default_config()
data = np.load(config.data_path)
surfaces = data["surface"]
returns = data["ret"]

test_dataset = VolSurfaceDataset(
    surfaces, config.history_len, config.future_len,
    start_idx=config.test_start,
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

n_samples = 50
max_batches = 20

# Generate samples
print(f"Generating samples (n_samples={n_samples}, max_batches={max_batches})...")
all_samples = []
all_gt = []
all_history = []

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
        history = batch["history"].to(device)
        future_gt = denormalize_iv(batch["future"].to(device))
        samples = model.sample(history, n_samples=n_samples)  # (B, S, T, 5, 5)
        all_samples.append(samples.cpu().numpy())
        all_gt.append(future_gt.cpu().numpy())
        all_history.append(denormalize_iv(history).cpu().numpy())
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{max_batches}")

cond_samples = np.concatenate(all_samples, axis=0)  # (N, S, T, 5, 5)
ground_truth = np.concatenate(all_gt, axis=0)  # (N, T, 5, 5)
history_arr = np.concatenate(all_history, axis=0)  # (N, T_hist, 5, 5)

N, S, T = cond_samples.shape[:3]
print(f"Total windows: N={N}, S={S}, T={T}")

# Compute per-window coverage (matching test suite exactly)
q05 = np.percentile(cond_samples, 5, axis=1)   # (N, T, 5, 5)
q95 = np.percentile(cond_samples, 95, axis=1)   # (N, T, 5, 5)
covered = (ground_truth >= q05) & (ground_truth <= q95)  # (N, T, 5, 5)
per_window_cov = covered.mean(axis=(1, 2, 3))  # (N,)

WINDOW_FLOOR = 0.50
bad_mask = per_window_cov < WINDOW_FLOOR
n_bad = int(bad_mask.sum())
pct_bad = n_bad / N

print(f"\n=== WINDOW FLOOR ANALYSIS ===")
print(f"Bad windows (coverage < {WINDOW_FLOOR:.0%}): {n_bad}/{N} ({pct_bad:.1%})")
print(f"Worst window: idx={per_window_cov.argmin()}, cov={per_window_cov.min():.3f}")
print(f"P10 coverage: {np.percentile(per_window_cov, 10):.3f}")
print(f"Coverage distribution: min={per_window_cov.min():.3f}, P25={np.percentile(per_window_cov, 25):.3f}, "
      f"P50={np.percentile(per_window_cov, 50):.3f}, P75={np.percentile(per_window_cov, 75):.3f}, "
      f"max={per_window_cov.max():.3f}")

# Map bad windows to global time indices
test_start = config.test_start  # 4540
bad_indices = np.where(bad_mask)[0]
bad_global_indices = test_start + bad_indices  # global index of start of history
bad_future_start = bad_global_indices + config.history_len  # start of future period

print(f"\n=== BAD WINDOW LOCATIONS ===")
print(f"Test start index: {test_start}")
print(f"Bad window local indices (first 20): {bad_indices[:20].tolist()}")
print(f"Bad window global indices (future start, first 20): {bad_future_start[:20].tolist()}")

# Check for temporal clustering
if len(bad_indices) > 1:
    gaps = np.diff(bad_indices)
    print(f"\nGaps between consecutive bad windows:")
    print(f"  min gap: {gaps.min()}, max gap: {gaps.max()}, median: {np.median(gaps):.0f}")
    print(f"  Mean gap: {gaps.mean():.1f}")
    # Check if they cluster (many gaps of 1 = consecutive)
    n_consecutive = (gaps == 1).sum()
    print(f"  Consecutive pairs: {n_consecutive}/{len(gaps)}")

    # Find clusters (runs of consecutive bad windows)
    clusters = []
    start = bad_indices[0]
    for i in range(1, len(bad_indices)):
        if bad_indices[i] - bad_indices[i-1] > 5:  # gap > 5 means new cluster
            clusters.append((start, bad_indices[i-1]))
            start = bad_indices[i]
    clusters.append((start, bad_indices[-1]))

    print(f"\n  Clusters (gap > 5): {len(clusters)}")
    for i, (s, e) in enumerate(clusters):
        n_in = sum(1 for idx in bad_indices if s <= idx <= e)
        global_s = test_start + s + config.history_len
        global_e = test_start + e + config.history_len
        print(f"    Cluster {i+1}: local [{s}, {e}] (n={n_in}), global future [{global_s}, {global_e}]")

# Compute per-window realized vol-of-vol (regime proxy)
# vol_of_vol = std of rolling 5-day realized vol within the 30-day history
hist_atm = history_arr[:, :, 2, 2]  # (N, 30) — ATM 3M cell
hist_changes = np.diff(hist_atm, axis=1)  # (N, 29)
# Rolling 5-day realized vol
window_size = 5
rolling_vols = []
for i in range(hist_changes.shape[1] - window_size + 1):
    rolling_vols.append(np.std(hist_changes[:, i:i+window_size], axis=1))
rolling_vols = np.stack(rolling_vols, axis=1)  # (N, 25)
vol_of_vol = np.std(rolling_vols, axis=1)  # (N,)

# Also compute mean IV level for regime classification
mean_iv = history_arr[:, -1, :, :].mean(axis=(1, 2))  # (N,) last day mean IV

# Classify regime based on vol_of_vol quartiles
q25_vov = np.percentile(vol_of_vol, 25)
q75_vov = np.percentile(vol_of_vol, 75)
calm = vol_of_vol < q25_vov
turb = vol_of_vol > q75_vov
mid = ~calm & ~turb

print(f"\n=== REGIME ANALYSIS ===")
print(f"Vol-of-vol quartiles: Q25={q25_vov:.6f}, Q75={q75_vov:.6f}")
for regime_name, regime_mask in [("calm", calm), ("mid", mid), ("turb", turb)]:
    n_regime = regime_mask.sum()
    n_bad_in_regime = (regime_mask & bad_mask).sum()
    rate = n_bad_in_regime / n_regime if n_regime > 0 else 0
    mean_cov = per_window_cov[regime_mask].mean() if n_regime > 0 else 0
    print(f"  {regime_name:5s}: {n_regime:4d} windows, {n_bad_in_regime:3d} bad "
          f"({rate:.1%}), mean cov={mean_cov:.3f}")

# High IV vs low IV
q50_iv = np.median(mean_iv)
low_iv = mean_iv < q50_iv
high_iv = ~low_iv
print(f"\nMedian IV level: {q50_iv:.4f}")
for iv_name, iv_mask in [("low_iv", low_iv), ("high_iv", high_iv)]:
    n_iv = iv_mask.sum()
    n_bad_in = (iv_mask & bad_mask).sum()
    rate = n_bad_in / n_iv if n_iv > 0 else 0
    print(f"  {iv_name:7s}: {n_iv:4d} windows, {n_bad_in:3d} bad ({rate:.1%})")

# Per-cell analysis: which cells have worst coverage in bad windows?
print(f"\n=== PER-CELL COVERAGE IN BAD WINDOWS ===")
if n_bad > 0:
    bad_covered = covered[bad_mask]  # (n_bad, T, 5, 5)
    bad_cell_cov = bad_covered.mean(axis=(0, 1))  # (5, 5)
    good_covered = covered[~bad_mask]
    good_cell_cov = good_covered.mean(axis=(0, 1))  # (5, 5)

    LABELS_K = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
    LABELS_T = ["1M", "3M", "6M", "1Y", "2Y"]

    print("Bad window cell coverage:")
    for r in range(5):
        row = "  " + " ".join(f"{bad_cell_cov[r,c]:.2f}" for c in range(5))
        print(f"  {LABELS_T[r]}: {row}")

    print("\nGood window cell coverage:")
    for r in range(5):
        row = "  " + " ".join(f"{good_cell_cov[r,c]:.2f}" for c in range(5))
        print(f"  {LABELS_T[r]}: {row}")

    print("\nDelta (bad - good):")
    delta = bad_cell_cov - good_cell_cov
    for r in range(5):
        row = "  " + " ".join(f"{delta[r,c]:+.2f}" for c in range(5))
        print(f"  {LABELS_T[r]}: {row}")

# Per-horizon analysis in bad windows
print(f"\n=== PER-HORIZON COVERAGE IN BAD WINDOWS ===")
if n_bad > 0:
    for h in [0, 6, 13, 29]:
        bad_h_cov = covered[bad_mask, h].mean()
        good_h_cov = covered[~bad_mask, h].mean()
        print(f"  h={h+1:2d}: bad={bad_h_cov:.3f}, good={good_h_cov:.3f}, delta={bad_h_cov-good_h_cov:+.3f}")

# Ensemble spread analysis in bad vs good windows
print(f"\n=== ENSEMBLE SPREAD ANALYSIS ===")
spread = cond_samples.std(axis=1)  # (N, T, 5, 5) — per-sample std
bad_spread = spread[bad_mask].mean()
good_spread = spread[~bad_mask].mean()
print(f"  Bad window mean spread: {bad_spread:.5f}")
print(f"  Good window mean spread: {good_spread:.5f}")
print(f"  Ratio (bad/good): {bad_spread/good_spread:.3f}")

# GT change magnitude in bad vs good windows
gt_change_mag = np.abs(np.diff(ground_truth, axis=1)).mean(axis=(1, 2, 3))  # (N,)
bad_change = gt_change_mag[bad_mask].mean()
good_change = gt_change_mag[~bad_mask].mean()
print(f"\nGT daily change magnitude:")
print(f"  Bad windows: {bad_change:.6f}")
print(f"  Good windows: {good_change:.6f}")
print(f"  Ratio: {bad_change/good_change:.3f}")

# GT total displacement in bad vs good windows
gt_displacement = np.abs(ground_truth[:, -1] - ground_truth[:, 0]).mean(axis=(1, 2))  # (N,)
bad_disp = gt_displacement[bad_mask].mean()
good_disp = gt_displacement[~bad_mask].mean()
print(f"\nGT 30-day displacement:")
print(f"  Bad windows: {bad_disp:.6f}")
print(f"  Good windows: {good_disp:.6f}")
print(f"  Ratio: {bad_disp/good_disp:.3f}")

# Save results
results = {
    "methodology": {
        "window_definition": "Each window is a (history=30d, future=30d) pair from the test set, stride-1",
        "coverage_definition": "Fraction of (T=30 timesteps x 5x5 cells = 750) points where GT falls inside 90% CI",
        "bad_window_threshold": 0.50,
        "gate": "< 5% of windows should have coverage < 50%",
        "total_windows": int(N),
        "n_samples_per_window": int(S),
        "test_start_idx": int(test_start),
    },
    "results_144b": {
        "n_bad_windows": int(n_bad),
        "pct_bad": float(pct_bad),
        "worst_window_cov": float(per_window_cov.min()),
        "p10_cov": float(np.percentile(per_window_cov, 10)),
        "coverage_distribution": {
            "min": float(per_window_cov.min()),
            "p25": float(np.percentile(per_window_cov, 25)),
            "p50": float(np.percentile(per_window_cov, 50)),
            "p75": float(np.percentile(per_window_cov, 75)),
            "max": float(per_window_cov.max()),
        },
    },
    "clustering": {
        "n_clusters_gap5": len(clusters) if len(bad_indices) > 1 else 0,
        "clusters": [(int(s), int(e)) for s, e in clusters] if len(bad_indices) > 1 else [],
        "n_consecutive_pairs": int(n_consecutive) if len(bad_indices) > 1 else 0,
        "median_gap": float(np.median(gaps)) if len(bad_indices) > 1 else None,
    },
    "regime_breakdown": {
        "calm": {
            "n_windows": int(calm.sum()),
            "n_bad": int((calm & bad_mask).sum()),
            "bad_rate": float((calm & bad_mask).sum() / calm.sum()) if calm.sum() > 0 else 0,
        },
        "mid": {
            "n_windows": int(mid.sum()),
            "n_bad": int((mid & bad_mask).sum()),
            "bad_rate": float((mid & bad_mask).sum() / mid.sum()) if mid.sum() > 0 else 0,
        },
        "turb": {
            "n_windows": int(turb.sum()),
            "n_bad": int((turb & bad_mask).sum()),
            "bad_rate": float((turb & bad_mask).sum() / turb.sum()) if turb.sum() > 0 else 0,
        },
    },
    "spread_analysis": {
        "bad_mean_spread": float(bad_spread),
        "good_mean_spread": float(good_spread),
        "ratio": float(bad_spread / good_spread),
    },
    "gt_change_analysis": {
        "bad_change_mag": float(bad_change),
        "good_change_mag": float(good_change),
        "ratio": float(bad_change / good_change),
        "bad_displacement": float(bad_disp),
        "good_displacement": float(good_disp),
        "displacement_ratio": float(bad_disp / good_disp),
    },
}

with open(output_dir / "window_floor_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_dir / 'window_floor_analysis.json'}")
