#!/usr/bin/env python
"""
Independent re-evaluation of 152e one-shot factored transformer.

Tasks:
1. Load checkpoint, generate samples, verify ALL metrics independently
2. Investigate eff_rank gap (5.57 vs GT 7.61) — per-horizon breakdown
3. Spread structure analysis — per-cell, per-horizon

This is NOT the training script's in-process eval. This loads the saved
checkpoint fresh and computes everything from scratch.
"""
import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import ks_2samp, kurtosis, spearmanr

# ── Load model from checkpoint ──
device = "cuda"
torch.manual_seed(42)
np.random.seed(42)

ckpt = torch.load("models/backfill/flow_152e/best_model.pt",
                   weights_only=False, map_location=device)
cfg = ckpt["config"]
train_mean = ckpt["train_mean"]
train_std = ckpt["train_std"]

# Rebuild architecture from scratch
import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_oneshot_flow import FactoredVelocityTransformer

model = FactoredVelocityTransformer(
    n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
    d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()
print(f"Loaded 152e best_model (epoch {ckpt['epoch']})")
print(f"Config: {cfg}")

# ── Load GT data ──
data = np.load("data/vol_surface_with_ret.npz")
surfaces = data["surface"]
H, F_LEN = 30, 30

train_end = 4040
gt_futures = [surfaces[i+H:i+H+F_LEN].reshape(-1)
              for i in range(train_end - H - F_LEN + 1)]
gt_data = np.array(gt_futures, dtype=np.float32)

# ── Generate samples independently ──
print("\nGenerating 2000 samples via 8-step ODE...")
DIM = cfg["dim"]
n_steps = cfg["n_steps"]
n_eval = 2000
eval_batch = 64

all_samples = []
with torch.no_grad():
    for si in range(0, n_eval, eval_batch):
        eb = min(eval_batch, n_eval - si)
        x = torch.randn(eb, DIM, device=device)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((eb,), step * dt, device=device)
            x = x + model(x, t) * dt
        all_samples.append(x.cpu().numpy())

samples_std = np.concatenate(all_samples)  # standardized
samples = samples_std * train_std + train_mean  # denormalized
samples = np.clip(samples, 0, 1)
print(f"Generated: {samples.shape}")

# ══════════════════════════════════════════════════════════
# TASK 1: Independent metric verification
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 1: Independent Metric Verification")
print("=" * 60)

samples_3d = samples.reshape(-1, F_LEN, 25)
gt_3d = gt_data.reshape(-1, F_LEN, 25)

# Daily changes
gen_ch = np.diff(samples_3d, axis=1).reshape(-1, 25)
gt_ch = np.diff(gt_3d, axis=1).reshape(-1, 25)

gen_corr = np.corrcoef(gen_ch.T)
gt_corr = np.corrcoef(gt_ch.T)

def eff_rank(corr):
    ev = np.linalg.eigvalsh(corr)[::-1]
    ev = np.maximum(ev, 0)
    p = ev / (ev.sum() + 1e-10); p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))

er_gen = eff_rank(gen_corr)
er_gt = eff_rank(gt_corr)

gt_vecs = np.linalg.eigh(gt_corr)[1][:, ::-1]
gen_vecs = np.linalg.eigh(gen_corr)[1][:, ::-1]
pc_aligns = [abs(float(np.dot(gt_vecs[:, i], gen_vecs[:, i]))) for i in range(5)]

frob = float(np.linalg.norm(gen_corr - gt_corr, 'fro'))

ks_results = []
for c in range(25):
    stat, _ = ks_2samp(gen_ch[:, c], gt_ch[:, c])
    ks_results.append(stat)
ks_pass = sum(1 for s in ks_results if s < 0.15)

kurt_gen = kurtosis(gen_ch.flatten(), fisher=True)
kurt_gt = kurtosis(gt_ch.flatten(), fisher=True)
kurt_ratio = kurt_gen / (kurt_gt + 1e-6)

# Compare with training-time claims
claims = {
    "eff_rank": (er_gen, 5.57, 0.3),
    "PC1": (pc_aligns[0], 0.994, 0.02),
    "PC2": (pc_aligns[1], 0.985, 0.05),
    "KS_pass": (ks_pass, 25, 1),
    "kurt_ratio": (kurt_ratio, 1.113, 0.15),
    "frob": (frob, 2.9, 0.5),
}
all_verified = True
for name, (actual, claimed, tol) in claims.items():
    ok = abs(actual - claimed) <= tol
    if not ok: all_verified = False
    print(f"  {'PASS' if ok else 'FAIL'}: {name} = {actual:.3f} (claimed {claimed}, tol ±{tol})")

print(f"\n  All PC alignments: {[f'{a:.3f}' for a in pc_aligns]}")
print(f"  GT eff_rank: {er_gt:.2f}")
print(f"  Overall: {'ALL VERIFIED' if all_verified else 'SOME DISCREPANCIES'}")

# ══════════════════════════════════════════════════════════
# TASK 2: eff_rank gap investigation — per-horizon breakdown
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2: eff_rank Gap Investigation (5.57 vs GT 7.61)")
print("=" * 60)

# Per-horizon: compute eff_rank of cross-cell correlation at each timestep
horizons = [0, 4, 9, 14, 19, 24, 29]  # h=1, 5, 10, 15, 20, 25, 30
print("\n  Per-horizon eff_rank (cross-cell correlation at each frame):")
print(f"  {'h':>4s}  {'gen_eff':>8s}  {'gt_eff':>8s}  {'ratio':>6s}")

per_h_results = {}
for h in horizons:
    # Cross-cell correlation at this specific horizon
    gen_frame = samples_3d[:, h, :]  # (N, 25)
    gt_frame = gt_3d[:, h, :]
    gc = np.corrcoef(gen_frame.T)
    gtc = np.corrcoef(gt_frame.T)
    er_g = eff_rank(gc)
    er_gt_h = eff_rank(gtc)
    ratio = er_g / er_gt_h
    per_h_results[h+1] = {"gen": er_g, "gt": er_gt_h, "ratio": ratio}
    print(f"  h={h+1:2d}   {er_g:8.2f}   {er_gt_h:8.2f}   {ratio:6.3f}")

# Per-horizon: eff_rank of daily changes at specific horizons
print("\n  Per-horizon eff_rank (daily changes ending at each frame):")
for h in [4, 9, 14, 19, 24, 29]:
    gen_dch = samples_3d[:, h, :] - samples_3d[:, h-1, :]
    gt_dch = gt_3d[:, h, :] - gt_3d[:, h-1, :]
    gc = np.corrcoef(gen_dch.T)
    gtc = np.corrcoef(gt_dch.T)
    er_g = eff_rank(gc)
    er_gt_h = eff_rank(gtc)
    print(f"  Δh={h+1:2d}  gen={er_g:.2f}  gt={er_gt_h:.2f}  ratio={er_g/er_gt_h:.3f}")

# Eigenvalue spectrum comparison
gen_eigvals = np.linalg.eigvalsh(gen_corr)[::-1]
gen_eigvals = np.maximum(gen_eigvals, 0)
gt_eigvals = np.linalg.eigvalsh(gt_corr)[::-1]
gt_eigvals = np.maximum(gt_eigvals, 0)

print(f"\n  Top-5 eigenvalue comparison (daily changes correlation):")
print(f"  {'PC':>4s}  {'GT':>8s}  {'Gen':>8s}  {'Gen/GT':>8s}")
for i in range(5):
    r = gen_eigvals[i] / (gt_eigvals[i] + 1e-10)
    print(f"  PC{i+1}   {gt_eigvals[i]:8.3f}   {gen_eigvals[i]:8.3f}   {r:8.3f}")

# ══════════════════════════════════════════════════════════
# TASK 3: Spread structure analysis
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 3: Spread Structure Analysis")
print("=" * 60)

# Per-horizon spread (std across samples)
spreads_per_h = samples_3d.std(axis=0)  # (30, 25) — std across N samples
mean_spread_per_h = spreads_per_h.mean(axis=1)  # (30,)

print("\n  Mean spread (std across samples) by horizon:")
for h in [0, 4, 9, 14, 19, 24, 29]:
    print(f"  h={h+1:2d}: {mean_spread_per_h[h]:.5f}")

# GT spread for comparison
gt_spreads = gt_3d.std(axis=0).mean(axis=1)
print("\n  GT spread (std across windows) by horizon:")
for h in [0, 4, 9, 14, 19, 24, 29]:
    print(f"  h={h+1:2d}: {gt_spreads[h]:.5f}")

# Per-cell spread at h=1 and h=30
print("\n  Per-cell spread at h=1 (5x5 grid):")
for i in range(5):
    print(f"    {' '.join(f'{spreads_per_h[0, i*5+j]:.4f}' for j in range(5))}")

print("\n  Per-cell spread at h=30 (5x5 grid):")
for i in range(5):
    print(f"    {' '.join(f'{spreads_per_h[29, i*5+j]:.4f}' for j in range(5))}")

# Ratio h30/h1 per cell
ratio_grid = spreads_per_h[29] / (spreads_per_h[0] + 1e-8)
print("\n  Spread ratio h30/h1 per cell (>1 = growing, <1 = shrinking):")
for i in range(5):
    print(f"    {' '.join(f'{ratio_grid[i*5+j]:.3f}' for j in range(5))}")
n_growing = (ratio_grid > 1.0).sum()
print(f"  Cells with growing spread: {n_growing}/25")

# ── Save all results ──
results = {
    "experiment_id": "152e",
    "verification_type": "independent_eval",
    "timestamp": "2026-03-24",
    "claims_verified": [
        {"claim": f"{k}={v[1]}", "actual": round(v[0], 4),
         "verified": bool(abs(v[0]-v[1]) <= v[2]),
         "tolerance": v[2]}
        for k, v in claims.items()
    ],
    "independent_metrics": {
        "eff_rank": round(er_gen, 4), "gt_eff_rank": round(er_gt, 4),
        "pc_alignments": [round(a, 4) for a in pc_aligns],
        "ks_pass": ks_pass, "ks_stats": [round(s, 4) for s in ks_results],
        "kurt_ratio": round(kurt_ratio, 4),
        "frobenius": round(frob, 4),
    },
    "per_horizon_eff_rank": {str(k): {kk: round(vv, 4) for kk, vv in v.items()}
                             for k, v in per_h_results.items()},
    "spread_h1": round(float(mean_spread_per_h[0]), 6),
    "spread_h30": round(float(mean_spread_per_h[29]), 6),
    "cells_with_growing_spread": int(n_growing),
    "eigenvalues_gen_top5": [round(float(v), 4) for v in gen_eigvals[:5]],
    "eigenvalues_gt_top5": [round(float(v), 4) for v in gt_eigvals[:5]],
    "all_claims_verified": all_verified,
}

out_dir = Path("results/validations/2026-03-24")
with open(out_dir / "verification_results/152e_independent_eval.json", "w") as f:
    json.dump(results, f, indent=2)
with open(out_dir / "analysis/152e_reeval/full_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_dir}/verification_results/152e_independent_eval.json")
print(f"Analysis saved to {out_dir}/analysis/152e_reeval/full_metrics.json")
