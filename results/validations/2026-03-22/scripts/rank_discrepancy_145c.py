#!/usr/bin/env python
"""145c Eff_rank Discrepancy Investigation.

Training reports eff_rank 3.82->4.36 (Gram matrix of K ensemble members).
Test suite reports rank_ratio = 0.319 (correlation matrix of 25 cells' daily changes).

These are DIFFERENT metrics measuring different things. This script computes both
and explains the discrepancy.
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
output_dir = Path("results/validations/2026-03-22/analysis/145c_rank_discrepancy")
output_dir.mkdir(parents=True, exist_ok=True)


def load_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        sp_cfg = {k: v for k, v in cfg.items()
                  if k in SinglePassConfig.__dataclass_fields__}
        sp_config = SinglePassConfig(**sp_cfg)
    else:
        sp_config = cfg
    model = SinglePassBlockAR(sp_config)
    model.load_state_dict(ckpt[key] if (key := "model_state_dict") in ckpt else ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model, ckpt


def eff_rank_entropy(eigvals):
    """Effective rank via eigenvalue entropy (Roy & Vetterli, 2007)."""
    eigvals = np.maximum(eigvals, 0)
    p = eigvals / (eigvals.sum() + 1e-10)
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def compute_gram_eff_rank(samples):
    """Compute eff_rank of K x K Gram matrix of ensemble members.

    This is what the training DPP loss measures.
    samples: (B, K, T, H, W)
    Returns: mean eff_rank across batch
    """
    B, K, T, H, W = samples.shape
    flat = samples.reshape(B, K, -1)  # (B, K, T*H*W)
    flat_centered = flat - flat.mean(axis=1, keepdims=True)  # (B, K, D)
    D = flat_centered.shape[2]

    eff_ranks = []
    for b in range(B):
        # K x K Gram matrix: G = X X^T / D
        gram = flat_centered[b] @ flat_centered[b].T / D  # (K, K)
        gram += 1e-6 * np.eye(K)
        eigvals = np.linalg.eigvalsh(gram)[::-1]
        eff_ranks.append(eff_rank_entropy(eigvals))

    return float(np.mean(eff_ranks)), eff_ranks


def compute_daily_change_corr_eff_rank(samples, ground_truth):
    """Compute eff_rank of 25x25 correlation matrix of daily changes across cells.

    This is what the test suite measures in Suite 9.
    samples: (N, K, T, H, W) — one sample trajectory per window
    ground_truth: (N, T, H, W)
    Returns: gt_eff_rank, gen_eff_rank, correlation matrices
    """
    N, K, T, H, W = samples.shape
    n_cells = H * W

    # GT correlation matrix
    gt_changes = np.diff(ground_truth, axis=1)  # (N, T-1, H, W)
    gt_flat = gt_changes.reshape(-1, n_cells)  # (N*(T-1), 25)
    gt_corr = np.corrcoef(gt_flat.T)  # (25, 25)

    # Gen correlation matrix (average over multiple sample indices)
    n_corr_samples = min(5, K)
    gen_corr_matrices = []
    for s in range(n_corr_samples):
        gen_changes = np.diff(samples[:, s], axis=1)  # (N, T-1, H, W)
        gen_flat = gen_changes.reshape(-1, n_cells)  # (N*(T-1), 25)
        gen_corr = np.corrcoef(gen_flat.T)  # (25, 25)
        gen_corr_matrices.append(gen_corr)
    gen_corr_avg = np.mean(gen_corr_matrices, axis=0)

    gt_eigvals = np.linalg.eigvalsh(gt_corr)[::-1]
    gen_eigvals = np.linalg.eigvalsh(gen_corr_avg)[::-1]

    gt_er = eff_rank_entropy(gt_eigvals)
    gen_er = eff_rank_entropy(gen_eigvals)

    # Also compute participation ratio for comparison
    gt_eigvals_pos = np.maximum(gt_eigvals, 0)
    gen_eigvals_pos = np.maximum(gen_eigvals, 0)
    gt_pr = float((gt_eigvals_pos.sum()**2) / (np.sum(gt_eigvals_pos**2) + 1e-12))
    gen_pr = float((gen_eigvals_pos.sum()**2) / (np.sum(gen_eigvals_pos**2) + 1e-12))

    return {
        "gt_eff_rank": gt_er,
        "gen_eff_rank": gen_er,
        "rank_ratio": gen_er / gt_er if gt_er > 1e-6 else float('inf'),
        "gt_participation_ratio": gt_pr,
        "gen_participation_ratio": gen_pr,
        "gt_pc1_var": float(gt_eigvals_pos[0] / (gt_eigvals_pos.sum() + 1e-10)),
        "gen_pc1_var": float(gen_eigvals_pos[0] / (gen_eigvals_pos.sum() + 1e-10)),
        "gt_eigvals_top5": gt_eigvals[:5].tolist(),
        "gen_eigvals_top5": gen_eigvals[:5].tolist(),
        "gt_corr": gt_corr,
        "gen_corr": gen_corr_avg,
    }


# === Load both models ===
print("=" * 60)
print("145c Eff_rank Discrepancy Investigation")
print("=" * 60)

models_to_compare = {
    "144b": "models/backfill/afcrps_144b/best_model.pt",
    "145c": "models/backfill/afcrps_145c/best_model.pt",
}

# Load data
config = get_default_config()
data = np.load(config.data_path)
surfaces = data["surface"]

test_dataset = VolSurfaceDataset(
    surfaces, config.history_len, config.future_len,
    start_idx=config.test_start,
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

n_samples = 50
max_batches = 20

results = {}

for model_name, model_path in models_to_compare.items():
    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")

    model, ckpt = load_model(model_path)

    # Generate samples
    print(f"Generating samples (n_samples={n_samples}, max_batches={max_batches})...")
    all_samples = []
    all_gt = []

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
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1}/{max_batches}")

    cond_samples = np.concatenate(all_samples, axis=0)  # (N, S, T, 5, 5)
    ground_truth = np.concatenate(all_gt, axis=0)  # (N, T, 5, 5)
    N, S, T = cond_samples.shape[:3]
    print(f"  N={N}, S={S}, T={T}")

    # === Metric A: Gram matrix eff_rank (what training DPP loss measures) ===
    # K x K Gram matrix of ensemble members
    print("\n  --- Metric A: Gram Matrix Eff_rank (Training Metric) ---")
    # Use first 8 members to match K=8 in training
    K_train = min(8, S)
    gram_er, gram_ers = compute_gram_eff_rank(cond_samples[:, :K_train])
    print(f"  Mean Gram eff_rank (K={K_train}): {gram_er:.3f}")
    print(f"  Distribution: min={min(gram_ers):.3f}, median={np.median(gram_ers):.3f}, max={max(gram_ers):.3f}")

    # === Metric B: Daily change correlation eff_rank (what test suite measures) ===
    print("\n  --- Metric B: Daily Change Correlation Eff_rank (Test Suite Metric) ---")
    corr_results = compute_daily_change_corr_eff_rank(cond_samples, ground_truth)
    print(f"  GT eff_rank: {corr_results['gt_eff_rank']:.3f}")
    print(f"  Gen eff_rank: {corr_results['gen_eff_rank']:.3f}")
    print(f"  Rank ratio: {corr_results['rank_ratio']:.3f}")
    print(f"  GT PC1 variance: {corr_results['gt_pc1_var']:.1%}")
    print(f"  Gen PC1 variance: {corr_results['gen_pc1_var']:.1%}")
    print(f"  GT participation ratio: {corr_results['gt_participation_ratio']:.3f}")
    print(f"  Gen participation ratio: {corr_results['gen_participation_ratio']:.3f}")
    print(f"  GT top-5 eigenvalues: {[f'{e:.3f}' for e in corr_results['gt_eigvals_top5']]}")
    print(f"  Gen top-5 eigenvalues: {[f'{e:.3f}' for e in corr_results['gen_eigvals_top5']]}")

    # === Metric C: Per-member daily change correlation matrix ===
    # For each ensemble member, what does the 25x25 correlation look like?
    print("\n  --- Metric C: Per-Member Correlation Structure ---")
    per_member_ranks = []
    for s in range(min(5, S)):
        gen_changes = np.diff(cond_samples[:, s], axis=1)
        gen_flat = gen_changes.reshape(-1, 25)
        gen_corr = np.corrcoef(gen_flat.T)
        eigvals = np.linalg.eigvalsh(gen_corr)[::-1]
        er = eff_rank_entropy(eigvals)
        per_member_ranks.append(er)
    print(f"  Per-member eff_ranks: {[f'{r:.3f}' for r in per_member_ranks]}")
    print(f"  Std across members: {np.std(per_member_ranks):.4f}")

    # === Metric D: Cross-member correlation of daily changes ===
    # Do different members produce similar daily changes?
    print("\n  --- Metric D: Cross-Member Daily Change Similarity ---")
    # For ATM cell (2,2), compute correlation between member pairs
    member_changes = np.diff(cond_samples[:, :8, :, 2, 2], axis=2)  # (N, K, T-1)
    member_flat = member_changes.reshape(8, -1)  # (K, N*(T-1))
    cross_member_corr = np.corrcoef(member_flat)
    mask = np.triu(np.ones((8, 8), dtype=bool), k=1)
    mean_cross_corr = float(cross_member_corr[mask].mean())
    print(f"  Mean cross-member correlation (ATM cell): {mean_cross_corr:.4f}")

    # For all cells
    all_cell_cross_corr = []
    for r in range(5):
        for c in range(5):
            mc = np.diff(cond_samples[:, :8, :, r, c], axis=2)  # (N, K, T-1)
            mf = mc.reshape(8, -1)
            cc = np.corrcoef(mf)
            all_cell_cross_corr.append(float(cc[mask].mean()))
    mean_all_cross = np.mean(all_cell_cross_corr)
    print(f"  Mean cross-member correlation (all cells): {mean_all_cross:.4f}")
    print(f"  This tells us how much diversity DPP actually created between members")

    results[model_name] = {
        "gram_eff_rank": {
            "mean": gram_er,
            "min": float(min(gram_ers)),
            "median": float(np.median(gram_ers)),
            "max": float(max(gram_ers)),
            "K": K_train,
        },
        "daily_change_corr": {
            "gt_eff_rank": corr_results["gt_eff_rank"],
            "gen_eff_rank": corr_results["gen_eff_rank"],
            "rank_ratio": corr_results["rank_ratio"],
            "gt_pc1_var": corr_results["gt_pc1_var"],
            "gen_pc1_var": corr_results["gen_pc1_var"],
            "gt_participation_ratio": corr_results["gt_participation_ratio"],
            "gen_participation_ratio": corr_results["gen_participation_ratio"],
            "gt_eigvals_top5": corr_results["gt_eigvals_top5"],
            "gen_eigvals_top5": corr_results["gen_eigvals_top5"],
        },
        "per_member_eff_ranks": per_member_ranks,
        "cross_member_corr_atm": mean_cross_corr,
        "cross_member_corr_all_cells": mean_all_cross,
    }

# === Comparison ===
print("\n" + "=" * 60)
print("COMPARISON: 144b vs 145c")
print("=" * 60)

print("\nMetric A: Gram Matrix Eff_rank (Training DPP metric)")
print(f"  144b: {results['144b']['gram_eff_rank']['mean']:.3f}")
print(f"  145c: {results['145c']['gram_eff_rank']['mean']:.3f}")
print(f"  Delta: {results['145c']['gram_eff_rank']['mean'] - results['144b']['gram_eff_rank']['mean']:+.3f}")

print("\nMetric B: Daily Change Corr Eff_rank (Test Suite metric)")
print(f"  144b: {results['144b']['daily_change_corr']['gen_eff_rank']:.3f} "
      f"(ratio={results['144b']['daily_change_corr']['rank_ratio']:.3f})")
print(f"  145c: {results['145c']['daily_change_corr']['gen_eff_rank']:.3f} "
      f"(ratio={results['145c']['daily_change_corr']['rank_ratio']:.3f})")
print(f"  Delta: {results['145c']['daily_change_corr']['gen_eff_rank'] - results['144b']['daily_change_corr']['gen_eff_rank']:+.3f}")

print("\nMetric D: Cross-Member Correlation")
print(f"  144b: {results['144b']['cross_member_corr_all_cells']:.4f}")
print(f"  145c: {results['145c']['cross_member_corr_all_cells']:.4f}")

print("\n" + "=" * 60)
print("EXPLANATION")
print("=" * 60)
print("""
The DPP loss operates on the K x K Gram matrix of ensemble MEMBERS.
It diversifies the K members relative to each other in the full
(T*H*W)-dimensional output space.

The test suite measures the 25x25 correlation matrix of daily CHANGES
across CELLS. This measures whether the model reproduces the factor
structure of the volatility surface (how cells co-move).

These are fundamentally different quantities:
- Gram eff_rank (Metric A): "How diverse are the K ensemble members?"
  Higher = members explore more orthogonal directions in output space.
- Corr eff_rank (Metric B): "How many independent factors drive cell co-movement?"
  Higher = more independent sources of variation across the 25 cells.

WHY DPP doesn't transfer:
The DPP loss makes members different from EACH OTHER, but each individual
member still has the same internal correlation structure (all 25 cells
still move together in rank-1 fashion). Diversifying members along the
dominant PC1 direction produces high Gram rank but doesn't change the
WITHIN-member cell correlation structure.

To fix Metric B, you need a loss that directly penalizes the cross-cell
correlation structure of individual members (e.g., variogram score, or
an explicit correlation matching loss).
""")

# Save results
with open(output_dir / "rank_discrepancy_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_dir / 'rank_discrepancy_analysis.json'}")
