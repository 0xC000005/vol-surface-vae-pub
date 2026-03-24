#!/usr/bin/env bash
# 152b PCA Alignment Verification
# Verifies that GT-aligned cross-cell correlation structure transfers from
# unconditional flow matching (152a) to conditional AR flow matching (152b).
#
# Usage: cd /home/max/Documents/vol-surface-vae-pub && bash results/validations/2026-03-24/scripts/152b_pca_alignment.sh

set -euo pipefail
cd /home/max/Documents/vol-surface-vae-pub

OUTPUT_DIR="results/validations/2026-03-24/analysis/152b_pca"
RESULT_JSON="results/validations/2026-03-24/verification_results/152b_pca_alignment.json"
MODEL_PATH="models/backfill/flow_152b/best_model.pt"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$RESULT_JSON")"

PYTHONPATH=. python -u - "$MODEL_PATH" "$OUTPUT_DIR" "$RESULT_JSON" <<'PYTHON_SCRIPT'
import sys
import json
import time
import numpy as np
import torch

MODEL_PATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
RESULT_JSON = sys.argv[3]

print("=" * 60)
print("152b PCA Alignment Verification")
print("=" * 60)

# ── Configuration ──
MAX_BATCHES = 10
N_SAMPLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PCS = 5  # number of principal components to compare

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Load model ──
print(f"\nLoading model from {MODEL_PATH}...")
from experiments.backfill.block_ar.eval_ar_flow import load_ar_flow_model
from diffusion.block_ar.single_pass_ar import denormalize_iv, normalize_iv
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from torch.utils.data import DataLoader

model, ckpt = load_ar_flow_model(MODEL_PATH, DEVICE)
print(f"  Epoch: {ckpt.get('epoch', '?')}")
print(f"  Val loss: {ckpt.get('val_loss', '?'):.4f}")

# ── Load test data ──
config = get_default_config()
data = np.load(config.data_path)
surfaces = data["surface"]

test_dataset = VolSurfaceDataset(
    surfaces, config.history_len, config.future_len,
    start_idx=config.test_start,
)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
print(f"  Test windows: {len(test_dataset)}")

# ── Generate samples ──
print(f"\nGenerating samples (max_batches={MAX_BATCHES}, n_samples={N_SAMPLES})...")
t0 = time.time()

all_samples = []
all_gt = []
all_history = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= MAX_BATCHES:
            break
        history = batch["history"].to(DEVICE)
        future_gt = denormalize_iv(batch["future"].to(DEVICE))

        samples = model.sample_batched(history, n_samples=N_SAMPLES)
        history_denorm = denormalize_iv(history)

        all_samples.append(samples.cpu().numpy())
        all_gt.append(future_gt.cpu().numpy())
        all_history.append(history_denorm.cpu().numpy())

        print(f"  Batch {batch_idx+1}/{MAX_BATCHES}: samples {samples.shape}")

cond_samples = np.concatenate(all_samples)  # (N, 50, 30, 5, 5)
ground_truth = np.concatenate(all_gt)       # (N, 30, 5, 5)
history_arr = np.concatenate(all_history)    # (N, 30, 5, 5)

gen_time = time.time() - t0
print(f"\n  Generated: {cond_samples.shape} in {gen_time:.1f}s")
print(f"  GT shape: {ground_truth.shape}")

# ── Compute daily changes ──
print("\n--- Computing Daily Changes ---")

# GT daily changes: use test data directly
# Concatenate history last frame + future to get continuous path, then diff
gt_paths = np.concatenate([history_arr[:, -1:, :, :], ground_truth], axis=1)  # (N, 31, 5, 5)
gt_changes = np.diff(gt_paths, axis=1)  # (N, 30, 5, 5)
gt_changes_flat = gt_changes.reshape(-1, 25)  # (N*30, 25)
print(f"  GT changes: {gt_changes_flat.shape}")

# Generated daily changes: for each ensemble member, compute changes
# cond_samples: (N, K, 30, 5, 5)
N, K, T, H, W = cond_samples.shape

# Build paths for each member: history_last + sample trajectory
history_last = history_arr[:, -1:, :, :]  # (N, 1, 5, 5)
history_last_exp = np.expand_dims(history_last, axis=1)  # (N, 1, 1, 5, 5)
history_last_exp = np.broadcast_to(history_last_exp, (N, K, 1, H, W))

gen_paths = np.concatenate([history_last_exp, cond_samples], axis=2)  # (N, K, 31, 5, 5)
gen_changes = np.diff(gen_paths, axis=2)  # (N, K, 30, 5, 5)
gen_changes_flat = gen_changes.reshape(-1, 25)  # (N*K*30, 25)
print(f"  Generated changes: {gen_changes_flat.shape}")

# ── Compute cross-cell correlation matrices ──
print("\n--- Computing Cross-Cell Correlation Matrices ---")

def compute_corr_matrix(data):
    """Compute 25x25 correlation matrix from (n_obs, 25) data."""
    # Remove mean
    data_centered = data - data.mean(axis=0, keepdims=True)
    # Covariance
    cov = np.cov(data_centered, rowvar=False)
    # Correlation
    std = np.sqrt(np.diag(cov))
    std[std < 1e-10] = 1e-10
    corr = cov / np.outer(std, std)
    return corr, cov

gt_corr, gt_cov = compute_corr_matrix(gt_changes_flat)
gen_corr, gen_cov = compute_corr_matrix(gen_changes_flat)

print(f"  GT correlation: mean={gt_corr.mean():.4f}, diag_mean={np.diag(gt_corr).mean():.4f}")
print(f"  Gen correlation: mean={gen_corr.mean():.4f}, diag_mean={np.diag(gen_corr).mean():.4f}")

# ── PCA / Eigendecomposition ──
print("\n--- PCA / Eigendecomposition ---")

gt_eigenvalues, gt_eigenvectors = np.linalg.eigh(gt_cov)
gen_eigenvalues, gen_eigenvectors = np.linalg.eigh(gen_cov)

# Sort descending
gt_idx = np.argsort(gt_eigenvalues)[::-1]
gen_idx = np.argsort(gen_eigenvalues)[::-1]
gt_eigenvalues = gt_eigenvalues[gt_idx]
gt_eigenvectors = gt_eigenvectors[:, gt_idx]
gen_eigenvalues = gen_eigenvalues[gen_idx]
gen_eigenvectors = gen_eigenvectors[:, gen_idx]

# Normalize eigenvalues (proportion of variance)
gt_var_explained = gt_eigenvalues / gt_eigenvalues.sum()
gen_var_explained = gen_eigenvalues / gen_eigenvalues.sum()

print(f"\n  Top {N_PCS} eigenvalues (proportion of variance):")
print(f"  {'PC':<5} {'GT':>10} {'Gen':>10} {'Ratio':>10}")
for i in range(N_PCS):
    ratio = gen_var_explained[i] / gt_var_explained[i] if gt_var_explained[i] > 0 else 0
    print(f"  PC{i+1:<3} {gt_var_explained[i]:>10.4f} {gen_var_explained[i]:>10.4f} {ratio:>10.3f}")

# ── PC Alignment (dot product of eigenvectors) ──
print(f"\n--- PC Alignment (|dot product|) ---")
alignments = []
for i in range(N_PCS):
    alignment = abs(np.dot(gt_eigenvectors[:, i], gen_eigenvectors[:, i]))
    alignments.append(alignment)
    label = "GOOD" if alignment > 0.9 else ("MODERATE" if alignment > 0.7 else "POOR")
    print(f"  PC{i+1}: {alignment:.4f} [{label}]")

# ── Effective Rank ──
print("\n--- Effective Rank ---")

def effective_rank(eigenvalues):
    """Shannon entropy-based effective rank."""
    eigenvalues = eigenvalues[eigenvalues > 0]
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

gt_eff_rank = effective_rank(gt_eigenvalues)
gen_eff_rank = effective_rank(gen_eigenvalues)
eff_rank_ratio = gen_eff_rank / gt_eff_rank

print(f"  GT effective rank: {gt_eff_rank:.3f}")
print(f"  Gen effective rank: {gen_eff_rank:.3f}")
print(f"  Ratio (gen/GT): {eff_rank_ratio:.3f}")

# ── Frobenius distance between correlation matrices ──
frob_dist = np.linalg.norm(gen_corr - gt_corr, 'fro')
frob_norm_gt = np.linalg.norm(gt_corr, 'fro')
frob_relative = frob_dist / frob_norm_gt
print(f"\n--- Frobenius Distance ---")
print(f"  Absolute: {frob_dist:.4f}")
print(f"  Relative (dist/||GT||): {frob_relative:.4f}")

# ── Correlation ratio (mean off-diagonal gen / mean off-diagonal GT) ──
mask = ~np.eye(25, dtype=bool)
gt_mean_offdiag = gt_corr[mask].mean()
gen_mean_offdiag = gen_corr[mask].mean()
corr_ratio = gen_mean_offdiag / gt_mean_offdiag if abs(gt_mean_offdiag) > 1e-6 else 0
print(f"\n--- Correlation Ratio ---")
print(f"  GT mean off-diagonal: {gt_mean_offdiag:.4f}")
print(f"  Gen mean off-diagonal: {gen_mean_offdiag:.4f}")
print(f"  Ratio: {corr_ratio:.4f}")

# ── Summary & Comparison ──
print("\n" + "=" * 60)
print("COMPARISON WITH PRIOR RESULTS")
print("=" * 60)

# H1b repulsive loss reference values
h1b_pc1 = 0.43
h1b_eff_rank_ratio = 1.90  # from Exp 151c
# 152a unconditional flow matching reference
flow_152a_pc1 = 1.000
flow_152a_pc2 = 0.994

print(f"\n  {'Metric':<25} {'H1b (151c)':<15} {'152a (uncond)':<15} {'152b (cond AR)':<15}")
print(f"  {'-'*70}")
print(f"  {'PC1 alignment':<25} {h1b_pc1:<15.3f} {flow_152a_pc1:<15.3f} {alignments[0]:<15.4f}")
print(f"  {'PC2 alignment':<25} {'N/A':<15} {flow_152a_pc2:<15.3f} {alignments[1]:<15.4f}")
print(f"  {'Eff rank ratio':<25} {h1b_eff_rank_ratio:<15.2f} {'~1.0':<15} {eff_rank_ratio:<15.3f}")

# ── Determine pass/fail ──
# PC1 > 0.9 means GT-aligned structure transferred
pc1_pass = alignments[0] > 0.9
pc2_pass = alignments[1] > 0.9
eff_rank_pass = 0.5 <= eff_rank_ratio <= 2.0
better_than_h1b = alignments[0] > h1b_pc1

overall_pass = pc1_pass and better_than_h1b

print(f"\n--- VERDICT ---")
print(f"  PC1 > 0.9:           {'PASS' if pc1_pass else 'FAIL'} ({alignments[0]:.4f})")
print(f"  PC2 > 0.9:           {'PASS' if pc2_pass else 'FAIL'} ({alignments[1]:.4f})")
print(f"  Eff rank ratio OK:   {'PASS' if eff_rank_pass else 'FAIL'} ({eff_rank_ratio:.3f})")
print(f"  Better than H1b:     {'PASS' if better_than_h1b else 'FAIL'} ({alignments[0]:.4f} vs {h1b_pc1})")
print(f"  OVERALL:             {'PASS' if overall_pass else 'FAIL'}")

# ── Save raw data ──
np.savez(
    f"{OUTPUT_DIR}/pca_data.npz",
    gt_eigenvalues=gt_eigenvalues,
    gen_eigenvalues=gen_eigenvalues,
    gt_eigenvectors=gt_eigenvectors,
    gen_eigenvectors=gen_eigenvectors,
    gt_corr=gt_corr,
    gen_corr=gen_corr,
    gt_var_explained=gt_var_explained,
    gen_var_explained=gen_var_explained,
    alignments=np.array(alignments),
)
print(f"\n  Raw data saved to {OUTPUT_DIR}/pca_data.npz")

# ── Save JSON result ──
result = {
    "verification": "152b_pca_alignment",
    "model_path": MODEL_PATH,
    "model_epoch": int(ckpt.get("epoch", -1)),
    "model_val_loss": float(ckpt.get("val_loss", -1)),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "config": {
        "max_batches": MAX_BATCHES,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "device": DEVICE,
        "n_test_windows": int(cond_samples.shape[0]),
        "n_generated_changes": int(gen_changes_flat.shape[0]),
        "n_gt_changes": int(gt_changes_flat.shape[0]),
    },
    "eigenvalues": {
        "gt_top5": [float(v) for v in gt_eigenvalues[:N_PCS]],
        "gen_top5": [float(v) for v in gen_eigenvalues[:N_PCS]],
        "gt_var_explained_top5": [float(v) for v in gt_var_explained[:N_PCS]],
        "gen_var_explained_top5": [float(v) for v in gen_var_explained[:N_PCS]],
    },
    "pc_alignments": {
        f"PC{i+1}": float(alignments[i]) for i in range(N_PCS)
    },
    "effective_rank": {
        "gt": float(gt_eff_rank),
        "gen": float(gen_eff_rank),
        "ratio": float(eff_rank_ratio),
    },
    "correlation": {
        "gt_mean_offdiag": float(gt_mean_offdiag),
        "gen_mean_offdiag": float(gen_mean_offdiag),
        "ratio": float(corr_ratio),
        "frobenius_distance": float(frob_dist),
        "frobenius_relative": float(frob_relative),
    },
    "comparison": {
        "h1b_repulsive_pc1": h1b_pc1,
        "h1b_eff_rank_ratio": h1b_eff_rank_ratio,
        "flow_152a_uncond_pc1": flow_152a_pc1,
        "flow_152a_uncond_pc2": flow_152a_pc2,
        "improvement_over_h1b": float(alignments[0] - h1b_pc1),
        "retention_from_152a": float(alignments[0] / flow_152a_pc1) if flow_152a_pc1 > 0 else 0,
    },
    "verdict": {
        "pc1_gt_0.9": bool(pc1_pass),
        "pc2_gt_0.9": bool(pc2_pass),
        "eff_rank_ratio_ok": bool(eff_rank_pass),
        "better_than_h1b": bool(better_than_h1b),
        "overall_pass": bool(overall_pass),
    },
    "claim": "The GT-aligned diversity from Stage 1 (152a) transfers to conditional AR forecasting (152b)",
    "evidence_summary": (
        f"PC1 alignment = {alignments[0]:.4f} "
        f"(vs H1b = {h1b_pc1}, 152a uncond = {flow_152a_pc1}). "
        f"Eff rank ratio = {eff_rank_ratio:.3f}. "
        f"Frobenius relative = {frob_relative:.4f}. "
        f"{'Claim SUPPORTED' if overall_pass else 'Claim NOT SUPPORTED'}."
    ),
}

with open(RESULT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"  Result JSON saved to {RESULT_JSON}")
print(f"\nDone.")
PYTHON_SCRIPT
