#!/bin/bash
set -euo pipefail

# Verification script for Exp 148c noise_scale_head weight analysis
# Claim: "noise_scale_head learned 0.75x uniform suppression (CV=1.6%)"
# Run from repo root: bash results/validations/2026-03-23/scripts/148c_noise_scale_analysis.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"

MODEL_PATH="${REPO_ROOT}/models/backfill/afcrps_148c/best_model.pt"
OUTPUT_DIR="${REPO_ROOT}/results/validations/2026-03-23/analysis/148c_noise_scale"
OUTPUT_JSON="${OUTPUT_DIR}/noise_scale_analysis.json"

mkdir -p "${OUTPUT_DIR}"

echo "REPO_ROOT: ${REPO_ROOT}"
echo "MODEL_PATH: ${MODEL_PATH}"

python - "${MODEL_PATH}" "${OUTPUT_JSON}" "${OUTPUT_DIR}" <<'PYEOF'
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import os

model_path = sys.argv[1]
output_json = sys.argv[2]
output_dir  = sys.argv[3]

print("=" * 60)
print("148c noise_scale_head Analysis")
print("=" * 60)

# 1. Load checkpoint
ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
state = ckpt["model_state_dict"]

# 2. Extract weight and bias
weight = state["noise_scale_head.weight"]   # (25, 128)
bias   = state["noise_scale_head.bias"]     # (25,)
print(f"\nWeight shape: {weight.shape}")
print(f"Bias shape:   {bias.shape}")

# 3. Baseline scale = softplus(bias)  [no conditioning, zero input]
baseline_scale = F.softplus(bias)           # (25,)
bs_np = baseline_scale.detach().numpy()

bs_mean = float(bs_np.mean())
bs_std  = float(bs_np.std())
bs_cv   = float(bs_std / bs_mean * 100)    # percent
bs_min  = float(bs_np.min())
bs_max  = float(bs_np.max())

print(f"\n--- Baseline scale (softplus(bias), zero conditioning) ---")
print(f"  Mean : {bs_mean:.4f}")
print(f"  Std  : {bs_std:.4f}")
print(f"  CV   : {bs_cv:.2f}%")
print(f"  Min  : {bs_min:.4f}")
print(f"  Max  : {bs_max:.4f}")
print(f"  Values: {np.round(bs_np, 4).tolist()}")

# 4. Weight matrix statistics
w_np = weight.detach().numpy()
w_norm      = float(np.linalg.norm(w_np))      # Frobenius norm
w_max_abs   = float(np.abs(w_np).max())

print(f"\n--- Weight matrix statistics ---")
print(f"  Frobenius norm  : {w_norm:.4f}")
print(f"  Max abs weight  : {w_max_abs:.4f}")

# 5. Simulate 1000 random conditioning vectors (standard normal, dim=128)
torch.manual_seed(42)
n_sim = 1000
cond_dim = weight.shape[1]   # 128
z = torch.randn(n_sim, cond_dim)                     # (1000, 128)
logits = z @ weight.T + bias.unsqueeze(0)            # (1000, 25)
scales = F.softplus(logits)                          # (1000, 25)

scales_np = scales.detach().numpy()

# per-cell mean/std over simulation runs
cell_mean = scales_np.mean(axis=0)   # (25,)
cell_std  = scales_np.std(axis=0)    # (25,)

# global (all cells, all samples)
global_mean  = float(scales_np.mean())
global_std   = float(scales_np.std())
global_cv    = float(scales_np.std() / scales_np.mean() * 100)
global_min   = float(scales_np.min())
global_max   = float(scales_np.max())
global_p5    = float(np.percentile(scales_np, 5))
global_p95   = float(np.percentile(scales_np, 95))

print(f"\n--- Simulated full noise_scale (n=1000 random cond vectors) ---")
print(f"  Global mean : {global_mean:.4f}")
print(f"  Global std  : {global_std:.4f}")
print(f"  Global CV   : {global_cv:.2f}%")
print(f"  Global min  : {global_min:.4f}")
print(f"  Global max  : {global_max:.4f}")
print(f"  p5 / p95    : {global_p5:.4f} / {global_p95:.4f}")

# 6. Claim verification flags
CLAIM_MEAN_LOW  = 0.743   # 0.755 - 0.012
CLAIM_MEAN_HIGH = 0.767   # 0.755 + 0.012
CLAIM_CV_MAX    = 5.0
CLAIM_WEIGHT_NORM_LOW  = 14.0
CLAIM_WEIGHT_NORM_HIGH = 17.0

claim_mean_ok   = CLAIM_MEAN_LOW  <= bs_mean <= CLAIM_MEAN_HIGH
claim_cv_ok     = bs_cv < CLAIM_CV_MAX
claim_wnorm_ok  = CLAIM_WEIGHT_NORM_LOW <= w_norm <= CLAIM_WEIGHT_NORM_HIGH

print(f"\n--- Claim verification ---")
print(f"  Claim: baseline scale ~ 0.755 +/- 0.012  => mean={bs_mean:.4f}  PASS={claim_mean_ok}")
print(f"  Claim: CV < 5%                           => CV={bs_cv:.2f}%     PASS={claim_cv_ok}")
print(f"  Claim: weight norm ~ 15.57               => norm={w_norm:.4f}   PASS={claim_wnorm_ok}")
overall_pass = claim_mean_ok and claim_cv_ok and claim_wnorm_ok
print(f"  Overall PASS: {overall_pass}")

# 7. Save results
results = {
    "model_path": model_path,
    "experiment": "148c",
    "claim": "noise_scale_head learned 0.75x uniform suppression (CV=1.6%)",
    "weight_shape": list(weight.shape),
    "bias_shape": list(bias.shape),
    "baseline_scale": {
        "description": "softplus(bias) -- scale with zero conditioning input",
        "mean": bs_mean,
        "std":  bs_std,
        "cv_pct": bs_cv,
        "min": bs_min,
        "max": bs_max,
        "per_cell": np.round(bs_np, 6).tolist()
    },
    "weight_stats": {
        "frobenius_norm": w_norm,
        "max_abs_weight": w_max_abs
    },
    "simulated_full_scale": {
        "description": "softplus(W*z + b), z~N(0,I), n=1000 samples, seed=42",
        "n_sim": n_sim,
        "cond_dim": cond_dim,
        "global_mean": global_mean,
        "global_std": global_std,
        "global_cv_pct": global_cv,
        "global_min": global_min,
        "global_max": global_max,
        "p5": global_p5,
        "p95": global_p95,
        "per_cell_mean": np.round(cell_mean, 6).tolist(),
        "per_cell_std":  np.round(cell_std,  6).tolist()
    },
    "claim_verification": {
        "baseline_mean_in_range_743_to_767": claim_mean_ok,
        "baseline_cv_below_5pct": claim_cv_ok,
        "weight_norm_in_range_14_to_17": claim_wnorm_ok,
        "overall_pass": overall_pass
    }
}

os.makedirs(output_dir, exist_ok=True)
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_json}")
PYEOF

echo ""
echo "Analysis complete. JSON at: ${OUTPUT_JSON}"
