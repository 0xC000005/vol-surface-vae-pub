#!/usr/bin/env bash
# Verification script: per-cell noise modulation analysis for model 149c
# Contract: runs analysis and saves results to verification_results/149c_percell_scale.json

set -e

REPO=/home/max/Documents/vol-surface-vae-pub
cd "$REPO"

PYTHONPATH=. python3 - <<'PYEOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

MODEL_PATH = "models/backfill/afcrps_149c/best_model.pt"
OUTPUT_JSON = "results/validations/2026-03-23/verification_results/149c_percell_scale.json"
ANALYSIS_DIR = "results/validations/2026-03-23/analysis/149c_percell_scale"

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── 1. Load checkpoint ──────────────────────────────────────────────────────
print("Loading checkpoint...")
ck = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
state = ck["model_state_dict"]
cfg   = ck["config"]

noise_dim      = cfg["noise_dim"]        # 32
noise_embed_dim = cfg["noise_embed_dim"] # 64

# ── 2. Extract weights ──────────────────────────────────────────────────────
# noise_embed_proj: Linear(32→64) + SiLU + Linear(64→32)
ne_w0 = state["frame_decoder.noise_embed_proj.0.weight"]   # (64, 32)
ne_b0 = state["frame_decoder.noise_embed_proj.0.bias"]     # (64,)
ne_w2 = state["frame_decoder.noise_embed_proj.2.weight"]   # (32, 64)
ne_b2 = state["frame_decoder.noise_embed_proj.2.bias"]     # (32,)

# percell_noise_scale: Linear(32→50) + SiLU + Linear(50→25)
pc_w0 = state["frame_decoder.percell_noise_scale.0.weight"]  # (50, 32)
pc_b0 = state["frame_decoder.percell_noise_scale.0.bias"]    # (50,)
pc_w2 = state["frame_decoder.percell_noise_scale.2.weight"]  # (25, 50)
pc_b2 = state["frame_decoder.percell_noise_scale.2.bias"]    # (25,)

print(f"noise_embed_proj:   ({ne_w0.shape}) → SiLU → ({ne_w2.shape})")
print(f"percell_noise_scale: ({pc_w0.shape}) → SiLU → ({pc_w2.shape})")

# ── 3. Baseline scale: softplus(last_bias) at zero activation ───────────────
# At z=0 input, noise_embed = ne_b2 (after full MLP forward pass is not zero,
# but the CONCEPTUAL baseline at bias=0 activation is just softplus(pc_b2))
baseline_scale = F.softplus(pc_b2).numpy()
print(f"\nBaseline scale (softplus(last_bias)):")
print(f"  Mean over 25 cells: {baseline_scale.mean():.4f}")
print(f"  Min:  {baseline_scale.min():.4f}, Max: {baseline_scale.max():.4f}")

# ── 4. Monte Carlo simulation: 10000 z ~ N(0, I_32) ─────────────────────────
torch.manual_seed(42)
N = 10000
z = torch.randn(N, noise_dim)  # (10000, 32)

# Forward through noise_embed_proj
with torch.no_grad():
    h = F.linear(z, ne_w0, ne_b0)         # (N, 64)
    h = F.silu(h)
    noise_embed = F.linear(h, ne_w2, ne_b2)  # (N, 32)

    # Forward through percell_noise_scale
    h2 = F.linear(noise_embed, pc_w0, pc_b0)  # (N, 50)
    h2 = F.silu(h2)
    logits = F.linear(h2, pc_w2, pc_b2)        # (N, 25)
    scales = F.softplus(logits)                  # (N, 25)

scales_np = scales.numpy()  # (10000, 25)
print(f"\nMonte Carlo scales shape: {scales_np.shape}")
print(f"Global stats: mean={scales_np.mean():.4f}, std={scales_np.std():.4f}")

# ── 5. Per-cell statistics ───────────────────────────────────────────────────
percentiles = [5, 25, 50, 75, 95]
per_cell_stats = {}
for cell_idx in range(25):
    row = int(cell_idx // 5)
    col = int(cell_idx % 5)
    cell_scales = scales_np[:, cell_idx]
    pctls = np.percentile(cell_scales, percentiles).tolist()
    per_cell_stats[f"cell_{cell_idx:02d}_r{row}c{col}"] = {
        "mean":    float(cell_scales.mean()),
        "std":     float(cell_scales.std()),
        "P5":      pctls[0],
        "P25":     pctls[1],
        "P50":     pctls[2],
        "P75":     pctls[3],
        "P95":     pctls[4],
        "baseline_softplus_bias": float(baseline_scale[cell_idx]),
    }

# Print summary table
print(f"\n{'Cell':>6} | {'Mean':>6} | {'Std':>5} | {'P5':>5} | {'P25':>5} | {'P50':>5} | {'P75':>5} | {'P95':>5} | {'Baseline':>8}")
print("-" * 75)
for cell_idx in range(25):
    row = int(cell_idx // 5)
    col = int(cell_idx % 5)
    key = f"cell_{cell_idx:02d}_r{row}c{col}"
    s = per_cell_stats[key]
    print(f"r{row}c{col:>2}  | {s['mean']:>6.3f} | {s['std']:>5.3f} | {s['P5']:>5.3f} | "
          f"{s['P25']:>5.3f} | {s['P50']:>5.3f} | {s['P75']:>5.3f} | {s['P95']:>5.3f} | "
          f"{s['baseline_softplus_bias']:>8.4f}")

# ── 6. Key question: cells with P95 > 3.0 ───────────────────────────────────
high_p95_cells = {k: v for k, v in per_cell_stats.items() if v["P95"] > 3.0}
print(f"\n--- KEY QUESTION: Cells with P95 > 3.0 (fat tail injectors) ---")
if high_p95_cells:
    for k, v in sorted(high_p95_cells.items(), key=lambda x: -x[1]["P95"]):
        print(f"  {k}: P95={v['P95']:.3f}, P50={v['P50']:.3f}, mean={v['mean']:.3f}")
else:
    print("  None — no cells with P95 > 3.0")

# ── 7. Fraction of draws above thresholds ───────────────────────────────────
thresholds = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
threshold_stats = {}
print(f"\n--- Fraction of (noise_draw, cell) pairs with scale > threshold ---")
for thr in thresholds:
    frac_any_cell = float((scales_np > thr).any(axis=1).mean())     # at least 1 cell
    frac_all_cells = float((scales_np > thr).all(axis=1).mean())    # all cells
    frac_elementwise = float((scales_np > thr).mean())               # any pair
    threshold_stats[f"gt_{thr}"] = {
        "frac_any_cell": frac_any_cell,
        "frac_all_cells": frac_all_cells,
        "frac_elementwise": frac_elementwise,
    }
    print(f"  > {thr:5.1f}:  any_cell={frac_any_cell:.4f}  all_cells={frac_all_cells:.4f}  "
          f"elementwise={frac_elementwise:.4f}")

# ── 8. Noise embedding statistics (sanity check) ───────────────────────────
noise_embed_np = noise_embed.numpy()
print(f"\nNoise embed stats (input to percell MLP):")
print(f"  Mean: {noise_embed_np.mean():.4f}, Std: {noise_embed_np.std():.4f}")
print(f"  Min: {noise_embed_np.min():.4f}, Max: {noise_embed_np.max():.4f}")

# ── 9. Weight norm analysis ────────────────────────────────────────────────
w0_norms = pc_w0.norm(dim=1).numpy()   # per-hidden-neuron (50,)
w2_norms = pc_w2.norm(dim=1).numpy()   # per-output-cell (25,)
print(f"\nWeight norms:")
print(f"  Layer 0 (→50) row norms: mean={w0_norms.mean():.4f}, max={w0_norms.max():.4f}")
print(f"  Layer 2 (→25) row norms: mean={w2_norms.mean():.4f}, max={w2_norms.max():.4f}")
print(f"  Last bias range: [{pc_b2.min().item():.4f}, {pc_b2.max().item():.4f}]")
print(f"  Last bias mean: {pc_b2.mean().item():.4f} (softplus → {F.softplus(pc_b2).mean().item():.4f})")

# ── 10. Comparison note: 146b ──────────────────────────────────────────────
# 146b uses factor noise but has no noise_scale_cond or percell_noise_scale
# (ar_noise_scale_cond=False, no per-cell learned scale).
# 146b relies on ar_factor_noise for diversity.
print("\n--- Comparison: 146b ---")
print("146b has NO percell_noise_scale (ar_noise_scale_cond=False)")
print("146b uses factor noise (ar_factor_noise>0) for diversity injection")
print("149c introduces percell_noise_scale as a learned per-cell heteroscedasticity")

# ── 11. Save full JSON ──────────────────────────────────────────────────────
results = {
    "model": "149c",
    "model_path": MODEL_PATH,
    "architecture": {
        "noise_dim": noise_dim,
        "noise_embed_dim": noise_embed_dim,
        "percell_noise_scale": "Linear(32→50) + SiLU + Linear(50→25) → softplus",
        "noise_embed_proj": "Linear(32→64) + SiLU + Linear(64→32)",
    },
    "baseline_scale": {
        "description": "softplus(last_layer_bias) per cell — scale at zero activation",
        "values": baseline_scale.tolist(),
        "mean": float(baseline_scale.mean()),
        "min":  float(baseline_scale.min()),
        "max":  float(baseline_scale.max()),
        "note": "Init target was softplus(0.541)≈1.0; deviation from 1.0 shows learned drift",
    },
    "monte_carlo": {
        "n_samples": N,
        "input": "z ~ N(0, I_32), passed through noise_embed_proj",
        "global_mean":  float(scales_np.mean()),
        "global_std":   float(scales_np.std()),
        "global_P5":    float(np.percentile(scales_np, 5)),
        "global_P50":   float(np.percentile(scales_np, 50)),
        "global_P95":   float(np.percentile(scales_np, 95)),
        "global_max":   float(scales_np.max()),
    },
    "per_cell_stats": per_cell_stats,
    "high_p95_cells": high_p95_cells,
    "threshold_stats": threshold_stats,
    "weight_norms": {
        "layer0_row_norm_mean": float(w0_norms.mean()),
        "layer0_row_norm_max":  float(w0_norms.max()),
        "layer2_row_norm_mean": float(w2_norms.mean()),
        "layer2_row_norm_max":  float(w2_norms.max()),
        "last_bias_min": float(pc_b2.min().item()),
        "last_bias_max": float(pc_b2.max().item()),
        "last_bias_mean": float(pc_b2.mean().item()),
    },
    "comparison_146b": {
        "has_percell_noise_scale": False,
        "has_noise_scale_cond": False,
        "diversity_source": "ar_factor_noise (factor structure, not per-cell scale)",
        "note": "146b eff_rank +54% came from factor noise, not heteroscedastic scale",
    },
    "interpretation": {
        "fat_tail_risk": len(high_p95_cells) > 0,
        "fat_tail_cells_count": len(high_p95_cells),
        "scale_range_summary": {
            f"frac_gt_{thr}": threshold_stats[f"gt_{thr}"]["frac_elementwise"]
            for thr in thresholds
        },
    },
}

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved verification JSON → {OUTPUT_JSON}")

# Also save numpy array for further analysis
np.save(f"{ANALYSIS_DIR}/scales_10000x25.npy", scales_np)
np.save(f"{ANALYSIS_DIR}/noise_embed_10000x32.npy", noise_embed_np)
print(f"Saved numpy arrays → {ANALYSIS_DIR}/")

PYEOF

echo "Script completed successfully."
