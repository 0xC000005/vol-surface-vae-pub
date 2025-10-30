"""
Comprehensive tests to verify posterior collapse in VAE models.

Tests implemented:
1. KL Divergence Analysis - Check if KL is suspiciously low
2. Posterior Variance Statistics - Compare empirical vs N(0,1)
3. Active Units Test - Count dimensions actually used
4. Decoder Sensitivity Test - Measure output response to latent perturbations
5. Reconstruction-KL visualization
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from vae.cvae_with_mem_randomized import CVAEMemRand
from datetime import datetime

print("="*80)
print("POSTERIOR COLLAPSE VERIFICATION")
print("="*80)

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
CONTEXT_LEN = 5
MODELS = ["no_ex", "ex_no_loss", "ex_loss"]
MODEL_LABELS = {
    "no_ex": "No EX",
    "ex_no_loss": "EX No Loss",
    "ex_loss": "EX Loss"
}

# [1] Load empirical latents
print("\n[1/6] Loading empirical latents...")

latent_data = {}
for model_name in MODELS:
    latents_path = f"{BASE_MODEL_DIR}/{model_name}_empirical_latents.npz"
    latents = np.load(latents_path)

    z_mean_pool = latents["z_mean_pool"]  # (5817, 5)
    z_log_var_pool = latents["z_log_var_pool"]  # (5817, 5)

    latent_data[model_name] = {
        "z_mean": z_mean_pool,
        "z_log_var": z_log_var_pool
    }

    print(f"  {model_name}: {z_mean_pool.shape[0]} samples, {z_mean_pool.shape[1]} dims")

# [2] Test 1: KL Divergence Analysis
print("\n[2/6] Test 1: KL Divergence Analysis...")
print("\nFor N(0,1) prior: KL(q||p) = 0.5 * (μ² + σ² - log(σ²) - 1)")

kl_results = {}

for model_name in MODELS:
    z_mean = latent_data[model_name]["z_mean"]  # (N, D)
    z_log_var = latent_data[model_name]["z_log_var"]  # (N, D)

    # KL per sample per dimension
    kl_per_sample = 0.5 * (z_mean**2 + np.exp(z_log_var) - z_log_var - 1)  # (N, D)

    # Average KL per dimension across dataset
    kl_per_dim = np.mean(kl_per_sample, axis=0)  # (D,)
    total_kl = np.sum(kl_per_dim)

    kl_results[model_name] = {
        "kl_per_dim": kl_per_dim,
        "total_kl": total_kl,
        "kl_per_sample": kl_per_sample
    }

    print(f"\n  {MODEL_LABELS[model_name]}:")
    print(f"    KL per dimension: {kl_per_dim}")
    print(f"    Total KL: {total_kl:.3f} (expected ~{z_mean.shape[1]/2:.1f} for well-calibrated)")
    print(f"    Average KL per dimension: {total_kl/z_mean.shape[1]:.3f} (expected ~0.5)")

    if total_kl < 0.5:
        print(f"    ⚠ WARNING: Very low KL suggests severe posterior collapse!")
    elif total_kl < z_mean.shape[1] * 0.3:
        print(f"    ⚠ WARNING: Low KL suggests posterior collapse")

# [3] Test 2: Posterior Variance Statistics
print("\n[3/6] Test 2: Posterior Variance Statistics...")
print("Expected for N(0,1): mean(z_log_var) ≈ 0, exp(z_log_var) ≈ 1")

variance_results = {}

for model_name in MODELS:
    z_log_var = latent_data[model_name]["z_log_var"]
    variance = np.exp(z_log_var)

    variance_results[model_name] = {
        "mean_log_var": np.mean(z_log_var),
        "std_log_var": np.std(z_log_var),
        "mean_variance": np.mean(variance),
        "std_variance": np.std(variance)
    }

    print(f"\n  {MODEL_LABELS[model_name]}:")
    print(f"    mean(z_log_var): {np.mean(z_log_var):.3f} (expected ~0)")
    print(f"    std(z_log_var): {np.std(z_log_var):.3f}")
    print(f"    mean(variance): {np.mean(variance):.3f} (expected ~1)")
    print(f"    std(variance): {np.std(variance):.3f}")

    if np.mean(z_log_var) < -1.0:
        print(f"    ⚠ WARNING: Very low log variance (< -1) suggests collapse")
        print(f"      Actual variance ≈ {np.mean(variance):.4f} << 1.0")

# [4] Test 3: Active Units Test
print("\n[4/6] Test 3: Active Units Test...")
print("Active dimension = var(z_mean) > 0.01 across dataset")

active_units_results = {}

for model_name in MODELS:
    z_mean = latent_data[model_name]["z_mean"]

    # Variance of z_mean across dataset for each dimension
    z_mean_variance = np.var(z_mean, axis=0)  # (D,)

    # Count active dimensions (arbitrary threshold)
    threshold = 0.01
    active_dims = np.sum(z_mean_variance > threshold)

    active_units_results[model_name] = {
        "z_mean_variance": z_mean_variance,
        "active_dims": active_dims,
        "total_dims": len(z_mean_variance)
    }

    print(f"\n  {MODEL_LABELS[model_name]}:")
    print(f"    Variance of z_mean per dimension: {z_mean_variance}")
    print(f"    Active dimensions: {active_dims}/{len(z_mean_variance)}")

    if active_dims < len(z_mean_variance):
        print(f"    ⚠ WARNING: {len(z_mean_variance) - active_dims} dimensions inactive (collapsed)")

# [5] Test 4: Decoder Sensitivity Test (MOST IMPORTANT!)
print("\n[5/6] Test 4: Decoder Sensitivity Test...")
print("Perturb z by ±1σ, ±2σ, ±3σ and measure output change")

# Load data for context
data = np.load("data/vol_surface_with_ret.npz")
surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.stack([ret_data, skew_data, slope_data], axis=-1)

dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
dates_full = pd.to_datetime(dates_df["date"].values)

# Select test days: 3 normal, 3 crisis
test_days_config = [
    {"idx": 1000, "label": "Normal 1 (2004)"},
    {"idx": 1500, "label": "Normal 2 (2006)"},
    {"idx": 4000, "label": "Normal 3 (2016)"},
    {"idx": 2050, "label": "Crisis 1 (Sept 2008)"},
    {"idx": 2100, "label": "Crisis 2 (Oct 2008)"},
    {"idx": 5000, "label": "COVID (2020)"},
]

sensitivity_results = {}

for model_name in MODELS:
    print(f"\n  {MODEL_LABELS[model_name]}:")

    # Load model
    model_path = f"{BASE_MODEL_DIR}/{model_name}.pt"
    model_data = torch.load(model_path, weights_only=False)
    model_config = model_data["model_config"]
    model_config["mem_dropout"] = 0.
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()

    use_ex_feats = model_config["ex_feats_dim"] > 0

    # Load empirical latents
    z_mean_pool = latent_data[model_name]["z_mean"]
    z_log_var_pool = latent_data[model_name]["z_log_var"]

    day_sensitivities = []

    for day_config in test_days_config:
        day_idx = day_config["idx"]
        pool_idx = day_idx - CONTEXT_LEN

        # Get latent for this day
        z_mean_day = z_mean_pool[pool_idx]  # (5,)
        z_log_var_day = z_log_var_pool[pool_idx]  # (5,)
        z_std_day = np.exp(0.5 * z_log_var_day)

        # Prepare context
        surf_ctx = torch.from_numpy(surf_data[day_idx-CONTEXT_LEN:day_idx]).float()
        ctx_data = {"surface": surf_ctx.unsqueeze(0)}

        if use_ex_feats:
            ex_ctx = torch.from_numpy(ex_data[day_idx-CONTEXT_LEN:day_idx]).float()
            ctx_data["ex_feats"] = ex_ctx.unsqueeze(0)

        # Generate with mean latent (z = μ)
        z_tensor_mean = torch.zeros((1, CONTEXT_LEN + 1, model.config["latent_dim"]))
        z_tensor_mean[0, -1, :] = torch.from_numpy(z_mean_day)

        with torch.no_grad():
            if use_ex_feats:
                surf_mean, _ = model.get_surface_given_conditions(ctx_data, z=z_tensor_mean)
            else:
                surf_mean = model.get_surface_given_conditions(ctx_data, z=z_tensor_mean)

        surf_mean = surf_mean.cpu().numpy()[0, 0]  # (5, 5)

        # Perturb each dimension by ±1σ, ±2σ, ±3σ
        perturbations = [1, 2, 3]
        max_changes = []

        for perturb_scale in perturbations:
            changes = []

            for dim in range(model.config["latent_dim"]):
                # Perturb this dimension
                z_perturbed = z_mean_day.copy()
                z_perturbed[dim] += perturb_scale * z_std_day[dim]

                z_tensor_perturbed = torch.zeros((1, CONTEXT_LEN + 1, model.config["latent_dim"]))
                z_tensor_perturbed[0, -1, :] = torch.from_numpy(z_perturbed)

                with torch.no_grad():
                    if use_ex_feats:
                        surf_perturbed, _ = model.get_surface_given_conditions(ctx_data, z=z_tensor_perturbed)
                    else:
                        surf_perturbed = model.get_surface_given_conditions(ctx_data, z=z_tensor_perturbed)

                surf_perturbed = surf_perturbed.cpu().numpy()[0, 0]  # (5, 5)

                # Compute change
                change = np.mean(np.abs(surf_perturbed - surf_mean))
                changes.append(change)

            max_changes.append(np.max(changes))

        day_sensitivities.append({
            "label": day_config["label"],
            "sensitivity_1sigma": max_changes[0],
            "sensitivity_2sigma": max_changes[1],
            "sensitivity_3sigma": max_changes[2]
        })

        print(f"    {day_config['label']:20s}: Δ(±1σ)={max_changes[0]:.6f}, Δ(±2σ)={max_changes[1]:.6f}, Δ(±3σ)={max_changes[2]:.6f}")

    sensitivity_results[model_name] = day_sensitivities

    avg_2sigma = np.mean([d["sensitivity_2sigma"] for d in day_sensitivities])
    print(f"    Average sensitivity (±2σ): {avg_2sigma:.6f}")

    if avg_2sigma < 0.001:
        print(f"    ✗ SEVERE COLLAPSE: Decoder nearly deterministic!")
    elif avg_2sigma < 0.005:
        print(f"    ⚠ WARNING: Very low decoder sensitivity")

# [6] Create visualization
print("\n[6/6] Creating visualization...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: KL divergence per dimension
for i, model_name in enumerate(MODELS):
    ax = fig.add_subplot(gs[0, i])
    kl_per_dim = kl_results[model_name]["kl_per_dim"]
    ax.bar(range(len(kl_per_dim)), kl_per_dim, color='blue', alpha=0.7)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Expected (0.5)')
    ax.set_xlabel('Latent Dimension', fontsize=10)
    ax.set_ylabel('KL Divergence', fontsize=10)
    ax.set_title(f'{MODEL_LABELS[model_name]}\nKL per Dimension (Total={kl_results[model_name]["total_kl"]:.2f})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Row 2: z_log_var histogram
for i, model_name in enumerate(MODELS):
    ax = fig.add_subplot(gs[1, i])
    z_log_var = latent_data[model_name]["z_log_var"].flatten()
    ax.hist(z_log_var, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    ax.axvline(np.mean(z_log_var), color='green', linestyle='-', linewidth=2,
               label=f'Actual ({np.mean(z_log_var):.2f})')
    ax.set_xlabel('log(variance)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{MODEL_LABELS[model_name]}\nPosterior Log Variance Distribution',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Row 3: Decoder sensitivity
for i, model_name in enumerate(MODELS):
    ax = fig.add_subplot(gs[2, i])

    days = [d["label"] for d in sensitivity_results[model_name]]
    sens_1sigma = [d["sensitivity_1sigma"] for d in sensitivity_results[model_name]]
    sens_2sigma = [d["sensitivity_2sigma"] for d in sensitivity_results[model_name]]
    sens_3sigma = [d["sensitivity_3sigma"] for d in sensitivity_results[model_name]]

    x = np.arange(len(days))
    width = 0.25

    ax.bar(x - width, sens_1sigma, width, label='±1σ', alpha=0.8)
    ax.bar(x, sens_2sigma, width, label='±2σ', alpha=0.8)
    ax.bar(x + width, sens_3sigma, width, label='±3σ', alpha=0.8)

    ax.set_ylabel('Output Change (MAE)', fontsize=10)
    ax.set_title(f'{MODEL_LABELS[model_name]}\nDecoder Sensitivity to Latent Perturbations',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add reference line (typical residual std ~0.01)
    ax.axhline(0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Typical residual')

fig.suptitle('Posterior Collapse Verification Tests', fontsize=16, fontweight='bold')

output_path = 'posterior_collapse_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# [7] Generate diagnostic report
print("\n" + "="*80)
print("DIAGNOSTIC REPORT")
print("="*80)

report_lines = []
report_lines.append("="*80)
report_lines.append("POSTERIOR COLLAPSE VERIFICATION REPORT")
report_lines.append("="*80)

for model_name in MODELS:
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"{MODEL_LABELS[model_name]}")
    report_lines.append(f"{'='*80}")

    # Test 1: KL Divergence
    total_kl = kl_results[model_name]["total_kl"]
    expected_kl = 5 / 2  # latent_dim / 2
    report_lines.append(f"\n[Test 1] KL Divergence:")
    report_lines.append(f"  Total KL: {total_kl:.3f} (expected ~{expected_kl:.1f})")
    report_lines.append(f"  Per dimension: {kl_results[model_name]['kl_per_dim']}")
    if total_kl < 0.5:
        report_lines.append(f"  ✗ SEVERE COLLAPSE: KL extremely low!")
    elif total_kl < expected_kl * 0.5:
        report_lines.append(f"  ⚠ Posterior collapse detected (KL < 50% of expected)")
    else:
        report_lines.append(f"  ✓ KL appears reasonable")

    # Test 2: Variance Statistics
    mean_log_var = variance_results[model_name]["mean_log_var"]
    mean_variance = variance_results[model_name]["mean_variance"]
    report_lines.append(f"\n[Test 2] Posterior Variance:")
    report_lines.append(f"  mean(z_log_var): {mean_log_var:.3f} (expected ~0)")
    report_lines.append(f"  mean(variance): {mean_variance:.3f} (expected ~1)")
    if mean_log_var < -1.0:
        report_lines.append(f"  ⚠ Very low variance suggests collapse (variance = {mean_variance:.4f})")
    else:
        report_lines.append(f"  ✓ Variance appears reasonable")

    # Test 3: Active Units
    active_dims = active_units_results[model_name]["active_dims"]
    total_dims = active_units_results[model_name]["total_dims"]
    report_lines.append(f"\n[Test 3] Active Units:")
    report_lines.append(f"  Active dimensions: {active_dims}/{total_dims}")
    if active_dims < total_dims:
        report_lines.append(f"  ⚠ {total_dims - active_dims} dimensions collapsed")
    else:
        report_lines.append(f"  ✓ All dimensions active")

    # Test 4: Decoder Sensitivity
    avg_2sigma = np.mean([d["sensitivity_2sigma"] for d in sensitivity_results[model_name]])
    report_lines.append(f"\n[Test 4] Decoder Sensitivity:")
    report_lines.append(f"  Average output change (±2σ perturbation): {avg_2sigma:.6f}")
    report_lines.append(f"  Typical residual std: ~0.01")
    report_lines.append(f"  Ratio (sensitivity / residual): {avg_2sigma / 0.01:.3f}")
    if avg_2sigma < 0.001:
        report_lines.append(f"  ✗ SEVERE DECODER COLLAPSE: Nearly deterministic!")
    elif avg_2sigma < 0.005:
        report_lines.append(f"  ⚠ Low decoder sensitivity")
    else:
        report_lines.append(f"  ✓ Decoder responds to latent variation")

    # Overall diagnosis
    report_lines.append(f"\n[DIAGNOSIS]:")
    collapse_score = 0
    if total_kl < 1.0:
        collapse_score += 1
    if mean_log_var < -1.0:
        collapse_score += 1
    if active_dims < total_dims:
        collapse_score += 1
    if avg_2sigma < 0.005:
        collapse_score += 2  # Decoder sensitivity is most important

    if collapse_score >= 4:
        report_lines.append(f"  ✗ SEVERE POSTERIOR COLLAPSE CONFIRMED")
    elif collapse_score >= 2:
        report_lines.append(f"  ⚠ Moderate posterior collapse detected")
    else:
        report_lines.append(f"  ✓ No significant posterior collapse")

report = "\n".join(report_lines)

report_path = 'posterior_collapse_report.txt'
with open(report_path, 'w') as f:
    f.write(report)

print("\n" + report)
print(f"\n✓ Report saved to: {report_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {output_path}")
print(f"  - {report_path}")
