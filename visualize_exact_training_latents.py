"""
Visualize teacher forcing with EXACT training latents (2008-2010).

Tests CI calibration hypothesis: If we use the exact z_mean and z_log_var that
the model learned during training, should the 90% CIs be well-calibrated (~10% violations)?

Key difference from other generation methods:
- Baseline: z ~ N(0,1) → 54% violations
- Empirical pool + noise: z ~ Empirical(pool) + noise → 37% violations
- THIS: z ~ N(μ_exact[day], σ²_exact[day]) → UNKNOWN (testing)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from vae.cvae_with_mem_randomized import CVAEMemRand
from datetime import datetime

print("="*80)
print("CI CALIBRATION TEST: EXACT TRAINING LATENTS (2008-2010)")
print("="*80)

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
CONTEXT_LEN = 5
NUM_SAMPLES = 1000

# Grid points to visualize
GRID_POINTS = {
    "ATM 3-Month": (1, 2),
    "ATM 1-Year": (3, 2),
    "OTM Put 1-Year": (3, 0)
}

# Models to test
MODELS = ["no_ex", "ex_no_loss", "ex_loss"]
MODEL_LABELS = {
    "no_ex": "No EX",
    "ex_no_loss": "EX No Loss",
    "ex_loss": "EX Loss"
}

# [1] Load data
print("\n[1/5] Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
surf_data = data["surface"]  # (5822, 5, 5)
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.stack([ret_data, skew_data, slope_data], axis=-1)

# Load dates
dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
dates_full = pd.to_datetime(dates_df["date"].values)

print(f"  Surface data shape: {surf_data.shape}")
print(f"  Date range: {dates_full[0]} to {dates_full[-1]}")

# Filter to 2008-2010 (within training set)
dates_gen = dates_full[CONTEXT_LEN:]  # Dates we can generate for (offset by ctx_len)
mask_2008_2010 = (dates_gen >= '2008-01-01') & (dates_gen <= '2010-12-31')
indices_2008_2010 = np.where(mask_2008_2010)[0]
dates_2008_2010 = dates_gen[mask_2008_2010]

print(f"\n  2008-2010 period:")
print(f"    {len(dates_2008_2010)} trading days")
print(f"    From {dates_2008_2010[0]} to {dates_2008_2010[-1]}")
print(f"    Index range: {indices_2008_2010[0]} to {indices_2008_2010[-1]}")

# [2] Helper function to generate with exact latents
def generate_with_exact_latents(model, surf_data, ex_data, day_idx,
                                z_mean_pool, z_log_var_pool,
                                num_samples, use_ex_feats):
    """
    Generate predictions using the exact learned latent distribution for a specific day.

    Args:
        day_idx: Actual day index in full dataset (e.g., 2008 for 2008-01-02)
        z_mean_pool: (5817, 5) - learned means
        z_log_var_pool: (5817, 5) - learned log variances

    Returns:
        surfaces: (num_samples, 5, 5)
        ex_feats: (num_samples, 3) or None
    """
    # Map day_idx to pool index (offset by ctx_len)
    pool_idx = day_idx - CONTEXT_LEN

    # Get exact learned distribution for this day
    z_mean_day = z_mean_pool[pool_idx]  # (5,)
    z_log_var_day = z_log_var_pool[pool_idx]  # (5,)

    # Sample: z = μ + σ*ε where σ = exp(0.5 * log_var)
    eps = np.random.randn(num_samples, model.config["latent_dim"])
    z_std = np.exp(0.5 * z_log_var_day)
    z_samples = z_mean_day + z_std * eps  # (num_samples, 5)

    # Prepare context data (convert to float32 for model)
    surf_ctx = torch.from_numpy(surf_data[day_idx-CONTEXT_LEN:day_idx]).float()
    ctx_data = {
        "surface": surf_ctx.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # (B, T, 5, 5)
    }

    if use_ex_feats:
        ex_ctx = torch.from_numpy(ex_data[day_idx-CONTEXT_LEN:day_idx]).float()
        ctx_data["ex_feats"] = ex_ctx.unsqueeze(0).repeat(num_samples, 1, 1)

    # Prepare z tensor: (B, T, latent_dim)
    z_tensor = torch.zeros((num_samples, CONTEXT_LEN + 1, model.config["latent_dim"]))
    z_tensor[:, -1, :] = torch.from_numpy(z_samples)  # Only future timestep matters

    # Generate
    model.eval()
    with torch.no_grad():
        if use_ex_feats and model.config.get("ex_feats_dim", 0) > 0:
            surf, ex_feat = model.get_surface_given_conditions(ctx_data, z=z_tensor)
            return surf.cpu().numpy(), ex_feat.cpu().numpy()
        else:
            surf = model.get_surface_given_conditions(ctx_data, z=z_tensor)
            return surf.cpu().numpy(), None

# [3] Generate for all models
print("\n[2/5] Generating predictions with exact training latents...")

generated_data = {}

for model_name in MODELS:
    print(f"\n  Model: {model_name}")

    # Load model
    model_path = f"{BASE_MODEL_DIR}/{model_name}.pt"
    model_data = torch.load(model_path, weights_only=False)
    model_config = model_data["model_config"]
    model_config["mem_dropout"] = 0.
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()

    # Load empirical latents
    latents_path = f"{BASE_MODEL_DIR}/{model_name}_empirical_latents.npz"
    latents = np.load(latents_path)
    z_mean_pool = latents["z_mean_pool"]
    z_log_var_pool = latents["z_log_var_pool"]

    use_ex_feats = model_config["ex_feats_dim"] > 0

    print(f"    Latent pool shape: {z_mean_pool.shape}")
    print(f"    Generating {len(indices_2008_2010)} days...")

    # Storage
    num_days = len(indices_2008_2010)
    all_surfaces = np.zeros((num_days, NUM_SAMPLES, 5, 5))
    all_ex_feats = np.zeros((num_days, NUM_SAMPLES, 3)) if use_ex_feats else None

    # Generate for each day
    for i, idx in enumerate(indices_2008_2010):
        actual_day_idx = idx + CONTEXT_LEN  # Convert to actual dataset index

        if i % 100 == 0:
            print(f"      Day {i}/{num_days} ({dates_2008_2010[i].strftime('%Y-%m-%d')})")

        surf, ex_feat = generate_with_exact_latents(
            model, surf_data, ex_data, actual_day_idx,
            z_mean_pool, z_log_var_pool, NUM_SAMPLES, use_ex_feats
        )

        all_surfaces[i] = surf.squeeze(1) if surf.shape[1] == 1 else surf
        if ex_feat is not None:
            all_ex_feats[i] = ex_feat.squeeze(1) if ex_feat.shape[1] == 1 else ex_feat

    generated_data[model_name] = {
        "surfaces": all_surfaces,
        "ex_feats": all_ex_feats
    }

    print(f"    ✓ Generated shape: {all_surfaces.shape}")

# [4] Compute CI violations
print("\n[3/5] Computing CI violations...")

violation_stats = []

for model_name in MODELS:
    print(f"\n  {MODEL_LABELS[model_name]}:")

    generated_surfaces = generated_data[model_name]["surfaces"]

    for grid_name, (row, col) in GRID_POINTS.items():
        # Extract actual and generated at grid point
        actual_indices = indices_2008_2010 + CONTEXT_LEN
        actual_values = surf_data[actual_indices, row, col]  # (num_days,)
        generated_values = generated_surfaces[:, :, row, col]  # (num_days, num_samples)

        # Compute CI bounds
        p05 = np.percentile(generated_values, 5, axis=1)
        p95 = np.percentile(generated_values, 95, axis=1)

        # Check violations
        violations = (actual_values < p05) | (actual_values > p95)
        num_violations = np.sum(violations)
        violation_rate = 100.0 * num_violations / len(actual_values)

        violation_stats.append({
            "model": MODEL_LABELS[model_name],
            "grid_point": grid_name,
            "num_violations": num_violations,
            "total_days": len(actual_values),
            "violation_rate": violation_rate,
            "row": row,
            "col": col
        })

        print(f"    {grid_name:20s}: {num_violations}/{len(actual_values)} = {violation_rate:5.2f}%")

# [5] Create visualization
print("\n[4/5] Creating visualization...")

fig, axes = plt.subplots(3, 3, figsize=(22, 14))
fig.suptitle(
    'Teacher Forcing with EXACT Training Latents (2008-2010)\n' +
    'Testing CI Calibration: z ~ N(μ_train[day], σ²_train[day])',
    fontsize=16, fontweight='bold'
)

for i, model_name in enumerate(MODELS):
    generated_surfaces = generated_data[model_name]["surfaces"]

    for j, (grid_name, (row, col)) in enumerate(GRID_POINTS.items()):
        ax = axes[i, j]

        # Extract data
        actual_indices = indices_2008_2010 + CONTEXT_LEN
        actual_values = surf_data[actual_indices, row, col]
        generated_values = generated_surfaces[:, :, row, col]

        # Compute percentiles
        p05 = np.percentile(generated_values, 5, axis=1)
        p50 = np.percentile(generated_values, 50, axis=1)
        p95 = np.percentile(generated_values, 95, axis=1)

        # Detect violations
        violations = (actual_values < p05) | (actual_values > p95)
        violation_rate = 100.0 * np.sum(violations) / len(actual_values)

        # Plot CI band
        ax.fill_between(dates_2008_2010, p05, p95, alpha=0.3, color='blue',
                        label='90% CI (Exact Latents)')
        ax.plot(dates_2008_2010, p50, 'b--', linewidth=0.8, label='Median', alpha=0.7)
        ax.plot(dates_2008_2010, actual_values, 'k-', linewidth=1.2, label='Actual', zorder=3)

        # Highlight violations
        if np.any(violations):
            ax.scatter(dates_2008_2010[violations], actual_values[violations],
                      color='red', s=20, marker='o', alpha=0.7, zorder=4)

        # Highlight Sept 2008 crisis
        ax.axvspan(datetime(2008, 9, 1), datetime(2008, 9, 30),
                   alpha=0.15, color='orange', label='Sept 2008 Crisis' if i==0 and j==0 else None)

        # Format
        title = f'{MODEL_LABELS[model_name]} - {grid_name}\nViolations: {violation_rate:.1f}% (Target: 10%)'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Implied Volatility', fontsize=9)
        ax.grid(True, alpha=0.3)

        if i == 0 and j == 0:
            ax.legend(loc='upper left', fontsize=7)

        if i == 2:
            ax.set_xlabel('Date', fontsize=9)

plt.tight_layout()

output_path = 'exact_training_latents_2008_2010.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# [6] Generate statistics report
print("\n[5/5] Generating statistics report...")

report_lines = []
report_lines.append("="*80)
report_lines.append("CI CALIBRATION TEST: EXACT TRAINING LATENTS")
report_lines.append("="*80)
report_lines.append(f"\nPeriod: 2008-2010 ({len(dates_2008_2010)} trading days)")
report_lines.append(f"Method: z ~ N(μ_exact[day], σ²_exact[day])")
report_lines.append(f"Samples per day: {NUM_SAMPLES}")
report_lines.append(f"\nHypothesis: Using the EXACT latent distribution learned during")
report_lines.append(f"training should produce well-calibrated 90% CIs (10% violation rate)")
report_lines.append("\n")

report_lines.append(f"{'Model':<15} {'Grid Point':<20} {'Violations':<15} {'Rate':<10} {'vs Target':<10}")
report_lines.append("-" * 80)

for stat in violation_stats:
    rate = stat['violation_rate']
    deviation = rate - 10.0
    report_lines.append(
        f"{stat['model']:<15} {stat['grid_point']:<20} "
        f"{stat['num_violations']}/{stat['total_days']:<10} "
        f"{rate:>6.2f}%   {deviation:>+6.2f}%"
    )

# Summary statistics
all_rates = [s['violation_rate'] for s in violation_stats]
report_lines.append("\n" + "="*80)
report_lines.append("SUMMARY STATISTICS")
report_lines.append("="*80)
report_lines.append(f"Average violation rate: {np.mean(all_rates):.2f}%")
report_lines.append(f"Std deviation: {np.std(all_rates):.2f}%")
report_lines.append(f"Min violation rate: {np.min(all_rates):.2f}%")
report_lines.append(f"Max violation rate: {np.max(all_rates):.2f}%")
report_lines.append(f"\nTarget rate: 10.0%")
report_lines.append(f"Average deviation from target: {np.mean(all_rates) - 10:.2f}%")

report_lines.append("\n" + "="*80)
report_lines.append("COMPARISON WITH OTHER METHODS")
report_lines.append("="*80)
report_lines.append(f"Baseline N(0,1) (OOD):           ~54% violations")
report_lines.append(f"Empirical pool + noise=2.0 (OOD): ~37% violations")
report_lines.append(f"Exact training latents (in-dist): {np.mean(all_rates):.2f}% violations")

report_lines.append("\n" + "="*80)
report_lines.append("INTERPRETATION")
report_lines.append("="*80)

avg_rate = np.mean(all_rates)
if avg_rate < 15:
    report_lines.append("✓ CIs are well-calibrated! Using exact training latents achieves ~10% violations.")
    report_lines.append("  This confirms the primary issue is latent sampling method.")
    report_lines.append("  Solution: Use learned posteriors instead of N(0,1) during generation.")
elif avg_rate < 30:
    report_lines.append("⚠ CIs are partially calibrated but still above target.")
    report_lines.append("  Suggests both sampling method AND decoder calibration contribute to issues.")
    report_lines.append("  May need both better sampling and architectural improvements.")
else:
    report_lines.append("✗ CIs remain severely miscalibrated even with exact training latents.")
    report_lines.append("  This suggests deeper architectural problems beyond just sampling:")
    report_lines.append("  - Decoder uncertainty may be fundamentally miscalibrated")
    report_lines.append("  - Weak KL regularization (1e-5) may prevent learning good posteriors")
    report_lines.append("  - Teacher forcing mismatch affects more than just latent sampling")

report = "\n".join(report_lines)

# Save report
report_path = 'exact_training_latents_statistics.txt'
with open(report_path, 'w') as f:
    f.write(report)

# Print to console
print("\n" + report)
print(f"\n✓ Report saved to: {report_path}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {output_path}")
print(f"  - {report_path}")
