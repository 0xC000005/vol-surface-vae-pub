"""
Residual analysis for exact training latents generation.

Since MSE is excellent, examine residuals to understand why CIs are miscalibrated:
- Are residuals normally distributed?
- Is there heteroscedasticity (changing variance)?
- Are there systematic biases?
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from vae.cvae_with_mem_randomized import CVAEMemRand
from datetime import datetime

print("="*80)
print("RESIDUAL ANALYSIS: EXACT TRAINING LATENTS (2008-2010)")
print("="*80)

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
CONTEXT_LEN = 5
NUM_SAMPLES = 1000

# Grid points
GRID_POINTS = {
    "ATM 3-Month": (1, 2),
    "ATM 1-Year": (3, 2),
    "OTM Put 1-Year": (3, 0)
}

MODELS = ["no_ex", "ex_no_loss", "ex_loss"]
MODEL_LABELS = {
    "no_ex": "No EX",
    "ex_no_loss": "EX No Loss",
    "ex_loss": "EX Loss"
}

# [1] Load data
print("\n[1/4] Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.stack([ret_data, skew_data, slope_data], axis=-1)

dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
dates_full = pd.to_datetime(dates_df["date"].values)

# Filter to 2008-2010
dates_gen = dates_full[CONTEXT_LEN:]
mask_2008_2010 = (dates_gen >= '2008-01-01') & (dates_gen <= '2010-12-31')
indices_2008_2010 = np.where(mask_2008_2010)[0]
dates_2008_2010 = dates_gen[mask_2008_2010]

print(f"  {len(dates_2008_2010)} trading days from {dates_2008_2010[0]} to {dates_2008_2010[-1]}")

# [2] Generate with exact latents
print("\n[2/4] Generating predictions...")

def generate_with_exact_latents(model, surf_data, ex_data, day_idx,
                                z_mean_pool, z_log_var_pool,
                                num_samples, use_ex_feats):
    pool_idx = day_idx - CONTEXT_LEN
    z_mean_day = z_mean_pool[pool_idx]
    z_log_var_day = z_log_var_pool[pool_idx]

    eps = np.random.randn(num_samples, model.config["latent_dim"])
    z_std = np.exp(0.5 * z_log_var_day)
    z_samples = z_mean_day + z_std * eps

    surf_ctx = torch.from_numpy(surf_data[day_idx-CONTEXT_LEN:day_idx]).float()
    ctx_data = {"surface": surf_ctx.unsqueeze(0).repeat(num_samples, 1, 1, 1)}

    if use_ex_feats:
        ex_ctx = torch.from_numpy(ex_data[day_idx-CONTEXT_LEN:day_idx]).float()
        ctx_data["ex_feats"] = ex_ctx.unsqueeze(0).repeat(num_samples, 1, 1)

    z_tensor = torch.zeros((num_samples, CONTEXT_LEN + 1, model.config["latent_dim"]))
    z_tensor[:, -1, :] = torch.from_numpy(z_samples)

    model.eval()
    with torch.no_grad():
        if use_ex_feats and model.config.get("ex_feats_dim", 0) > 0:
            surf, _ = model.get_surface_given_conditions(ctx_data, z=z_tensor)
        else:
            surf = model.get_surface_given_conditions(ctx_data, z=z_tensor)
        return surf.cpu().numpy()

generated_data = {}

for model_name in MODELS:
    print(f"  Model: {model_name}")

    model_path = f"{BASE_MODEL_DIR}/{model_name}.pt"
    model_data = torch.load(model_path, weights_only=False)
    model_config = model_data["model_config"]
    model_config["mem_dropout"] = 0.
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()

    latents_path = f"{BASE_MODEL_DIR}/{model_name}_empirical_latents.npz"
    latents = np.load(latents_path)
    z_mean_pool = latents["z_mean_pool"]
    z_log_var_pool = latents["z_log_var_pool"]

    use_ex_feats = model_config["ex_feats_dim"] > 0

    num_days = len(indices_2008_2010)
    all_surfaces = np.zeros((num_days, NUM_SAMPLES, 5, 5))

    for i, idx in enumerate(indices_2008_2010):
        actual_day_idx = idx + CONTEXT_LEN
        surf = generate_with_exact_latents(
            model, surf_data, ex_data, actual_day_idx,
            z_mean_pool, z_log_var_pool, NUM_SAMPLES, use_ex_feats
        )
        all_surfaces[i] = surf.squeeze(1) if surf.shape[1] == 1 else surf

    generated_data[model_name] = all_surfaces

# [3] Compute residuals
print("\n[3/4] Computing residuals...")

residual_data = {}

for model_name in MODELS:
    generated_surfaces = generated_data[model_name]
    residual_data[model_name] = {}

    for grid_name, (row, col) in GRID_POINTS.items():
        actual_indices = indices_2008_2010 + CONTEXT_LEN
        actual_values = surf_data[actual_indices, row, col]
        generated_values = generated_surfaces[:, :, row, col]

        # Statistics
        pred_mean = np.mean(generated_values, axis=1)
        pred_median = np.median(generated_values, axis=1)
        pred_std = np.std(generated_values, axis=1)

        # Residuals
        residuals_mean = actual_values - pred_mean
        residuals_median = actual_values - pred_median

        residual_data[model_name][grid_name] = {
            "actual": actual_values,
            "pred_mean": pred_mean,
            "pred_median": pred_median,
            "pred_std": pred_std,
            "residuals_mean": residuals_mean,
            "residuals_median": residuals_median,
            "generated_values": generated_values
        }

        print(f"  {MODEL_LABELS[model_name]} - {grid_name}:")
        print(f"    Mean residual: {np.mean(residuals_mean):.6f}")
        print(f"    Residual std: {np.std(residuals_mean):.6f}")
        print(f"    Mean predicted std: {np.mean(pred_std):.6f}")
        print(f"    Ratio (actual/predicted): {np.std(residuals_mean)/np.mean(pred_std):.3f}")

# [4] Create visualizations
print("\n[4/4] Creating visualizations...")

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

for i, model_name in enumerate(MODELS):
    for j, (grid_name, (row, col)) in enumerate(GRID_POINTS.items()):
        data = residual_data[model_name][grid_name]
        residuals = data["residuals_mean"]
        pred_mean = data["pred_mean"]
        pred_std = data["pred_std"]

        # Column 0: Residuals vs Time
        ax1 = fig.add_subplot(gs[j, 0])
        ax1.scatter(dates_2008_2010, residuals, alpha=0.3, s=10, color='blue')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        ax1.axhline(np.mean(pred_std), color='green', linestyle='--', linewidth=1, label=f'Avg pred std: {np.mean(pred_std):.4f}')
        ax1.axhline(-np.mean(pred_std), color='green', linestyle='--', linewidth=1)
        ax1.axvspan(datetime(2008, 9, 1), datetime(2008, 9, 30), alpha=0.15, color='orange')
        ax1.set_ylabel('Residual', fontsize=9)
        ax1.set_title(f'{MODEL_LABELS[model_name]} - {grid_name}\nResiduals Over Time', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7)
        if j == 2:
            ax1.set_xlabel('Date', fontsize=9)

        # Column 1: Residuals vs Predicted
        ax2 = fig.add_subplot(gs[j, 1])
        ax2.scatter(pred_mean, residuals, alpha=0.3, s=10, color='blue')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Predicted Value', fontsize=9)
        ax2.set_ylabel('Residual', fontsize=9)
        ax2.set_title(f'Residuals vs Predicted', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if j == 2:
            ax2.set_xlabel('Predicted Value', fontsize=9)

        # Column 2: Histogram of residuals
        ax3 = fig.add_subplot(gs[j, 2])
        ax3.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)

        # Overlay predicted std normal
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        pred_normal = stats.norm.pdf(x_range, 0, np.mean(pred_std))
        actual_normal = stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals))
        ax3.plot(x_range, pred_normal, 'g-', linewidth=2, label=f'N(0, {np.mean(pred_std):.4f})')
        ax3.plot(x_range, actual_normal, 'r--', linewidth=2, label=f'N({np.mean(residuals):.4f}, {np.std(residuals):.4f})')

        ax3.set_xlabel('Residual', fontsize=9)
        ax3.set_ylabel('Density', fontsize=9)
        ax3.set_title(f'Residual Distribution', fontsize=10, fontweight='bold')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # Column 3: QQ plot
        ax4 = fig.add_subplot(gs[j, 3])
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title(f'Q-Q Plot', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)

fig.suptitle('Residual Analysis: Exact Training Latents (2008-2010)', fontsize=16, fontweight='bold')

output_path = 'residual_analysis_exact_latents.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# [5] Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

for model_name in MODELS:
    print(f"\n{MODEL_LABELS[model_name]}:")
    for grid_name, (row, col) in GRID_POINTS.items():
        data = residual_data[model_name][grid_name]
        residuals = data["residuals_mean"]
        pred_std = data["pred_std"]

        # Normality test (Shapiro-Wilk)
        _, p_shapiro = stats.shapiro(residuals)

        # Mean is zero test (t-test)
        _, p_ttest = stats.ttest_1samp(residuals, 0)

        # Compare variance
        actual_std = np.std(residuals)
        avg_pred_std = np.mean(pred_std)
        ratio = actual_std / avg_pred_std

        print(f"\n  {grid_name}:")
        print(f"    Normality (Shapiro p-value): {p_shapiro:.4f} {'✓' if p_shapiro > 0.05 else '✗'}")
        print(f"    Zero mean (t-test p-value): {p_ttest:.4f} {'✓' if p_ttest > 0.05 else '✗'}")
        print(f"    Actual residual std: {actual_std:.6f}")
        print(f"    Avg predicted std: {avg_pred_std:.6f}")
        print(f"    Ratio (actual/predicted): {ratio:.3f}x")
        if ratio > 1.5:
            print(f"    ⚠ Model underestimates uncertainty by {ratio:.1f}x")
        elif ratio < 0.67:
            print(f"    ⚠ Model overestimates uncertainty by {1/ratio:.1f}x")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated: {output_path}")
