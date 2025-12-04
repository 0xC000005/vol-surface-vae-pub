import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
import os

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
OUTPUT_DIR = "results/2024_1213/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_LEN = 5
START_DAY = 5
NUM_DAYS = 1000  # Generate 1000 days
NUM_PATHS = 10   # Generate 10 independent paths

print("="*80)
print("Autoregressive Return Generation Test")
print("="*80)

# Load ground truth data
print("\nLoading ground truth data...")
ground_truth_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = ground_truth_data["surface"]
ret_data = ground_truth_data["ret"]
skew_data = ground_truth_data["skews"]
slope_data = ground_truth_data["slopes"]
ex_data = np.concatenate([
    ret_data[..., np.newaxis],
    skew_data[..., np.newaxis],
    slope_data[..., np.newaxis]
], axis=-1)

# Load dates
dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")

# Extract last 1000 days for visualization
total_days_needed = START_DAY + NUM_DAYS + CONTEXT_LEN
start_idx = total_days_needed - NUM_DAYS - CONTEXT_LEN
ground_truth_returns = ret_data[start_idx:start_idx + NUM_DAYS + CONTEXT_LEN]
ground_truth_surfaces = vol_surf_data[start_idx:start_idx + NUM_DAYS + CONTEXT_LEN]
ground_truth_ex = ex_data[start_idx:start_idx + NUM_DAYS + CONTEXT_LEN]
dates = pd.to_datetime(dates_df["date"].values[start_idx:start_idx + NUM_DAYS + CONTEXT_LEN])

print(f"Period: {dates[0]} to {dates[-1]}")
print(f"Total days: {len(dates)}")

# Load EX Loss model
print("\nLoading EX Loss model...")
model_path = f"{BASE_MODEL_DIR}/ex_loss.pt"
model_data = torch.load(model_path, weights_only=False)
model_config = model_data["model_config"]
model_config["mem_dropout"] = 0.0
model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
print("Model loaded successfully")

# Autoregressive generation function
def generate_autoregressive_path(model, init_surfaces, init_ex_feats, num_days, seed):
    """
    Generate autoregressive path where model uses its own predictions as context.

    Args:
        model: CVAEMemRand model
        init_surfaces: (CONTEXT_LEN, 5, 5) initial context surfaces
        init_ex_feats: (CONTEXT_LEN, 3) initial context features
        num_days: number of days to generate
        seed: random seed for this path

    Returns:
        generated_returns: (num_days,) array of generated returns
        generated_surfaces: (num_days, 5, 5) array of generated surfaces
    """
    set_seeds(seed)

    # Initialize context with real data
    ctx_surfaces = init_surfaces.copy()  # (CONTEXT_LEN, 5, 5)
    ctx_ex_feats = init_ex_feats.copy()  # (CONTEXT_LEN, 3)

    generated_returns = np.zeros(num_days)
    generated_surfaces = np.zeros((num_days, 5, 5))

    for day in range(num_days):
        # Prepare context
        ctx_data = {
            "surface": torch.from_numpy(ctx_surfaces).unsqueeze(0).float(),  # (1, CONTEXT_LEN, 5, 5)
            "ex_feats": torch.from_numpy(ctx_ex_feats).unsqueeze(0).float()  # (1, CONTEXT_LEN, 3)
        }

        # Generate next day (surface and features)
        with torch.no_grad():
            next_surface, next_ex_feats = model.get_surface_given_conditions(ctx_data)

        # Extract generated data
        next_surface = next_surface.cpu().numpy()[0, 0, :, :]  # (5, 5)
        next_ex_feats = next_ex_feats.cpu().numpy()[0, 0, :]  # (3,)
        next_return = next_ex_feats[0]

        # Store generated values
        generated_returns[day] = next_return
        generated_surfaces[day] = next_surface

        # Update context: slide window (drop oldest, add newest)
        ctx_surfaces = np.concatenate([ctx_surfaces[1:], next_surface[np.newaxis, :, :]], axis=0)
        ctx_ex_feats = np.concatenate([ctx_ex_feats[1:], next_ex_feats[np.newaxis, :]], axis=0)

    return generated_returns, generated_surfaces

# Generate 10 autoregressive paths
print(f"\nGenerating {NUM_PATHS} autoregressive paths...")
all_generated_returns = np.zeros((NUM_PATHS, NUM_DAYS))

for path_idx in range(NUM_PATHS):
    print(f"  Generating path {path_idx + 1}/{NUM_PATHS}...")

    # Use real data as initial context
    init_surfaces = ground_truth_surfaces[:CONTEXT_LEN]  # First CONTEXT_LEN days
    init_ex_feats = ground_truth_ex[:CONTEXT_LEN]

    # Generate autoregressive path
    gen_returns, gen_surfaces = generate_autoregressive_path(
        model, init_surfaces, init_ex_feats, NUM_DAYS, seed=1000 + path_idx
    )

    all_generated_returns[path_idx] = gen_returns

print("Generation complete!")

# Extract ground truth returns (excluding initial context)
gt_returns = ground_truth_returns[CONTEXT_LEN:]
dates_plot = dates[CONTEXT_LEN:]

# Compute ACF function
def compute_acf(x, nlags):
    """Compute autocorrelation function"""
    x = np.array(x)
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)

    acf = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        acf[k] = np.dot(x[:-k], x[k:]) / len(x) / c0

    return acf

# Compute ACF for all paths
nlags = 50
gt_acf_returns = compute_acf(gt_returns, nlags)
gt_acf_squared = compute_acf(gt_returns ** 2, nlags)

gen_acf_returns = np.zeros((NUM_PATHS, nlags + 1))
gen_acf_squared = np.zeros((NUM_PATHS, nlags + 1))

for path_idx in range(NUM_PATHS):
    gen_acf_returns[path_idx] = compute_acf(all_generated_returns[path_idx], nlags)
    gen_acf_squared[path_idx] = compute_acf(all_generated_returns[path_idx] ** 2, nlags)

mean_gen_acf_returns = np.mean(gen_acf_returns, axis=0)
mean_gen_acf_squared = np.mean(gen_acf_squared, axis=0)

# Plot 1: Time series of returns
print("\nCreating Plot 1: Return paths...")
fig1, ax = plt.subplots(1, 1, figsize=(16, 6))

# Plot ground truth
ax.plot(dates_plot, gt_returns, 'k-', linewidth=2.5, label='Ground Truth', zorder=10)

# Plot 10 generated paths
colors = plt.cm.tab10(np.linspace(0, 1, NUM_PATHS))
for path_idx in range(NUM_PATHS):
    ax.plot(dates_plot, all_generated_returns[path_idx],
            color=colors[path_idx], alpha=0.6, linewidth=1.0,
            label=f'Generated Path {path_idx + 1}' if path_idx < 3 else None)

# Formatting
ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax.set_title(f'Autoregressive Return Generation (EX Loss Model)\n{NUM_PATHS} Independent Paths | Context Length = {CONTEXT_LEN} days',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Daily Log Return', fontsize=12, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax.legend(loc='upper left', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file_1 = f"{OUTPUT_DIR}/autoregressive_returns_paths.png"
plt.savefig(output_file_1, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file_1}")
plt.close()

# Plot 2: ACF comparison (2 subplots)
print("Creating Plot 2: ACF comparison...")
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confidence interval (95%)
conf_interval = 1.96 / np.sqrt(NUM_DAYS)

# Subplot 1: ACF of returns
ax1 = axes[0]
ax1.plot(range(nlags + 1), gt_acf_returns, 'ko-', linewidth=2, markersize=4, label='Ground Truth')
ax1.plot(range(nlags + 1), mean_gen_acf_returns, 'ro-', linewidth=2, markersize=4,
         label=f'Generated (Mean of {NUM_PATHS} paths)')

# Plot individual paths with transparency
for path_idx in range(NUM_PATHS):
    ax1.plot(range(nlags + 1), gen_acf_returns[path_idx],
             color='red', alpha=0.15, linewidth=0.8)

# Confidence bands
ax1.axhline(conf_interval, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='95% CI')
ax1.axhline(-conf_interval, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
ax1.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

ax1.set_title('ACF of Returns\n(Should be ≈ 0, no autocorrelation)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Lag', fontsize=11)
ax1.set_ylabel('Autocorrelation', fontsize=11)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, nlags)

# Subplot 2: ACF of squared returns
ax2 = axes[1]
ax2.plot(range(nlags + 1), gt_acf_squared, 'ko-', linewidth=2, markersize=4, label='Ground Truth')
ax2.plot(range(nlags + 1), mean_gen_acf_squared, 'ro-', linewidth=2, markersize=4,
         label=f'Generated (Mean of {NUM_PATHS} paths)')

# Plot individual paths with transparency
for path_idx in range(NUM_PATHS):
    ax2.plot(range(nlags + 1), gen_acf_squared[path_idx],
             color='red', alpha=0.15, linewidth=0.8)

# Confidence bands
ax2.axhline(conf_interval, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='95% CI')
ax2.axhline(-conf_interval, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

ax2.set_title('ACF of Squared Returns (Volatility Clustering)\n(Should be > 0, autocorrelated)',
              fontsize=12, fontweight='bold')
ax2.set_xlabel('Lag', fontsize=11)
ax2.set_ylabel('Autocorrelation', fontsize=11)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, nlags)

fig2.suptitle(f'Autocorrelation Analysis: Autoregressive Generation (EX Loss Model)\n{NUM_PATHS} Independent Paths | {NUM_DAYS} Days',
              fontsize=14, fontweight='bold', y=1.00)

plt.tight_layout()
output_file_2 = f"{OUTPUT_DIR}/autoregressive_acf_comparison.png"
plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file_2}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)
print(f"\nGround Truth Returns:")
print(f"  Mean: {np.mean(gt_returns):.6f}")
print(f"  Std:  {np.std(gt_returns):.6f}")
print(f"  ACF[1]: {gt_acf_returns[1]:.4f}")
print(f"  ACF[1] (squared): {gt_acf_squared[1]:.4f}")

print(f"\nGenerated Returns (Average across {NUM_PATHS} paths):")
print(f"  Mean: {np.mean(all_generated_returns):.6f}")
print(f"  Std:  {np.std(all_generated_returns):.6f}")
print(f"  ACF[1]: {mean_gen_acf_returns[1]:.4f}")
print(f"  ACF[1] (squared): {mean_gen_acf_squared[1]:.4f}")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print(f"\nGenerated files:")
print(f"  1. {output_file_1}")
print(f"  2. {output_file_2}")
print(f"\nKey insights to look for:")
print(f"  - ACF of returns should be close to 0 (no autocorrelation)")
print(f"  - ACF of squared returns should be > 0 (volatility clustering)")
print(f"  - Generated paths should match ground truth ACF patterns")
