"""
Diagnose marginal distribution variance mismatch.

Compares variance statistics between ground truth and model predictions (p50)
to verify whether models are tracking temporal variance correctly.

If variance ratios (model_std / GT_std) are close to 1.0, models are working correctly.
"""
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Test one representative case
PERIOD = 'insample'  # or 'oos'
HORIZON = 1  # 1, 7, 14, or 30

# Grid point to analyze in detail
TEST_GRID_I = 2  # Moneyness index (0-4)
TEST_GRID_J = 2  # Maturity index (0-4)

moneyness_labels = ['0.70', '0.85', '1.00', '1.15', '1.30']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

# ============================================================================
# Load Data
# ============================================================================

print("=" * 80)
print(f"MARGINAL VARIANCE DIAGNOSTIC: {PERIOD.upper()} H{HORIZON}")
print("=" * 80)
print()

# File paths
if PERIOD == 'insample':
    oracle_file = "models/backfill/insample_reconstruction_16yr.npz"
    vae_file = "models/backfill/vae_prior_insample_16yr.npz"
    econ_file = "results/econometric_backfill/econometric_backfill_insample.npz"
else:  # oos
    oracle_file = "models/backfill/oos_reconstruction_16yr.npz"
    vae_file = "models/backfill/vae_prior_oos_16yr.npz"
    econ_file = "results/econometric_backfill/econometric_backfill_oos.npz"

print("Loading data...")
oracle_data = np.load(oracle_file)
vae_data = np.load(vae_file)
econ_data = np.load(econ_file)
gt_data = np.load("data/vol_surface_with_ret.npz")

# Extract reconstructions and indices
recon_key = f'recon_h{HORIZON}'
indices_key = f'indices_h{HORIZON}'

# Extract p50 (median) quantile - index 1
print(f"\nExtracting p50 quantile (index 1 of dim 1)...")
oracle_recon = oracle_data[recon_key]  # (N, 3, 5, 5)
vae_recon = vae_data[recon_key]  # (N, 3, 5, 5)
econ_recon = econ_data[recon_key]  # (N, 3, 5, 5)

print(f"  Oracle shape: {oracle_recon.shape}")
print(f"  VAE shape: {vae_recon.shape}")
print(f"  Econ shape: {econ_recon.shape}")

oracle_p50 = oracle_recon[:, 1, :, :]  # (N, 5, 5)
vae_p50 = vae_recon[:, 1, :, :]  # (N, 5, 5)
econ_p50 = econ_recon[:, 1, :, :]  # (N, 5, 5)

# Get indices and ground truth
indices = oracle_data[indices_key]
gt = gt_data["surface"][indices]  # (N, 5, 5)

n_days = gt.shape[0]
print(f"\nNumber of days: {n_days}")
print(f"  Ground truth shape: {gt.shape}")
print(f"  Oracle p50 shape: {oracle_p50.shape}")
print(f"  VAE p50 shape: {vae_p50.shape}")
print(f"  Econ p50 shape: {econ_p50.shape}")

# Verify alignment (use minimum length to avoid index errors)
min_len = min(gt.shape[0], oracle_p50.shape[0], vae_p50.shape[0], econ_p50.shape[0])
print(f"\nUsing {min_len} days (minimum across all datasets)")

gt = gt[:min_len]
oracle_p50 = oracle_p50[:min_len]
vae_p50 = vae_p50[:min_len]
econ_p50 = econ_p50[:min_len]

# ============================================================================
# Detailed Analysis: Single Grid Point
# ============================================================================

print()
print("=" * 80)
print(f"DETAILED ANALYSIS: Grid Point ({TEST_GRID_I}, {TEST_GRID_J})")
print(f"  Moneyness: {moneyness_labels[TEST_GRID_I]}")
print(f"  Maturity: {maturity_labels[TEST_GRID_J]}")
print("=" * 80)
print()

# Extract values for this grid point
gt_vals = gt[:, TEST_GRID_I, TEST_GRID_J]
oracle_vals = oracle_p50[:, TEST_GRID_I, TEST_GRID_J]
vae_vals = vae_p50[:, TEST_GRID_I, TEST_GRID_J]
econ_vals = econ_p50[:, TEST_GRID_I, TEST_GRID_J]

# Print first 20 values
print("First 20 values:")
print(f"{'Day':<5} {'GT':<10} {'Oracle':<10} {'VAE':<10} {'Econ':<10}")
print("-" * 50)
for i in range(min(20, len(gt_vals))):
    print(f"{i:<5} {gt_vals[i]:<10.6f} {oracle_vals[i]:<10.6f} {vae_vals[i]:<10.6f} {econ_vals[i]:<10.6f}")

print()
print("Statistics:")
print("-" * 80)
print(f"{'Metric':<15} {'Ground Truth':<15} {'Oracle':<15} {'VAE Prior':<15} {'Econometric':<15}")
print("-" * 80)

# Mean
gt_mean = np.mean(gt_vals)
oracle_mean = np.mean(oracle_vals)
vae_mean = np.mean(vae_vals)
econ_mean = np.mean(econ_vals)
print(f"{'Mean':<15} {gt_mean:<15.6f} {oracle_mean:<15.6f} {vae_mean:<15.6f} {econ_mean:<15.6f}")

# Std
gt_std = np.std(gt_vals)
oracle_std = np.std(oracle_vals)
vae_std = np.std(vae_vals)
econ_std = np.std(econ_vals)
print(f"{'Std':<15} {gt_std:<15.6f} {oracle_std:<15.6f} {vae_std:<15.6f} {econ_std:<15.6f}")

# Variance ratio (model / GT)
oracle_var_ratio = oracle_std / gt_std if gt_std > 0 else np.nan
vae_var_ratio = vae_std / gt_std if gt_std > 0 else np.nan
econ_var_ratio = econ_std / gt_std if gt_std > 0 else np.nan
print(f"{'Var Ratio':<15} {'1.000':<15} {oracle_var_ratio:<15.6f} {vae_var_ratio:<15.6f} {econ_var_ratio:<15.6f}")

# Min/Max
print(f"{'Min':<15} {np.min(gt_vals):<15.6f} {np.min(oracle_vals):<15.6f} {np.min(vae_vals):<15.6f} {np.min(econ_vals):<15.6f}")
print(f"{'Max':<15} {np.max(gt_vals):<15.6f} {np.max(oracle_vals):<15.6f} {np.max(vae_vals):<15.6f} {np.max(econ_vals):<15.6f}")

# Range
gt_range = np.max(gt_vals) - np.min(gt_vals)
oracle_range = np.max(oracle_vals) - np.min(oracle_vals)
vae_range = np.max(vae_vals) - np.min(vae_vals)
econ_range = np.max(econ_vals) - np.min(econ_vals)
print(f"{'Range':<15} {gt_range:<15.6f} {oracle_range:<15.6f} {vae_range:<15.6f} {econ_range:<15.6f}")

# ============================================================================
# Grid-Level Summary
# ============================================================================

print()
print("=" * 80)
print("VARIANCE RATIO SUMMARY: ALL 25 GRID POINTS")
print("=" * 80)
print()

# Compute variance ratios for all grid points
oracle_var_ratios = np.zeros((5, 5))
vae_var_ratios = np.zeros((5, 5))
econ_var_ratios = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        gt_std_ij = np.std(gt[:, i, j])
        oracle_std_ij = np.std(oracle_p50[:, i, j])
        vae_std_ij = np.std(vae_p50[:, i, j])
        econ_std_ij = np.std(econ_p50[:, i, j])

        oracle_var_ratios[i, j] = oracle_std_ij / gt_std_ij if gt_std_ij > 0 else np.nan
        vae_var_ratios[i, j] = vae_std_ij / gt_std_ij if gt_std_ij > 0 else np.nan
        econ_var_ratios[i, j] = econ_std_ij / gt_std_ij if gt_std_ij > 0 else np.nan

print("Oracle Variance Ratios (model_std / GT_std):")
print("Rows: Moneyness [0.70, 0.85, 1.00, 1.15, 1.30]")
print("Cols: Maturity [1M, 3M, 6M, 1Y, 2Y]")
print()
print("      1M      3M      6M      1Y      2Y")
for i in range(5):
    print(f"{moneyness_labels[i]}  ", end="")
    for j in range(5):
        print(f"{oracle_var_ratios[i, j]:6.3f}  ", end="")
    print()

print()
print("VAE Prior Variance Ratios (model_std / GT_std):")
print("      1M      3M      6M      1Y      2Y")
for i in range(5):
    print(f"{moneyness_labels[i]}  ", end="")
    for j in range(5):
        print(f"{vae_var_ratios[i, j]:6.3f}  ", end="")
    print()

print()
print("Econometric Variance Ratios (model_std / GT_std):")
print("      1M      3M      6M      1Y      2Y")
for i in range(5):
    print(f"{moneyness_labels[i]}  ", end="")
    for j in range(5):
        print(f"{econ_var_ratios[i, j]:6.3f}  ", end="")
    print()

# Overall statistics
print()
print("=" * 80)
print("OVERALL VARIANCE RATIO STATISTICS")
print("=" * 80)
print()

print(f"Oracle:")
print(f"  Mean variance ratio:   {np.nanmean(oracle_var_ratios):.4f}")
print(f"  Median variance ratio: {np.nanmedian(oracle_var_ratios):.4f}")
print(f"  Min variance ratio:    {np.nanmin(oracle_var_ratios):.4f}")
print(f"  Max variance ratio:    {np.nanmax(oracle_var_ratios):.4f}")

print(f"\nVAE Prior:")
print(f"  Mean variance ratio:   {np.nanmean(vae_var_ratios):.4f}")
print(f"  Median variance ratio: {np.nanmedian(vae_var_ratios):.4f}")
print(f"  Min variance ratio:    {np.nanmin(vae_var_ratios):.4f}")
print(f"  Max variance ratio:    {np.nanmax(vae_var_ratios):.4f}")

print(f"\nEconometric:")
print(f"  Mean variance ratio:   {np.nanmean(econ_var_ratios):.4f}")
print(f"  Median variance ratio: {np.nanmedian(econ_var_ratios):.4f}")
print(f"  Min variance ratio:    {np.nanmin(econ_var_ratios):.4f}")
print(f"  Max variance ratio:    {np.nanmax(econ_var_ratios):.4f}")

# ============================================================================
# Interpretation
# ============================================================================

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

mean_oracle_ratio = np.nanmean(oracle_var_ratios)
mean_vae_ratio = np.nanmean(vae_var_ratios)
mean_econ_ratio = np.nanmean(econ_var_ratios)

if 0.9 <= mean_oracle_ratio <= 1.1:
    print("✓ Oracle: EXCELLENT variance tracking (ratio 0.9-1.1)")
else:
    print(f"⚠ Oracle: Variance ratio {mean_oracle_ratio:.3f} outside expected range")

if 0.9 <= mean_vae_ratio <= 1.1:
    print("✓ VAE Prior: EXCELLENT variance tracking (ratio 0.9-1.1)")
else:
    print(f"⚠ VAE Prior: Variance ratio {mean_vae_ratio:.3f} outside expected range")

if 0.9 <= mean_econ_ratio <= 1.1:
    print("✓ Econometric: EXCELLENT variance tracking (ratio 0.9-1.1)")
else:
    print(f"⚠ Econometric: Variance ratio {mean_econ_ratio:.3f} outside expected range")

print()
print("CONCLUSION:")
if mean_oracle_ratio > 0.9 and mean_vae_ratio > 0.9:
    print("Models are tracking ground truth temporal variance correctly.")
    print("If visualizations show very narrow model distributions, it may be a")
    print("plotting artifact (binning, scaling, or overlapping histograms).")
    print()
    print("The p50 predictions SHOULD have similar variance to ground truth")
    print("because they track day-to-day changes. Within-day uncertainty is")
    print("captured by the p05-p95 spread, not by the p50 marginal distribution.")
else:
    print("Models are NOT tracking ground truth variance correctly.")
    print("Further investigation needed into model calibration.")

print()
