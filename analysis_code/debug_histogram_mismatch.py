"""
Debug why histograms show GT wider than models despite variance ratio ~1.0
"""
import numpy as np
import matplotlib.pyplot as plt

# Load data for one case
period = 'oos'
horizon = 1

print("="*80)
print(f"DEBUGGING: {period.upper()} H{horizon}")
print("="*80)
print()

# Load files
if period == 'insample':
    oracle_file = "models/backfill/insample_reconstruction_16yr.npz"
    vae_file = "models/backfill/vae_prior_insample_16yr.npz"
    econ_file = "results/econometric_backfill/econometric_backfill_insample.npz"
else:
    oracle_file = "models/backfill/oos_reconstruction_16yr.npz"
    vae_file = "models/backfill/vae_prior_oos_16yr.npz"
    econ_file = "results/econometric_backfill/econometric_backfill_oos.npz"

oracle_data = np.load(oracle_file)
vae_data = np.load(vae_file)
econ_data = np.load(econ_file)
gt_data = np.load("data/vol_surface_with_ret.npz")

recon_key = f'recon_h{horizon}'
indices_key = f'indices_h{horizon}'

# Extract WITHOUT alignment (what the viz script does)
oracle_p50 = oracle_data[recon_key][:, 1, :, :]
vae_p50 = vae_data[recon_key][:, 1, :, :]
econ_p50 = econ_data[recon_key][:, 1, :, :]

indices = oracle_data[indices_key]
gt = gt_data["surface"][indices]

print("Original shapes (unaligned):")
print(f"  GT: {gt.shape}")
print(f"  Oracle: {oracle_p50.shape}")
print(f"  VAE: {vae_p50.shape}")
print(f"  Econ: {econ_p50.shape}")
print()

# Test grid point: ATM 6M (2, 2)
i, j = 2, 2
print(f"Testing grid point ({i},{j}) - ATM 6M")
print()

gt_vals_unaligned = gt[:, i, j]
oracle_vals_unaligned = oracle_p50[:, i, j]
vae_vals_unaligned = vae_p50[:, i, j]
econ_vals_unaligned = econ_p50[:, i, j]

print("Unaligned statistics:")
print(f"  GT:     mean={np.mean(gt_vals_unaligned):.6f}, std={np.std(gt_vals_unaligned):.6f}, n={len(gt_vals_unaligned)}")
print(f"  Oracle: mean={np.mean(oracle_vals_unaligned):.6f}, std={np.std(oracle_vals_unaligned):.6f}, n={len(oracle_vals_unaligned)}")
print(f"  VAE:    mean={np.mean(vae_vals_unaligned):.6f}, std={np.std(vae_vals_unaligned):.6f}, n={len(vae_vals_unaligned)}")
print(f"  Econ:   mean={np.mean(econ_vals_unaligned):.6f}, std={np.std(econ_vals_unaligned):.6f}, n={len(econ_vals_unaligned)}")
print()

# NOW align properly
min_len = min(gt.shape[0], oracle_p50.shape[0], vae_p50.shape[0], econ_p50.shape[0])
print(f"Aligning to {min_len} days...")
print()

gt_aligned = gt[:min_len]
oracle_aligned = oracle_p50[:min_len]
vae_aligned = vae_p50[:min_len]
econ_aligned = econ_p50[:min_len]

gt_vals = gt_aligned[:, i, j]
oracle_vals = oracle_aligned[:, i, j]
vae_vals = vae_aligned[:, i, j]
econ_vals = econ_aligned[:, i, j]

print("Aligned statistics:")
print(f"  GT:     mean={np.mean(gt_vals):.6f}, std={np.std(gt_vals):.6f}, n={len(gt_vals)}")
print(f"  Oracle: mean={np.mean(oracle_vals):.6f}, std={np.std(oracle_vals):.6f}, n={len(oracle_vals)}")
print(f"  VAE:    mean={np.mean(vae_vals):.6f}, std={np.std(vae_vals):.6f}, n={len(vae_vals)}")
print(f"  Econ:   mean={np.mean(econ_vals):.6f}, std={np.std(econ_vals):.6f}, n={len(econ_vals)}")
print()

print("Variance ratios (aligned):")
gt_std = np.std(gt_vals)
print(f"  Oracle / GT: {np.std(oracle_vals) / gt_std:.4f}")
print(f"  VAE / GT:    {np.std(vae_vals) / gt_std:.4f}")
print(f"  Econ / GT:   {np.std(econ_vals) / gt_std:.4f}")
print()

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Unaligned (what viz script shows)
ax = axes[0]
bins = 40
alpha = 0.4
ax.hist(gt_vals_unaligned, bins=bins, alpha=alpha, color='black', density=True, label='GT')
ax.hist(oracle_vals_unaligned, bins=bins, alpha=alpha, color='blue', density=True, label='Oracle')
ax.hist(vae_vals_unaligned, bins=bins, alpha=alpha, color='red', density=True, label='VAE')
ax.hist(econ_vals_unaligned, bins=bins, alpha=alpha, color='green', density=True, label='Econ')
ax.set_title(f'UNALIGNED (current viz script)\nGT std={np.std(gt_vals_unaligned):.4f}, VAE std={np.std(vae_vals_unaligned):.4f}', fontweight='bold')
ax.set_xlabel('IV')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Aligned (correct comparison)
ax = axes[1]
ax.hist(gt_vals, bins=bins, alpha=alpha, color='black', density=True, label='GT')
ax.hist(oracle_vals, bins=bins, alpha=alpha, color='blue', density=True, label='Oracle')
ax.hist(vae_vals, bins=bins, alpha=alpha, color='red', density=True, label='VAE')
ax.hist(econ_vals, bins=bins, alpha=alpha, color='green', density=True, label='Econ')
ax.set_title(f'ALIGNED (correct)\nGT std={np.std(gt_vals):.4f}, VAE std={np.std(vae_vals):.4f}, ratio={np.std(vae_vals)/np.std(gt_vals):.4f}', fontweight='bold')
ax.set_xlabel('IV')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/debug_histogram_alignment.png', dpi=150, bbox_inches='tight')
print("Saved comparison plot to: results/debug_histogram_alignment.png")
print()

# Check if alignment matters
print("DIAGNOSIS:")
diff_unaligned = np.abs(np.std(gt_vals_unaligned) - np.std(vae_vals_unaligned))
diff_aligned = np.abs(np.std(gt_vals) - np.std(vae_vals))
print(f"  Std difference (unaligned): {diff_unaligned:.6f}")
print(f"  Std difference (aligned):   {diff_aligned:.6f}")

if diff_aligned < diff_unaligned * 0.8:
    print("\n✓ ALIGNMENT FIXES THE ISSUE!")
    print("  The visualization script needs to align data lengths before plotting.")
else:
    print("\n⚠ Alignment doesn't fully explain the visual mismatch.")
    print("  Other factors (binning, overlapping colors) may be at play.")
