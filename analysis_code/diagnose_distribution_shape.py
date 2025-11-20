"""
Diagnose distribution SHAPE differences (not just variance).

Computes kurtosis and skewness to understand why GT appears flat/spread
while models appear peaked, even when variance is similar.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

PERIOD = 'oos'  # or 'insample'
HORIZON = 1
TEST_GRID_I = 2  # ATM
TEST_GRID_J = 2  # 6M

moneyness_labels = ['0.70', '0.85', '1.00', '1.15', '1.30']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

output_dir = Path("tables/distribution_shape_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Load Data
# ============================================================================

print("="*80)
print(f"DISTRIBUTION SHAPE DIAGNOSTIC: {PERIOD.upper()} H{HORIZON}")
print("="*80)
print()

if PERIOD == 'insample':
    oracle_file = "models_backfill/insample_reconstruction_16yr.npz"
    vae_file = "models_backfill/vae_prior_insample_16yr.npz"
    econ_file = "tables/econometric_backfill/econometric_backfill_insample.npz"
else:
    oracle_file = "models_backfill/oos_reconstruction_16yr.npz"
    vae_file = "models_backfill/vae_prior_oos_16yr.npz"
    econ_file = "tables/econometric_backfill/econometric_backfill_oos.npz"

oracle_data = np.load(oracle_file)
vae_data = np.load(vae_file)
econ_data = np.load(econ_file)
gt_data = np.load("data/vol_surface_with_ret.npz")

recon_key = f'recon_h{HORIZON}'
indices_key = f'indices_h{HORIZON}'

oracle_p50 = oracle_data[recon_key][:, 1, :, :]
vae_p50 = vae_data[recon_key][:, 1, :, :]
econ_p50 = econ_data[recon_key][:, 1, :, :]

indices = oracle_data[indices_key]
gt = gt_data["surface"][indices]

# Align lengths
min_len = min(gt.shape[0], oracle_p50.shape[0], vae_p50.shape[0], econ_p50.shape[0])
gt = gt[:min_len]
oracle_p50 = oracle_p50[:min_len]
vae_p50 = vae_p50[:min_len]
econ_p50 = econ_p50[:min_len]

print(f"Data loaded: {min_len} days")
print()

# ============================================================================
# Detailed Analysis: Single Grid Point
# ============================================================================

i, j = TEST_GRID_I, TEST_GRID_J
print(f"Grid Point ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity")
print("="*80)
print()

gt_vals = gt[:, i, j]
oracle_vals = oracle_p50[:, i, j]
vae_vals = vae_p50[:, i, j]
econ_vals = econ_p50[:, i, j]

# Compute distributional statistics
def compute_stats(data, name):
    """Compute comprehensive distributional statistics."""
    return {
        'name': name,
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data, fisher=False),  # fisher=False gives Pearson (normal=3)
        'excess_kurtosis': stats.kurtosis(data, fisher=True),  # fisher=True gives excess (normal=0)
        'p01': np.percentile(data, 1),
        'p05': np.percentile(data, 5),
        'p25': np.percentile(data, 25),
        'p50': np.percentile(data, 50),
        'p75': np.percentile(data, 75),
        'p95': np.percentile(data, 95),
        'p99': np.percentile(data, 99),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'range': np.max(data) - np.min(data),
    }

gt_stats = compute_stats(gt_vals, 'Ground Truth')
oracle_stats = compute_stats(oracle_vals, 'Oracle')
vae_stats = compute_stats(vae_vals, 'VAE Prior')
econ_stats = compute_stats(econ_vals, 'Econometric')

# Print statistics
print("DISTRIBUTIONAL STATISTICS")
print("-"*80)
print(f"{'Metric':<20} {'GT':<12} {'Oracle':<12} {'VAE':<12} {'Econometric':<12}")
print("-"*80)
print(f"{'Mean':<20} {gt_stats['mean']:<12.6f} {oracle_stats['mean']:<12.6f} {vae_stats['mean']:<12.6f} {econ_stats['mean']:<12.6f}")
print(f"{'Std':<20} {gt_stats['std']:<12.6f} {oracle_stats['std']:<12.6f} {vae_stats['std']:<12.6f} {econ_stats['std']:<12.6f}")
print()
print(f"{'Skewness':<20} {gt_stats['skewness']:<12.4f} {oracle_stats['skewness']:<12.4f} {vae_stats['skewness']:<12.4f} {econ_stats['skewness']:<12.4f}")
print(f"{'Kurtosis':<20} {gt_stats['kurtosis']:<12.4f} {oracle_stats['kurtosis']:<12.4f} {vae_stats['kurtosis']:<12.4f} {econ_stats['kurtosis']:<12.4f}")
print(f"{'Excess Kurtosis':<20} {gt_stats['excess_kurtosis']:<12.4f} {oracle_stats['excess_kurtosis']:<12.4f} {vae_stats['excess_kurtosis']:<12.4f} {econ_stats['excess_kurtosis']:<12.4f}")
print()
print(f"{'p01':<20} {gt_stats['p01']:<12.6f} {oracle_stats['p01']:<12.6f} {vae_stats['p01']:<12.6f} {econ_stats['p01']:<12.6f}")
print(f"{'p25':<20} {gt_stats['p25']:<12.6f} {oracle_stats['p25']:<12.6f} {vae_stats['p25']:<12.6f} {econ_stats['p25']:<12.6f}")
print(f"{'p50 (median)':<20} {gt_stats['p50']:<12.6f} {oracle_stats['p50']:<12.6f} {vae_stats['p50']:<12.6f} {econ_stats['p50']:<12.6f}")
print(f"{'p75':<20} {gt_stats['p75']:<12.6f} {oracle_stats['p75']:<12.6f} {vae_stats['p75']:<12.6f} {econ_stats['p75']:<12.6f}")
print(f"{'p99':<20} {gt_stats['p99']:<12.6f} {oracle_stats['p99']:<12.6f} {vae_stats['p99']:<12.6f} {econ_stats['p99']:<12.6f}")
print()
print(f"{'IQR (p75-p25)':<20} {gt_stats['iqr']:<12.6f} {oracle_stats['iqr']:<12.6f} {vae_stats['iqr']:<12.6f} {econ_stats['iqr']:<12.6f}")
print(f"{'Range':<20} {gt_stats['range']:<12.6f} {oracle_stats['range']:<12.6f} {vae_stats['range']:<12.6f} {econ_stats['range']:<12.6f}")
print()

# Interpretation
print("="*80)
print("INTERPRETATION")
print("="*80)
print()

print("Kurtosis interpretation (Pearson, normal=3.0):")
for stat_dict in [gt_stats, oracle_stats, vae_stats, econ_stats]:
    k = stat_dict['kurtosis']
    if k > 4.0:
        shape = "VERY PEAKED (leptokurtic)"
    elif k > 3.5:
        shape = "Peaked (leptokurtic)"
    elif k >= 2.5:
        shape = "Normal-ish"
    elif k > 2.0:
        shape = "Flat (platykurtic)"
    else:
        shape = "VERY FLAT (platykurtic)"
    print(f"  {stat_dict['name']:<15}: {k:.2f} → {shape}")
print()

print("Skewness interpretation:")
for stat_dict in [gt_stats, oracle_stats, vae_stats, econ_stats]:
    s = stat_dict['skewness']
    if abs(s) < 0.5:
        shape = "Approximately symmetric"
    elif s > 0:
        shape = f"Right-skewed (long right tail)"
    else:
        shape = f"Left-skewed (long left tail)"
    print(f"  {stat_dict['name']:<15}: {s:+.2f} → {shape}")
print()

# ============================================================================
# Visualization
# ============================================================================

print("Creating visualization...")
print()

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Overlaid histograms (current viz)
ax1 = fig.add_subplot(gs[0, :])
bins = 40
alpha = 0.4
ax1.hist(gt_vals, bins=bins, alpha=alpha, color='black', density=True, label='Ground Truth', edgecolor='black', linewidth=1.5)
ax1.hist(oracle_vals, bins=bins, alpha=alpha, color='blue', density=True, label='Oracle', edgecolor='blue', linewidth=1.5)
ax1.hist(vae_vals, bins=bins, alpha=alpha, color='red', density=True, label='VAE Prior', edgecolor='red', linewidth=1.5)
ax1.hist(econ_vals, bins=bins, alpha=alpha, color='green', density=True, label='Econometric', edgecolor='green', linewidth=1.5)
ax1.set_title(f'Overlaid Histograms (current visualization)\n{PERIOD.upper()} H{HORIZON}, Grid ({i},{j})', fontsize=14, fontweight='bold')
ax1.set_xlabel('Implied Volatility')
ax1.set_ylabel('Density')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2-5: Individual distributions
for idx, (vals, stats_dict, color) in enumerate([
    (gt_vals, gt_stats, 'black'),
    (oracle_vals, oracle_stats, 'blue'),
    (vae_vals, vae_stats, 'red'),
    (econ_vals, econ_stats, 'green')
]):
    row = 1 + idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])

    ax.hist(vals, bins=bins, alpha=0.7, color=color, density=True, edgecolor=color, linewidth=1.5)
    ax.axvline(stats_dict['mean'], color='darkred', linestyle='--', linewidth=2, label=f"Mean={stats_dict['mean']:.4f}")
    ax.axvline(stats_dict['p50'], color='darkblue', linestyle=':', linewidth=2, label=f"Median={stats_dict['p50']:.4f}")

    # Add statistics text
    stats_text = (
        f"Std: {stats_dict['std']:.4f}\n"
        f"Skew: {stats_dict['skewness']:+.2f}\n"
        f"Kurt: {stats_dict['kurtosis']:.2f}\n"
        f"IQR: {stats_dict['iqr']:.4f}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))

    ax.set_title(stats_dict['name'], fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('Implied Volatility')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Distribution Shape Analysis: {PERIOD.upper()} H{HORIZON}\n'
             f'Grid ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity',
             fontsize=16, fontweight='bold')

output_file = output_dir / f'{PERIOD}_h{HORIZON}_shape_analysis_grid_{i}_{j}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# Summary: Why GT Looks Flat vs Models Peaked
# ============================================================================

print()
print("="*80)
print("SUMMARY: WHY GT LOOKS FLAT WHILE MODELS LOOK PEAKED")
print("="*80)
print()

print("Even with similar variance, distributions can have different SHAPES:")
print()
print(f"1. GT Kurtosis: {gt_stats['kurtosis']:.2f}")
print(f"   Oracle Kurtosis: {oracle_stats['kurtosis']:.2f}")
print(f"   VAE Kurtosis: {vae_stats['kurtosis']:.2f}")
print()

kurt_diff_oracle = oracle_stats['kurtosis'] - gt_stats['kurtosis']
kurt_diff_vae = vae_stats['kurtosis'] - gt_stats['kurtosis']

if kurt_diff_oracle > 0.5 or kurt_diff_vae > 0.5:
    print("✓ MODELS ARE MORE PEAKED (higher kurtosis):")
    print(f"  - Oracle is {kurt_diff_oracle:+.2f} more peaked than GT")
    print(f"  - VAE is {kurt_diff_vae:+.2f} more peaked than GT")
    print()
    print("  Interpretation:")
    print("  - Models produce predictions clustered around mean (tall peak)")
    print("  - GT has more spread-out values (flatter distribution)")
    print("  - This happens because models 'average' across regimes")
    print("  - GT mixes crisis + normal periods → more dispersed")
else:
    print("⚠ Kurtosis difference is small (<0.5)")
    print("  Visual difference might be due to:")
    print("  - Histogram binning artifacts")
    print("  - Overlapping transparent colors")
    print("  - Skewness differences")

print()
print(f"2. Std comparison (for reference):")
print(f"   GT: {gt_stats['std']:.6f}")
print(f"   Oracle: {oracle_stats['std']:.6f} (ratio: {oracle_stats['std']/gt_stats['std']:.4f})")
print(f"   VAE: {vae_stats['std']:.6f} (ratio: {vae_stats['std']/gt_stats['std']:.4f})")
print()
