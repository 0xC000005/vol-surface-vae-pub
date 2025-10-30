"""
Visualize marginal distributions of training vs validation data.

Creates histograms comparing:
- 25 volatility surface grid points (5x5 layout)
- 1 return distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
surfaces = data["surface"]  # Shape: (5822, 5, 5)
returns = data["ret"]  # Shape: (5822,)

# Split train/validation
train_surfaces = surfaces[:4000]  # (4000, 5, 5)
val_surfaces = surfaces[4000:5000]  # (1000, 5, 5)
train_returns = returns[:4000]  # (4000,)
val_returns = returns[4000:5000]  # (1000,)

print(f"Training samples: {len(train_surfaces)}")
print(f"Validation samples: {len(val_surfaces)}")

# Create figure with GridSpec for flexible layout
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(6, 5, figure=fig, hspace=0.4, wspace=0.3)

# Moneyness and maturity labels (you can adjust these based on actual grid)
moneyness_labels = ['0.8', '0.9', '1.0', '1.1', '1.2']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

# Plot 25 volatility surface grid points (5x5)
print("Creating volatility surface histograms...")
for i in range(5):  # moneyness
    for j in range(5):  # maturity
        ax = fig.add_subplot(gs[i, j])

        # Extract data for this grid point
        train_vals = train_surfaces[:, i, j]
        val_vals = val_surfaces[:, i, j]

        # Determine common bin range
        all_vals = np.concatenate([train_vals, val_vals])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)

        # Plot histograms
        ax.hist(train_vals, bins=bins, alpha=0.6, label='Train', color='blue', density=True)
        ax.hist(val_vals, bins=bins, alpha=0.6, label='Val', color='orange', density=True)

        # Add statistics
        train_mean = train_vals.mean()
        val_mean = val_vals.mean()
        train_std = train_vals.std()
        val_std = val_vals.std()

        # Title with grid position
        ax.set_title(f'M={moneyness_labels[j]}, T={maturity_labels[i]}\n'
                    f'Train: μ={train_mean:.3f}, σ={train_std:.3f}\n'
                    f'Val: μ={val_mean:.3f}, σ={val_std:.3f}',
                    fontsize=8)

        ax.set_xlabel('Implied Vol', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # Add legend only on first subplot
        if i == 0 and j == 0:
            ax.legend(fontsize=8)

# Plot returns distribution (bottom, spanning full width)
print("Creating return histogram...")
ax_ret = fig.add_subplot(gs[5, :])

# Determine common bin range for returns
all_returns = np.concatenate([train_returns, val_returns])
bins_ret = np.linspace(all_returns.min(), all_returns.max(), 60)

# Plot histograms
ax_ret.hist(train_returns, bins=bins_ret, alpha=0.6, label='Train', color='blue', density=True)
ax_ret.hist(val_returns, bins=bins_ret, alpha=0.6, label='Val', color='orange', density=True)

# Add statistics
train_ret_mean = train_returns.mean()
val_ret_mean = val_returns.mean()
train_ret_std = train_returns.std()
val_ret_std = val_returns.std()

ax_ret.set_title(f'Daily Log Returns\n'
                f'Train: μ={train_ret_mean:.5f}, σ={train_ret_std:.5f} | '
                f'Val: μ={val_ret_mean:.5f}, σ={val_ret_std:.5f}',
                fontsize=12, fontweight='bold')
ax_ret.set_xlabel('Log Return', fontsize=10)
ax_ret.set_ylabel('Density', fontsize=10)
ax_ret.legend(fontsize=10)
ax_ret.grid(True, alpha=0.3)

# Main title
fig.suptitle('Marginal Distributions: Training vs Validation Data\n'
            '(5×5 Volatility Surface Grid + Returns)',
            fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_path = 'marginal_dist_train_val.png'
print(f"Saving figure to {output_path}...")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# Also save as PDF
output_path_pdf = 'marginal_dist_train_val.pdf'
print(f"Saving PDF to {output_path_pdf}...")
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"✓ Saved: {output_path_pdf}")

print("\nDone! Generated 26 histograms (25 surface grid points + 1 return distribution)")
