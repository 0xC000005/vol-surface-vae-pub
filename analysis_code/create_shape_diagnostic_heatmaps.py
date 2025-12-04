"""
Create 5×5 heatmaps showing kurtosis and skewness differences across volatility surface.

Visualizes spatial patterns of distribution shape mismatch:
- Which grid points have worst kurtosis mismatch?
- Which grid points have worst skewness mismatch?
- How does the pattern vary by period and horizon?

Reads from: results/distribution_analysis/shape_diagnostics/shape_statistics_comprehensive.csv
Outputs to: results/distribution_analysis/shape_diagnostics/
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

csv_file = Path("results/distribution_analysis/shape_diagnostics/shape_statistics_comprehensive.csv")
output_dir = Path("results/distribution_analysis/shape_diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

moneyness_labels = ['0.70', '0.85', '1.00', '1.15', '1.30']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

# ============================================================================
# Load Data
# ============================================================================

print("="*80)
print("DISTRIBUTION SHAPE HEATMAP GENERATION")
print("="*80)
print()

print(f"Loading data from: {csv_file}")
df = pd.read_csv(csv_file)
print(f"  Loaded {len(df)} rows")
print()

# ============================================================================
# Helper Functions
# ============================================================================

def create_kurtosis_heatmap(period, horizon, df):
    """Create heatmap showing kurtosis differences for all models."""

    subset = df[(df['period'] == period) & (df['horizon'] == horizon)]

    # Create 5×5 grids
    oracle_kurt_diff = np.zeros((5, 5))
    vae_kurt_diff = np.zeros((5, 5))
    econ_kurt_diff = np.zeros((5, 5))

    for idx, row in subset.iterrows():
        i = row['moneyness_idx']
        j = row['maturity_idx']
        oracle_kurt_diff[i, j] = row['oracle_kurt_diff']
        vae_kurt_diff[i, j] = row['vae_kurt_diff']
        econ_kurt_diff[i, j] = row['econ_kurt_diff']

    # Create figure with 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'Kurtosis Difference Heatmaps: {period.upper()} H{horizon}\n'
                 f'Model Kurtosis - GT Kurtosis (positive = model more peaked)',
                 fontsize=16, fontweight='bold')

    # Determine colormap range (symmetric around 0)
    vmax = max(abs(oracle_kurt_diff).max(), abs(vae_kurt_diff).max(), abs(econ_kurt_diff).max())
    vmin = -vmax
    cmap = 'RdBu_r'  # Red = positive (more peaked), Blue = negative (flatter)

    # Plot Oracle
    ax = axes[0]
    im = ax.imshow(oracle_kurt_diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'Oracle\nMean: {oracle_kurt_diff.mean():+.2f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    # Annotate values
    for i in range(5):
        for j in range(5):
            color = 'white' if abs(oracle_kurt_diff[i, j]) > vmax*0.6 else 'black'
            ax.text(j, i, f'{oracle_kurt_diff[i, j]:+.1f}',
                   ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    # Plot VAE Prior
    ax = axes[1]
    im = ax.imshow(vae_kurt_diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'VAE Prior\nMean: {vae_kurt_diff.mean():+.2f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    for i in range(5):
        for j in range(5):
            color = 'white' if abs(vae_kurt_diff[i, j]) > vmax*0.6 else 'black'
            ax.text(j, i, f'{vae_kurt_diff[i, j]:+.1f}',
                   ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    # Plot Econometric
    ax = axes[2]
    im = ax.imshow(econ_kurt_diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'Econometric\nMean: {econ_kurt_diff.mean():+.2f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    for i in range(5):
        for j in range(5):
            color = 'white' if abs(econ_kurt_diff[i, j]) > vmax*0.6 else 'black'
            ax.text(j, i, f'{econ_kurt_diff[i, j]:+.1f}',
                   ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, aspect=40)
    cbar.set_label('Kurtosis Difference (Model - GT)', fontsize=12)
    cbar.ax.axvline(0, color='black', linewidth=2, linestyle='--', label='Perfect match')

    # Add interpretation guide
    guide_text = (
        "Interpretation:\n"
        "  Red (+): Model MORE PEAKED than GT (leptokurtic, concentrated around mean)\n"
        "  Blue (-): Model FLATTER than GT (platykurtic, more dispersed)\n"
        "  White (0): Perfect shape match"
    )
    fig.text(0.5, 0.01, guide_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    # Save
    output_file = output_dir / f'{period}_h{horizon}_kurtosis_heatmap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_skewness_heatmap(period, horizon, df):
    """Create heatmap showing skewness differences for all models."""

    subset = df[(df['period'] == period) & (df['horizon'] == horizon)]

    # Create 5×5 grids
    oracle_skew_diff = np.zeros((5, 5))
    vae_skew_diff = np.zeros((5, 5))
    econ_skew_diff = np.zeros((5, 5))

    for idx, row in subset.iterrows():
        i = row['moneyness_idx']
        j = row['maturity_idx']
        oracle_skew_diff[i, j] = row['oracle_skew_diff']
        vae_skew_diff[i, j] = row['vae_skew_diff']
        econ_skew_diff[i, j] = row['econ_skew_diff']

    # Create figure with 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'Skewness Difference Heatmaps: {period.upper()} H{horizon}\n'
                 f'Model Skewness - GT Skewness (positive = model more right-skewed)',
                 fontsize=16, fontweight='bold')

    # Determine colormap range (symmetric around 0)
    vmax = max(abs(oracle_skew_diff).max(), abs(vae_skew_diff).max(), abs(econ_skew_diff).max())
    vmin = -vmax
    cmap = 'PiYG'  # Purple = negative (left-skewed), Green = positive (right-skewed)

    # Plot Oracle
    ax = axes[0]
    im = ax.imshow(oracle_skew_diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'Oracle\nMean: {oracle_skew_diff.mean():+.3f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    # Annotate values
    for i in range(5):
        for j in range(5):
            color = 'white' if abs(oracle_skew_diff[i, j]) > vmax*0.6 else 'black'
            ax.text(j, i, f'{oracle_skew_diff[i, j]:+.2f}',
                   ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    # Plot VAE Prior
    ax = axes[1]
    im = ax.imshow(vae_skew_diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'VAE Prior\nMean: {vae_skew_diff.mean():+.3f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    for i in range(5):
        for j in range(5):
            color = 'white' if abs(vae_skew_diff[i, j]) > vmax*0.6 else 'black'
            ax.text(j, i, f'{vae_skew_diff[i, j]:+.2f}',
                   ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    # Plot Econometric
    ax = axes[2]
    im = ax.imshow(econ_skew_diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'Econometric\nMean: {econ_skew_diff.mean():+.3f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    for i in range(5):
        for j in range(5):
            color = 'white' if abs(econ_skew_diff[i, j]) > vmax*0.6 else 'black'
            ax.text(j, i, f'{econ_skew_diff[i, j]:+.2f}',
                   ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, aspect=40)
    cbar.set_label('Skewness Difference (Model - GT)', fontsize=12)
    cbar.ax.axvline(0, color='black', linewidth=2, linestyle='--', label='Perfect match')

    # Add interpretation guide
    guide_text = (
        "Interpretation:\n"
        "  Green (+): Model MORE RIGHT-SKEWED than GT (longer right tail)\n"
        "  Purple (-): Model MORE LEFT-SKEWED than GT (longer left tail)\n"
        "  White (0): Perfect skewness match"
    )
    fig.text(0.5, 0.01, guide_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    # Save
    output_file = output_dir / f'{period}_h{horizon}_skewness_heatmap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    periods = ['insample', 'oos']
    horizons = [1, 7, 14, 30]

    print("Generating kurtosis heatmaps...")
    print()

    for period in periods:
        for horizon in horizons:
            try:
                create_kurtosis_heatmap(period, horizon, df)
            except Exception as e:
                print(f"ERROR: Failed to create kurtosis heatmap for {period} H{horizon}")
                print(f"  {str(e)}")

    print()
    print("Generating skewness heatmaps...")
    print()

    for period in periods:
        for horizon in horizons:
            try:
                create_skewness_heatmap(period, horizon, df)
            except Exception as e:
                print(f"ERROR: Failed to create skewness heatmap for {period} H{horizon}")
                print(f"  {str(e)}")

    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated heatmaps in: {output_dir}")
    print()
    print("Files generated:")
    for period in periods:
        for horizon in horizons:
            kurt_file = f'{period}_h{horizon}_kurtosis_heatmap.png'
            skew_file = f'{period}_h{horizon}_skewness_heatmap.png'
            print(f"  ✓ {kurt_file}")
            print(f"  ✓ {skew_file}")
