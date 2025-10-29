import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_generated_surfaces(file_path, key="surfaces"):
    """Load generated surfaces from npz file."""
    data = np.load(file_path)
    return data[key]

def compute_marginal_distribution(surfaces, grid_row, grid_col):
    """
    Compute marginal distribution by pooling all days and samples.

    Args:
        surfaces: (num_days, num_samples, 5, 5) or (num_days, 5, 5)
        grid_row, grid_col: Grid point to analyze

    Returns:
        Flattened array of all values at this grid point
    """
    if len(surfaces.shape) == 4:
        # (num_days, num_samples, 5, 5) → pool all days and samples
        values = surfaces[:, :, grid_row, grid_col].flatten()
    elif len(surfaces.shape) == 3:
        # (num_days, 5, 5) → pool all days
        values = surfaces[:, grid_row, grid_col].flatten()
    else:
        raise ValueError(f"Unexpected shape: {surfaces.shape}")
    return values

def plot_marginal_comparison(actual_values, baseline_values, empirical_values,
                              model_name, grid_name, noise_scale, output_path):
    """
    Plot marginal distribution comparison: actual vs N(0,1) baseline vs empirical.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Histogram for actual
    ax.hist(actual_values, bins=50, density=True, alpha=0.4, color='black',
            label=f'Actual (n={len(actual_values)})', edgecolor='black')

    # Histogram for N(0,1) baseline
    ax.hist(baseline_values, bins=50, density=True, alpha=0.4, color='red',
            label=f'N(0,1) Baseline (n={len(baseline_values)})', edgecolor='red')

    # Histogram for empirical sampling
    ax.hist(empirical_values, bins=50, density=True, alpha=0.4, color='blue',
            label=f'Empirical (noise={noise_scale}) (n={len(empirical_values)})', edgecolor='blue')

    # KS test statistics
    ks_baseline = stats.ks_2samp(actual_values, baseline_values)
    ks_empirical = stats.ks_2samp(actual_values, empirical_values)

    ax.set_xlabel('Implied Volatility', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{model_name} - {grid_name}\nKS: Baseline={ks_baseline.statistic:.4f}, Empirical={ks_empirical.statistic:.4f}',
                fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return ks_baseline.statistic, ks_empirical.statistic

def main():
    print("="*70)
    print("Marginal Distribution Verification")
    print("="*70)

    # Parameters
    base_folder = "test_spx/2024_11_09"
    noise_scale = 0.3
    start_day = 5
    days_to_generate = 5810

    # Grid points to analyze
    grid_points = [
        {"name": "ATM 3-Month", "row": 2, "col": 2},
        {"name": "ATM 1-Year", "row": 2, "col": 4},
        {"name": "OTM Put 1-Year", "row": 1, "col": 4}
    ]

    # Models to analyze
    models = ["no_ex", "ex_no_loss", "ex_loss"]

    # Load actual data
    print("\nLoading actual data...")
    data = np.load("data/vol_surface_with_ret.npz")
    actual_surfaces = data["surface"][start_day:start_day+days_to_generate]  # (5810, 5, 5)
    print(f"  Actual surfaces shape: {actual_surfaces.shape}")

    # Results storage
    results = []

    # Process each model and grid point
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        # Load generations
        baseline_path = f"{base_folder}/{model_name}_gen5.npz"
        empirical_path = f"{base_folder}/{model_name}_empirical_gen5_noise{noise_scale}.npz"

        print(f"  Loading baseline: {baseline_path}")
        baseline_surfaces = load_generated_surfaces(baseline_path)  # (5810, 1000, 5, 5)

        print(f"  Loading empirical: {empirical_path}")
        empirical_surfaces = load_generated_surfaces(empirical_path)  # (5810, 1000, 5, 5)

        print(f"  Baseline shape: {baseline_surfaces.shape}")
        print(f"  Empirical shape: {empirical_surfaces.shape}")

        for grid_point in grid_points:
            grid_name = grid_point["name"]
            grid_row = grid_point["row"]
            grid_col = grid_point["col"]

            print(f"\n  Grid point: {grid_name} [{grid_row}, {grid_col}]")

            # Compute marginal distributions
            actual_marginal = compute_marginal_distribution(actual_surfaces, grid_row, grid_col)
            baseline_marginal = compute_marginal_distribution(baseline_surfaces, grid_row, grid_col)
            empirical_marginal = compute_marginal_distribution(empirical_surfaces, grid_row, grid_col)

            print(f"    Actual:    n={len(actual_marginal):7d}, mean={np.mean(actual_marginal):.4f}, std={np.std(actual_marginal):.4f}")
            print(f"    Baseline:  n={len(baseline_marginal):7d}, mean={np.mean(baseline_marginal):.4f}, std={np.std(baseline_marginal):.4f}")
            print(f"    Empirical: n={len(empirical_marginal):7d}, mean={np.mean(empirical_marginal):.4f}, std={np.std(empirical_marginal):.4f}")

            # Plot comparison
            output_path = f"{base_folder}/marginal_dist_{model_name}_{grid_name.replace(' ', '_').replace('-', '_').lower()}.png"
            ks_baseline, ks_empirical = plot_marginal_comparison(
                actual_marginal, baseline_marginal, empirical_marginal,
                model_name, grid_name, noise_scale, output_path
            )

            print(f"    KS Baseline:  {ks_baseline:.4f}")
            print(f"    KS Empirical: {ks_empirical:.4f}")
            print(f"    ✓ Saved plot: {output_path}")

            # Store results
            results.append({
                "model": model_name,
                "grid_point": grid_name,
                "ks_baseline": ks_baseline,
                "ks_empirical": ks_empirical,
                "improvement": ks_baseline - ks_empirical  # Positive = empirical is better
            })

    # Print summary
    print("\n" + "="*70)
    print("Summary: KS Test Statistics (lower = better match to actual)")
    print("="*70)
    print(f"{'Model':<15} {'Grid Point':<20} {'Baseline':<12} {'Empirical':<12} {'Improvement':<12}")
    print("-"*70)
    for result in results:
        improvement_str = f"+{result['improvement']:.4f}" if result['improvement'] > 0 else f"{result['improvement']:.4f}"
        print(f"{result['model']:<15} {result['grid_point']:<20} {result['ks_baseline']:<12.4f} "
              f"{result['ks_empirical']:<12.4f} {improvement_str:<12}")

    # Overall assessment
    avg_improvement = np.mean([r['improvement'] for r in results])
    print("-"*70)
    print(f"Average improvement: {avg_improvement:+.4f}")

    if avg_improvement > 0:
        print("\n✓ Empirical sampling produces marginal distributions CLOSER to actual")
    else:
        print("\n✗ Empirical sampling produces marginal distributions FURTHER from actual")

    print("\n" + "="*70)
    print("Marginal Distribution Verification Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
