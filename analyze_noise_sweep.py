import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

def compute_ci_violations_from_file(surfaces_path, actual_surfaces, grid_row, grid_col, last_n_days=1000):
    """
    Compute CI violations from generated surface file.

    Returns dict with violation statistics.
    """
    # Load generated surfaces
    data = np.load(surfaces_path)
    generated_surfaces = data["surfaces"]  # (num_days, 1000, 5, 5)

    # Focus on last N days
    actual_test = actual_surfaces[-last_n_days:]
    generated_test = generated_surfaces[-last_n_days:]

    # Extract grid point
    actual_values = actual_test[:, grid_row, grid_col]  # (1000,)
    generated_values = generated_test[:, :, grid_row, grid_col]  # (1000, 1000)

    # Compute CI bounds
    p05 = np.percentile(generated_values, 5, axis=1)
    p95 = np.percentile(generated_values, 95, axis=1)

    # Check violations
    below_ci = actual_values < p05
    above_ci = actual_values > p95
    outside_ci = below_ci | above_ci

    # Compute statistics
    num_violations = np.sum(outside_ci)
    pct_violations = 100.0 * num_violations / last_n_days
    pct_below = 100.0 * np.sum(below_ci) / last_n_days
    pct_above = 100.0 * np.sum(above_ci) / last_n_days

    # CI width
    ci_width = p95 - p05
    mean_ci_width = np.mean(ci_width)

    return {
        "violations": pct_violations,
        "below": pct_below,
        "above": pct_above,
        "ci_width": mean_ci_width,
        "num_violations": num_violations,
        "num_days": last_n_days
    }

def main():
    print("="*70)
    print("Noise Sweep Analysis")
    print("="*70)

    # Parameters
    base_folder = "test_spx/2024_11_09"
    noise_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    models = ["no_ex", "ex_no_loss", "ex_loss"]
    grid_points = [
        {"name": "ATM 3-Month", "row": 2, "col": 2},
        {"name": "ATM 1-Year", "row": 2, "col": 4},
        {"name": "OTM Put 1-Year", "row": 1, "col": 4}
    ]

    # Load actual data
    print("\nLoading actual data...")
    data = np.load("data/vol_surface_with_ret.npz")
    actual_surfaces = data["surface"][5:5+5810]  # (5810, 5, 5)
    print(f"  Actual surfaces shape: {actual_surfaces.shape}")

    # Collect results
    results = []

    print("\nProcessing all combinations...")
    for model_name in models:
        print(f"\n  Model: {model_name}")
        for grid in grid_points:
            print(f"    Grid: {grid['name']}")
            for noise in noise_values:
                empirical_path = f"{base_folder}/{model_name}_empirical_gen5_noise{noise}.npz"

                try:
                    stats = compute_ci_violations_from_file(
                        empirical_path, actual_surfaces,
                        grid["row"], grid["col"]
                    )

                    results.append({
                        "model": model_name,
                        "grid_point": grid["name"],
                        "noise": noise,
                        "violations": stats["violations"],
                        "below_p05": stats["below"],
                        "above_p95": stats["above"],
                        "ci_width": stats["ci_width"]
                    })

                    print(f"      noise={noise}: {stats['violations']:.2f}% violations, CI width={stats['ci_width']:.4f}")

                except FileNotFoundError:
                    print(f"      noise={noise}: FILE NOT FOUND - {empirical_path}")
                    continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv("noise_sweep_results.csv", index=False)
    print(f"\n✓ Results saved to: noise_sweep_results.csv")

    # Create visualizations
    print("\nGenerating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Noise vs Violation Rate (all models, all grid points)
    ax1 = plt.subplot(2, 2, 1)
    for model in models:
        for grid in grid_points:
            subset = df[(df["model"] == model) & (df["grid_point"] == grid["name"])]
            if len(subset) > 0:
                label = f"{model} - {grid['name']}"
                ax1.plot(subset["noise"], subset["violations"], marker='o', label=label, linewidth=2)

    ax1.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (10%)')
    ax1.set_xlabel('Noise Scale', fontsize=12)
    ax1.set_ylabel('CI Violation Rate (%)', fontsize=12)
    ax1.set_title('Noise Scale vs CI Violation Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')

    # Plot 2: Noise vs CI Width (all models, all grid points)
    ax2 = plt.subplot(2, 2, 2)
    for model in models:
        for grid in grid_points:
            subset = df[(df["model"] == model) & (df["grid_point"] == grid["name"])]
            if len(subset) > 0:
                label = f"{model} - {grid['name']}"
                ax2.plot(subset["noise"], subset["ci_width"], marker='o', label=label, linewidth=2)

    ax2.set_xlabel('Noise Scale', fontsize=12)
    ax2.set_ylabel('Mean CI Width', fontsize=12)
    ax2.set_title('Noise Scale vs CI Width (Coverage-Precision Trade-off)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')

    # Plot 3: Average violations by noise (across all models/grid points)
    ax3 = plt.subplot(2, 2, 3)
    avg_violations = df.groupby("noise")["violations"].mean()
    std_violations = df.groupby("noise")["violations"].std()
    ax3.errorbar(avg_violations.index, avg_violations.values, yerr=std_violations.values,
                marker='o', linewidth=2, capsize=5, capthick=2, label='Average ± Std')
    ax3.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (10%)')
    ax3.set_xlabel('Noise Scale', fontsize=12)
    ax3.set_ylabel('Average CI Violation Rate (%)', fontsize=12)
    ax3.set_title('Average Violations Across All Models/Grid Points', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Plot 4: Violation asymmetry (above vs below)
    ax4 = plt.subplot(2, 2, 4)
    avg_below = df.groupby("noise")["below_p05"].mean()
    avg_above = df.groupby("noise")["above_p95"].mean()
    x = np.arange(len(noise_values))
    width = 0.35
    ax4.bar(x - width/2, avg_below, width, label='Below p05', alpha=0.7)
    ax4.bar(x + width/2, avg_above, width, label='Above p95', alpha=0.7)
    ax4.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Target (5% each)')
    ax4.set_xlabel('Noise Scale', fontsize=12)
    ax4.set_ylabel('Violation Rate (%)', fontsize=12)
    ax4.set_title('Violation Asymmetry: Below p05 vs Above p95', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(n) for n in noise_values])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("noise_sweep_summary.png", dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: noise_sweep_summary.png")

    # Find optimal noise
    print("\n" + "="*70)
    print("OPTIMAL NOISE IDENTIFICATION")
    print("="*70)

    # Find noise closest to 10% target (averaged across all)
    avg_violations = df.groupby("noise")["violations"].mean()
    distance_from_target = np.abs(avg_violations - 10)
    optimal_noise = distance_from_target.idxmin()
    optimal_violations = avg_violations[optimal_noise]

    print(f"\nOptimal noise_scale: {optimal_noise}")
    print(f"Average violations at optimal: {optimal_violations:.2f}%")
    print(f"Distance from target (10%): {abs(optimal_violations - 10):.2f}%")

    # Show per-model results at optimal noise
    print(f"\nBreakdown at optimal noise={optimal_noise}:")
    optimal_df = df[df["noise"] == optimal_noise]
    for model in models:
        model_df = optimal_df[optimal_df["model"] == model]
        avg_viol = model_df["violations"].mean()
        print(f"  {model:12s}: {avg_viol:.2f}% (avg across grid points)")

    # Create summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE: Violations by Noise Scale")
    print("="*70)
    summary_table = df.groupby("noise").agg({
        "violations": ["mean", "std", "min", "max"],
        "ci_width": ["mean"]
    }).round(2)
    print(summary_table)

    # Save markdown summary
    with open("noise_sweep_summary.md", 'w') as f:
        f.write("# Noise Sweep Summary\n\n")
        f.write(f"**Optimal noise_scale:** {optimal_noise}\n\n")
        f.write(f"**Average violations at optimal:** {optimal_violations:.2f}%\n\n")
        f.write(f"**Distance from target (10%):** {abs(optimal_violations - 10):.2f}%\n\n")
        f.write("\n## Full Results by Noise Scale\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary Statistics\n\n")
        f.write(summary_table.to_markdown())

    print(f"\n✓ Summary saved to: noise_sweep_summary.md")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - noise_sweep_results.csv")
    print("  - noise_sweep_summary.png")
    print("  - noise_sweep_summary.md")

if __name__ == "__main__":
    main()
