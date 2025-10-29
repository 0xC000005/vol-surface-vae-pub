import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_ci_comparison_timeseries(actual, baseline_gen, empirical_gen, dates,
                                   model_name, grid_name, noise_scale, output_path):
    """
    Plot time series with CI bands comparing baseline vs empirical sampling.
    """
    # Compute CIs
    baseline_p05 = np.percentile(baseline_gen, 5, axis=1)
    baseline_p95 = np.percentile(baseline_gen, 95, axis=1)
    baseline_mean = np.mean(baseline_gen, axis=1)

    empirical_p05 = np.percentile(empirical_gen, 5, axis=1)
    empirical_p95 = np.percentile(empirical_gen, 95, axis=1)
    empirical_mean = np.mean(empirical_gen, axis=1)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Subplot 1: Baseline (N(0,1))
    ax = axes[0]
    ax.plot(dates, actual, 'k-', linewidth=1.5, label='Actual', zorder=3)
    ax.fill_between(dates, baseline_p05, baseline_p95, alpha=0.3, color='red', label='90% CI (N(0,1))')
    ax.plot(dates, baseline_mean, 'r--', linewidth=1, label='Mean (N(0,1))', alpha=0.7)
    ax.set_ylabel('Implied Volatility', fontsize=11)
    ax.set_title(f'{model_name} - {grid_name} - Baseline N(0,1) Sampling', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Empirical
    ax = axes[1]
    ax.plot(dates, actual, 'k-', linewidth=1.5, label='Actual', zorder=3)
    ax.fill_between(dates, empirical_p05, empirical_p95, alpha=0.3, color='blue', label=f'90% CI (Empirical, noise={noise_scale})')
    ax.plot(dates, empirical_mean, 'b--', linewidth=1, label=f'Mean (Empirical)', alpha=0.7)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Implied Volatility', fontsize=11)
    ax.set_title(f'{model_name} - {grid_name} - Empirical Latent Sampling', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_violation_rate_comparison(results, output_path):
    """
    Bar plot comparing violation rates across models and grid points.
    """
    models = list(set([r['model'] for r in results]))
    grid_points = list(set([r['grid_point'] for r in results]))

    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for i, model_name in enumerate(models):
        ax = axes[i]
        model_results = [r for r in results if r['model'] == model_name]

        x = np.arange(len(grid_points))
        width = 0.35

        baseline_violations = [r['baseline_violations'] for r in model_results]
        empirical_violations = [r['empirical_violations'] for r in model_results]

        bars1 = ax.bar(x - width/2, baseline_violations, width, label='Baseline (N(0,1))', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, empirical_violations, width, label='Empirical', color='blue', alpha=0.7)

        # Add target line at 10%
        ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (10%)')

        ax.set_xlabel('Grid Point', fontsize=11)
        ax.set_ylabel('CI Violation Rate (%)', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([gp.replace(' ', '\n') for gp in grid_points], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("="*70)
    print("Comparison Visualization: Baseline vs Empirical Sampling")
    print("="*70)

    # Parameters
    base_folder = "test_spx/2024_11_09"
    noise_scale = 0.3
    start_day = 5
    days_to_generate = 5810
    PLOT_LAST_N = 200  # Plot last 200 days for clarity

    # Grid points
    grid_points = [
        {"name": "ATM 3-Month", "row": 2, "col": 2},
        {"name": "ATM 1-Year", "row": 2, "col": 4},
        {"name": "OTM Put 1-Year", "row": 1, "col": 4}
    ]

    # Models
    models = ["no_ex", "ex_no_loss", "ex_loss"]

    # Load actual data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    actual_surfaces = data["surface"][start_day:start_day+days_to_generate]

    # Load SPX data for dates
    spx_data = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    dates = pd.to_datetime(spx_data['date'].values[start_day:start_day+days_to_generate])

    results = []

    # Generate time series plots
    print("\nGenerating time series comparisons...")
    for model_name in models:
        print(f"\n  Model: {model_name}")

        # Load generations
        baseline_path = f"{base_folder}/{model_name}_gen5.npz"
        empirical_path = f"{base_folder}/{model_name}_empirical_gen5_noise{noise_scale}.npz"

        baseline_surfaces = np.load(baseline_path)["surfaces"]
        empirical_surfaces = np.load(empirical_path)["surfaces"]

        # Focus on last N days for plotting
        actual_plot = actual_surfaces[-PLOT_LAST_N:]
        baseline_plot = baseline_surfaces[-PLOT_LAST_N:]
        empirical_plot = empirical_surfaces[-PLOT_LAST_N:]
        dates_plot = dates[-PLOT_LAST_N:]

        for grid_point in grid_points:
            grid_name = grid_point["name"]
            grid_row = grid_point["row"]
            grid_col = grid_point["col"]

            print(f"    Grid: {grid_name}")

            actual_values = actual_plot[:, grid_row, grid_col]
            baseline_gen = baseline_plot[:, :, grid_row, grid_col]
            empirical_gen = empirical_plot[:, :, grid_row, grid_col]

            output_path = f"{base_folder}/timeseries_{model_name}_{grid_name.replace(' ', '_').replace('-', '_').lower()}.png"
            plot_ci_comparison_timeseries(
                actual_values, baseline_gen, empirical_gen, dates_plot,
                model_name, grid_name, noise_scale, output_path
            )
            print(f"      ✓ Saved: {output_path}")

            # Compute violation rates for bar chart
            baseline_p05 = np.percentile(baseline_gen, 5, axis=1)
            baseline_p95 = np.percentile(baseline_gen, 95, axis=1)
            empirical_p05 = np.percentile(empirical_gen, 5, axis=1)
            empirical_p95 = np.percentile(empirical_gen, 95, axis=1)

            baseline_violations = 100 * np.mean((actual_values < baseline_p05) | (actual_values > baseline_p95))
            empirical_violations = 100 * np.mean((actual_values < empirical_p05) | (actual_values > empirical_p95))

            results.append({
                "model": model_name,
                "grid_point": grid_name,
                "baseline_violations": baseline_violations,
                "empirical_violations": empirical_violations
            })

    # Generate bar chart comparison
    print("\nGenerating violation rate comparison bar chart...")
    bar_chart_path = f"{base_folder}/violation_rates_comparison.png"
    plot_violation_rate_comparison(results, bar_chart_path)
    print(f"  ✓ Saved: {bar_chart_path}")

    print("\n" + "="*70)
    print("Comparison Visualization Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
