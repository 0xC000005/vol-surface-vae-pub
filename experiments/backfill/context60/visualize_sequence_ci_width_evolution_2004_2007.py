"""
Oracle vs Prior Full Sequence CI Width Evolution Visualization (2004-2007)

Shows how confidence interval width evolves across the entire forecast horizon
(days 1-60 or 1-90) for teacher-forced predictions during 2004-2007.

This visualization reveals:
1. How CI width changes across the forecast horizon
2. Where oracle vs prior diverge in the sequence
3. Why NPZ avg/max metrics show different patterns than endpoint predictions

Usage:
    python experiments/backfill/context60/visualize_sequence_ci_width_evolution_2004_2007.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta


# Configuration
START_DATE = "2004-01-01"
END_DATE = "2007-12-31"
BASE_DIR = Path("results/context60_baseline")
PREDICTIONS_DIR = BASE_DIR / "predictions/teacher_forcing"
OUTPUT_DIR = BASE_DIR / "visualizations/2004_2007_confirmation"

# ATM 6M grid indices
MONEYNESS_IDX = 2  # K/S = 1.00
MATURITY_IDX = 2   # 6-month


def load_prediction_data(horizon):
    """
    Load oracle and prior prediction files for given horizon.

    Args:
        horizon: int (60 or 90)

    Returns:
        oracle_data, prior_data: npz file contents
    """
    oracle_file = PREDICTIONS_DIR / f"oracle/vae_tf_insample_h{horizon}.npz"
    prior_file = PREDICTIONS_DIR / f"prior/vae_tf_insample_h{horizon}.npz"

    oracle_data = np.load(oracle_file)
    prior_data = np.load(prior_file)

    return oracle_data, prior_data


def extract_ci_width_sequences(data, horizon):
    """
    Extract CI width (p95 - p05) for ATM 6M across full forecast horizon.

    Args:
        data: npz data with 'surfaces' key
        horizon: int (60 or 90)

    Returns:
        ci_width: (num_sequences, horizon) array of CI widths
    """
    surfaces = data['surfaces']  # (N, T, 3, 5, 5)

    # Extract ATM 6M quantiles
    p05 = surfaces[:, :, 0, MONEYNESS_IDX, MATURITY_IDX]  # (N, T)
    p95 = surfaces[:, :, 2, MONEYNESS_IDX, MATURITY_IDX]  # (N, T)

    ci_width = p95 - p05  # (N, T)

    return ci_width


def get_dates_for_sequences(indices, horizon):
    """
    Convert sequence indices to dates.

    Args:
        indices: array of starting indices
        horizon: int (60 or 90)

    Returns:
        dates: array of datetime objects for sequence start dates
    """
    # Base date (training data starts 2000-01-03)
    base_date = datetime(2000, 1, 3)

    # Convert indices to dates
    dates = np.array([base_date + timedelta(days=int(idx)) for idx in indices])

    return dates


def filter_to_period(ci_width, dates, start_date, end_date):
    """
    Filter sequences to 2004-2007 period.

    Args:
        ci_width: (N, T) array of CI widths
        dates: array of datetime objects
        start_date: string "YYYY-MM-DD"
        end_date: string "YYYY-MM-DD"

    Returns:
        filtered_ci_width: (N_filtered, T) array
        filtered_dates: array of datetime objects
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    mask = (dates >= start_dt) & (dates <= end_dt)

    return ci_width[mask], dates[mask]


def plot_individual_trajectories(ax, oracle_ci, prior_ci, dates, horizon, num_examples=3):
    """
    Plot individual sequence trajectories for oracle and prior.

    Args:
        ax: matplotlib axis
        oracle_ci: (N, T) array
        prior_ci: (N, T) array
        dates: array of datetime objects
        horizon: int
        num_examples: int, number of example sequences to plot
    """
    # Select evenly spaced examples
    n_sequences = len(dates)
    example_indices = np.linspace(0, n_sequences-1, num_examples, dtype=int)

    forecast_days = np.arange(1, horizon+1)

    for i, idx in enumerate(example_indices):
        alpha = 0.7 if i == 0 else 0.5
        label_o = 'Oracle' if i == 0 else None
        label_p = 'Prior' if i == 0 else None

        ax.plot(forecast_days, oracle_ci[idx], color='blue',
                alpha=alpha, linewidth=1.5, label=label_o)
        ax.plot(forecast_days, prior_ci[idx], color='orange',
                alpha=alpha, linewidth=1.5, label=label_p)

    ax.set_xlabel('Forecast Day', fontsize=11)
    ax.set_ylabel('CI Width (p95 - p05)', fontsize=11)
    ax.set_title(f'H={horizon}: Individual Sequence Trajectories',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # Add date labels for examples
    example_dates = dates[example_indices]
    dates_text = "Examples:\n" + "\n".join([d.strftime('%Y-%m-%d') for d in example_dates])
    ax.text(0.98, 0.02, dates_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_average_trajectory(ax, oracle_ci, prior_ci, horizon):
    """
    Plot average CI width trajectory with percentile bands.

    Args:
        ax: matplotlib axis
        oracle_ci: (N, T) array
        prior_ci: (N, T) array
        horizon: int
    """
    forecast_days = np.arange(1, horizon+1)

    # Compute statistics across sequences (axis 0)
    oracle_mean = oracle_ci.mean(axis=0)
    oracle_p25 = np.percentile(oracle_ci, 25, axis=0)
    oracle_p75 = np.percentile(oracle_ci, 75, axis=0)

    prior_mean = prior_ci.mean(axis=0)
    prior_p25 = np.percentile(prior_ci, 25, axis=0)
    prior_p75 = np.percentile(prior_ci, 75, axis=0)

    # Plot oracle
    ax.plot(forecast_days, oracle_mean, color='blue', linewidth=2, label='Oracle mean')
    ax.fill_between(forecast_days, oracle_p25, oracle_p75,
                     color='blue', alpha=0.2, label='Oracle IQR')

    # Plot prior
    ax.plot(forecast_days, prior_mean, color='orange', linewidth=2, label='Prior mean')
    ax.fill_between(forecast_days, prior_p25, prior_p75,
                     color='orange', alpha=0.2, label='Prior IQR')

    ax.set_xlabel('Forecast Day', fontsize=11)
    ax.set_ylabel('CI Width (p95 - p05)', fontsize=11)
    ax.set_title(f'H={horizon}: Average Trajectory (2004-2007)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Statistics box
    oracle_overall_mean = oracle_ci.mean()
    prior_overall_mean = prior_ci.mean()
    ratio = prior_overall_mean / oracle_overall_mean

    stats_text = (f"Overall Mean:\n"
                 f"Oracle: {oracle_overall_mean:.5f}\n"
                 f"Prior: {prior_overall_mean:.5f}\n"
                 f"Ratio: {ratio:.3f}×")

    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_ratio_evolution(ax, oracle_ci, prior_ci, horizon):
    """
    Plot how Prior/Oracle ratio evolves across forecast horizon.

    Args:
        ax: matplotlib axis
        oracle_ci: (N, T) array
        prior_ci: (N, T) array
        horizon: int
    """
    forecast_days = np.arange(1, horizon+1)

    # Compute mean ratio at each forecast day
    ratio_mean = (prior_ci / oracle_ci).mean(axis=0)
    ratio_p25 = np.percentile(prior_ci / oracle_ci, 25, axis=0)
    ratio_p75 = np.percentile(prior_ci / oracle_ci, 75, axis=0)

    # Plot
    ax.plot(forecast_days, ratio_mean, color='purple', linewidth=2, label='Mean ratio')
    ax.fill_between(forecast_days, ratio_p25, ratio_p75,
                     color='purple', alpha=0.2, label='IQR')

    # Reference line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1,
               alpha=0.5, label='Equal (1.0)')

    ax.set_xlabel('Forecast Day', fontsize=11)
    ax.set_ylabel('Prior/Oracle Ratio', fontsize=11)
    ax.set_title(f'H={horizon}: CI Width Ratio Evolution',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Statistics
    overall_ratio = ratio_mean.mean()
    pct_prior_wider = (ratio_mean > 1.0).sum() / len(ratio_mean) * 100

    stats_text = (f"Mean ratio: {overall_ratio:.3f}×\n"
                 f"Days prior>oracle: {pct_prior_wider:.1f}%")

    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_heatmap(ax, ci_width, dates, horizon, title, cmap='viridis'):
    """
    Plot heatmap of CI width by date and forecast day.

    Args:
        ax: matplotlib axis
        ci_width: (N, T) array
        dates: array of datetime objects
        horizon: int
        title: string
        cmap: colormap name
    """
    # Create meshgrid for plotting
    forecast_days = np.arange(1, horizon+1)

    # Plot heatmap
    im = ax.imshow(ci_width.T, aspect='auto', cmap=cmap, origin='lower',
                   extent=[mdates.date2num(dates[0]), mdates.date2num(dates[-1]),
                          forecast_days[0], forecast_days[-1]])

    # Format x-axis as dates
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Forecast Day', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CI Width', fontsize=10)

    return im


def main():
    """Main execution pipeline"""
    print("="*80)
    print("FULL SEQUENCE CI WIDTH EVOLUTION: 2004-2007")
    print("Oracle vs Prior Teacher Forcing Predictions")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    for horizon in [60, 90]:
        print(f"\n{'='*80}")
        print(f"PROCESSING HORIZON {horizon} DAYS")
        print(f"{'='*80}")

        # Load data
        print(f"\nLoading prediction data for H={horizon}...")
        oracle_data, prior_data = load_prediction_data(horizon)
        print(f"  Oracle file loaded: {oracle_data['surfaces'].shape}")
        print(f"  Prior file loaded: {prior_data['surfaces'].shape}")

        # Extract CI widths
        print(f"\nExtracting CI width sequences...")
        oracle_ci = extract_ci_width_sequences(oracle_data, horizon)
        prior_ci = extract_ci_width_sequences(prior_data, horizon)
        print(f"  Oracle CI shape: {oracle_ci.shape}")
        print(f"  Prior CI shape: {prior_ci.shape}")

        # Get dates
        dates = get_dates_for_sequences(oracle_data['indices'], horizon)
        print(f"  Date range: {dates[0]} to {dates[-1]}")

        # Filter to 2004-2007
        print(f"\nFiltering to 2004-2007...")
        oracle_ci_filtered, dates_filtered = filter_to_period(
            oracle_ci, dates, START_DATE, END_DATE
        )
        prior_ci_filtered, _ = filter_to_period(
            prior_ci, dates, START_DATE, END_DATE
        )
        print(f"  Filtered sequences: {len(dates_filtered)}")
        print(f"  Date range: {dates_filtered[0]} to {dates_filtered[-1]}")

        # Create 3-panel analysis plot
        print(f"\nCreating 3-panel analysis plot...")
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        plot_individual_trajectories(axes[0], oracle_ci_filtered, prior_ci_filtered,
                                     dates_filtered, horizon, num_examples=3)
        plot_average_trajectory(axes[1], oracle_ci_filtered, prior_ci_filtered, horizon)
        plot_ratio_evolution(axes[2], oracle_ci_filtered, prior_ci_filtered, horizon)

        plt.tight_layout()
        output_file = OUTPUT_DIR / f"sequence_ci_evolution_h{horizon}_2004_2007.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file.name}")

        # Summary statistics
        print(f"\nSummary Statistics for H={horizon}:")
        print("-" * 40)

        oracle_mean = oracle_ci_filtered.mean()
        prior_mean = prior_ci_filtered.mean()
        oracle_max = oracle_ci_filtered.max(axis=1).mean()
        prior_max = prior_ci_filtered.max(axis=1).mean()

        ratio_avg = prior_mean / oracle_mean
        ratio_max = prior_max / oracle_max

        print(f"  Average CI Width (across all days and sequences):")
        print(f"    Oracle: {oracle_mean:.5f}")
        print(f"    Prior:  {prior_mean:.5f}")
        print(f"    Ratio:  {ratio_avg:.3f}× (prior/oracle)")
        print(f"\n  Maximum CI Width (avg of max across sequences):")
        print(f"    Oracle: {oracle_max:.5f}")
        print(f"    Prior:  {prior_max:.5f}")
        print(f"    Ratio:  {ratio_max:.3f}× (prior/oracle)")

        # Day-by-day ratio statistics
        daily_ratios = (prior_ci_filtered / oracle_ci_filtered).mean(axis=0)
        days_prior_wider = (daily_ratios > 1.0).sum()

        print(f"\n  Forecast Day Analysis:")
        print(f"    Days where prior > oracle: {days_prior_wider}/{horizon} ({days_prior_wider/horizon*100:.1f}%)")
        print(f"    Mean daily ratio: {daily_ratios.mean():.3f}×")
        print(f"    Min daily ratio: {daily_ratios.min():.3f}× (day {daily_ratios.argmin()+1})")
        print(f"    Max daily ratio: {daily_ratios.max():.3f}× (day {daily_ratios.argmax()+1})")

    # Create combined heatmap comparison
    print(f"\n{'='*80}")
    print("CREATING HEATMAP COMPARISON")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    for i, horizon in enumerate([60, 90]):
        # Reload and filter data
        oracle_data, prior_data = load_prediction_data(horizon)
        oracle_ci = extract_ci_width_sequences(oracle_data, horizon)
        prior_ci = extract_ci_width_sequences(prior_data, horizon)
        dates = get_dates_for_sequences(oracle_data['indices'], horizon)

        oracle_ci_filtered, dates_filtered = filter_to_period(
            oracle_ci, dates, START_DATE, END_DATE
        )
        prior_ci_filtered, _ = filter_to_period(
            prior_ci, dates, START_DATE, END_DATE
        )

        # Plot heatmaps
        plot_heatmap(axes[i, 0], oracle_ci_filtered, dates_filtered, horizon,
                    f'Oracle H={horizon}: CI Width', cmap='Blues')
        plot_heatmap(axes[i, 1], prior_ci_filtered, dates_filtered, horizon,
                    f'Prior H={horizon}: CI Width', cmap='Oranges')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sequence_ci_heatmap_2004_2007.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")

    # Create difference heatmaps
    print(f"\nCreating difference heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    for i, horizon in enumerate([60, 90]):
        # Reload and filter data
        oracle_data, prior_data = load_prediction_data(horizon)
        oracle_ci = extract_ci_width_sequences(oracle_data, horizon)
        prior_ci = extract_ci_width_sequences(prior_data, horizon)
        dates = get_dates_for_sequences(oracle_data['indices'], horizon)

        oracle_ci_filtered, dates_filtered = filter_to_period(
            oracle_ci, dates, START_DATE, END_DATE
        )
        prior_ci_filtered, _ = filter_to_period(
            prior_ci, dates, START_DATE, END_DATE
        )

        # Compute difference
        diff = prior_ci_filtered - oracle_ci_filtered

        # Plot difference heatmap
        plot_heatmap(axes[i], diff, dates_filtered, horizon,
                    f'H={horizon}: Prior - Oracle CI Width', cmap='RdBu_r')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sequence_ci_difference_2004_2007.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  1. sequence_ci_evolution_h60_2004_2007.png  (3-panel analysis)")
    print("  2. sequence_ci_evolution_h90_2004_2007.png  (3-panel analysis)")
    print("  3. sequence_ci_heatmap_2004_2007.png        (Oracle vs Prior heatmaps)")
    print("  4. sequence_ci_difference_2004_2007.png     (Difference heatmaps)")
    print("="*80)


if __name__ == "__main__":
    main()
