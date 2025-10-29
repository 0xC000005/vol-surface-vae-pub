import numpy as np
import matplotlib.pyplot as plt

def compute_ci_violations(actual, generated, percentile_low=5, percentile_high=95):
    """
    Compute CI violation rates.

    Args:
        actual: (num_days,) actual values
        generated: (num_days, num_samples) generated samples
        percentile_low: Lower percentile for CI (default 5 for 90% CI)
        percentile_high: Upper percentile for CI (default 95 for 90% CI)

    Returns:
        Dictionary with violation statistics
    """
    num_days = len(actual)

    # Compute CI bounds from samples
    p_low = np.percentile(generated, percentile_low, axis=1)  # (num_days,)
    p_high = np.percentile(generated, percentile_high, axis=1)  # (num_days,)

    # Check violations
    below_ci = actual < p_low
    above_ci = actual > p_high
    outside_ci = below_ci | above_ci

    # Compute statistics
    num_violations = np.sum(outside_ci)
    num_below = np.sum(below_ci)
    num_above = np.sum(above_ci)
    pct_violations = 100.0 * num_violations / num_days
    pct_below = 100.0 * num_below / num_days
    pct_above = 100.0 * num_above / num_days

    # CI width statistics
    ci_width = p_high - p_low
    mean_ci_width = np.mean(ci_width)

    return {
        "num_days": num_days,
        "num_violations": num_violations,
        "num_below": num_below,
        "num_above": num_above,
        "pct_violations": pct_violations,
        "pct_below": pct_below,
        "pct_above": pct_above,
        "mean_ci_width": mean_ci_width,
        "p_low": p_low,
        "p_high": p_high
    }

def main():
    print("="*70)
    print("CI Calibration Evaluation: Baseline vs Empirical Sampling")
    print("="*70)

    # Parameters
    base_folder = "test_spx/2024_11_09"
    noise_scale = 0.3
    start_day = 5
    days_to_generate = 5810
    LAST_N_DAYS = 1000  # Evaluate on last 1000 days (test set)

    # Grid points to analyze
    grid_points = [
        {"name": "ATM 3-Month", "row": 2, "col": 2},
        {"name": "ATM 1-Year", "row": 2, "col": 4},
        {"name": "OTM Put 1-Year", "row": 1, "col": 4}
    ]

    # Models to analyze
    models = ["no_ex", "ex_no_loss", "ex_loss"]

    # Load actual data
    print(f"\nLoading actual data...")
    data = np.load("data/vol_surface_with_ret.npz")
    actual_surfaces = data["surface"][start_day:start_day+days_to_generate]  # (5810, 5, 5)
    print(f"  Shape: {actual_surfaces.shape}")
    print(f"  Evaluating on last {LAST_N_DAYS} days")

    # Results storage
    results = []

    # Process each model
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        # Load generations
        baseline_path = f"{base_folder}/{model_name}_gen5.npz"
        empirical_path = f"{base_folder}/{model_name}_empirical_gen5_noise{noise_scale}.npz"

        print(f"  Loading baseline: {baseline_path}")
        baseline_data = np.load(baseline_path)
        baseline_surfaces = baseline_data["surfaces"]  # (5810, 1000, 5, 5)

        print(f"  Loading empirical: {empirical_path}")
        empirical_data = np.load(empirical_path)
        empirical_surfaces = empirical_data["surfaces"]  # (5810, 1000, 5, 5)

        # Focus on last N days
        actual_test = actual_surfaces[-LAST_N_DAYS:]
        baseline_test = baseline_surfaces[-LAST_N_DAYS:]
        empirical_test = empirical_surfaces[-LAST_N_DAYS:]

        for grid_point in grid_points:
            grid_name = grid_point["name"]
            grid_row = grid_point["row"]
            grid_col = grid_point["col"]

            print(f"\n  Grid point: {grid_name} [{grid_row}, {grid_col}]")

            # Extract data at grid point
            actual_values = actual_test[:, grid_row, grid_col]  # (1000,)
            baseline_generated = baseline_test[:, :, grid_row, grid_col]  # (1000, 1000)
            empirical_generated = empirical_test[:, :, grid_row, grid_col]  # (1000, 1000)

            # Compute CI violations
            baseline_ci = compute_ci_violations(actual_values, baseline_generated)
            empirical_ci = compute_ci_violations(actual_values, empirical_generated)

            # Print results
            print(f"    Baseline (N(0,1)):")
            print(f"      Violations: {baseline_ci['num_violations']}/{baseline_ci['num_days']} ({baseline_ci['pct_violations']:.2f}%)")
            print(f"      Below p05:  {baseline_ci['num_below']} ({baseline_ci['pct_below']:.2f}%)")
            print(f"      Above p95:  {baseline_ci['num_above']} ({baseline_ci['pct_above']:.2f}%)")
            print(f"      Mean CI width: {baseline_ci['mean_ci_width']:.4f}")

            print(f"    Empirical (noise={noise_scale}):")
            print(f"      Violations: {empirical_ci['num_violations']}/{empirical_ci['num_days']} ({empirical_ci['pct_violations']:.2f}%)")
            print(f"      Below p05:  {empirical_ci['num_below']} ({empirical_ci['pct_below']:.2f}%)")
            print(f"      Above p95:  {empirical_ci['num_above']} ({empirical_ci['pct_above']:.2f}%)")
            print(f"      Mean CI width: {empirical_ci['mean_ci_width']:.4f}")

            # Compute improvement
            improvement = baseline_ci['pct_violations'] - empirical_ci['pct_violations']
            improvement_str = f"{improvement:+.2f}%"
            if improvement > 0:
                print(f"    ✓ Improvement: {improvement_str} (closer to 10% target)")
            else:
                print(f"    ✗ Worse: {improvement_str} (further from 10% target)")

            # Store results
            results.append({
                "model": model_name,
                "grid_point": grid_name,
                "baseline_violations": baseline_ci['pct_violations'],
                "empirical_violations": empirical_ci['pct_violations'],
                "baseline_ci_width": baseline_ci['mean_ci_width'],
                "empirical_ci_width": empirical_ci['mean_ci_width'],
                "improvement": improvement
            })

    # Print summary table
    print("\n" + "="*70)
    print("Summary: CI Violation Rates (target: 10%)")
    print("="*70)
    print(f"{'Model':<15} {'Grid Point':<20} {'Baseline':<12} {'Empirical':<12} {'Improvement':<12}")
    print("-"*70)
    for result in results:
        improvement_str = f"{result['improvement']:+.2f}%"
        print(f"{result['model']:<15} {result['grid_point']:<20} "
              f"{result['baseline_violations']:<11.2f}% {result['empirical_violations']:<11.2f}% {improvement_str:<12}")

    # Overall statistics
    print("-"*70)
    avg_baseline = np.mean([r['baseline_violations'] for r in results])
    avg_empirical = np.mean([r['empirical_violations'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])

    print(f"{'Average':<15} {'All':<20} {avg_baseline:<11.2f}% {avg_empirical:<11.2f}% {avg_improvement:+.2f}%")

    # Assessment
    print("\n" + "="*70)
    print("Assessment:")
    print("="*70)
    print(f"Target violation rate: 10%")
    print(f"Baseline average: {avg_baseline:.2f}%")
    print(f"Empirical average: {avg_empirical:.2f}%")
    print(f"Average improvement: {avg_improvement:+.2f}%")

    if avg_empirical < avg_baseline:
        distance_to_target_baseline = abs(avg_baseline - 10)
        distance_to_target_empirical = abs(avg_empirical - 10)
        if distance_to_target_empirical < distance_to_target_baseline:
            print(f"\n✓ Empirical sampling IMPROVES calibration")
            print(f"  Distance to target reduced from {distance_to_target_baseline:.2f}% to {distance_to_target_empirical:.2f}%")
        else:
            print(f"\n⚠ Empirical sampling reduces violations but overshoots target")
    else:
        print(f"\n✗ Empirical sampling WORSENS calibration")

    print("\n" + "="*70)
    print("CI Calibration Evaluation Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
