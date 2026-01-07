"""
Test Log-Return Approach for Horizon Variance

Goal: Verify that modeling log-returns (instead of levels) solves the horizon
variance issue where CI coverage is ~50% instead of 90%.

Hypothesis:
- Log-returns have constant variance (stationary)
- Model's constant z_logvar would be CORRECT for log-returns
- Horizon effect emerges naturally from summing independent samples
- CI width should grow as √H, coverage should be ~90%

Tests:
1. Log-return space: CI coverage should be ~90% at all horizons
2. Transformed-back space: CI width should grow with horizon, coverage ~90%
3. Compare to level-based approach (current model)
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_data():
    """Load IV surface data."""
    data = np.load(project_root / "data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)
    return surfaces


def compute_log_returns(surfaces):
    """Compute log-returns: Δlog(IV) = log(IV_t) - log(IV_{t-1})."""
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]  # (N-1, 5, 5)
    return log_returns, log_surfaces


def test_stationarity(log_returns):
    """Verify log-returns are stationary (constant variance across time)."""
    print("=" * 70)
    print("TEST 0: Stationarity of Log-Returns")
    print("=" * 70)
    print()

    n = len(log_returns)
    chunk_size = n // 5

    print("Variance of single-day log-returns by time period:")
    variances = []
    for i in range(5):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = log_returns[start:end]
        var = chunk.var()
        variances.append(var)
        print(f"  Period {i+1} (days {start:4d}-{end:4d}): var = {var:.5f}")

    ratio = max(variances) / min(variances)
    print()
    print(f"Variance ratio (max/min): {ratio:.2f}x")
    print(f"Status: {'PASS - relatively stable' if ratio < 3 else 'WARN - significant variation'}")
    print()

    return np.mean(variances)


def test_log_return_space_ci(log_returns, horizons=[1, 7, 14, 30], n_samples=1000):
    """
    Test 1: CI coverage in log-return space.

    For each horizon H:
    - Sample N trajectories of H log-returns from empirical distribution
    - Compute 90% CI from samples
    - Check if ground truth falls within CI
    """
    print("=" * 70)
    print("TEST 1: Log-Return Space CI Coverage")
    print("=" * 70)
    print()
    print("Simulating VAE by sampling from empirical log-return distribution.")
    print("For each horizon, we sample N=1000 trajectories and check CI coverage.")
    print()

    # Flatten log-returns for sampling (treat all grid points independently for now)
    flat_log_returns = log_returns.reshape(-1)  # All log-returns pooled

    results = {}

    for H in horizons:
        # For H-step forecast, we need H consecutive log-returns
        # Ground truth: actual H-step cumulative log-return
        n_test_points = min(500, len(log_returns) - H)

        coverage_count = 0
        ci_widths = []

        for t in range(0, n_test_points, 10):  # Sample every 10th point
            # Ground truth cumulative log-return from t to t+H
            gt_cumsum = log_returns[t:t+H].sum(axis=0)  # (5, 5)

            # Sample N trajectories
            samples = []
            for _ in range(n_samples):
                # Sample H independent log-returns
                sampled_returns = np.random.choice(flat_log_returns, size=(H, 5, 5))
                cumsum = sampled_returns.sum(axis=0)  # (5, 5)
                samples.append(cumsum)

            samples = np.array(samples)  # (N, 5, 5)

            # Compute 90% CI (5th and 95th percentile)
            p05 = np.percentile(samples, 5, axis=0)
            p95 = np.percentile(samples, 95, axis=0)

            # Check coverage (for each grid point)
            in_ci = (gt_cumsum >= p05) & (gt_cumsum <= p95)
            coverage_count += in_ci.mean()  # Average across grid

            # CI width
            ci_widths.append((p95 - p05).mean())

        n_tests = n_test_points // 10
        coverage = coverage_count / n_tests
        avg_ci_width = np.mean(ci_widths)

        results[H] = {
            'coverage': coverage,
            'ci_width': avg_ci_width,
        }

        status = "PASS" if abs(coverage - 0.90) < 0.10 else "FAIL"
        print(f"H={H:2d}: Coverage = {coverage*100:.1f}% (target: 90%), "
              f"CI width = {avg_ci_width:.4f} [{status}]")

    print()
    return results


def test_transformed_back_space(log_returns, log_surfaces, horizons=[1, 7, 14, 30], n_samples=1000):
    """
    Test 2: CI coverage and width in transformed-back IV space.

    For each horizon H:
    - Start from initial log(IV)
    - Sample H log-returns, sum them, exp() back to IV
    - Check CI coverage and width growth
    """
    print("=" * 70)
    print("TEST 2: Transformed-Back Space (IV Levels)")
    print("=" * 70)
    print()
    print("Sample log-returns, sum, then exp() back to IV levels.")
    print("Expected: CI width grows with √H, coverage ~90%.")
    print()

    flat_log_returns = log_returns.reshape(-1)
    surfaces = np.exp(log_surfaces)

    results = {}
    base_width = None

    for H in horizons:
        n_test_points = min(500, len(log_returns) - H)

        coverage_count = 0
        ci_widths = []

        for t in range(0, n_test_points, 10):
            # Starting point
            initial_log = log_surfaces[t]  # (5, 5)

            # Ground truth IV at t+H
            gt_iv = surfaces[t + H]  # (5, 5)

            # Sample N trajectories
            samples = []
            for _ in range(n_samples):
                sampled_returns = np.random.choice(flat_log_returns, size=(H, 5, 5))
                cumsum = sampled_returns.sum(axis=0)
                final_log = initial_log + cumsum
                final_iv = np.exp(final_log)  # Transform back - always positive!
                samples.append(final_iv)

            samples = np.array(samples)  # (N, 5, 5)

            # Compute 90% CI
            p05 = np.percentile(samples, 5, axis=0)
            p95 = np.percentile(samples, 95, axis=0)

            # Check coverage
            in_ci = (gt_iv >= p05) & (gt_iv <= p95)
            coverage_count += in_ci.mean()

            # CI width (in IV space)
            ci_widths.append((p95 - p05).mean())

            # Check for negative values (should be 0!)
            assert (samples > 0).all(), "Negative IV values found!"

        n_tests = n_test_points // 10
        coverage = coverage_count / n_tests
        avg_ci_width = np.mean(ci_widths)

        if base_width is None:
            base_width = avg_ci_width
            width_ratio = 1.0
        else:
            width_ratio = avg_ci_width / base_width

        expected_ratio = np.sqrt(H / horizons[0])

        results[H] = {
            'coverage': coverage,
            'ci_width': avg_ci_width,
            'width_ratio': width_ratio,
            'expected_ratio': expected_ratio,
        }

        status = "PASS" if abs(coverage - 0.90) < 0.10 else "FAIL"
        print(f"H={H:2d}: Coverage = {coverage*100:.1f}%, "
              f"CI width = {avg_ci_width:.4f}, "
              f"Width ratio = {width_ratio:.2f}x (expected ~{expected_ratio:.2f}x) [{status}]")

    print()
    print("Negative IV values: 0 (guaranteed by exp() transform)")
    print()

    return results


def test_level_baseline(surfaces, horizons=[1, 7, 14, 30], n_samples=1000):
    """
    Test 3: Baseline - what happens with level-based approach (current model).

    Simulates current model behavior:
    - Sample from empirical level-change distribution
    - Constant variance at all horizons
    """
    print("=" * 70)
    print("TEST 3: Level-Based Baseline (Current Model Simulation)")
    print("=" * 70)
    print()
    print("Simulating current model: sample level changes directly.")
    print("Expected: CI coverage degrades at longer horizons.")
    print()

    level_changes = surfaces[1:] - surfaces[:-1]  # (N-1, 5, 5)
    flat_changes = level_changes.reshape(-1)

    results = {}
    base_width = None

    for H in horizons:
        n_test_points = min(500, len(level_changes) - H)

        coverage_count = 0
        ci_widths = []

        for t in range(0, n_test_points, 10):
            initial_iv = surfaces[t]
            gt_iv = surfaces[t + H]

            # Current model: predicts H-step change directly with constant variance
            # Simulate by sampling single-step changes (which have smaller variance)
            # This mimics the problem: model variance doesn't grow with H
            samples = []
            for _ in range(n_samples):
                # Sample single-step change (not H-step!)
                sampled_change = np.random.choice(flat_changes, size=(5, 5))
                final_iv = initial_iv + sampled_change  # Can go negative!
                samples.append(final_iv)

            samples = np.array(samples)

            p05 = np.percentile(samples, 5, axis=0)
            p95 = np.percentile(samples, 95, axis=0)

            in_ci = (gt_iv >= p05) & (gt_iv <= p95)
            coverage_count += in_ci.mean()

            ci_widths.append((p95 - p05).mean())

            # Count negative values
            n_negative = (samples < 0).sum()

        n_tests = n_test_points // 10
        coverage = coverage_count / n_tests
        avg_ci_width = np.mean(ci_widths)

        if base_width is None:
            base_width = avg_ci_width
        width_ratio = avg_ci_width / base_width

        results[H] = {
            'coverage': coverage,
            'ci_width': avg_ci_width,
            'width_ratio': width_ratio,
        }

        # Coverage should degrade as H increases (because we use constant variance)
        print(f"H={H:2d}: Coverage = {coverage*100:.1f}%, "
              f"CI width = {avg_ci_width:.4f}, "
              f"Width ratio = {width_ratio:.2f}x (FLAT - this is the problem!)")

    print()
    return results


def print_summary(log_return_results, transformed_results, baseline_results):
    """Print comparison summary."""
    print("=" * 70)
    print("SUMMARY: Log-Return vs Level-Based Approach")
    print("=" * 70)
    print()
    print("                    | Log-Return (IV space) | Level-Based (baseline)")
    print("-" * 70)

    for H in [1, 7, 14, 30]:
        if H in transformed_results and H in baseline_results:
            tr = transformed_results[H]
            bl = baseline_results[H]
            print(f"H={H:2d} Coverage       | {tr['coverage']*100:5.1f}%                 | {bl['coverage']*100:5.1f}%")
            print(f"H={H:2d} CI Width Ratio | {tr['width_ratio']:5.2f}x (vs {tr['expected_ratio']:.2f}x exp) | {bl['width_ratio']:5.2f}x (FLAT)")
            print("-" * 70)

    print()
    print("CONCLUSION:")

    # Check if log-return approach is better
    log_return_coverage = np.mean([transformed_results[H]['coverage'] for H in [1, 7, 14, 30]])
    baseline_coverage = np.mean([baseline_results[H]['coverage'] for H in [1, 7, 14, 30]])

    if log_return_coverage > baseline_coverage + 0.1:
        print(f"  Log-return approach significantly improves CI coverage!")
        print(f"  Average coverage: {log_return_coverage*100:.1f}% vs {baseline_coverage*100:.1f}%")
        print(f"  Recommendation: IMPLEMENT LOG-RETURN APPROACH")
    elif log_return_coverage > baseline_coverage:
        print(f"  Log-return approach slightly improves CI coverage.")
        print(f"  Average coverage: {log_return_coverage*100:.1f}% vs {baseline_coverage*100:.1f}%")
    else:
        print(f"  Log-return approach does not improve coverage.")
        print(f"  May need additional changes (e.g., regime-dependent variance).")

    print()


def main():
    print("=" * 70)
    print("LOG-RETURN VARIANCE TEST")
    print("=" * 70)
    print()
    print("Testing whether log-returns solve the horizon variance issue.")
    print()

    # Load data
    surfaces = load_data()
    print(f"Loaded {len(surfaces)} days of IV surfaces, shape: {surfaces.shape}")
    print()

    # Compute log-returns
    log_returns, log_surfaces = compute_log_returns(surfaces)
    print(f"Computed {len(log_returns)} log-returns")
    print(f"Log-return range: [{log_returns.min():.4f}, {log_returns.max():.4f}]")
    print(f"Log-return mean: {log_returns.mean():.6f}")
    print(f"Log-return std: {log_returns.std():.4f}")
    print()

    # Test 0: Stationarity
    test_stationarity(log_returns)

    # Test 1: Log-return space CI
    log_return_results = test_log_return_space_ci(log_returns)

    # Test 2: Transformed-back space
    transformed_results = test_transformed_back_space(log_returns, log_surfaces)

    # Test 3: Level-based baseline
    baseline_results = test_level_baseline(surfaces)

    # Summary
    print_summary(log_return_results, transformed_results, baseline_results)


if __name__ == "__main__":
    main()
