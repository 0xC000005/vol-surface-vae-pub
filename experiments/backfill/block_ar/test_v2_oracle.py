#!/usr/bin/env python
"""Oracle validation for test_block_ar_requirements_v2.py

Feeds GT data as predictions to verify all test suites are valid.
Three oracle variants:

Oracle A (Perfect): 50 near-identical copies of GT future + tiny noise (std=1e-6)
  Expected: Surface, Time Series, Block-AR, Cointegration, Distributional, Cross-Cell PASS
  Expected FAIL: CI Coverage (100% > 95% gate), Regime (100% > 95% gate)

Oracle B (Cross-window unconditional): For each window, 50 random OTHER GT futures
  Expected: All of A plus CI Coverage PASS

Oracle C (Local-window conditional): For each window i, 50 GT futures from windows i-25 to i+24
  Expected: ALL suites PASS including Suite 9 (cross-cell correlation)

Suite 3 (Conditionality) is SKIPPED — requires a model object.
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_surface_validity_tests,
    run_ci_coverage_tests,
    run_time_series_tests,
    run_block_ar_tests,
    run_cointegration_tests,
    run_regime_coverage_tests,
    run_distributional_fidelity_tests,
    run_cross_cell_correlation_tests,
    print_summary,
    convert_to_serializable,
)


# =========================================================================
# Data Loading
# =========================================================================

def build_test_data():
    """Load raw data and build test windows matching the test script."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    returns = data["ret"]       # (N,) daily returns

    history_len = 30
    future_len = 30
    test_start = 4540

    # Build windows from test_start onward (matching VolSurfaceDataset)
    local_surfaces = surfaces[test_start:]
    total_len = history_len + future_len
    n_windows = len(local_surfaces) - total_len + 1

    histories = []
    futures = []
    for i in range(n_windows):
        hist = local_surfaces[i : i + history_len]
        fut = local_surfaces[i + history_len : i + history_len + future_len]
        histories.append(hist)
        futures.append(fut)

    histories = np.stack(histories)  # (N, 30, 5, 5)
    futures = np.stack(futures)      # (N, 30, 5, 5)

    print(f"Test data: {n_windows} windows")
    print(f"  Histories: {histories.shape}, range [{histories.min():.4f}, {histories.max():.4f}]")
    print(f"  Futures:   {futures.shape}, range [{futures.min():.4f}, {futures.max():.4f}]")
    print(f"  Returns:   {returns.shape}")

    return histories, futures, returns, test_start


# =========================================================================
# Oracle Constructors
# =========================================================================

def build_oracle_a(futures, n_samples=50):
    """Oracle A (Perfect): 50 copies of GT + tiny noise (std=1e-6)."""
    oracle = np.stack(
        [futures + np.random.normal(0, 1e-6, futures.shape) for _ in range(n_samples)],
        axis=1,
    )  # (N, 50, 30, 5, 5)
    print(f"\nOracle A (Perfect): {oracle.shape}")
    return oracle


def build_oracle_b(futures, n_samples=50):
    """Oracle B (Cross-window unconditional): random OTHER GT futures as ensemble."""
    N = futures.shape[0]
    oracle = np.zeros((N, n_samples, *futures.shape[1:]))
    rng = np.random.RandomState(42)
    for i in range(N):
        indices = rng.choice(N, n_samples, replace=True)
        oracle[i] = futures[indices]
    print(f"\nOracle B (Cross-window): {oracle.shape}")
    return oracle


def build_oracle_c(futures, n_samples=50, window_half=25):
    """Oracle C (Local-window conditional): nearby GT windows (i-25 to i+24)."""
    N = futures.shape[0]
    oracle = np.zeros((N, n_samples, *futures.shape[1:]))
    rng = np.random.RandomState(42)
    for i in range(N):
        lo = max(0, i - window_half)
        hi = min(N, i + window_half)
        indices = rng.choice(range(lo, hi), n_samples, replace=True)
        oracle[i] = futures[indices]
    print(f"\nOracle C (Local window +/-{window_half}): {oracle.shape}")
    return oracle


# =========================================================================
# Test Runner
# =========================================================================

def run_oracle_test(name, cond_samples, ground_truth, history, returns, test_start):
    """Run all applicable test suites on oracle data."""
    print(f"\n{'#' * 70}")
    print(f"# ORACLE TEST: {name}")
    print(f"{'#' * 70}")
    print(f"  cond_samples: {cond_samples.shape}")
    print(f"  ground_truth: {ground_truth.shape}")
    print(f"  history: {history.shape}")

    results = {}

    # Suite 1: Surface Validity
    results["surface"] = run_surface_validity_tests(cond_samples, ground_truth)

    # Suite 2: CI Coverage
    results["coverage"] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Suite 3: Conditionality -- SKIPPED (needs model object)
    results["conditionality"] = {
        "width_ratio": 0.0,
        "turb_calm_ratio": 0.0,
        "turb_calm_pass": False,
        "mae_reduction_pct": 0.0,
        "mae_pass": False,
        "growing_uncertainty_monotonic": False,
        "worst_cell_mae_reduction": 0.0,
        "worst_cell_mae_pass": True,
        "per_regime_conditionality": {},
        "pass": False,
        "_skipped": True,
        "_reason": "Requires model for unconditional baseline",
    }

    # Suite 4: Time Series Properties
    results["time_series"] = run_time_series_tests(cond_samples, ground_truth)

    # Suite 5: Block-AR Specific
    results["block_ar"] = run_block_ar_tests(cond_samples, block_size=10)

    # Suite 6: Cointegration
    results["cointegration"] = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=test_start,
        history_len=30,
        future_len=30,
    )

    # Suite 7: Regime Coverage
    results["regime_coverage"] = run_regime_coverage_tests(
        cond_samples, ground_truth, history,
    )

    # Suite 8: Distributional Fidelity
    results["distributional"] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history,
    )

    # Suite 9: Cross-Cell Correlation
    results["cross_cell_correlation"] = run_cross_cell_correlation_tests(
        cond_samples, ground_truth,
    )

    # Print summary
    print_summary(results)

    return results


# =========================================================================
# Comparison Table
# =========================================================================

def extract_pass_fail(results):
    """Extract a compact pass/fail table from results."""
    table = {}
    suites = [
        ("1. Surface Validity",  "surface",               "overall_pass"),
        ("2. CI Coverage",       "coverage",              "pass"),
        ("3. Conditionality",    "conditionality",        "pass"),
        ("4. Time Series",       "time_series",           "overall_pass"),
        ("5. Block-AR Boundary", "block_ar",              "overall_pass"),
        ("6. Cointegration",     "cointegration",         "pass"),
        ("7. Regime Coverage",   "regime_coverage",       "overall_pass"),
        ("8. Distributional",    "distributional",        "overall_pass"),
        ("9. Cross-Cell Corr",   "cross_cell_correlation","overall_pass"),
    ]
    for label, key, pass_key in suites:
        if key in results:
            if results[key].get("_skipped", False):
                table[label] = "SKIP"
            else:
                table[label] = "PASS" if results[key].get(pass_key, False) else "FAIL"
        else:
            table[label] = "N/A"
    return table


def print_comparison_table(oracle_pf):
    """Print a comparison table across all oracle variants."""
    print("\n" + "=" * 90)
    print("ORACLE TEST COMPARISON TABLE (v2)")
    print("=" * 90)

    oracle_names = list(oracle_pf.keys())
    suites = [
        "1. Surface Validity",
        "2. CI Coverage",
        "3. Conditionality",
        "4. Time Series",
        "5. Block-AR Boundary",
        "6. Cointegration",
        "7. Regime Coverage",
        "8. Distributional",
        "9. Cross-Cell Corr",
    ]

    # Header
    header = f"{'Suite':<25s}"
    for name in oracle_names:
        header += f" {name:>18s}"
    print(header)
    print("-" * len(header))

    # Rows
    for suite in suites:
        row = f"{suite:<25s}"
        for name in oracle_names:
            pf = oracle_pf[name].get(suite, "N/A")
            row += f" {pf:>18s}"
        print(row)

    print("-" * len(header))

    # Count passes per oracle (excluding SKIP)
    count_row = f"{'PASS count':<25s}"
    for name in oracle_names:
        n_pass = sum(1 for s in suites if oracle_pf[name].get(s) == "PASS")
        n_total = sum(1 for s in suites if oracle_pf[name].get(s) != "SKIP")
        count_row += f"{'%d/%d' % (n_pass, n_total):>19s}"
    print(count_row)
    print("=" * 90)


def print_detailed_diagnostics(all_results):
    """Print detailed diagnostic info for any FAIL results."""
    print("\nKEY DIAGNOSTIC METRICS FOR FAILURES:")
    print("-" * 60)

    for oracle_name, results in all_results.items():
        pf_table = extract_pass_fail(results)
        failures = [s for s, v in pf_table.items() if v == "FAIL"]
        if not failures:
            print(f"\n  {oracle_name}: ALL NON-SKIPPED SUITES PASS")
            continue

        print(f"\n  {oracle_name} failures:")
        for suite_label in failures:
            if "1." in suite_label:
                s = results["surface"]
                print(
                    f"    Suite 1: explosion={s['explosion']['explosion_total_rate']:.3%}, "
                    f"calendar={s['calendar']['calendar_avg_violation_rate']:.3%}, "
                    f"butterfly={s['butterfly']['butterfly_avg_violation_rate']:.3%}"
                )
            elif "2." in suite_label:
                c = results["coverage"]
                print(
                    f"    Suite 2: overall_90={c['overall'].get(0.9, 0):.3%}, "
                    f"worst_cell_pass={c.get('worst_cell_pass', 'N/A')}"
                )
                for h, p in c.get("horizon_pass", {}).items():
                    cov = c["per_horizon"].get(h, {}).get(0.9, 0)
                    worst = c.get("worst_cell_per_horizon", {}).get(h, 0)
                    best = c.get("best_cell_per_horizon", {}).get(h, 0)
                    print(
                        f"      h={h}: cov={cov:.3%}, worst_cell={worst:.3%}, "
                        f"best_cell={best:.3%} {'PASS' if p else 'FAIL'}"
                    )
            elif "4." in suite_label:
                ts = results["time_series"]
                print(
                    f"    Suite 4: acf_corr={ts['acf']['acf_correlation']:.3f}, "
                    f"kurt_ratio={ts['kurtosis']['kurtosis_ratio']:.3f}"
                )
            elif "5." in suite_label:
                ba = results["block_ar"]
                print(
                    f"    Suite 5: boundary_ratio="
                    f"{ba['boundary_smoothness']['boundary_ratio']:.3f}, "
                    f"monotonic={ba['growing_uncertainty']['monotonic']}"
                )
            elif "6." in suite_label:
                co = results["cointegration"]
                print(
                    f"    Suite 6: gen_rate={co['gen_pass_rate']:.3%}, "
                    f"gt_rate={co['gt_pass_rate']:.3%}, "
                    f"ratio={co['gen_gt_ratio']:.3f}, "
                    f"worst_cell={co.get('worst_cell_ratio', 0):.3f}"
                )
            elif "7." in suite_label:
                rc = results["regime_coverage"]
                print(
                    f"    Suite 7: L1={rc['layer1_pass']}, L2={rc['layer2_pass']}, "
                    f"L3={rc['layer3_pass']} "
                    f"(catastrophic={rc['layer3_catastrophic_rate']:.3%})"
                )
            elif "8." in suite_label:
                df = results["distributional"]
                print(
                    f"    Suite 8: KS_changes={df['ks_test']['n_pass']}/25, "
                    f"KS_levels={df['ks_level_test']['n_pass']}/25, "
                    f"bias_frac={df['median_bias']['n_pass']}/25, "
                    f"bias_mag={df['median_bias']['n_mag_pass']}/25, "
                    f"window_floor={df['window_floor']['pass']}, "
                    f"explosion={df['explosion']['pass']}, "
                    f"mae={df['cell_mae']['n_pass']}/25"
                )
            elif "9." in suite_label:
                xc = results["cross_cell_correlation"]
                print(
                    f"    Suite 9: corr_ratio={xc['corr_ratio']:.3f}, "
                    f"rank_ratio={xc['rank_ratio']:.3f}"
                )


# =========================================================================
# Main
# =========================================================================

def main():
    np.random.seed(42)

    # Load data
    histories, futures, returns, test_start = build_test_data()

    # Limit to first 1280 windows for speed (matches v1 oracle)
    max_windows = 1280
    if len(futures) > max_windows:
        print(f"\nLimiting to first {max_windows} windows (of {len(futures)}) for speed")
        histories = histories[:max_windows]
        futures = futures[:max_windows]

    # Build oracle samples
    oracle_a = build_oracle_a(futures, n_samples=50)
    oracle_b = build_oracle_b(futures, n_samples=50)
    oracle_c = build_oracle_c(futures, n_samples=50, window_half=25)

    # Run tests
    all_results = {}
    all_pf = {}

    for name, oracle_samples in [
        ("A: Perfect", oracle_a),
        ("B: CrossWin", oracle_b),
        ("C: LocalWin", oracle_c),
    ]:
        results = run_oracle_test(
            name, oracle_samples, futures, histories, returns, test_start,
        )
        all_results[name] = results
        all_pf[name] = extract_pass_fail(results)

    # Comparison table
    print_comparison_table(all_pf)
    print_detailed_diagnostics(all_results)

    # =====================================================================
    # Key assertions
    # =====================================================================
    oracle_c_pf = all_pf["C: LocalWin"]
    # Oracle C should pass ALL non-skipped suites
    oracle_c_non_skip = {
        s: v for s, v in oracle_c_pf.items() if v != "SKIP"
    }
    oracle_c_pass_count = sum(1 for v in oracle_c_non_skip.values() if v == "PASS")
    oracle_c_total = len(oracle_c_non_skip)
    oracle_c_all_pass = oracle_c_pass_count == oracle_c_total

    print(f"\n{'=' * 70}")
    print(f"ORACLE C ASSERTION: {oracle_c_pass_count}/{oracle_c_total} non-skipped suites PASS")
    if oracle_c_all_pass:
        print("ASSERTION PASSED: Oracle C passes ALL non-skipped suites.")
    else:
        failed = [s for s, v in oracle_c_non_skip.items() if v != "PASS"]
        print(f"ASSERTION FAILED: Oracle C fails: {failed}")
        print("These suites have bugs or unreasonable thresholds in v2!")
    print(f"{'=' * 70}")

    # Save results
    output_dir = Path("results/block_ar/oracle_test_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, results in all_results.items():
        safe_name = name.replace(": ", "_").replace(" ", "_")
        with open(output_dir / f"oracle_{safe_name}.json", "w") as f:
            json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to {output_dir}/")

    # Hard assertion for CI
    assert oracle_c_all_pass, (
        f"Oracle C must pass ALL non-skipped suites -- test is broken if it doesn't. "
        f"Failed: {[s for s, v in oracle_c_non_skip.items() if v != 'PASS']}"
    )


if __name__ == "__main__":
    main()
