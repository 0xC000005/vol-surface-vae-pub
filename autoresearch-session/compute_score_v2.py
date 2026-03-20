"""Composite metric for autoresearch session.

Reads summary.json from a test results directory and computes a single
score that balances suite pass count with continuous sub-metrics.

Usage:
    python autoresearch-session/compute_score.py results/block_ar/XXX_30d/summary.json
"""

import json
import sys


def compute_score(summary_path: str) -> dict:
    """Compute composite score from summary.json."""
    with open(summary_path) as f:
        summary = json.load(f)

    # Count passing suites (9 total)
    suite_map = {
        "surface": summary.get("surface", {}).get("overall_pass", False),
        "ci_coverage": summary.get("coverage", {}).get("pass", False),
        "conditionality": summary.get("conditionality", {}).get("pass", False),
        "time_series": summary.get("time_series", {}).get("overall_pass", False),
        "block_ar": summary.get("block_ar", {}).get("overall_pass", False),
        "cointegration": summary.get("cointegration", {}).get("pass", False),
        "regime_coverage": summary.get("regime_coverage", {}).get("overall_pass", False),
        "distributional": summary.get("distributional", {}).get("overall_pass", False),
        "cross_cell": summary.get("cross_cell_correlation", {}).get("overall_pass", False),
    }
    suites_passed = sum(1 for v in suite_map.values() if v)

    # --- Continuous sub-metrics ---

    # Suite 2: CI coverage
    cov = summary.get("coverage", {})
    ci_90 = cov.get("overall", {}).get("0.9", 0.85)
    worst_cell_pass = cov.get("worst_cell_pass", False)
    ci_component = max(0, (ci_90 - 0.85)) * 20  # 0-3 range
    ci_cell_component = 3.0 if worst_cell_pass else 0.0

    # Suite 7: regime coverage
    regime = summary.get("regime_coverage", {})
    l3_catastrophic = regime.get("layer3_n_catastrophic", 600)
    # Layer 2 failures: per-regime per-horizon entries with [70%, 95%] gates
    l2 = regime.get("layer2_regime_cell", {})
    n_failing_l2 = 0
    total_l2 = 0
    LAYER2_LOW = 0.70
    LAYER2_HIGH = 0.95
    for regime_name, horizons in l2.items():
        if isinstance(horizons, dict):
            for h, v in horizons.items():
                total_l2 += 1
                if isinstance(v, dict):
                    worst = v.get("worst", 0.0)
                    best = v.get("best", 1.0)
                    low_pass = worst >= LAYER2_LOW
                    high_pass = best <= LAYER2_HIGH
                    if not (low_pass and high_pass):
                        n_failing_l2 += 1
    regime_l2_component = max(0, (max(total_l2, 1) - n_failing_l2) / max(total_l2, 1)) * 3  # 0-3
    # Layer 3: catastrophic failures (fewer = better, out of ~600)
    regime_l3_component = max(0, (600 - l3_catastrophic) / 600) * 2  # 0-2

    # Suite 8: distributional
    dist = summary.get("distributional", {})
    ks_daily = dist.get("ks_test", {})
    ks_daily_pass = ks_daily.get("n_pass", 0)
    ks_component = (ks_daily_pass / 25) * 5  # 0-5

    ks_level = dist.get("ks_level_test", {})
    ks_levels_pass = ks_level.get("n_pass", 0)
    ks_levels_component = (ks_levels_pass / 25) * 3  # 0-3

    median = dist.get("median_bias", {})
    median_pass = median.get("n_pass", 0)
    median_component = (median_pass / 25) * 2  # 0-2

    # Suite 9: cross-cell correlation
    xcell = summary.get("cross_cell_correlation", {})
    corr_ratio = xcell.get("corr_ratio", 0.0)
    xcell_component = max(0, min(3.0, (1.0 - abs(corr_ratio - 1.0)) * 3.0))  # 0-3, peaks at ratio=1.0

    # Suite 4: time series kurtosis
    ts = summary.get("time_series", {})
    kurt_info = ts.get("kurtosis", {})
    kurtosis_ratio = kurt_info.get("kurtosis_ratio", 1.0)
    kurt_distance = abs(kurtosis_ratio - 1.0)
    kurt_component = max(0, 2.0 - kurt_distance * 4)  # 0-2

    # Suite 5: growing uncertainty
    bar = summary.get("block_ar", {})
    growing_unc = 3.0 if bar.get("overall_pass", False) else 0.0

    # Suite 6: cointegration
    coint = summary.get("cointegration", {})
    coint_ratio = coint.get("gen_gt_ratio", 0.0)
    coint_component = min(coint_ratio, 1.0) * 3  # 0-3

    # Composite
    total = (
        suites_passed * 10              # 0-90, dominant signal (9 suites)
        + ci_component                   # 0-3, CI coverage
        + ci_cell_component              # 0/3, per-cell CI
        + regime_l2_component            # 0-3, regime layer 2
        + regime_l3_component            # 0-2, regime layer 3
        + ks_component                   # 0-5, KS daily
        + ks_levels_component            # 0-3, KS levels
        + kurt_component                 # 0-2, kurtosis
        + growing_unc                    # 0/3, growing uncertainty
        + coint_component                # 0-3, cointegration
        + median_component               # 0-2, median bias
        + xcell_component                # 0-3, cross-cell correlation
    )

    return {
        "total_score": round(total, 2),
        "max_possible": 122.0,
        "suites_passed": suites_passed,
        "suite_detail": suite_map,
        "components": {
            "suite_pass": suites_passed * 10,
            "ci_coverage": round(ci_component, 3),
            "ci_cell": ci_cell_component,
            "regime_l2": round(regime_l2_component, 3),
            "regime_l3": round(regime_l3_component, 3),
            "ks_daily": round(ks_component, 3),
            "ks_levels": round(ks_levels_component, 3),
            "kurtosis": round(kurt_component, 3),
            "growing_unc": growing_unc,
            "cointegration": round(coint_component, 3),
            "median_bias": round(median_component, 3),
            "xcell_corr": round(xcell_component, 3),
        },
        "raw_metrics": {
            "ci_90": round(ci_90, 4),
            "worst_cell_pass": worst_cell_pass,
            "regime_l2_failing": n_failing_l2,
            "regime_l2_total": total_l2,
            "regime_l3_catastrophic": l3_catastrophic,
            "ks_daily_pass": ks_daily_pass,
            "ks_levels_pass": ks_levels_pass,
            "kurtosis_ratio": round(kurtosis_ratio, 4),
            "coint_gen_gt_ratio": round(coint_ratio, 4),
            "median_bias_pass": median_pass,
            "xcell_corr_ratio": round(corr_ratio, 4),
        }
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_score.py <summary.json>")
        sys.exit(1)

    result = compute_score(sys.argv[1])
    print(json.dumps(result, indent=2))
    print(f"\n=== COMPOSITE SCORE: {result['total_score']} / {result['max_possible']} "
          f"(suites: {result['suites_passed']}/9) ===")
