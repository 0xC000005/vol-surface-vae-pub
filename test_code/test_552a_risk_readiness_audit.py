import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.audit_552a_risk_manager_readiness import (
    factor_readiness,
    lower_only_coverage_pass,
    lower_only_regime_pass,
    score_candidate,
)


def _candidate_result() -> dict:
    return {
        "summary": {"n_pass": 8, "failed_suites": ["coverage", "regime_coverage"]},
        "surface": {"overall_pass": True},
        "coverage": {
            "overall": {"0.9": 0.93},
            "horizon_pass": {"1": True, "7": True, "14": True, "30": True},
            "worst_cell_per_horizon": {"1": 0.72, "7": 0.74, "14": 0.78, "30": 0.80},
            "best_cell_per_horizon": {"1": 1.0, "7": 0.98, "14": 0.97, "30": 0.99},
        },
        "conditionality": {"overall_pass": True, "mae_reduction_pct": 6.0},
        "time_series": {"overall_pass": True},
        "block_ar": {"overall_pass": True},
        "cointegration": {"overall_pass": True},
        "regime_coverage": {
            "layer1_pass": True,
            "layer2_regime_cell": {
                "calm": {"1": {"worst": 0.72, "best": 1.0}},
                "turb": {"1": {"worst": 0.71, "best": 1.0}},
            },
            "layer3_pass": True,
            "layer2_n_passing": 0,
            "layer2_n_total": 8,
        },
        "distributional_fidelity": {
            "ks_test": {"pass": True, "n_pass": 25},
            "ks_level_test": {"pass": False, "n_pass": 13},
            "median_bias": {"pass": True},
            "window_floor": {"pass": True},
            "explosion": {"pass": True},
            "cell_mae": {"pass": True},
            "overall_pass": False,
        },
        "cross_cell_correlation": {"overall_pass": True},
        "mean_reversion": {"overall_pass": True},
        "pathwise_jump_realism": {"overall_pass": True},
    }


def test_lower_only_coverage_ignores_overcoverage_but_requires_lower_bounds() -> None:
    result = _candidate_result()

    assert lower_only_coverage_pass(result["coverage"])["pass"] is True

    result["coverage"]["worst_cell_per_horizon"]["30"] = 0.66
    assert lower_only_coverage_pass(result["coverage"])["pass"] is False


def test_lower_only_regime_ignores_best_cell_overcoverage() -> None:
    result = _candidate_result()

    assert lower_only_regime_pass(result["regime_coverage"])["pass"] is True

    result["regime_coverage"]["layer2_regime_cell"]["turb"]["1"]["worst"] = 0.62
    assert lower_only_regime_pass(result["regime_coverage"])["pass"] is False


def test_score_candidate_separates_stress_readiness_from_original_suite() -> None:
    result = _candidate_result()
    scored = score_candidate("demo", result)

    assert scored["name"] == "demo"
    assert scored["stress_pass"] is True
    assert scored["original_n_pass"] == 8
    assert "level_ks_warning" in scored["warnings"]


def test_score_candidate_accepts_risk_state_allocation_as_conditionality_evidence() -> None:
    result = _candidate_result()
    result["conditionality"] = {"overall_pass": False, "mae_reduction_pct": 1.0}
    result["risk_state_allocation"] = {
        "overall_pass": True,
        "observable_state_response_pass": True,
        "oracle_future_alignment_pass": False,
        "overall_pass_rule": "observable_state_response_only_future_signal_absent",
    }

    scored = score_candidate("risk_state_demo", result)

    assert scored["conditionality"]["pass"] is True
    assert scored["conditionality"]["risk_state_allocation_pass"] is True
    assert "conditionality_borderline" not in scored["warnings"]


def test_factor_readiness_reports_available_non_iv_factors_and_model_gap() -> None:
    readiness = factor_readiness(["surface", "ret", "price", "slopes", "skews", "levels"])

    assert readiness["available_non_iv_factors"] == ["ret", "price", "slopes", "skews", "levels"]
    assert readiness["current_generator_scope"] == "iv_surface_only"
    assert readiness["multifactor_ready"] is False
