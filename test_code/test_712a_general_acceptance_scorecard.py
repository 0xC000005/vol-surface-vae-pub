import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.acceptance_scorecard_712a import (
    evaluate_general_scorecard,
    score_anchor_panel,
    score_framework_gate,
    score_iv_suite,
    score_joint_panel,
)


def _iv_result(failed_suites=None, risk_state=True):
    return {
        "summary": {
            "n_pass": 10 if failed_suites else 11,
            "n_total": 11,
            "failed_suites": failed_suites or [],
        },
        "conditionality": {"overall_pass": False},
        "risk_state_allocation": {"overall_pass": risk_state},
    }


def _panel_summary(n_factors=3):
    return {
        "finite_rate": 1.0,
        "n_factors": n_factors,
        "factor_delta_ks_mean": 0.11,
        "factor_delta_ks_pass_020": n_factors,
        "factor_tail_q99_ratio_median": 1.02,
        "factor_tail_q99_pass_05_20": n_factors,
        "factor_factor_corr": {
            "upper_corr": 0.74,
            "gt_mean_abs": 0.20,
            "gen_mean_abs": 0.13,
        },
        "iv_factor_corr": {
            "matrix_corr": 0.68,
            "gt_mean_abs": 0.18,
            "gen_mean_abs": 0.11,
        },
        "conditional_panel": {
            "median_mae_reduction_vs_rolled_pct": 8.0,
            "history_activity_width_spearman": 0.22,
        },
    }


def _framework_manifest():
    shared = {
        "generated_coordinate": "state_normalized_innovation",
        "normalization_family": "support_aware_history_scale",
        "temporal_factorization": "ar_daily",
        "backend": "flow_matching",
        "stochastic_source": "shared_noise",
        "shared_core": "state_encoder_transition",
        "scalar_loss_terms": ["fm", "rollout_energy"],
        "loss_weights": {"fm": 1.0, "rollout_energy": 0.2},
        "sampler": "same_flow_sampler",
        "training_protocol": "same_epochs_same_sampling",
    }
    return {
        "framework_id": "demo_framework",
        "post_hoc_glued_decks": False,
        "task_specific_loss_recipes": False,
        "scope_values": {
            "iv_only": dict(shared, input_head="iv", decoder_head="iv"),
            "anchor_only": dict(shared, input_head="anchor", decoder_head="anchor"),
            "joint": dict(shared, input_head="joint", decoder_head="joint"),
        },
    }


def test_iv_suite_replaces_old_conditionality_failure_with_risk_state_gate() -> None:
    scored = score_iv_suite(_iv_result(failed_suites=["conditionality"], risk_state=True))

    assert scored["pass"] is True
    assert scored["effective_failed_suites"] == []
    assert scored["risk_state_allocation_pass"] is True


def test_iv_suite_does_not_hide_non_conditional_failures() -> None:
    scored = score_iv_suite(
        _iv_result(failed_suites=["conditionality", "coverage"], risk_state=True)
    )

    assert scored["pass"] is False
    assert scored["effective_failed_suites"] == ["coverage"]


def test_iv_suite_treats_old_cointegration_as_monitoring_only() -> None:
    scored = score_iv_suite(
        _iv_result(failed_suites=["conditionality", "cointegration"], risk_state=True)
    )

    assert scored["pass"] is True
    assert scored["effective_failed_suites"] == []
    assert scored["monitoring_only_suites"] == ["cointegration"]
    assert (
        scored["cointegration_policy"]
        == "old_iv_ewma_cointegration_monitoring_only_until_new_gate_defined"
    )


def test_anchor_panel_requires_marginals_tails_dependency_and_conditioning() -> None:
    scored = score_anchor_panel(_panel_summary())

    assert scored["pass"] is True
    assert scored["checks"]["factor_delta_ks"]["pass"] is True
    assert scored["checks"]["conditional_panel"]["pass"] is True

    bad = _panel_summary()
    bad["factor_tail_q99_pass_05_20"] = 1
    assert score_anchor_panel(bad)["pass"] is False


def test_joint_panel_requires_anchor_quality_and_iv_factor_coherence() -> None:
    scored = score_joint_panel(_panel_summary())

    assert scored["pass"] is True
    assert scored["checks"]["iv_factor_corr"]["pass"] is True

    bad = _panel_summary()
    bad["iv_factor_corr"]["matrix_corr"] = 0.1
    assert score_joint_panel(bad)["pass"] is False


def test_framework_gate_allows_heads_but_rejects_scope_specific_loss_recipes() -> None:
    scored = score_framework_gate(_framework_manifest())

    assert scored["pass"] is True
    assert scored["allowed_scope_differences"] == ["decoder_head", "input_head"]

    bad = _framework_manifest()
    bad["scope_values"]["joint"]["loss_weights"] = {"fm": 1.0, "rollout_energy": 0.8}
    assert score_framework_gate(bad)["pass"] is False


def test_general_scorecard_requires_all_scope_and_framework_gates() -> None:
    report = evaluate_general_scorecard(
        iv_result=_iv_result(failed_suites=["conditionality"], risk_state=True),
        anchor_panel=_panel_summary(),
        joint_panel=_panel_summary(),
        framework_manifest=_framework_manifest(),
    )

    assert report["overall_pass"] is True
    assert report["gate_passes"] == {
        "iv": True,
        "anchor": True,
        "joint": True,
        "framework": True,
    }
