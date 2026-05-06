import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_casebook_acceptance_audit import (
    audit_case,
    build_acceptance_audit,
    render_acceptance_markdown,
)


def _case(
    *,
    window_id: str = "w1",
    target_cosine: float = 0.9,
    energy_improvement: float = 0.2,
    crps_improvement: float = 0.1,
    coverage: float = 0.65,
    mean_path_improvement: float = -0.1,
    analogue_similarity: float = 0.88,
    validation_errors: list[dict] | None = None,
    validation_warnings: list[dict] | None = None,
) -> dict:
    return {
        "window_id": window_id,
        "regime_tags": ["risk_off_stress"],
        "input_narrative": "Risk-off stress with equities lower and volatility higher.",
        "market_implications": [
            {"market": "SPX", "direction": "down", "magnitude": "large"},
            {"market": "VIX", "direction": "up", "magnitude": "large"},
        ],
        "grounding": {
            "external_news_used": False,
            "validation_errors": validation_errors or [],
            "validation_warnings": validation_warnings or [],
            "non_observed_catalysts": [],
        },
        "bridge": {
            "target_cosine": target_cosine,
            "true_rank_test_pool": 4,
            "hard_negative_gap": 0.8,
        },
        "historical_analogues": [
            {
                "window_id": "a1",
                "similarity": analogue_similarity,
                "primary_narrative": "Prior risk-off analogue.",
            },
            {
                "window_id": "a2",
                "similarity": 0.82,
                "primary_narrative": "Another analogue.",
            },
            {
                "window_id": "a3",
                "similarity": 0.81,
                "primary_narrative": "Third analogue.",
            },
        ],
        "scores": {
            "narrative_generator_topk": {
                "energy_score_improvement_vs_persistence": energy_improvement,
                "ensemble_crps_improvement_vs_persistence": crps_improvement,
                "coverage_80": coverage,
                "mean_path_mae_improvement_vs_persistence": mean_path_improvement,
            }
        },
    }


def _casebook(cases: list[dict]) -> dict:
    return {
        "title": "Casebook",
        "summary": {"case_count": len(cases)},
        "cases": cases,
    }


def test_audit_case_passes_distributional_scenario_with_point_path_warning() -> None:
    result = audit_case(_case())

    assert result["status"] == "warning"
    assert "point_path_lags_persistence" in result["warning_codes"]
    assert "product_framing" in result["bottleneck_tags"]
    assert result["rule_counts"] == {"fail": 0, "pass": 5, "warning": 1}


def test_audit_case_fails_unsupported_labels_and_weak_bridge() -> None:
    result = audit_case(
        _case(
            target_cosine=0.6,
            validation_errors=[
                {"code": "external_catalyst_requires_grounding", "severity": "error"}
            ],
        )
    )

    assert result["status"] == "fail"
    assert "label_validation_error" in result["failure_codes"]
    assert "weak_bridge_alignment" in result["failure_codes"]
    assert {"label_quality", "bridge_alignment"} <= set(result["bottleneck_tags"])


def test_build_acceptance_audit_summarizes_status_regime_and_bottlenecks() -> None:
    audit = build_acceptance_audit(
        _casebook(
            [
                _case(window_id="warn_case"),
                _case(
                    window_id="fail_case", target_cosine=0.5, energy_improvement=-0.1
                ),
            ]
        ),
        title="Acceptance audit",
    )

    assert audit["summary"]["case_count"] == 2
    assert audit["summary"]["status_counts"] == {"fail": 1, "warning": 1}
    assert audit["summary"]["regime_status_counts"]["risk_off_stress"] == {
        "fail": 1,
        "warning": 1,
    }
    assert audit["summary"]["bottleneck_counts"]["bridge_alignment"] == 1


def test_render_acceptance_markdown_contains_gate_and_case_table() -> None:
    audit = build_acceptance_audit(_casebook([_case()]))
    markdown = render_acceptance_markdown(audit)

    assert "# Risk Manager Casebook Acceptance Audit" in markdown
    assert "## Production Gate" in markdown
    assert "| Window | Status | Regimes | Bottlenecks | Key Issues |" in markdown
    assert "point_path_lags_persistence" in markdown
