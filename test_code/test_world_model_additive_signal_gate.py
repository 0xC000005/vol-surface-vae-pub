import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.analyze_additive_signal_gate import (
    summarize_additive_signal_gate,
)


def test_summarize_additive_signal_gate_keeps_raw_exact_state_as_guardrail():
    taxonomy = {
        "family_summaries": {
            "path_shape_or_risk_width": {
                "n_targets": 2,
                "barlow_best_raw_wins": 2,
                "barlow_adds_to_raw_last": 2,
            },
            "persistence_or_exact_state_dominated": {
                "n_targets": 2,
                "barlow_best_raw_wins": 0,
                "barlow_adds_to_raw_last": 1,
            },
        },
        "rows": [
            {"target_family": "path_shape_or_risk_width"},
            {"target_family": "path_shape_or_risk_width"},
            {"target_family": "persistence_or_exact_state_dominated"},
            {"target_family": "persistence_or_exact_state_dominated"},
        ],
    }
    regime = {
        "decision": {
            "accuracy_gate_passed": False,
            "balanced_signal_present": True,
        },
        "scaled_barlow_vs_raw_last": {
            "macro_recall_delta": 0.17,
            "accuracy_delta": -0.08,
        },
    }
    exact_state = {
        "iv_surface_gap": {
            "raw_surface_iv_mse": 0.005,
            "scale_barlow_iv_mse": 0.015,
            "scale_to_raw_iv_mse_ratio": 3.0,
        },
        "decision": {
            "exact_iv_retention_gap_confirmed": True,
        },
    }

    summary = summarize_additive_signal_gate(
        taxonomy=taxonomy,
        regime=regime,
        exact_state=exact_state,
    )

    assert summary["feature_surfaces"] == [
        "raw_only",
        "learned_only",
        "raw_plus_learned",
    ]
    assert summary["gate_layers"]["exact_state_guardrail"]["status"] == "FAIL"
    assert summary["gate_layers"]["path_shape_risk_width"]["status"] == "PASS"
    assert summary["gate_layers"]["regime_balanced_signal"]["status"] == "PARTIAL"
    assert summary["decision"]["promotion_decision"] == "DO_NOT_PROMOTE"
    assert summary["decision"]["part_b_blocked"] is True
