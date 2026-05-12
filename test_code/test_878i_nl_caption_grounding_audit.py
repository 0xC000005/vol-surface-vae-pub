import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_caption_grounding_audit import (
    build_caption_grounding_audit,
)


def test_caption_grounding_audit_flags_rejected_and_low_margin_windows():
    pipeline_report = {
        "rejected_label_windows": ["w_rejected"],
        "label_validation": [
            {
                "window_id": "w_rejected",
                "errors": [
                    {
                        "code": "external_catalyst_requires_grounding",
                        "message": "external catalyst problem",
                        "severity": "error",
                    }
                ],
            }
        ],
        "narrative_bundles": [
            {
                "window_id": "w_rejected",
                "narratives": [
                    {
                        "text": "Rejected story",
                        "observed_fact_tokens": "SPX: DOWN LARGE",
                        "unsupported_claims": ["specific policy cause"],
                    }
                ],
                "narrative_catalysts": [
                    {"label": "policy shock", "grounding_status": "unsupported"}
                ],
            },
            {
                "window_id": "w_low_margin",
                "narratives": [
                    {
                        "text": "Low margin story",
                        "observed_fact_tokens": "VIX: UP LARGE",
                        "unsupported_claims": [],
                    }
                ],
                "narrative_catalysts": [],
            },
        ],
    }
    bridge_report = {
        "evaluation": {
            "hard_negative_separation": {
                "windows": [
                    {
                        "window_id": "w_low_margin",
                        "hard_margin": 0.10,
                        "negative_gap": 0.20,
                    },
                    {
                        "window_id": "w_ok",
                        "hard_margin": 0.90,
                        "negative_gap": 0.80,
                    },
                ]
            }
        }
    }

    audit = build_caption_grounding_audit(
        pipeline_report,
        bridge_report,
        low_margin_threshold=0.50,
        low_gap_threshold=0.50,
    )

    assert audit["summary"]["case_count"] == 2
    assert audit["summary"]["rejected_label_count"] == 1
    assert audit["summary"]["low_margin_count"] == 1
    cases = {case["window_id"]: case for case in audit["cases"]}
    assert cases["w_rejected"]["recommended_action"] == "regenerate_or_repair_label"
    assert cases["w_rejected"]["unsupported_claim_count"] == 1
    assert cases["w_low_margin"]["recommended_action"] == "inspect_hard_negatives"


def test_caption_grounding_audit_sorts_low_margin_cases_first():
    pipeline_report = {"narrative_bundles": []}
    bridge_report = {
        "evaluation": {
            "hard_negative_separation": {
                "windows": [
                    {"window_id": "w2", "hard_margin": 0.30, "negative_gap": 0.40},
                    {"window_id": "w1", "hard_margin": 0.10, "negative_gap": 0.40},
                ]
            }
        }
    }

    audit = build_caption_grounding_audit(
        pipeline_report,
        bridge_report,
        low_margin_threshold=0.50,
        low_gap_threshold=0.50,
    )

    assert [case["window_id"] for case in audit["cases"]] == ["w1", "w2"]
