import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
    APP_CSS,
    DEFAULT_PREFIX_BRIDGE_ADAPTER,
    DEFAULT_PREFIX_BRIDGE_ARRAYS,
    DEFAULT_PREFIX_BRIDGE_REPORT,
    DEFAULT_PREFIX_FULL_START_BRIDGE_ARRAYS,
    DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
    DEFAULT_PREFIX_SUPPORT_BANK_ARRAYS,
    DEFAULT_PREFIX_SUPPORT_BANK_REPORT,
    DEFAULT_STORY,
    DEMO_TABLE_CLASS,
    apply_live_top3_90_posterior_ensemble,
    apply_live_support_gated_ensemble_calibration,
    analogues_table,
    analogue_scope_choices,
    boss_demo_live_casebook_table,
    boss_demo_pack_markdown,
    boss_demo_status_strip,
    build_start_state_payload,
    build_prefix_latent_run_args,
    build_run_args,
    cached_prefix_casebook_choices,
    cached_prefix_casebook_update,
    export_historical_start_json_for_app,
    fan_chart_figure,
    historical_start_candidate_choices,
    historical_start_candidate_to_index,
    implications_table,
    prefix_diagnostic_start_table,
    prefix_condition_implications_table,
    prefix_condition_warnings_table,
    prefix_latent_status_markdown,
    prefix_latent_product_status_markdown,
    prefix_trust_interpretation,
    prefix_visible_warning_lines,
    preview_start_state_json,
    prefix_selected_start_table,
    recommended_narrative_choices,
    recommended_narrative_text,
    prefix_shift_factor_table,
    prefix_start_candidates_table,
    prefix_user_start_table,
    prefix_validation_table,
    prefix_variant_table,
    prefix_warning_component_table,
    refresh_fan_chart,
    resolve_launch_auth,
    build_demo,
    run_live_openai_prefix_for_app,
    preview_live_openai_start_for_app,
    run_prefix_latent_for_app,
    run_story_for_app,
    scenario_summary_html,
    scenario_table,
    status_markdown,
    validation_gate_markdown,
    validation_gate_table,
    warnings_table,
)
from experiments.backfill.block_ar.nl_prefix_latent_temporal_grounding_testflight import (
    ConditionOnlyGroundingResult,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (
    _allocate_weighted_sample_counts,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (
    _load_bridge_adapter,
)


def _report() -> dict:
    return {
        "artifact_paths": {
            "json": "outputs/story_smoke_report.json",
            "markdown": "outputs/story_smoke_report.md",
        },
        "story": DEFAULT_STORY,
        "grounding": {
            "narrative_frame": "fragile risk-on rebound",
            "market_implications": [
                {
                    "market": "SPX",
                    "direction": "up",
                    "magnitude": "small",
                    "confidence": "high",
                    "inferred": False,
                    "evidence": ["equities are recovering"],
                },
                {
                    "market": "VIX",
                    "direction": "down",
                    "magnitude": "small",
                    "confidence": "high",
                    "inferred": False,
                    "evidence": ["volatility is compressing"],
                },
            ],
            "grounding_warnings": [
                {
                    "severity": "warning",
                    "code": "interpretive_phrase",
                    "message": "Risk-on rebound is an interpretation.",
                }
            ],
        },
        "condition_diagnostics": {
            "condition_dim": 128,
            "query_condition_norm": 11.1,
            "top_cosine": 0.9,
            "top_gap": 0.02,
        },
        "relevance": {
            "status": "pass",
            "reason": "story maps to a nearby historical condition cluster",
        },
        "hard_case_gate": {"status": "pass", "reason": "no hard-case overlap"},
        "historical_analogues": [
            {
                "window_id": "joint39_val_0031",
                "cosine": 0.9,
                "weight": 0.55,
                "manifest_split": "train",
                "implication_alignment": {"match_rate": 0.67, "status": "warning"},
                "primary_narrative": "A clear risk-on regime.",
            }
        ],
        "generation": {
            "generated_state_shape": [3, 2, 30, 39],
            "finite_rate": 1.0,
            "path_quantiles": [
                {
                    "market": "SPX",
                    "display_name": "SPX",
                    "analogue_key": "ALL",
                    "analogue_label": "All retrieved analogues",
                    "days": [1, 2],
                    "p10": [-1.0, -2.0],
                    "p50": [0.1, 0.2],
                    "p90": [1.0, 2.0],
                    "mean": [0.2, 0.3],
                },
                {
                    "market": "SPX",
                    "display_name": "SPX",
                    "analogue_key": "RANK_1",
                    "analogue_label": "Analogue 1: joint39_val_0031",
                    "window_id": "joint39_val_0031",
                    "days": [1, 2],
                    "p10": [1.0, 2.0],
                    "p50": [1.1, 2.2],
                    "p90": [1.4, 2.6],
                    "mean": [1.2, 2.3],
                    "realized_path": [0.8, 2.4],
                    "sample_paths": [
                        {"label": "Generated path 1", "values": [0.9, 2.0]},
                        {"label": "Generated path 2", "values": [1.3, 2.5]},
                    ],
                },
                {
                    "market": "IV_ATM_3M",
                    "display_name": "IV ATM 3M (K=1.00)",
                    "analogue_key": "ALL",
                    "analogue_label": "All retrieved analogues",
                    "cell": {
                        "row": 1,
                        "col": 2,
                        "maturity": "3M",
                        "moneyness": "1.00",
                    },
                    "days": [1, 2],
                    "p10": [-0.02, -0.03],
                    "p50": [0.01, 0.02],
                    "p90": [0.04, 0.05],
                    "mean": [0.02, 0.03],
                },
            ],
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": 1.2,
                    "p10": -0.4,
                    "p90": 2.1,
                }
            ],
        },
    }


def _boss_pack() -> dict:
    return {
        "artifact_paths": {"summary_markdown": "outputs/boss_demo_pack.md"},
        "validation_snapshot": {
            "run_count": 9,
            "improved_crps_rows": 9,
            "improved_energy_rows": 9,
            "mean_crps_improvement_vs_persistence": 0.1469,
        },
        "live_casebook_snapshot": {
            "case_count": 3,
            "pass_count": 3,
            "total_openai_tokens": 5609,
            "min_support_candidate_count": 8,
            "grounding_models": ["gpt-5.4-mini"],
            "embedding_models": ["text-embedding-3-small"],
            "case_rows": [
                {
                    "case_name": "safe_haven_gold_bid_18",
                    "expected_start_index": 18,
                    "overall_status": "pass",
                    "condition_only_validation_status": "pass",
                    "forward_warning_count": 1,
                    "support_candidate_count": 8,
                    "summary_path": "safe_haven/gradio_api_smoke_summary.json",
                }
            ],
        },
        "fixed_start_caption_audit_snapshot": {
            "status": "pass",
            "professional_minus_start_only": {
                "factor_terminal_ks": 0.2611,
                "portfolio_terminal_ks": 0.3750,
            },
            "professional_minus_simple": {"path_energy": 4.4262},
        },
    }


def _prefix_report() -> dict:
    return {
        "artifact_paths": {
            "report": "outputs/prefix_latent_story_smoke_report.json",
            "markdown": "outputs/prefix_latent_story_smoke_report.md",
            "arrays": "outputs/prefix_latent_story_smoke_arrays.npz",
        },
        "cached_query": {
            "window_id": "joint39_val_0370",
            "kind": "revised_market_description",
            "narrative_text": "Risk-on market tape with tighter spreads.",
            "text_memory_dim": 128,
            "condition_source": "condition_only_openai_story",
            "grounding": {
                "market_implications": [
                    {
                        "market": "SPX",
                        "direction": "up",
                        "magnitude": "small",
                        "confidence": "high",
                        "horizon": "current_state",
                        "evidence": ["equities are recovering"],
                    }
                ],
                "non_conditioning_forward_language": [
                    {
                        "phrase": "volatility could reverse",
                        "reason": "future-looking stress phrase",
                        "severity": "warning",
                    }
                ],
                "grounding_warnings": [],
            },
            "memory_prior": {
                "support_diversity_policy": {
                    "policy": "direction_checked_latent_temporal_diverse_support",
                    "requested_top_k": 8,
                    "selected_count": 4,
                    "direction_gate": True,
                    "latent_max_pairwise_cosine": 0.95,
                    "temporal_min_index_gap": 30,
                    "temporal_non_overlap_enforced": True,
                    "padding_with_temporal_overlaps": False,
                },
                "candidate_details": [
                    {
                        "rank": 1,
                        "window_id": "joint39_val_0269",
                        "bridge_local_index": 269,
                        "source_index": 4762,
                        "history_end_date": "2020-03-12",
                        "manifest_split": "train",
                        "weight": 0.42,
                        "memory_support_cosine": 0.887,
                        "start_distance_z": 6.94,
                        "recent_prefix_alignment_score": 0.8,
                        "recent_prefix_mismatches": 1,
                        "recent_prefix_checked": 5,
                        "combined_score": 0.73,
                    }
                ],
            },
        },
        "variant_rows": [
            {
                "variant": "original",
                "query_window_id": "joint39_val_0370",
                "start_window_id": "joint39_val_0370",
                "start_distance_z": 0.0,
                "memory_support_cosine": 0.812,
                "start_selection_method": "",
                "case_role": "diagnostic_original_start",
                "is_operational": False,
            },
            {
                "variant": "nearest_train_start",
                "query_window_id": "joint39_val_0370",
                "start_window_id": "joint39_val_0269",
                "start_window_index": 269,
                "start_distance_z": 6.94,
                "memory_support_cosine": 0.887,
                "start_selection_method": "max_memory_inside_start_threshold",
                "case_role": "operational_selected_start",
                "is_operational": True,
            },
        ],
        "selected_start_state": {
            "values_by_name": {
                "factor:spx": 4200.0,
            }
        },
        "validation_gate": {
            "overall_status": "pass",
            "operational_status": "pass",
            "selected_start_status": "pass",
            "diagnostic_baseline_status": "pass",
            "stress_status": "pass",
            "endpoint_max_abs_error": 0.0,
            "warning_counts": {},
            "fail_counts": {},
            "cases": [
                {
                    "variant": "original",
                    "case_role": "operational_selected_start",
                    "query_window_index": 153,
                    "start_window_index": 153,
                    "status": "pass",
                    "input_memory_cosine": 0.899,
                    "start_distance_z": 0.0,
                    "terminal_mean_abs_delta_z": 0.0,
                    "warnings": [],
                    "failures": [],
                }
            ],
        },
        "condition_only_product_gate": {
            "production_decision": {
                "decision": "warn_and_continue_for_narrative_only",
                "ui_guidance": "Show the warning and allow explicit start input.",
            },
            "decompositions": [
                {
                    "start_mode": "balanced_memory_start",
                    "components": {
                        "support_prior": {
                            "status": "pass",
                            "mismatch_count": 0,
                        },
                        "memory_compatibility": {
                            "status": "pass",
                            "input_memory_cosine": 0.969,
                        },
                        "start_distance": {
                            "status": "pass",
                            "start_distance_z": 14.922,
                        },
                        "rollout_shift": {
                            "status": "warning",
                            "terminal_mean_abs_delta_z": 1.165,
                        },
                    },
                    "top_rollout_shift_factors": [
                        {
                            "factor": "SPX",
                            "terminal_abs_shift_z": 3.249,
                            "signed_terminal_shift_z": 3.249,
                            "mean_path_abs_shift_z": 2.714,
                        }
                    ],
                }
            ],
        },
        "start_reliability_gate": {
            "product_status": "pass",
            "decision": "Start has enough fixed-start evidence.",
            "warnings": [],
            "failures": [],
        },
        "generation": {
            "generated_state_shape": [2, 16, 30, 39],
            "finite_rate": 1.0,
            "rollout_temperature": 0.5,
            "rollout_fan_scale": 3.5,
            "sample_count": 16,
            "window_scores": [
                {
                    "variant": "nearest_train_start",
                    "start_window_index": 269,
                    "methods": {
                        "text_memory_plus_start_prefix_decoder": {
                            "ensemble_crps_z_improvement_vs_persistence": 0.145,
                            "energy_score_z_improvement_vs_persistence": 0.159,
                        }
                    },
                }
            ],
            "path_quantiles": [
                {
                    "market": "SPX",
                    "display_name": "SPX",
                    "analogue_key": "ALL",
                    "analogue_label": "All start variants",
                    "days": [1, 2],
                    "p10": [-1.0, -2.0],
                    "p50": [0.1, 0.2],
                    "p90": [1.0, 2.0],
                    "mean": [0.2, 0.3],
                },
                {
                    "market": "SPX",
                    "display_name": "SPX",
                    "analogue_key": "RANK_1",
                    "analogue_label": "Diagnostic baseline: joint39_val_0370",
                    "days": [1, 2],
                    "p10": [0.0, 0.1],
                    "p50": [0.3, 0.5],
                    "p90": [0.8, 1.1],
                    "mean": [0.4, 0.6],
                },
                {
                    "market": "SPX",
                    "display_name": "SPX",
                    "analogue_key": "RANK_2",
                    "analogue_label": "Selected start: joint39_val_0269",
                    "days": [1, 2],
                    "p10": [-0.5, -0.6],
                    "p50": [0.4, 0.7],
                    "p90": [1.2, 1.5],
                    "mean": [0.5, 0.8],
                },
            ],
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": -52.5,
                    "p10": -113.3,
                    "p90": 8.4,
                }
            ],
        },
    }


def _prefix_user_start_report() -> dict:
    report = _prefix_report()
    report["user_start_state"] = {
        "label": "today",
        "source_path": "tmp/today_start.json",
        "source_format": "values_by_name",
        "coordinate": "raw_state",
        "dimension": 39,
        "spec_names": ["iv:00", "factor:vix"],
    }
    report["variant_rows"][1] = {
        **report["variant_rows"][1],
        "variant": "user_start_state",
        "start_window_id": "today",
        "start_window_index": -1,
        "start_manifest_split": "user_supplied",
        "start_selection_method": "user_supplied_joint39_state",
        "nearest_train_start_window_index": 18,
        "max_abs_user_start_z": 1.817,
    }
    return report


def test_resolve_launch_auth_defaults_to_none_for_local_dev() -> None:
    assert resolve_launch_auth(env={}) is None


def test_resolve_launch_auth_reads_configured_env_names() -> None:
    auth = resolve_launch_auth(
        env={"DEMO_USER": "risk", "DEMO_PASSWORD": "manager"},
        user_env="DEMO_USER",
        password_env="DEMO_PASSWORD",
    )

    assert auth == ("risk", "manager")


def test_resolve_launch_auth_requires_both_values_when_enabled() -> None:
    try:
        resolve_launch_auth(
            env={"DEMO_USER": "risk"},
            user_env="DEMO_USER",
            password_env="DEMO_PASSWORD",
            require_auth=True,
        )
    except RuntimeError as error:
        assert "DEMO_PASSWORD" in str(error)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected missing password to fail")


def test_demo_css_keeps_tables_mobile_safe() -> None:
    assert "overflow-wrap: anywhere" in APP_CSS
    assert "@media (max-width: 900px)" in APP_CSS
    assert ".demo-responsive-row" in APP_CSS
    assert ".demo-shell" in APP_CSS
    assert "margin-left: auto" in APP_CSS
    assert "margin-right: auto" in APP_CSS
    assert "@media (max-width: 640px)" in APP_CSS
    assert f".{DEMO_TABLE_CLASS}" in APP_CSS
    assert "overflow-x: auto" in APP_CSS


def test_demo_places_product_workflow_before_diagnostics() -> None:
    source = inspect.getsource(build_demo)

    assert "Validation evidence and casebook details" not in source
    assert "boss_demo_status_strip" not in source
    assert "Advanced story processing" not in source
    assert "Approve starting level for scenario generation" not in source
    assert "prefix_casebook_choice" not in source
    assert "prefix_cached_condition_report" not in source
    assert "prefix_live_story" not in source
    assert "prefix_condition_only_story" not in source
    assert "prefix_approve_start" not in source
    assert "prefix_start_source" not in source
    assert "prefix_start_mode" not in source
    assert "Recommended start method" not in source
    assert "Use recommended start" not in source
    assert "Research Diagnostics" not in source
    assert "Story-Smoke Diagnostics" not in source
    assert "Start/support view" not in source
    assert "All retrieved analogues" not in source
    assert "Scenario samples" not in source
    assert "Validate Starting Level" not in source
    assert "Recommended narrative examples" in source
    assert "Reliability-checked demo starts" not in source
    assert "Grounded current/recent market claims" in source
    # P10: user-facing wording standardized on "historical analogues"; "support"
    # phrasing is reserved for audit surfaces (not in build_demo's primary copy).
    assert "Selected historical analogues" in source
    assert "Selected support regimes" not in source
    assert "support ensemble" not in source
    assert "top3/90" not in source
    assert "closest one-to-three historical regimes" in source
    assert "prefix_preview_button" not in source
    assert "preview_live_openai_start_for_app" not in source
    assert "validate it, then generate" not in source
    assert "approve a starting level" not in source
    assert "How to read this screen" in source
    assert "A historical start is the day-0 market level" in source
    # P9: the redundant "## Main Workflow" heading that bisected the inputs is gone.
    assert "## Main Workflow" not in source


def test_recommended_narrative_examples_fill_story_without_cached_casebook() -> None:
    choices = recommended_narrative_choices()
    labels = [label for label, _value in choices]

    assert labels[0] == "Type my own narrative"
    assert "Dollar squeeze/liquidation - short" in labels
    assert "Dollar squeeze/liquidation - full" in labels
    assert "Safe-haven risk-off - short" in labels
    assert recommended_narrative_text("", "my custom story") == "my custom story"
    selected = recommended_narrative_text("dollar_squeeze_liquidation__full", "")
    assert "dollar" in selected.lower()
    assert "liquidation" in selected.lower()
    assert "Mechanical summary:" not in selected
    assert "No-forecast caveat:" not in selected
    assert "will" not in selected.lower()


def test_projected_memory_bridge_loader_reads_checkpoint_hidden_dim() -> None:
    checkpoint = Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "stride5_14x14_retrieval_training_openai_holdout_991a_seed1/"
        "projected_memory/projected_memory_bridge_best.pt"
    )

    adapter = _load_bridge_adapter(
        checkpoint,
        embedding_dim=3072,
        condition_dim=128,
    )

    assert adapter.net[1].weight.shape == (256, 3072)
    assert adapter.net[3].weight.shape == (128, 256)


def test_build_prefix_latent_run_args_uses_clean_full_start_bridge_backend() -> None:
    args = build_prefix_latent_run_args(
        start_mode="explicit_start_window",
        samples=12,
        live_story=True,
        story="Dollar squeeze with oil and gold liquidation.",
    )

    # Start pool = the full train-region start bridge (4010 windows, 2000-2015,
    # incl. 2008), derived from the clean 939a numeric bank with a zeros
    # condition_vectors placeholder. Contamination hold STILL holds: this is NOT
    # the 990f/991a/projected_memory_14x14 contaminated bridge, and the query
    # adapter + embedding model are unchanged (1536->128, text-embedding-3-small).
    assert args.bridge_report == DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT
    assert "full_start_bridge" in args.bridge_report
    assert "991a_seed1" not in args.bridge_report
    assert "projected_memory_14x14" not in args.bridge_report
    assert args.bridge_arrays == DEFAULT_PREFIX_FULL_START_BRIDGE_ARRAYS
    assert "991a_seed1" not in args.bridge_arrays
    # Query adapter + embedding model unchanged (legacy clean 1536-d oracle).
    assert args.bridge_adapter == DEFAULT_PREFIX_BRIDGE_ADAPTER
    assert args.embedding_model == "text-embedding-3-small"
    # Retrieval pool stays the clean 939a support bank.
    assert "support_bank_train_all_939a" in str(args.support_bank_report)


def test_bridge_adapter_rejects_wrong_embedding_dim() -> None:
    # A 3072-d embedding fed into the 1536-d legacy adapter must fail loudly, not silently.
    import numpy as np
    import pytest
    import torch

    adapter = _load_bridge_adapter(
        DEFAULT_PREFIX_BRIDGE_ADAPTER, embedding_dim=1536, condition_dim=128
    )
    with pytest.raises((RuntimeError, ValueError)):
        adapter(torch.tensor(np.zeros((1, 3072), dtype="float32")))


def test_table_formatters_expose_demo_evidence() -> None:
    report = _report()
    prefix_report = _prefix_report()
    prefix_report["generation"]["start_only_baseline"] = {
        "terminal_delta_summary": [
            {
                "market": "SPX",
                "mean_terminal_delta": 10.0,
                "p10": -5.0,
                "p90": 25.0,
            }
        ]
    }

    assert implications_table(report).iloc[0]["Market"] == "SPX"
    assert warnings_table(report).iloc[0]["Code"] == "interpretive_phrase"
    assert analogues_table(report).iloc[0]["Implication Match"] == "0.670"
    assert scenario_table(report).iloc[0]["Market"] == "SPX"
    prefix_scenario = scenario_table(prefix_report)
    assert list(prefix_scenario.columns) == [
        "Market",
        "Baseline View",
        "Baseline Path Share",
        "Baseline Mean Move",
        "Narrative View",
        "Narrative Path Share",
        "Narrative Mean Move",
        "30d Change vs Baseline",
    ]
    assert prefix_scenario.iloc[0]["Market"] == "SPX"
    assert prefix_scenario.iloc[0]["Baseline View"] == "Up"
    assert prefix_scenario.iloc[0]["Baseline Path Share"] == "n/a"
    assert prefix_scenario.iloc[0]["Baseline Mean Move"] == "+10 pts / +0.9σ"
    assert prefix_scenario.iloc[0]["Narrative View"] == "Down"
    assert prefix_scenario.iloc[0]["Narrative Path Share"] == "n/a"
    assert prefix_scenario.iloc[0]["Narrative Mean Move"] == "-52 pts / -1.1σ"
    assert prefix_scenario.iloc[0]["30d Change vs Baseline"] == "More down than baseline"
    summary_html = scenario_summary_html(prefix_report)
    assert 'class="demo-dir demo-dir-down"' in summary_html
    assert '<span class="demo-dir-arrow">↓</span> Down' in summary_html
    flat_report = {
        "generation": {
            "terminal_delta_summary": [
                {
                    "market": "VIX",
                    "mean_terminal_delta": 0.0,
                    "p10": -1.0,
                    "p90": 1.0,
                }
            ],
            "start_only_baseline": {
                "terminal_delta_summary": [
                    {
                        "market": "VIX",
                        "mean_terminal_delta": 0.0,
                        "p10": -1.0,
                        "p90": 1.0,
                    }
                ]
            },
        }
    }
    assert scenario_table(flat_report).iloc[0]["Narrative View"] == "-"
    assert '<span class="demo-dir-arrow">-</span></span>' in scenario_summary_html(
        flat_report
    )
    less_down_report = {
        "generation": {
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": -5.0,
                    "p10": -20.0,
                    "p90": 10.0,
                }
            ],
            "start_only_baseline": {
                "terminal_delta_summary": [
                    {
                        "market": "SPX",
                        "mean_terminal_delta": -12.0,
                        "p10": -27.0,
                        "p90": 3.0,
                    }
                ]
            },
        }
    }
    less_down = scenario_table(less_down_report)
    assert less_down.iloc[0]["Baseline View"] == "Down"
    assert less_down.iloc[0]["Narrative View"] == "Down"
    assert less_down.iloc[0]["Narrative Path Share"] == "n/a"
    assert less_down.iloc[0]["Narrative Mean Move"] == "-5 pts / -0.4σ"
    assert less_down.iloc[0]["30d Change vs Baseline"] == "Less down than baseline"
    less_down_html = scenario_summary_html(less_down_report)
    assert 'class="demo-dir demo-dir-moderate"' in less_down_html
    # Relative-tilt labels carry no arrow glyph now (the words carry direction);
    # only the colour class + text remain.
    assert (
        '<span class="demo-dir demo-dir-moderate">Less down than baseline</span>'
        in less_down_html
    )
    probability_report = {
        "generation": {
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": 1.0,
                    "p10": -1.0,
                    "p90": 5.0,
                    "terminal_probability_up": 0.82,
                    "terminal_probability_down": 0.18,
                }
            ],
            "start_only_baseline": {
                "terminal_delta_summary": [
                    {
                        "market": "SPX",
                        "mean_terminal_delta": -2.0,
                        "p10": -7.0,
                        "p90": 2.0,
                        "terminal_probability_up": 0.30,
                        "terminal_probability_down": 0.70,
                    }
                ]
            },
        }
    }
    probability_table = scenario_table(probability_report)
    assert probability_table.iloc[0]["Baseline View"] == "Down"
    assert probability_table.iloc[0]["Baseline Path Share"] == "70% down"
    assert probability_table.iloc[0]["Baseline Mean Move"] == "-2 pts / -0.6σ"
    assert probability_table.iloc[0]["Narrative View"] == "Up"
    assert probability_table.iloc[0]["Narrative Path Share"] == "82% up"
    assert probability_table.iloc[0]["Narrative Mean Move"] == "+1 pts / +0.4σ"
    skewed_report = {
        "generation": {
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": 3.0,
                    "p10": -1.0,
                    "p90": 8.0,
                    "terminal_probability_up": 0.45,
                    "terminal_probability_down": 0.55,
                }
            ],
            "start_only_baseline": {"terminal_delta_summary": []},
        }
    }
    skewed_table = scenario_table(skewed_report)
    assert skewed_table.iloc[0]["Narrative View"] == "-"
    assert skewed_table.iloc[0]["Narrative Path Share"] == "55% down"
    assert skewed_table.iloc[0]["Narrative Mean Move"] == "+3 pts / +0.9σ"
    assert "Narrative" in analogues_table(report).columns


def test_boss_demo_pack_formatters_surface_live_readiness() -> None:
    strip = boss_demo_status_strip(_boss_pack())
    markdown = boss_demo_pack_markdown(_boss_pack())
    table = boss_demo_live_casebook_table(_boss_pack())

    assert "Validation evidence:" in strip
    assert "details below" in strip
    assert "Demo readiness evidence" in markdown
    assert "Live API casebook: `3/3` pass" in markdown
    assert "Fixed-start caption audit: `pass`" in markdown
    assert "Professional vs start-only portfolio KS delta `0.375`" in markdown
    assert "OpenAI tokens `5609`" in markdown
    assert "Forward-risk language is warning-only" in markdown
    assert table.iloc[0]["Case"] == "safe_haven_gold_bid_18"
    assert table.iloc[0]["Support"] == "8"


def test_boss_demo_pack_markdown_handles_missing_pack() -> None:
    markdown = boss_demo_pack_markdown({})

    assert "Status: `not available`" in markdown


def test_status_markdown_summarizes_relevance_and_artifacts() -> None:
    markdown = status_markdown(_report())

    assert "Relevance: `pass`" in markdown
    assert "Hard-case gate: `pass`" in markdown
    assert "Top analogue cosine: `0.900`" in markdown
    assert "story_smoke_report.md" in markdown


def test_validation_gate_formatters_explain_prefix_latent_qc() -> None:
    gate_report = {
        "gate": {
            "overall_status": "fail",
            "operational_status": "warning",
            "stress_status": "fail",
            "endpoint_max_abs_error": 0.0,
            "warning_counts": {"large_rollout_shift": 2},
            "fail_counts": {"rollout_shift_fail": 1},
            "hard_cases": [
                {
                    "variant": "farthest_train_start",
                    "query_window_index": 177,
                    "start_window_index": 6,
                    "input_memory_cosine": 0.7774,
                    "start_distance_z": 29.54,
                    "terminal_mean_abs_delta_z": 2.0067,
                    "status": "fail",
                    "warnings": ["large_start_distance"],
                    "failures": ["rollout_shift_fail"],
                }
            ],
        }
    }

    markdown = validation_gate_markdown(gate_report)
    table = validation_gate_table(gate_report)

    assert "Operational status: `warning`" in markdown
    assert "Stress status: `fail`" in markdown
    assert table.iloc[0]["Variant"] == "farthest_train_start"
    assert table.iloc[0]["Status"] == "fail"


def test_prefix_latent_live_smoke_formatters_show_current_run_gate() -> None:
    report = _prefix_report()

    markdown = prefix_latent_status_markdown(report)
    product_markdown = prefix_latent_product_status_markdown(report)
    variants = prefix_variant_table(report)
    selected = prefix_selected_start_table(report)
    diagnostic = prefix_diagnostic_start_table(report)
    validation = prefix_validation_table(report)

    assert "Selected start status: `pass`" in markdown
    assert "Research overall: `pass`" in markdown
    assert "Rollout temperature: `0.500`" in markdown
    assert "Scenario CRPS vs persistence: `+14.5%`" in markdown
    assert "Operational interpretation: `supported narrative scenario`" in markdown
    assert "Run record: `n/a`" in markdown
    assert "joint39_val_0370" in markdown
    assert variants.iloc[1]["Start Window"] == "joint39_val_0269"
    assert variants.iloc[1]["Memory Support"] == "0.887"
    assert variants.iloc[1]["Selection"] == "max_memory_inside_start_threshold"
    assert selected.iloc[0]["Starting Level"] == "2001-03-13"  # window 269 day-0 (end date) via full start bridge
    assert "Reliability" not in selected.columns
    assert selected.iloc[0]["Narrative match (cosine)"] == "0.887"
    assert "Product decision:" not in product_markdown
    assert "Starting level:" not in product_markdown
    assert "Scenario ready" in product_markdown
    assert "Review the fan chart and baseline-vs-narrative summary" in product_markdown
    assert "Narrative support:" not in product_markdown
    assert "Support rule:" not in product_markdown
    assert "Warning:" not in product_markdown
    assert "Forward-looking language excluded from conditioning" not in product_markdown
    assert "Start reliability:" not in product_markdown
    assert "Result note:" not in product_markdown
    assert diagnostic.iloc[0]["Start Window"] == "joint39_val_0370"
    assert validation.iloc[0]["Memory Cosine"] == "0.899"
    assert "warn_and_continue_for_narrative_only" in markdown
    assert "SPX (3.249z)" in markdown


def test_prefix_trust_interpretation_separates_warning_from_metric_failure() -> None:
    report = _prefix_report()
    report["validation_gate"]["selected_start_status"] = "warning"

    assert (
        prefix_trust_interpretation(report)
        == "usable with support/shift caveats; scenario CRPS improved vs persistence"
    )


def test_live_support_gated_calibration_updates_operational_scenario(
    tmp_path: Path,
) -> None:
    arrays_path = tmp_path / "prefix_arrays.npz"
    markdown_path = tmp_path / "prefix_report.md"
    markdown_path.write_text("# Prefix report\n", encoding="utf-8")
    states = np.zeros((1, 4, 2, 39), dtype=np.float32)
    requested_raw = np.zeros((1, 39), dtype=np.float32)
    delta_scale = np.ones((2, 39), dtype=np.float32)
    np.savez(
        arrays_path,
        generated_states=states,
        requested_raw=requested_raw,
        delta_scale=delta_scale,
    )
    report = {
        "artifact_paths": {"arrays": str(arrays_path), "markdown": str(markdown_path)},
        "cached_query": {
            "operational_memory_prior_variant_index": 0,
            "grounding": {
                "market_implications": [
                    {"market": "SPX", "direction": "up"},
                    {"market": "VIX", "direction": "down"},
                ]
            },
            "memory_prior": {
                "mode": "diverse_topk_narrative_start_checked",
                "direction_check": {"status": "pass"},
            },
        },
        "selected_start_state": {"variant_index": 0},
        "variant_rows": [
            {"is_operational": True, "start_window_id": "joint39_val_0040"}
        ],
        "generation": {
            "path_quantiles": [
                {
                    "market": "SPX",
                    "analogue_key": "RANK_1",
                    "analogue_label": "Selected start: joint39_val_0040",
                    "days": [1, 2],
                    "p10": [0.0, 0.0],
                    "p50": [0.0, 0.0],
                    "p90": [0.0, 0.0],
                    "mean": [0.0, 0.0],
                    "sample_paths": [
                        {"label": "Generated path 1", "values": [0.0, 0.0]}
                    ],
                }
            ],
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": 0.0,
                    "p10": 0.0,
                    "p90": 0.0,
                }
            ],
        },
    }

    calibrated = apply_live_support_gated_ensemble_calibration(report, beta=0.25)

    metadata = calibrated["generation"]["narrative_ensemble_calibration"]
    assert metadata["applied"] is True
    assert metadata["support_gate"] == 1.0
    assert metadata["active_direction_count"] == 2
    spx_summary = next(
        row
        for row in calibrated["generation"]["terminal_delta_summary"]
        if row["market"] == "SPX"
    )
    vix_summary = next(
        row
        for row in calibrated["generation"]["terminal_delta_summary"]
        if row["market"] == "VIX"
    )
    assert spx_summary["mean_terminal_delta"] == 0.25
    assert vix_summary["mean_terminal_delta"] == -0.25
    spx_path = calibrated["generation"]["path_quantiles"][0]
    assert spx_path["p50"][-1] == 0.25
    assert spx_path["sample_paths"][0]["values"][-1] == 0.25
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Live Demo Narrative Calibration" in markdown
    assert "retained for audit/replay only" in markdown
    assert "nearest-similar top3/90 posterior ensemble" in markdown


def test_live_top3_90_posterior_ensemble_selects_dominant_components(
    tmp_path: Path,
) -> None:
    arrays_path = tmp_path / "prefix_arrays.npz"
    markdown_path = tmp_path / "prefix_report.md"
    markdown_path.write_text("# Prefix report\n", encoding="utf-8")
    states = np.zeros((1, 6, 2, 39), dtype=np.float32)
    requested_raw = np.zeros((1, 39), dtype=np.float32)
    requested_raw[0, 25] = 100.0
    terminal_spx = np.array([101.0, 101.0, 102.0, 102.0, 103.0, 140.0])
    states[0, :, 0, 25] = requested_raw[0, 25]
    states[0, :, 1, 25] = terminal_spx
    np.savez(
        arrays_path,
        generated_states=states,
        requested_raw=requested_raw,
        rollout_component_variant_index=np.array([0, 0, 0, 0], dtype=np.int64),
        rollout_component_window_index=np.array([10, 20, 30, 40], dtype=np.int64),
        rollout_component_weight=np.array([0.50, 0.25, 0.15, 0.10], dtype=np.float64),
        rollout_component_sample_count=np.array([2, 2, 1, 1], dtype=np.int64),
    )
    report = {
        "artifact_paths": {"arrays": str(arrays_path), "markdown": str(markdown_path)},
        "cached_query": {
            "operational_memory_prior_variant_index": 0,
            "memory_prior": {
                "candidate_details": [
                    {
                        "window_index": 10,
                        "bridge_local_index": 10,
                        "window_id": "joint39_train_0010",
                        "memory_support_cosine": 0.90,
                        "history_end_date": "2016-02-10",
                        "manifest_split": "train",
                    },
                    {
                        "window_index": 20,
                        "bridge_local_index": 20,
                        "window_id": "joint39_train_0020",
                        "memory_support_cosine": 0.85,
                        "history_end_date": "2016-03-01",
                        "manifest_split": "train",
                    },
                    {
                        "window_index": 30,
                        "bridge_local_index": 30,
                        "window_id": "joint39_train_0030",
                        "memory_support_cosine": 0.80,
                        "history_end_date": "2016-03-21",
                        "manifest_split": "train",
                    },
                    {
                        "window_index": 40,
                        "bridge_local_index": 40,
                        "window_id": "joint39_train_0040",
                        "memory_support_cosine": 0.75,
                        "history_end_date": "2016-04-10",
                        "manifest_split": "train",
                    },
                ]
            },
        },
        "selected_start_state": {"variant_index": 0},
        "variant_rows": [
            {"is_operational": True, "start_window_id": "joint39_train_0001"}
        ],
        "generation": {"path_quantiles": [], "terminal_delta_summary": []},
    }

    updated = apply_live_top3_90_posterior_ensemble(report)

    posterior = updated["generation"]["posterior_ensemble"]
    assert posterior["applied"] is True
    assert posterior["default_analogue_key"] == "TOP3_90"
    assert posterior["selected_component_count"] == 3
    assert posterior["posterior_sample_count"] == 5
    assert np.isclose(posterior["base_weight_mass"], 0.90)
    assert [row["window_index"] for row in posterior["selected_support"]] == [
        10,
        20,
        30,
    ]
    spx_summary = next(
        row
        for row in updated["generation"]["terminal_delta_summary"]
        if row["market"] == "SPX"
    )
    assert np.isclose(spx_summary["mean_terminal_delta"], 1.8)
    assert np.isclose(spx_summary["terminal_probability_up"], 1.0)
    assert np.isclose(spx_summary["terminal_probability_down"], 0.0)
    assert spx_summary["terminal_sample_count"] == 5
    assert all(
        row["analogue_key"] == "TOP3_90"
        for row in updated["generation"]["path_quantiles"]
    )
    table = prefix_start_candidates_table(updated)
    assert set(table["Used For"]) == {"Narrative scenario"}
    assert list(table["Episode date"]) == [
        "2016-02-10",
        "2016-03-01",
        "2016-03-21",
    ]
    assert "2016-04-10" not in table["Episode date"].tolist()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Live Demo Top3/90 Ensemble" in markdown
    assert "current paper candidate" in markdown


def test_live_support_gated_calibration_blocks_start_only_support(
    tmp_path: Path,
) -> None:
    arrays_path = tmp_path / "prefix_arrays.npz"
    np.savez(
        arrays_path,
        generated_states=np.zeros((1, 2, 2, 39), dtype=np.float32),
        requested_raw=np.zeros((1, 39), dtype=np.float32),
        delta_scale=np.ones((2, 39), dtype=np.float32),
    )
    report = {
        "artifact_paths": {"arrays": str(arrays_path)},
        "cached_query": {
            "operational_memory_prior_variant_index": 0,
            "grounding": {
                "market_implications": [{"market": "SPX", "direction": "up"}]
            },
            "memory_prior": {"mode": "soft_topk_start_only"},
        },
        "selected_start_state": {"variant_index": 0},
        "generation": {"terminal_delta_summary": []},
    }

    calibrated = apply_live_support_gated_ensemble_calibration(report, beta=0.25)

    metadata = calibrated["generation"]["narrative_ensemble_calibration"]
    assert metadata["applied"] is False
    assert metadata["support_gate"] == 0.0
    assert metadata["skip_reason"] == "support_gate_blocked"


def test_prefix_condition_only_tables_show_used_and_excluded_language() -> None:
    report = _prefix_report()

    implications = prefix_condition_implications_table(report)
    warnings = prefix_condition_warnings_table(report)
    visible_warnings = prefix_visible_warning_lines(report)
    components = prefix_warning_component_table(report)
    factors = prefix_shift_factor_table(report)
    candidates = prefix_start_candidates_table(report)

    assert implications.iloc[0]["Market"] == "SPX"
    assert implications.iloc[0]["Horizon"] == "current_state"
    assert warnings.iloc[0]["Code"] == "non_conditioning_forward_language"
    assert "volatility could reverse" in warnings.iloc[0]["Message"]
    assert "Forward-looking language excluded from conditioning" in visible_warnings[0]
    assert "volatility could reverse" in visible_warnings[0]
    assert (
        components[components["Check"] == "rollout_shift"].iloc[0]["Status"]
        == "warning"
    )
    assert factors.iloc[0]["Factor"] == "SPX"
    assert candidates.iloc[0]["Episode date"] == "2020-03-12"
    assert candidates.iloc[0]["Used For"] == "Narrative scenario"
    assert candidates.iloc[0]["Weight"] == "0.420"
    assert candidates.iloc[0]["Narrative match (cosine)"] == "0.887"
    assert candidates.iloc[0]["Narrative directions"] == "🟠 4/5 matched"


def test_prefix_start_candidates_table_includes_start_only_baseline_support() -> None:
    report = _prefix_report()
    report["generation"]["start_only_baseline"] = {
        "support_candidates": [
            {
                "rank": 1,
                "window_id": "joint39_train_3942",
                "history_end_date": "2015-10-19",
                "weight": 0.689,
                "memory_support_cosine": 0.798,
                "start_distance_z": 0.868,
            }
        ]
    }

    table = prefix_start_candidates_table(report)

    assert table.iloc[0]["Used For"] == "Narrative scenario"
    assert table.iloc[-1]["Used For"] == "Start-only baseline"
    assert table.iloc[-1]["Episode date"] == "2015-10-19"
    assert table.iloc[-1]["Distance from start (σ)"] == "0.868"


def test_prefix_user_start_table_shows_supplied_start_diagnostics() -> None:
    table = prefix_user_start_table(_prefix_user_start_report())

    assert table.iloc[0]["Label"] == "today"
    assert table.iloc[0]["Format"] == "values_by_name"
    assert table.iloc[0]["Dimension"] == "39"
    assert table.iloc[0]["Nearest Train"] == "18"
    assert table.iloc[0]["Distance from start (σ)"] == "6.940"
    assert table.iloc[0]["Max Abs Z"] == "1.817"


def test_preview_start_state_json_shows_key_values(tmp_path) -> None:
    path = tmp_path / "start.json"
    path.write_text(
        json.dumps(
            {
                "label": "today",
                "coordinate": "raw_state",
                "values_by_name": {
                    "factor:spx": 5000.0,
                    "factor:vix": 18.5,
                    "iv:07": 0.22,
                },
            }
        ),
        encoding="utf-8",
    )

    status, table = preview_start_state_json(str(path))

    assert "Status: `ok`" in status
    rows = {row["Field"]: row["Value"] for row in table.to_dict("records")}
    assert rows["Label"] == "today"
    assert rows["Format"] == "values_by_name"
    assert rows["SPX"] == "5000.000"
    assert rows["VIX"] == "18.500"
    assert rows["IV ATM 3M"] == "0.220"


def test_preview_start_state_json_reports_errors() -> None:
    status, table = preview_start_state_json("/missing/start.json")

    assert "Status: `error`" in status
    assert table.empty


def test_export_historical_start_json_for_app_writes_template(tmp_path) -> None:
    bank = {
        "history_raw": [
            [[0.1, 1.0, 2.0], [0.2, 3.0, 4.0]],
            [[0.3, 5.0, 6.0], [0.4, 7.0, 8.0]],
        ],
        "spec_names": ["iv:00", "factor:spx", "factor:vix"],
        "metadata": {1: {"window_id": "joint39_val_0001"}},
    }

    status, path, preview = export_historical_start_json_for_app(
        "1",
        output_dir=tmp_path,
        bank_loader=lambda: bank,
    )

    assert "Status: `ok`" in status
    assert path.endswith("user_start_template_0001.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["label"] == "user_template_from_joint39_val_0001"
    assert payload["values_by_name"]["factor:spx"] == 7.0
    assert preview[preview["Field"] == "SPX"].iloc[0]["Value"] == "7.000"


def test_build_start_state_payload_rejects_wrong_length() -> None:
    payload = build_start_state_payload(
        label="today",
        spec_names=["iv:00", "factor:spx"],
        raw_state=[0.2, 5000.0],
    )
    assert payload["values_by_name"]["factor:spx"] == 5000.0

    try:
        build_start_state_payload(label="bad", spec_names=["iv:00"], raw_state=[1, 2])
    except ValueError as error:
        assert "raw_state length" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def test_historical_start_candidate_choices_use_memory_prior_metadata() -> None:
    choices = historical_start_candidate_choices(_prefix_report())

    assert choices == [("joint39_val_0269 | idx 269 | w 0.420 | start 6.940z", "269")]
    assert historical_start_candidate_to_index("269") == 269
    assert historical_start_candidate_to_index("269.0") == 269
    assert historical_start_candidate_to_index("") is None


def test_cached_prefix_casebook_controls_fill_story_start_and_report() -> None:
    choices = cached_prefix_casebook_choices()

    assert choices[0] == ("Typed story / current controls", "")
    dollar_value = [
        value for label, value in choices if "Dollar liquidity squeeze" in label
    ][0]
    (
        story,
        use_explicit_start,
        start_index,
        condition_only_story,
        live_story,
        condition_report,
        status,
    ) = cached_prefix_casebook_update(dollar_value)

    assert "dollar liquidity squeeze" in story.lower()
    assert use_explicit_start is True
    assert start_index in {0, 22, 77}
    assert condition_only_story is False
    assert live_story is False
    assert "condition_only_report_823a_dollar" in condition_report
    assert "OpenAI calls: `none" in status


def test_analogue_scope_choices_falls_back_to_path_quantile_scopes() -> None:
    choices = analogue_scope_choices(_prefix_report())

    assert choices == [
        ("Operational selected start", "RANK_2"),
    ]


def test_build_prefix_latent_run_args_sets_cached_smoke_controls() -> None:
    args = build_prefix_latent_run_args(
        start_mode="nearest_train_start",
        samples=12,
        output_dir="tmp/prefix",
    )

    assert args.start_mode == "nearest_train_start"
    assert args.samples == 12
    assert args.output_dir == "tmp/prefix"
    assert args.device == "cuda"
    assert args.memory_prior_mode == "cohesive_topk_narrative_start_checked"
    assert args.memory_prior_top_k == 8
    assert args.memory_prior_diverse_max_pairwise_cosine == 0.95
    assert args.memory_prior_diverse_min_index_gap == 30
    assert args.support_bank_report == (
        DEFAULT_PREFIX_SUPPORT_BANK_REPORT
        if Path(DEFAULT_PREFIX_SUPPORT_BANK_REPORT).exists()
        else None
    )
    assert args.support_bank_arrays == (
        DEFAULT_PREFIX_SUPPORT_BANK_ARRAYS
        if Path(DEFAULT_PREFIX_SUPPORT_BANK_ARRAYS).exists()
        else None
    )
    assert args.rollout_mixture_mode == "component_prefix_mixture"
    assert args.temperature == 0.5
    assert args.rollout_fan_scale == 3.5

    default_args = build_prefix_latent_run_args(
        start_mode="nearest_train_start",
        samples=12,
    )
    assert "risk_manager_story_gradio_demo" in default_args.output_dir

    live_args = build_prefix_latent_run_args(
        start_mode="nearest_train_start",
        samples=12,
        live_story=True,
        story="A live risk-manager story.",
    )
    assert live_args.live_story is True
    assert live_args.story == "A live risk-manager story."

    condition_args = build_prefix_latent_run_args(
        start_mode="balanced_memory_start",
        samples=4,
        condition_report="tmp/condition_only_report.json",
        explicit_start_window_index=22,
        start_state_json="tmp/today_start.json",
        skip_rollout=True,
    )
    assert condition_args.condition_report == "tmp/condition_only_report.json"
    assert condition_args.explicit_start_window_index == 22
    assert condition_args.start_state_json == "tmp/today_start.json"
    assert condition_args.skip_rollout is True


def test_build_run_args_sets_generator_controls() -> None:
    args = build_run_args(
        story="A risk-on recovery.",
        samples=3,
        top_k=4,
        skip_generator=True,
        output_dir="tmp/out",
    )

    assert args.story == "A risk-on recovery."
    assert args.samples == 3
    assert args.top_k == 4
    assert args.skip_generator is True
    assert args.output_dir == "tmp/out"
    assert args.chunk_size >= 8


def test_fan_chart_figure_uses_path_quantiles() -> None:
    figure = fan_chart_figure(_report(), "SPX")

    assert figure.layout.title.text == "SPX 30-day scenario fan (raw level)"
    assert len(figure.data) == 4
    assert list(figure.data[1].y) == [0.1, 0.2]


def test_fan_chart_figure_can_filter_to_one_analogue() -> None:
    figure = fan_chart_figure(_report(), "SPX", "RANK_1")

    assert figure.layout.title.text == "SPX 30-day scenario fan (raw level)"
    assert "Analogue 1: joint39_val_0031" in figure.layout.annotations[0].text
    assert list(figure.data[1].y) == [1.1, 2.2]
    assert [trace.name for trace in figure.data][-3:] == [
        "Generated sample paths",
        "Generated path 2",
        "Actual outcome (hindsight)",
    ]
    assert list(figure.data[-1].y) == [0.8, 2.4]


def test_fan_chart_figure_supports_selected_iv_cells() -> None:
    figure = fan_chart_figure(_report(), "IV_ATM_3M")

    assert (
        figure.layout.title.text == "IV ATM 3M (K=1.00) 30-day scenario fan (raw level)"
    )
    assert "3M / K=1.00" in figure.layout.annotations[0].text
    assert list(figure.data[1].y) == [0.01, 0.02]


def test_analogue_scope_choices_and_refresh_use_saved_report() -> None:
    choices = analogue_scope_choices(_report())
    figure = refresh_fan_chart(_report(), "SPX", "RANK_1")

    assert choices == [
        ("All retrieved analogues", "ALL"),
        ("Analogue 1: joint39_val_0031", "RANK_1"),
    ]
    assert list(figure.data[1].y) == [1.1, 2.2]


def test_run_story_for_app_can_use_injected_runner() -> None:
    def fake_runner(args: SimpleNamespace) -> dict:
        assert args.story == "A risk-on recovery."
        return _report()

    outputs = list(
        run_story_for_app(
            "A risk-on recovery.",
            samples=2,
            top_k=3,
            fan_market="SPX",
            analogue_scope="ALL",
            skip_generator=False,
            runner=fake_runner,
        )
    )[-1]

    assert "Risk Manager Story Smoke Test" in outputs[0]
    assert outputs[1].iloc[0]["Market"] == "SPX"
    assert outputs[3].iloc[0]["Window"] == "joint39_val_0031"
    assert outputs[5].iloc[0]["Market"] == "SPX"
    assert outputs[6].layout.title.text == "SPX 30-day scenario fan (raw level)"
    assert outputs[8] == _report()


def test_run_story_for_app_streams_visible_progress_before_runner_finishes() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append(args.story)
        return _report()

    stream = run_story_for_app(
        "A risk-on recovery.",
        samples=2,
        top_k=3,
        fan_market="SPX",
        analogue_scope="ALL",
        skip_generator=False,
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "Run started" in first[4]
    assert "OpenAI grounding" in first[4]
    assert calls == ["A risk-on recovery."]
    assert "Completed in" in final[4]


def test_run_prefix_latent_for_app_streams_progress_and_outputs_validation() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append((args.start_mode, args.samples))
        return _prefix_report()

    stream = run_prefix_latent_for_app(
        start_mode="nearest_train_start",
        samples=8,
        fan_market="SPX",
        analogue_scope="ALL",
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "Run started" in first[1]
    assert "30-day scenario generation" in first[1]
    assert calls == [("nearest_train_start", 8)]
    assert "Scenario ready" in final[1]
    assert "Completed in" not in final[1]
    assert final[2].iloc[0]["Starting Level"] == "2001-03-13"  # window 269 day-0 (end date) via full start bridge
    assert final[3].iloc[0]["Variant"] == "original"
    assert final[4].iloc[0]["Status"] == "pass"
    assert final[6].layout.title.text == "SPX 30-day scenario fan (raw level)"
    assert final[14].iloc[0]["Episode date"] == "2020-03-12"
    assert final[16]["choices"] == [
        ("joint39_val_0269 | idx 269 | w 0.420 | start 6.940z", "269")
    ]


def test_run_prefix_latent_for_app_attaches_start_only_baseline_summary() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append(args.memory_prior_mode)
        report = _prefix_report()
        if args.memory_prior_mode == "soft_topk_start_only":
            report["generation"]["terminal_delta_summary"] = [
                {
                    "market": "SPX",
                    "mean_terminal_delta": 10.0,
                    "p10": -5.0,
                    "p90": 25.0,
                }
            ]
        return report

    stream = run_prefix_latent_for_app(
        start_mode="explicit_start_window",
        samples=4,
        fan_market="SPX",
        analogue_scope="ALL",
        live_story=False,
        use_explicit_start=True,
        explicit_start_window_index=22,
        include_start_only_baseline=True,
        runner=fake_runner,
    )

    next(stream)
    final = list(stream)[-1]

    assert calls == [
        "cohesive_topk_narrative_start_checked",
        "soft_topk_start_only",
    ]
    baseline = final[8]["generation"]["start_only_baseline"]
    assert baseline["memory_prior_mode"] == "soft_topk_start_only"
    assert baseline["terminal_delta_summary"][0]["mean_terminal_delta"] == 10.0
    assert "Baseline View" in final[5]
    assert "Baseline Path Share" in final[5]
    assert "Baseline Mean Move" in final[5]
    assert "Narrative View" in final[5]
    assert "Narrative Path Share" in final[5]
    assert "Narrative Mean Move" in final[5]
    assert 'class="demo-dir demo-dir-down"' in final[5]
    assert '<span class="demo-dir-arrow">↓</span> Down' in final[5]
    assert "More down than baseline" in final[5]


def test_run_prefix_latent_for_app_does_not_require_start_approval_gate() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append(args)
        return _prefix_report()

    stream = run_prefix_latent_for_app(
        start_mode="balanced_memory_start",
        samples=8,
        fan_market="SPX",
        analogue_scope="ALL",
        approve_start=False,
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "Starting level approval required" not in first[1]
    assert calls
    assert "Scenario ready" in final[1]
    assert "Completed in" not in final[1]


def test_run_prefix_latent_for_app_can_preview_start_without_rollout() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append((args.skip_rollout, args.samples))
        return _prefix_report()

    stream = run_prefix_latent_for_app(
        start_mode="balanced_memory_start",
        samples=8,
        fan_market="SPX",
        analogue_scope="ALL",
        skip_rollout=True,
        approve_start=False,
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "scenario rollout skipped" in first[1]
    assert calls == [(True, 8)]
    assert "Scenario ready" in final[1]
    assert "Completed in" not in final[1]
    assert final[2].iloc[0]["Starting Level"] == "2001-03-13"  # window 269 day-0 (end date) via full start bridge


def test_live_openai_wrappers_force_production_story_path() -> None:
    calls = []

    def fake_grounder(story: str, **kwargs):
        return (
            ConditionOnlyGroundingResult.model_validate(
                {
                    "prompt_version": "condition_only_grounding_v1",
                    "narrative_frame": "risk-on recovery",
                    "current_market_state_summary": "Equities are recovering.",
                    "recent_regime_summary": (
                        "No separate recent-regime description stated beyond current conditions."
                    ),
                    "cleaned_conditioning_text": "Equities are recovering.",
                    "current_market_state_implications": [
                        {
                            "market": "SPX",
                            "direction": "up",
                            "magnitude": "small",
                            "confidence": "high",
                            "horizon": "current_state",
                            "target_use": "support_prior",
                            "evidence": ["Equities are recovering"],
                            "inferred": False,
                            "rationale": "The story states equities are recovering.",
                        }
                    ],
                    "recent_regime_implications": [],
                    "non_conditioning_forward_language": [],
                    "unsupported_claims": [],
                    "grounding_warnings": [],
                    "critique": [],
                }
            ),
            {"model": "fixture"},
        )

    def fake_condition_report_runner(args: SimpleNamespace) -> dict:
        return {
            "artifact_paths": {
                "report": "tmp/live_condition_report.json",
                "arrays": "tmp/live_condition_report_arrays.npz",
            },
            "cached_query": {"condition_source": "condition_only_openai_story"},
        }

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append(args)
        return _prefix_report()

    preview = preview_live_openai_start_for_app(
        samples=8,
        fan_market="SPX",
        analogue_scope="ALL",
        story="A current risk-on rebound.",
        explicit_start_window_index=22,
        runner=fake_runner,
        condition_grounder=fake_grounder,
        condition_report_runner=fake_condition_report_runner,
    )
    first = next(preview)
    final = list(preview)[-1]

    assert "user-selected historical start" in first[1]
    assert "user-selected historical start validation" not in first[1]
    assert "scenario rollout skipped" in first[1]
    assert calls[-1].live_story is False
    assert calls[-1].condition_report is not None
    assert calls[-1].start_mode == "explicit_start_window"
    assert calls[-1].explicit_start_window_index == 22
    assert calls[-1].skip_rollout is True
    assert final[2].iloc[0]["Starting Level"] == "2001-03-13"  # window 269 day-0 (end date) via full start bridge

    generated = run_live_openai_prefix_for_app(
        samples=8,
        fan_market="SPX",
        analogue_scope="ALL",
        story="A current risk-on rebound.",
        explicit_start_window_index=77,
        runner=fake_runner,
        condition_grounder=fake_grounder,
        condition_report_runner=fake_condition_report_runner,
    )
    next(generated)
    list(generated)

    assert calls[-1].live_story is False
    assert calls[-1].condition_report is not None
    assert calls[-1].explicit_start_window_index == 77
    assert calls[-1].skip_rollout is False


def test_live_openai_wrappers_require_manual_start_index() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append(args)
        return _prefix_report()

    stream = run_live_openai_prefix_for_app(
        samples=8,
        fan_market="SPX",
        analogue_scope="ALL",
        story="A current risk-on rebound.",
        explicit_start_window_index=None,
        runner=fake_runner,
    )
    first = next(stream)

    assert calls == []
    assert "Historical start required" in first[1]


def test_run_prefix_latent_for_app_can_pass_live_story_testflight() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append((args.live_story, args.story))
        report = _prefix_report()
        report["cached_query"]["condition_source"] = "live_openai_story"
        return report

    stream = run_prefix_latent_for_app(
        start_mode="nearest_train_start",
        samples=4,
        fan_market="SPX",
        analogue_scope="ALL",
        live_story=True,
        story="A live risk-manager story.",
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "OpenAI story check and embedding" in first[1]
    assert calls == [(True, "A live risk-manager story.")]
    assert "Scenario ready" in final[1]
    assert "Review the fan chart and baseline-vs-narrative summary" in final[1]
    assert "Narrative support:" not in final[1]
    assert "Support rule:" not in final[1]
    assert "Warning:" not in final[1]
    assert "Completed in" not in final[1]


def test_run_prefix_latent_for_app_can_use_explicit_historical_start() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append((args.start_mode, args.explicit_start_window_index))
        return _prefix_report()

    stream = run_prefix_latent_for_app(
        start_mode="balanced_memory_start",
        samples=4,
        fan_market="SPX",
        analogue_scope="ALL",
        use_explicit_start=True,
        explicit_start_window_index=22,
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "user-selected historical start" in first[1]
    assert "user-selected historical start validation" not in first[1]
    assert calls == [("explicit_start_window", 22)]
    assert "Scenario ready" in final[1]
    assert "Completed in" not in final[1]


def test_run_prefix_latent_for_app_can_use_user_start_json() -> None:
    calls = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append((args.start_mode, args.start_state_json))
        return _prefix_user_start_report()

    stream = run_prefix_latent_for_app(
        start_mode="balanced_memory_start",
        samples=4,
        fan_market="SPX",
        analogue_scope="ALL",
        use_user_start_state=True,
        start_state_json="tmp/today_start.json",
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "user-supplied start state" in first[1]
    assert "user-supplied start-state validation" not in first[1]
    assert calls == [("user_start_state", "tmp/today_start.json")]
    assert final[15].iloc[0]["Label"] == "today"


def test_run_prefix_latent_for_app_can_use_cached_condition_report(tmp_path) -> None:
    calls = []
    report_path = tmp_path / "condition_report.json"
    report_path.write_text("{}", encoding="utf-8")

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append(
            (
                args.condition_report,
                args.live_story,
                args.start_mode,
                args.explicit_start_window_index,
            )
        )
        return _prefix_report()

    stream = run_prefix_latent_for_app(
        start_mode="balanced_memory_start",
        samples=4,
        fan_market="SPX",
        analogue_scope="ALL",
        live_story=True,
        story="Cached condition story.",
        cached_condition_report=str(report_path),
        condition_only_story=True,
        use_explicit_start=True,
        explicit_start_window_index=77,
        runner=fake_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "cached story report" in first[1]
    assert calls == [(str(report_path), False, "explicit_start_window", 77)], final[7]
    assert "Scenario ready" in final[1]
    assert "Completed in" not in final[1]


def test_run_prefix_latent_for_app_can_use_condition_only_contract(tmp_path) -> None:
    calls = []

    def fake_grounder(story: str, **kwargs):
        return (
            ConditionOnlyGroundingResult.model_validate(
                {
                    "prompt_version": "condition_only_grounding_v1",
                    "narrative_frame": "risk-on recovery",
                    "current_market_state_summary": "Equities are recovering.",
                    "recent_regime_summary": (
                        "No separate recent-regime description stated beyond current conditions."
                    ),
                    "cleaned_conditioning_text": "Equities are recovering.",
                    "current_market_state_implications": [
                        {
                            "market": "SPX",
                            "direction": "up",
                            "magnitude": "small",
                            "confidence": "high",
                            "horizon": "current_state",
                            "target_use": "support_prior",
                            "evidence": ["Equities are recovering"],
                            "inferred": False,
                            "rationale": "The story states equities are recovering.",
                        }
                    ],
                    "recent_regime_implications": [],
                    "non_conditioning_forward_language": [
                        {
                            "phrase": "volatility could reverse",
                            "reason": "future-looking phrase",
                            "handling": "warning_only",
                            "severity": "warning",
                        }
                    ],
                    "unsupported_claims": [],
                    "grounding_warnings": [],
                    "critique": [],
                }
            ),
            {"model": "fixture"},
        )

    def fake_condition_report_runner(args: SimpleNamespace) -> dict:
        assert args.case_json.endswith("condition_only_grounding_case.json")
        assert Path(args.case_json).exists()
        return {
            "artifact_paths": {
                "report": str(tmp_path / "condition_only_report.json"),
                "arrays": str(tmp_path / "condition_only_report_arrays.npz"),
            },
            "cached_query": {"condition_source": "condition_only_openai_story"},
        }

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append((args.condition_report, args.live_story))
        return _prefix_report()

    stream = run_prefix_latent_for_app(
        start_mode="balanced_memory_start",
        samples=4,
        fan_market="SPX",
        analogue_scope="ALL",
        live_story=True,
        story="Equities are recovering. Volatility could reverse.",
        condition_only_story=True,
        runner=fake_runner,
        condition_grounder=fake_grounder,
        condition_report_runner=fake_condition_report_runner,
    )

    first = next(stream)
    final = list(stream)[-1]

    assert "OpenAI story check" in first[1]
    assert calls == [(str(tmp_path / "condition_only_report.json"), False)], final[7]
    assert "OpenAI was called" in final[8]["scope_note"]
    assert (
        final[8]["live_app_openai_conditioning"]["status"] == "fresh_condition_report"
    )
    assert final[8]["live_app_openai_conditioning"]["grounding_model"] == "fixture"
    assert final[10].iloc[0]["Market"] == "SPX"


def test_allocate_weighted_sample_counts_preserves_total_and_weights() -> None:
    counts = _allocate_weighted_sample_counts([0.5, 0.3, 0.2], 11)

    assert counts.tolist() == [6, 3, 2]
    assert int(counts.sum()) == 11


def test_prefix_selected_start_table_date_maps_in_range_index() -> None:
    import re

    from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
        _start_index_bounds,
        _start_window_calendar_label,
        prefix_selected_start_table,
    )

    lo, hi = _start_index_bounds()
    in_range = 22  # a real loaded window in the demo start bank
    assert lo <= in_range <= hi

    # The operational start window index resolves to a calendar DATE, not the
    # raw window id (locks the #7 "Starting Level" fix against regression).
    resolved = _start_window_calendar_label(in_range, "joint39_val_0040")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", resolved), resolved
    assert resolved != "joint39_val_0040"

    report = {
        "variant_rows": [
            {
                "is_operational": True,
                "start_window_id": "joint39_val_0040",
                "start_window_index": in_range,
                "memory_support_cosine": 0.79,
                "start_distance_z": 14.4,
                "start_manifest_split": "train",
            }
        ]
    }
    table = prefix_selected_start_table(report)
    assert table.iloc[0]["Starting Level"] == resolved
    # Primary view is now 2-col (date + narrative match); raw Index/Source/distance
    # were dropped to the audit JSON.
    assert list(table.columns) == ["Starting Level", "Narrative match (cosine)"]


# --- Track-D display-surface unit tests ------------------------------------


def test_terminal_level_text_formats_per_factor_units() -> None:
    from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
        _terminal_level_text,
    )

    assert _terminal_level_text("SPX", 1852.0594) == "1,852.1"  # index, thousands-sep 1dp
    assert _terminal_level_text("NIKKEI", 27423.96) == "27,424.0"
    assert _terminal_level_text("US2Y", 4.78) == "4.78%"  # rate level is percent-magnitude
    assert _terminal_level_text("BBB_OAS", 1.59) == "1.59%"
    assert _terminal_level_text("USDJPY", 136.18) == "136.18"  # FX, 2dp
    assert _terminal_level_text("VIX", 20.95) == "20.95"
    assert _terminal_level_text("CRUDE_OIL", 75.57) == "75.57"
    assert _terminal_level_text("GOLD", 1817.0) == "1,817.00"
    assert _terminal_level_text("IV_ATM_3M", 0.1576) == "15.76%"  # IV x100 + %
    assert _terminal_level_text("IV_SURFACE", 0.183) == "18.30%"
    assert _terminal_level_text("SPX", None) == "n/a"
    assert _terminal_level_text("SPX", float("nan")) == "n/a"


def test_support_ess_html_reports_effective_analogues() -> None:
    from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
        support_ess_html,
    )

    report = {
        "cached_query": {
            "memory_prior": {
                "candidate_details": [
                    {"weight": 0.5},
                    {"weight": 0.3},
                    {"weight": 0.2},
                ]
            }
        }
    }
    out = support_ess_html(report)
    # Kish ESS for [.5,.3,.2] = 1/0.38 ~= 2.6, leads with plain "Effective analogues:".
    assert "Effective analogues: 2.6" in out
    assert "ess-strip" in out
    assert "ESS " not in out  # the bare-acronym lead was removed

    na = support_ess_html({})
    assert "Effective analogues:" in na
    assert "n/a" in na


def test_support_hull_html_green_and_red_branches(monkeypatch) -> None:
    import experiments.backfill.block_ar.nl_hull_gate_inputs as hull_mod
    from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
        support_hull_html,
    )

    grounding = {
        "grounding": {"market_implications": [{"market": "SPX", "direction": "down"}]}
    }

    # Green: implied move stays inside the historical analogue hull.
    monkeypatch.setattr(
        hull_mod,
        "hull_label_from_grounding",
        lambda g, **k: {
            "ladder": [{"status": "ok", "pool_mahalanobis": 1.2}],
            "any_indeterminate": False,
            "any_infeasible": False,
        },
    )
    green = support_hull_html(grounding)
    assert "within precedent" in green
    assert "pool Mahalanobis" not in green  # jargon removed (#9)

    # Red: implied move exits the hull -> loud "no precedent" flag.
    monkeypatch.setattr(
        hull_mod,
        "hull_label_from_grounding",
        lambda g, **k: {
            "ladder": [{"status": "ok"}],
            "any_indeterminate": False,
            "any_infeasible": True,
            "leaves_hull_at_kappa": 2.0,
        },
    )
    red = support_hull_html(grounding)
    assert "No close historical precedent" in red
    assert "Details in Audit" not in red  # dangling pointer removed (#9)

    # Degrades to a neutral n/a strip when there is no grounding.
    assert "n/a" in support_hull_html({})


def test_full_start_date_picker_spans_train_region() -> None:
    from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
        _default_full_start_index,
        _start_index_bounds,
        full_start_date_choices,
    )

    choices = full_start_date_choices()
    # One entry per train-region window (2000-2015), positional indices.
    assert len(choices) == 4010
    values = [value for _label, value in choices]
    assert values == list(range(4010))
    labels = [label for label, _value in choices]
    # Labels show the DAY-0 date = calendar_end_date (last observed history day),
    # so window 0 reads 2000-02-14 and the final window reads 2016-01-26.
    assert labels[0].startswith("2000-02-14")
    assert labels[-1].startswith("2016-01-26")
    assert any("2008-" in label for label in labels)  # GFC day-0 dates are selectable

    lo, hi = _start_index_bounds()
    assert (lo, hi) == (0, 4009)

    # Default lands on the late-Oct-2008 GFC crash window (day-0 = 2008-10-24).
    label_by_index = {value: label for label, value in choices}
    assert label_by_index[_default_full_start_index()].startswith("2008-10-24")


def test_start_index_date_hint_in_and_out_of_range() -> None:
    import re

    from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
        _start_index_bounds,
        start_index_date_hint,
    )

    lo, hi = _start_index_bounds()
    assert lo <= 22 <= hi

    in_range = start_index_date_hint(22)
    assert "as of" in in_range
    assert re.search(r"\d{4}-\d{2}-\d{2}", in_range)

    out_of_range = start_index_date_hint(hi + 1000)
    assert "out of range" in out_of_range

    none_hint = start_index_date_hint(None)
    assert "enter a historical window" in none_hint


def test_friendly_error_message_maps_known_openai_errors() -> None:
    from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
        _friendly_error_message,
    )

    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APIError(Exception):
        pass

    class SomethingElse(Exception):
        pass

    assert "credentials" in _friendly_error_message(AuthenticationError())
    assert "rate-limited" in _friendly_error_message(RateLimitError())
    assert "did not respond in time" in _friendly_error_message(APITimeoutError())
    assert "Could not reach" in _friendly_error_message(APIConnectionError())
    assert "returned an error" in _friendly_error_message(APIError())
    # Unknown errors fall back to the generic technical status (empty mapping).
    assert _friendly_error_message(SomethingElse()) == ""


def test_live_openai_guards_block_paid_call_on_bad_input() -> None:
    calls: list = []

    def fake_runner(args: SimpleNamespace) -> dict:
        calls.append(args)
        return _prefix_report()

    # Empty narrative: short-circuit, never substitute a default, never call out.
    blank = list(
        run_live_openai_prefix_for_app(
            8,
            "SPX",
            "ALL",
            story="   ",
            explicit_start_window_index=22,
            runner=fake_runner,
        )
    )
    assert calls == []
    assert "Enter a market narrative" in blank[0][0]

    # Out-of-range day-0 index: short-circuit before the paid grounding call.
    out_of_range = list(
        run_live_openai_prefix_for_app(
            8,
            "SPX",
            "ALL",
            story="Risk-off stress.",
            explicit_start_window_index=10_000,
            runner=fake_runner,
        )
    )
    assert calls == []
    assert "out of range" in out_of_range[0][1]
