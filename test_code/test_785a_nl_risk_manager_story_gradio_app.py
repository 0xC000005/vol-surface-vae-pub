import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
    DEFAULT_STORY,
    analogues_table,
    analogue_scope_choices,
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
    prefix_trust_interpretation,
    preview_start_state_json,
    prefix_selected_start_table,
    prefix_shift_factor_table,
    prefix_start_candidates_table,
    prefix_user_start_table,
    prefix_validation_table,
    prefix_variant_table,
    prefix_warning_component_table,
    refresh_fan_chart,
    run_prefix_latent_for_app,
    run_story_for_app,
    scenario_table,
    status_markdown,
    validation_gate_markdown,
    validation_gate_table,
    warnings_table,
)
from experiments.backfill.block_ar.nl_prefix_latent_temporal_grounding_testflight import (
    ConditionOnlyGroundingResult,
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
                ]
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
        "generation": {
            "generated_state_shape": [2, 16, 30, 39],
            "finite_rate": 1.0,
            "rollout_temperature": 0.5,
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


def test_table_formatters_expose_demo_evidence() -> None:
    report = _report()

    assert implications_table(report).iloc[0]["Market"] == "SPX"
    assert warnings_table(report).iloc[0]["Code"] == "interpretive_phrase"
    assert analogues_table(report).iloc[0]["Implication Match"] == "0.670"
    assert scenario_table(report).iloc[0]["Market"] == "SPX"
    assert "Narrative" in analogues_table(report).columns


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
    variants = prefix_variant_table(report)
    selected = prefix_selected_start_table(report)
    diagnostic = prefix_diagnostic_start_table(report)
    validation = prefix_validation_table(report)

    assert "Selected-start: `pass`" in markdown
    assert "Research overall: `pass`" in markdown
    assert "Rollout temperature: `0.500`" in markdown
    assert "Scenario CRPS vs persistence: `+14.5%`" in markdown
    assert "Operational interpretation: `supported calibrated scenario`" in markdown
    assert "joint39_val_0370" in markdown
    assert variants.iloc[1]["Start Window"] == "joint39_val_0269"
    assert variants.iloc[1]["Memory Support"] == "0.887"
    assert variants.iloc[1]["Selection"] == "max_memory_inside_start_threshold"
    assert selected.iloc[0]["Start Window"] == "joint39_val_0269"
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


def test_prefix_condition_only_tables_show_used_and_excluded_language() -> None:
    report = _prefix_report()

    implications = prefix_condition_implications_table(report)
    warnings = prefix_condition_warnings_table(report)
    components = prefix_warning_component_table(report)
    factors = prefix_shift_factor_table(report)
    candidates = prefix_start_candidates_table(report)

    assert implications.iloc[0]["Market"] == "SPX"
    assert implications.iloc[0]["Horizon"] == "current_state"
    assert warnings.iloc[0]["Code"] == "non_conditioning_forward_language"
    assert "volatility could reverse" in warnings.iloc[0]["Message"]
    assert (
        components[components["Component"] == "rollout_shift"].iloc[0]["Status"]
        == "warning"
    )
    assert factors.iloc[0]["Factor"] == "SPX"
    assert candidates.iloc[0]["Window"] == "joint39_val_0269"
    assert candidates.iloc[0]["Weight"] == "0.420"
    assert candidates.iloc[0]["Alignment"] == "0.800 (1/5 mismatches)"


def test_prefix_user_start_table_shows_supplied_start_diagnostics() -> None:
    table = prefix_user_start_table(_prefix_user_start_report())

    assert table.iloc[0]["Label"] == "today"
    assert table.iloc[0]["Format"] == "values_by_name"
    assert table.iloc[0]["Dimension"] == "39"
    assert table.iloc[0]["Nearest Train"] == "18"
    assert table.iloc[0]["Start Distance"] == "6.940"
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
        ("All retrieved analogues", "ALL"),
        ("Diagnostic baseline: joint39_val_0370", "RANK_1"),
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
    assert args.memory_prior_mode == "soft_topk_combined"
    assert args.memory_prior_top_k == 8
    assert args.temperature == 0.5

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
    )
    assert condition_args.condition_report == "tmp/condition_only_report.json"
    assert condition_args.explicit_start_window_index == 22
    assert condition_args.start_state_json == "tmp/today_start.json"


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

    assert figure.layout.title.text == "SPX 30-day scenario fan"
    assert len(figure.data) == 4
    assert list(figure.data[1].y) == [0.1, 0.2]


def test_fan_chart_figure_can_filter_to_one_analogue() -> None:
    figure = fan_chart_figure(_report(), "SPX", "RANK_1")

    assert figure.layout.title.text == "SPX 30-day scenario fan"
    assert "Analogue 1: joint39_val_0031" in figure.layout.annotations[0].text
    assert list(figure.data[1].y) == [1.1, 2.2]
    assert [trace.name for trace in figure.data][-3:] == [
        "Generated path 1",
        "Generated path 2",
        "Realized future",
    ]
    assert list(figure.data[-1].y) == [0.8, 2.4]


def test_fan_chart_figure_supports_selected_iv_cells() -> None:
    figure = fan_chart_figure(_report(), "IV_ATM_3M")

    assert figure.layout.title.text == "IV ATM 3M (K=1.00) 30-day scenario fan"
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
    assert outputs[6].layout.title.text == "SPX 30-day scenario fan"
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

    assert "Prefix-latent run started" in first[1]
    assert "Calibrated rollout temperature: `0.50`" in first[1]
    assert calls == [("nearest_train_start", 8)]
    assert "Completed in" in final[1]
    assert final[2].iloc[0]["Variant"] == "nearest_train_start"
    assert final[3].iloc[0]["Variant"] == "original"
    assert final[4].iloc[0]["Status"] == "pass"
    assert final[6].layout.title.text == "SPX 30-day scenario fan"
    assert final[14].iloc[0]["Window"] == "joint39_val_0269"
    assert final[16]["choices"] == [
        ("joint39_val_0269 | idx 269 | w 0.420 | start 6.940z", "269")
    ]


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

    assert "OpenAI grounding and embedding" in first[1]
    assert calls == [(True, "A live risk-manager story.")]
    assert "live_openai_story" in final[1]


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

    assert "explicit_start_window" in first[1]
    assert calls == [("explicit_start_window", 22)]
    assert "Completed in" in final[1]


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

    assert "user_start_state" in first[1]
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

    assert "cached condition-only report" in first[1]
    assert calls == [(str(report_path), False, "explicit_start_window", 77)], final[7]
    assert "Completed in" in final[1]


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

    assert "condition-only OpenAI grounding" in first[1]
    assert calls == [(str(tmp_path / "condition_only_report.json"), False)], final[7]
    assert final[10].iloc[0]["Market"] == "SPX"
