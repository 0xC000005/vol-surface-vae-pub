import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (
    DEFAULT_STORY,
    analogues_table,
    analogue_scope_choices,
    build_prefix_latent_run_args,
    build_run_args,
    fan_chart_figure,
    implications_table,
    prefix_latent_status_markdown,
    prefix_validation_table,
    prefix_variant_table,
    refresh_fan_chart,
    run_prefix_latent_for_app,
    run_story_for_app,
    scenario_table,
    status_markdown,
    validation_gate_markdown,
    validation_gate_table,
    warnings_table,
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
        },
        "variant_rows": [
            {
                "variant": "original",
                "query_window_id": "joint39_val_0370",
                "start_window_id": "joint39_val_0370",
                "start_distance_z": 0.0,
                "memory_support_cosine": 0.812,
                "start_selection_method": "",
            },
            {
                "variant": "nearest_train_start",
                "query_window_id": "joint39_val_0370",
                "start_window_id": "joint39_val_0269",
                "start_distance_z": 6.94,
                "memory_support_cosine": 0.887,
                "start_selection_method": "max_memory_inside_start_threshold",
            },
        ],
        "validation_gate": {
            "overall_status": "pass",
            "operational_status": "pass",
            "stress_status": "pass",
            "endpoint_max_abs_error": 0.0,
            "warning_counts": {},
            "fail_counts": {},
            "cases": [
                {
                    "variant": "original",
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
        "generation": {
            "generated_state_shape": [2, 16, 30, 39],
            "finite_rate": 1.0,
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
                    "analogue_label": "Analogue 1: joint39_val_0370",
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
    validation = prefix_validation_table(report)

    assert "Overall: `pass`" in markdown
    assert "joint39_val_0370" in markdown
    assert variants.iloc[1]["Start Window"] == "joint39_val_0269"
    assert variants.iloc[1]["Memory Support"] == "0.887"
    assert variants.iloc[1]["Selection"] == "max_memory_inside_start_threshold"
    assert validation.iloc[0]["Memory Cosine"] == "0.899"


def test_analogue_scope_choices_falls_back_to_path_quantile_scopes() -> None:
    choices = analogue_scope_choices(_prefix_report())

    assert choices == [
        ("All retrieved analogues", "ALL"),
        ("Analogue 1: joint39_val_0370", "RANK_1"),
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
    assert calls == [("nearest_train_start", 8)]
    assert "Completed in" in final[1]
    assert final[2].iloc[0]["Variant"] == "original"
    assert final[3].iloc[0]["Status"] == "pass"
    assert final[5].layout.title.text == "SPX 30-day scenario fan"


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
