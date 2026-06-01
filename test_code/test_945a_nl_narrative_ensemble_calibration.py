from __future__ import annotations

import numpy as np
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_narrative_ensemble_calibration import (
    _analyze_grid,
    _per_start_promotion_gates,
    _per_start_narrative_metrics,
    _parse_start_root,
    _render_qualitative_markdown,
    _resolve_start_roots,
    _select_beta_candidate,
    _summarize_qualitative_response,
    _support_evidence_gate_from_report,
    apply_directional_delta_calibration,
    direction_vector_from_grounding,
    fit_directional_beta,
    path_with_start,
    plot_qualitative_response_panels,
)


def test_direction_vector_from_grounding_maps_directional_claims() -> None:
    grounding = {
        "market_implications": [
            {"market": "SPX", "direction": "up"},
            {"market": "VIX", "direction": "down"},
            {"market": "BBB_OAS", "direction": "tighter"},
            {"market": "Gold", "direction": "elevated"},
        ]
    }

    direction = direction_vector_from_grounding(grounding, factor_count=39)

    assert direction[25] == 1.0
    assert direction[38] == -1.0
    assert direction[35] == -1.0
    assert direction[37] == 1.0
    assert np.count_nonzero(direction) == 4


def test_directional_delta_calibration_is_bounded_and_anchor_preserving() -> None:
    samples = np.zeros((3, 4, 39), dtype=np.float32)
    delta_scale = np.ones((4, 39), dtype=np.float32)
    direction = np.zeros(39, dtype=np.float32)
    direction[25] = 1.0
    direction[38] = -1.0

    calibrated = apply_directional_delta_calibration(
        samples,
        delta_scale=delta_scale,
        direction_vector=direction,
        beta=0.2,
        alpha=1.0,
        beta_bound=0.15,
    )

    assert calibrated.shape == samples.shape
    assert np.allclose(calibrated[:, 0, 25], 0.15 / 4.0)
    assert np.allclose(calibrated[:, -1, 25], 0.15)
    assert np.allclose(calibrated[:, -1, 38], -0.15)
    assert np.allclose(calibrated[:, :, 0], 0.0)
    assert np.allclose(samples, 0.0)


def test_fit_directional_beta_selects_nonzero_when_direction_matches_target() -> None:
    samples = np.zeros((4, 3, 39), dtype=np.float32)
    delta_scale = np.ones((3, 39), dtype=np.float32)
    target = np.zeros((3, 39), dtype=np.float32)
    target[:, 25] = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    direction = np.zeros(39, dtype=np.float32)
    direction[25] = 1.0

    fit = fit_directional_beta(
        [
            {
                "samples": samples,
                "target": target,
                "delta_scale": delta_scale,
                "direction_vector": direction,
            }
        ],
        beta_grid=[0.0, 0.1, 0.2, 0.3],
        alpha=1.0,
        beta_bound=0.3,
    )

    assert fit["selected_beta"] > 0.0
    assert fit["candidates"][0]["beta"] == 0.0
    assert fit["selected"]["ensemble_crps_z_mean"] < fit["candidates"][0][
        "ensemble_crps_z_mean"
    ]


def test_fit_directional_beta_reports_effective_betas_within_bound() -> None:
    samples = np.zeros((4, 3, 39), dtype=np.float32)
    delta_scale = np.ones((3, 39), dtype=np.float32)
    target = np.zeros((3, 39), dtype=np.float32)
    direction = np.zeros(39, dtype=np.float32)
    direction[25] = 1.0

    fit = fit_directional_beta(
        [
            {
                "samples": samples,
                "target": target,
                "delta_scale": delta_scale,
                "direction_vector": direction,
            }
        ],
        beta_grid=[0.0, 0.25, 0.3],
        alpha=1.0,
        beta_bound=0.25,
    )

    assert [candidate["beta"] for candidate in fit["candidates"]] == [0.0, 0.25]
    assert fit["selected_beta"] <= 0.25


def test_quality_constrained_response_selector_prefers_stronger_feasible_beta() -> None:
    identity = {
        "ensemble_crps_z_mean": 1.0,
        "energy_score_z_mean": 1.0,
        "coverage_80_mean": 0.8,
    }
    candidates = [
        {
            "beta": 0.0,
            "ensemble_crps_z_mean": 1.0,
            "energy_score_z_mean": 1.0,
            "coverage_80_mean": 0.8,
            "identity": identity,
        },
        {
            "beta": 0.05,
            "ensemble_crps_z_mean": 0.99,
            "energy_score_z_mean": 0.99,
            "coverage_80_mean": 0.8,
            "identity": identity,
        },
        {
            "beta": 0.25,
            "ensemble_crps_z_mean": 1.01,
            "energy_score_z_mean": 1.01,
            "coverage_80_mean": 0.79,
            "identity": identity,
        },
    ]

    selected = _select_beta_candidate(
        candidates,
        selection_objective="quality_constrained_response",
        max_quality_regression_pct=0.02,
        min_coverage_delta=-0.03,
    )

    assert selected["beta"] == 0.25


def test_support_evidence_gate_blocks_rejected_support() -> None:
    rejected = {
        "variant_rows": [
            {
                "is_operational": True,
                "memory_prior_direction_status": "reject",
                "memory_prior_support_weighted_match_rate": 0.05,
            }
        ]
    }
    accepted = {
        "variant_rows": [
            {
                "is_operational": True,
                "memory_prior_direction_status": "pass",
                "memory_prior_support_weighted_match_rate": 1.0,
            }
        ]
    }
    warning_without_match_rate = {
        "variant_rows": [
            {
                "is_operational": True,
                "memory_prior_direction_status": "warning",
            }
        ]
    }
    start_only_even_if_pass = {
        "cached_query": {
            "memory_prior": {
                "mode": "soft_topk_start_only",
                "direction_check": {"status": "pass"},
            }
        }
    }

    assert _support_evidence_gate_from_report(rejected) == 0.0
    assert _support_evidence_gate_from_report(accepted) == 1.0
    assert _support_evidence_gate_from_report(warning_without_match_rate) == 1.0
    assert _support_evidence_gate_from_report(start_only_even_if_pass) == 0.0


def test_path_with_start_prepends_initial_level() -> None:
    states = np.asarray([[[11.0, 21.0], [12.0, 22.0]]], dtype=np.float32)
    start = np.asarray([10.0, 20.0], dtype=np.float32)

    path = path_with_start(states, start, factor_index=1)

    assert path.shape == (1, 3)
    assert np.allclose(path[0], [20.0, 21.0, 22.0])


def test_analyze_grid_reports_pairwise_feature_distances() -> None:
    def cell(start_level: float, narrative_shift: float) -> dict:
        start = np.full(39, start_level, dtype=np.float32)
        states = np.zeros((4, 2, 39), dtype=np.float32)
        states[:] = start[None, None, :]
        states[:, -1, 25] += narrative_shift + np.asarray(
            [-0.2, -0.05, 0.05, 0.2], dtype=np.float32
        )
        return {
            "states": states,
            "start": start,
            "support": [{"window_index": int(start_level + narrative_shift)}],
        }

    grid = {
        "s1": {"n1": cell(100.0, -1.0), "n2": cell(100.0, 1.0)},
        "s2": {"n1": cell(110.0, -1.0), "n2": cell(110.0, 1.0)},
    }

    result, detail = _analyze_grid(
        grid,
        starts=("s1", "s2"),
        cases=("n1", "n2"),
        policy="test_policy",
        feature_space="start_normalized",
    )

    assert result["mean_feature_distance_same_start_narrative"] > 0.0
    assert result["mean_feature_distance_same_narrative_start"] > 0.0
    assert detail["pairwise_feature_distances"]["same_start_narrative"] > 0.0


def test_summarize_qualitative_response_reports_relevant_raw_level_effects() -> None:
    def result(terminal_shift: float, *, effective_beta: float = 0.0) -> dict:
        start = np.zeros(39, dtype=np.float32)
        start[25] = 100.0
        start[38] = 20.0
        states = np.zeros((3, 2, 39), dtype=np.float32)
        states[:] = start[None, None, :]
        states[:, -1, 25] += terminal_shift + np.asarray([-1.0, 0.0, 1.0])
        states[:, -1, 38] -= terminal_shift
        return {
            "states": states,
            "start": start,
            "support": [{"window_index": 1}, {"window_index": 2}],
            "calibration": {
                "effective_beta": effective_beta,
                "support_evidence_gate": 1.0,
            },
        }

    summary = _summarize_qualitative_response(
        candidate_cases={
            "fragile_risk_on": result(1.0, effective_beta=0.25),
            "defensive_risk_off": result(-1.0, effective_beta=0.25),
        },
        null_cases={
            "fragile_risk_on": result(0.0),
            "defensive_risk_off": result(0.0),
        },
        relevant_factors={
            "fragile_risk_on": ("SPX", "VIX"),
            "defensive_risk_off": ("SPX", "VIX"),
        },
        reference_case="fragile_risk_on",
    )

    assert summary["headline"]["case_count"] == 2
    assert summary["headline"]["max_abs_candidate_minus_null_terminal_median"] > 0.0
    risk_off = next(row for row in summary["case_summaries"] if row["case"] == "defensive_risk_off")
    spx = next(row for row in risk_off["factor_summaries"] if row["factor"] == "SPX")
    assert spx["start_level"] == 100.0
    assert spx["terminal_median"] == 99.0
    assert spx["candidate_minus_null_terminal_median"] == -1.0
    assert spx["terminal_median_vs_reference"] == -2.0


def test_render_qualitative_markdown_names_raw_levels_and_null_control() -> None:
    summary = {
        "headline": {
            "case_count": 1,
            "max_abs_candidate_minus_null_terminal_median": 1.25,
            "max_abs_terminal_median_vs_reference": 0.0,
        },
        "case_summaries": [
            {
                "label": "Fragile risk-on",
                "support_count": 2,
                "effective_beta": 0.25,
                "support_evidence_gate": 1.0,
                "factor_summaries": [
                    {
                        "factor": "SPX",
                        "start_level": 100.0,
                        "terminal_median": 101.0,
                        "candidate_minus_null_terminal_median": 1.0,
                        "terminal_median_vs_reference": 0.0,
                    }
                ],
            }
        ],
    }

    markdown = _render_qualitative_markdown(summary)

    assert "raw market levels" in markdown
    assert "start-only null" in markdown
    assert "Fragile risk-on" in markdown
    assert "SPX" in markdown


def test_plot_qualitative_response_panels_writes_file(tmp_path: Path) -> None:
    start = np.zeros(39, dtype=np.float32)
    start[25] = 100.0
    start[38] = 20.0
    states = np.zeros((4, 2, 39), dtype=np.float32)
    states[:] = start[None, None, :]
    states[:, -1, 25] += np.asarray([-1.0, 0.0, 1.0, 2.0])
    candidate = {
        "fragile_risk_on": {"states": states, "start": start, "support": []},
    }
    null = {
        "fragile_risk_on": {"states": states - 0.25, "start": start, "support": []},
    }
    output = tmp_path / "panel.png"

    plot_qualitative_response_panels(
        candidate_cases=candidate,
        null_cases=null,
        output_path=output,
        relevant_factors={"fragile_risk_on": ("SPX", "VIX")},
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_per_start_narrative_metrics_report_each_fixed_start() -> None:
    def cell(start_level: float, narrative_shift: float) -> dict:
        start = np.full(39, start_level, dtype=np.float32)
        states = np.zeros((4, 2, 39), dtype=np.float32)
        states[:] = start[None, None, :]
        states[:, -1, 25] += narrative_shift + np.asarray(
            [-0.2, -0.05, 0.05, 0.2], dtype=np.float32
        )
        return {
            "states": states,
            "start": start,
            "support": [{"window_index": int(start_level + narrative_shift)}],
        }

    grid = {
        "s1": {"n1": cell(100.0, -1.0), "n2": cell(100.0, 1.0)},
        "s2": {"n1": cell(110.0, -1.0), "n2": cell(110.0, 1.0)},
    }

    summary = _per_start_narrative_metrics(
        grid,
        starts=("s1", "s2"),
        cases=("n1", "n2"),
    )

    assert [row["start"] for row in summary["rows"]] == ["s1", "s2"]
    assert summary["min_mean_factor_terminal_ks"] > 0.0
    assert summary["min_mean_portfolio_terminal_ks"] > 0.0
    assert summary["max_mean_support_jaccard"] == 0.0


def test_per_start_promotion_gates_enforce_minimum_response() -> None:
    gates = _per_start_promotion_gates(
        {
            "min_mean_factor_terminal_ks": 0.31,
            "min_mean_portfolio_terminal_ks": 0.39,
            "max_mean_support_jaccard": 0.04,
        }
    )

    assert gates == {
        "per_start_factor_ks_min_ge_0p20": True,
        "per_start_portfolio_ks_min_ge_0p20": True,
        "per_start_support_jaccard_max_le_0p25": True,
    }

    failed = _per_start_promotion_gates(
        {
            "min_mean_factor_terminal_ks": 0.19,
            "min_mean_portfolio_terminal_ks": 0.25,
            "max_mean_support_jaccard": 0.5,
        }
    )

    assert failed["per_start_factor_ks_min_ge_0p20"] is False
    assert failed["per_start_support_jaccard_max_le_0p25"] is False


def test_resolve_start_roots_allows_matched_fixed_start_decks(tmp_path: Path) -> None:
    start_a = tmp_path / "start_a"
    start_b = tmp_path / "start_b"
    start_a.mkdir()
    start_b.mkdir()

    assert _parse_start_root(f"s18={start_a}") == ("s18", str(start_a))

    resolved = _resolve_start_roots([f"s18={start_a}", f"s22={start_b}"])

    assert resolved == {"s18": str(start_a), "s22": str(start_b)}
