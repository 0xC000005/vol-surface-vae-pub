import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (
    _ks_statistic,
    _support_jaccard,
    _terminal_summary,
    build_story_smoke_command,
    condition_report_case_name,
    normalize_case_name,
)


def test_ks_statistic_detects_equal_and_disjoint_samples():
    assert _ks_statistic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert _ks_statistic(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 1.0


def test_support_jaccard_uses_selected_window_sets():
    left = [{"window_index": 1}, {"window_index": 2}]
    right = [{"window_index": 2}, {"window_index": 3}]

    assert _support_jaccard(left, right) == 1.0 / 3.0


def test_terminal_summary_reports_raw_level_quantiles():
    states = np.asarray(
        [
            [[10.0, 1.0], [12.0, 3.0]],
            [[20.0, 2.0], [22.0, 4.0]],
        ],
        dtype=np.float32,
    )

    summary = _terminal_summary(states, {"SPX": 0, "VIX": 1})

    assert summary["SPX"]["p50"] == 17.0
    assert summary["VIX"]["p10"] == 3.1


def test_build_story_smoke_command_contains_policy_contract(tmp_path):
    command = build_story_smoke_command(
        case_name="fragile_risk_on",
        policy_name="narrative_first_soft_direction_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=931,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode diverse_topk_combined" in joined
    assert "--start-distance-penalty 0.0" in joined
    assert "--memory-prior-diverse-min-index-gap 30" in joined
    assert "--samples 8" in joined
    assert "fragile_risk_on_start18/condition_only_report.json" in joined
    assert "fragile_risk_on/narrative_first_soft_direction_gap30" in joined


def test_case_names_accept_legacy_condition_report_aliases():
    assert normalize_case_name("fragile_risk_on_start18") == "fragile_risk_on"
    assert condition_report_case_name("fragile_risk_on") == "fragile_risk_on_start18"


def test_build_story_smoke_command_passes_external_support_bank(tmp_path):
    command = build_story_smoke_command(
        case_name="fragile_risk_on",
        policy_name="current_start_checked_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=931,
        support_bank_report="support/report.json",
        support_bank_arrays="support/arrays.npz",
    )

    joined = " ".join(command)
    assert "--support-bank-report support/report.json" in joined
    assert "--support-bank-arrays support/arrays.npz" in joined


def test_hard_direction_ablation_removes_start_penalty_but_keeps_direction_gate(
    tmp_path,
):
    command = build_story_smoke_command(
        case_name="fragile_risk_on",
        policy_name="narrative_first_hard_direction_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=931,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode diverse_topk_narrative_start_checked" in joined
    assert "--start-distance-penalty 0.0" in joined
    assert "--implication-alignment-weight 0.25" in joined


def test_response_aware_book_policy_uses_quality_guard_contract(tmp_path):
    command = build_story_smoke_command(
        case_name="commodity_inflation",
        policy_name="narrative_book_response_guard_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=931,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode narrative_book_quality_guard_926b" in joined
    assert "--memory-prior-quality-guard-candidate-pool-size 12" in joined
    assert "--memory-prior-quality-guard-max-mixtures 64" in joined
    assert "--memory-prior-quality-guard-min-candidate-mixtures 4" in joined


def test_portfolio_quality_guard_policy_uses_train_only_guard_contract(tmp_path):
    command = build_story_smoke_command(
        case_name="commodity_inflation",
        policy_name="portfolio_quality_guard_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=936,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode portfolio_quality_guard_924e" in joined
    assert "--start-distance-penalty 0.02" in joined
    assert "--implication-alignment-weight 0.25" in joined
    assert "--memory-prior-diverse-min-index-gap 30" in joined
    assert "--memory-prior-quality-guard-candidate-pool-size 12" in joined


def test_broad_response_guard_policy_uses_learned_support_contract(tmp_path):
    command = build_story_smoke_command(
        case_name="commodity_inflation",
        policy_name="broad_replay_response_guard_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=940,
        quality_guard_candidate_pool_size=20,
        quality_guard_max_mixtures=96,
        quality_guard_probability_temperature=0.15,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode broad_replay_response_guard_940a" in joined
    assert "--memory-prior-quality-guard-candidate-pool-size 20" in joined
    assert "--memory-prior-quality-guard-max-mixtures 96" in joined
    assert "--memory-prior-quality-guard-probability-temperature 0.15" in joined
    assert "--memory-prior-diverse-min-index-gap 30" in joined


def test_response_preview_policy_uses_two_stage_rollout_contract(tmp_path):
    command = build_story_smoke_command(
        case_name="commodity_inflation",
        policy_name="response_preview_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=931,
        response_preview_samples_per_component=4,
        response_preview_alpha=0.9,
        response_preview_temperature=0.8,
        response_preview_blend=0.35,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode diverse_topk_narrative_start_checked" in joined
    assert "--rollout-mixture-mode response_preview_component_mixture" in joined
    assert "--response-preview-samples-per-component 4" in joined
    assert "--response-preview-alpha 0.9" in joined
    assert "--response-preview-temperature 0.8" in joined
    assert "--response-preview-blend 0.35" in joined


def test_broad_response_preview_policy_combines_broad_guard_and_preview(tmp_path):
    command = build_story_smoke_command(
        case_name="commodity_inflation",
        policy_name="broad_response_preview_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=941,
        quality_guard_candidate_pool_size=20,
        quality_guard_max_mixtures=96,
        response_preview_samples_per_component=4,
        response_preview_blend=0.35,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode broad_replay_response_guard_940a" in joined
    assert "--rollout-mixture-mode response_preview_component_mixture" in joined
    assert "--memory-prior-quality-guard-candidate-pool-size 20" in joined
    assert "--memory-prior-quality-guard-max-mixtures 96" in joined
    assert "--response-preview-samples-per-component 4" in joined
    assert "--response-preview-blend 0.35" in joined


def test_broad_portfolio_response_preview_policy_uses_portfolio_objective(tmp_path):
    command = build_story_smoke_command(
        case_name="commodity_inflation",
        policy_name="broad_portfolio_response_preview_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=942,
        response_preview_samples_per_component=4,
        response_preview_blend=0.35,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode broad_replay_response_guard_940a" in joined
    assert "--rollout-mixture-mode response_preview_component_mixture" in joined
    assert "--response-preview-objective factor_portfolio" in joined


def test_broad_channel_portfolio_preview_policy_uses_channel_objective(tmp_path):
    command = build_story_smoke_command(
        case_name="commodity_inflation",
        policy_name="broad_channel_portfolio_preview_gap30",
        output_root=tmp_path,
        samples=8,
        decoder_steps=10,
        seed=943,
        response_preview_samples_per_component=4,
        response_preview_blend=0.35,
    )

    joined = " ".join(command)
    assert "--memory-prior-mode broad_replay_response_guard_940a" in joined
    assert "--rollout-mixture-mode response_preview_component_mixture" in joined
    assert "--response-preview-objective channel_portfolio" in joined
