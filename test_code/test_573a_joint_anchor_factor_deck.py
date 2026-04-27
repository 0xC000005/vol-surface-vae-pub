import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.generate_573a_joint_anchor_factor_deck import (
    anchor_factor_columns,
    build_joint_manifest,
    empirical_copula_diagnostics,
    empirical_copula_factor_quantiles,
    quantile_matched_indices,
    rank_matched_indices,
    select_panel_factor_paths,
)


def test_anchor_factor_columns_excludes_iv_cells() -> None:
    columns = ["iv:00", "iv:01", "factor:spx", "factor:spx_logret"]

    assert anchor_factor_columns(columns, iv_count=2) == [
        "factor:spx",
        "factor:spx_logret",
    ]


def test_select_panel_factor_paths_reconstructs_levels_and_preserves_stress_order() -> (
    None
):
    columns = ["iv:00", "factor:spx", "factor:spx_logret"]
    history = np.zeros((1, 2, 3), dtype=np.float32)
    history[0, -1, 1] = 100.0
    candidates = np.zeros((1, 6, 2, 3), dtype=np.float32)
    for idx in range(6):
        candidates[0, idx, :, 0] = float(idx)
        candidates[0, idx, :, 2] = np.log(1.0 + 0.01 * idx)

    selected = select_panel_factor_paths(
        panel_history=history,
        panel_candidates=candidates,
        columns=columns,
        n_select=3,
        iv_count=1,
    )

    assert selected.factor_scenarios.shape == (1, 3, 2, 2)
    assert selected.factor_columns == ["factor:spx", "factor:spx_logret"]
    assert selected.selected_indices[0].tolist() == [0, 2, 4]
    assert np.allclose(selected.factor_scenarios[0, 0, :, 0], [100.0, 100.0])
    assert selected.factor_scenarios[0, 2, -1, 0] > 100.0


def test_rank_matched_indices_maps_target_order_to_candidate_severity_ranks() -> None:
    candidate_severity = np.arange(6, dtype=np.float64)
    target_severity = np.array([0.2, 0.4, 0.9], dtype=np.float64)

    indices = rank_matched_indices(candidate_severity, target_severity)

    assert indices.tolist() == [0, 2, 5]


def test_quantile_matched_indices_preserves_stress_selection_quantiles() -> None:
    candidate_severity = np.arange(6, dtype=np.float64)
    target_quantiles = np.array([0.0, 0.45, 0.85], dtype=np.float64)

    indices = quantile_matched_indices(candidate_severity, target_quantiles)

    assert indices.tolist() == [0, 2, 4]


def test_select_panel_factor_paths_can_rank_match_to_selected_iv_severity() -> None:
    columns = ["iv:00", "factor:spx", "factor:spx_logret"]
    history = np.zeros((1, 2, 3), dtype=np.float32)
    history[0, -1, 1] = 100.0
    candidates = np.zeros((1, 6, 2, 3), dtype=np.float32)
    for idx in range(6):
        candidates[0, idx, :, 0] = float(idx)
        candidates[0, idx, :, 2] = np.log(1.0 + 0.01 * idx)

    selected = select_panel_factor_paths(
        panel_history=history,
        panel_candidates=candidates,
        columns=columns,
        n_select=3,
        iv_count=1,
        target_iv_path_mean=np.array([[0.2, 0.4, 0.9]], dtype=np.float32),
    )

    assert selected.selected_indices[0].tolist() == [0, 2, 5]
    assert np.allclose(
        selected.factor_scenarios[0, 2, :, 0], [105.0, 110.25], rtol=1e-5
    )


def test_select_panel_factor_paths_can_quantile_match_to_selected_iv_policy() -> None:
    columns = ["iv:00", "factor:spx", "factor:spx_logret"]
    history = np.zeros((1, 2, 3), dtype=np.float32)
    history[0, -1, 1] = 100.0
    candidates = np.zeros((1, 6, 2, 3), dtype=np.float32)
    for idx in range(6):
        candidates[0, idx, :, 0] = float(idx)
        candidates[0, idx, :, 2] = np.log(1.0 + 0.01 * idx)

    selected = select_panel_factor_paths(
        panel_history=history,
        panel_candidates=candidates,
        columns=columns,
        n_select=3,
        iv_count=1,
        target_iv_severity_quantiles=np.array([[0.0, 0.45, 0.85]], dtype=np.float32),
    )

    assert selected.selected_indices[0].tolist() == [0, 2, 4]
    assert (
        selected.pairing_policy == "candidate_quantile_matched_to_selected_iv_severity"
    )
    assert np.allclose(
        selected.internal_panel_severity_quantiles[0],
        np.array([0.0, 0.4, 0.8], dtype=np.float32),
    )


def test_empirical_copula_factor_quantiles_can_preserve_mixed_rank_cases() -> None:
    target_iv_quantiles = np.array([0.05, 0.15, 0.9], dtype=np.float64)
    historical_iv_quantiles = np.array(
        [0.02, 0.08, 0.12, 0.18, 0.82, 0.95], dtype=np.float64
    )
    historical_factor_quantiles = np.array(
        [0.85, 0.9, 0.1, 0.2, 0.75, 0.8], dtype=np.float64
    )

    sampled = empirical_copula_factor_quantiles(
        target_iv_quantiles,
        historical_iv_quantiles,
        historical_factor_quantiles,
        bins=3,
        seed=123,
    )

    assert sampled.shape == target_iv_quantiles.shape
    assert set(np.round(sampled[:2], 2)).issubset({0.85, 0.9, 0.1, 0.2})
    assert sampled[2] in {0.75, 0.8}
    assert np.any(np.abs(sampled - target_iv_quantiles) > 0.4)


def test_select_panel_factor_paths_can_quantile_match_factor_stress() -> None:
    columns = ["iv:00", "factor:spx", "factor:spx_logret"]
    history = np.zeros((1, 2, 3), dtype=np.float32)
    history[0, -1, 1] = 100.0
    candidates = np.zeros((1, 6, 2, 3), dtype=np.float32)
    for idx in range(6):
        candidates[0, idx, :, 0] = float(5 - idx)
        candidates[0, idx, :, 2] = np.log(1.0 + 0.01 * idx)

    selected = select_panel_factor_paths(
        panel_history=history,
        panel_candidates=candidates,
        columns=columns,
        n_select=3,
        iv_count=1,
        target_factor_severity_quantiles=np.array(
            [[0.0, 0.45, 0.85]], dtype=np.float32
        ),
    )

    assert selected.selected_indices[0].tolist() == [0, 2, 4]
    assert (
        selected.pairing_policy == "empirical_copula_matched_to_factor_stress_severity"
    )
    assert np.allclose(
        selected.internal_panel_factor_stress_quantiles[0],
        np.array([0.0, 0.4, 0.8], dtype=np.float32),
    )
    assert not np.allclose(
        selected.internal_panel_severity_quantiles[0],
        selected.internal_panel_factor_stress_quantiles[0],
    )


def test_empirical_copula_diagnostics_reports_selected_and_historical_mixing() -> None:
    diag = empirical_copula_diagnostics(
        iv_quantiles=np.array([0.1, 0.5, 0.9], dtype=np.float64),
        factor_quantiles=np.array([0.8, 0.5, 0.2], dtype=np.float64),
        historical_iv_quantiles=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        historical_factor_quantiles=np.array([0.8, 0.2, 0.1, 0.9], dtype=np.float64),
        bins=3,
    )

    assert diag["copula_bins"] == 3
    assert diag["selected_mixed_bucket_rate"] > 0.0
    assert diag["historical_mixed_bucket_rate"] > 0.0
    assert diag["historical_copula_windows"] == 4


def test_build_joint_manifest_is_json_serializable_and_separates_contracts() -> None:
    manifest = build_joint_manifest(
        history_start_index=10,
        history_end_index=40,
        history_len=30,
        future_len=30,
        samples=6,
        iv_candidate_count=192,
        factor_candidate_count=192,
        seed=573,
        iv_scenario_shape=(6, 30, 5, 5),
        factor_scenario_shape=(6, 30, 26),
        factor_columns=["factor:spx", "factor:spx_logret"],
        output_npz="demo.npz",
        iv_evidence_path="iv.json",
        factor_evidence_path="factor.json",
    )

    json.dumps(manifest)
    assert manifest["iv_risk_contract"]["risk_manager_acceptable"] is True
    assert manifest["joint_anchor_factor_contract"]["risk_manager_acceptable"] is True
    assert (
        manifest["probability_interpretation"]
        == "stress_scenario_set_not_calibrated_law"
    )
