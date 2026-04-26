import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.generate_573a_joint_anchor_factor_deck import (
    anchor_factor_columns,
    build_joint_manifest,
    select_panel_factor_paths,
)


def test_anchor_factor_columns_excludes_iv_cells() -> None:
    columns = ["iv:00", "iv:01", "factor:spx", "factor:spx_logret"]

    assert anchor_factor_columns(columns, iv_count=2) == ["factor:spx", "factor:spx_logret"]


def test_select_panel_factor_paths_reconstructs_levels_and_preserves_stress_order() -> None:
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
    assert manifest["probability_interpretation"] == "stress_scenario_set_not_calibrated_law"
