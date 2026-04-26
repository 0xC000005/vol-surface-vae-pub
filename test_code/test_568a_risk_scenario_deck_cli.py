import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.generate_568a_risk_scenario_deck import (
    build_manifest,
    severity_bucket_labels,
)


def test_severity_bucket_labels_match_stress_deck_contract() -> None:
    labels = severity_bucket_labels(48)

    assert labels.shape == (48,)
    assert labels[:16].tolist() == ["calm"] * 16
    assert labels[16:32].tolist() == ["central"] * 16
    assert labels[32:].tolist() == ["stress"] * 16


def test_build_manifest_is_json_serializable_and_disclaims_probabilities() -> None:
    path_mean_iv = np.array([0.12, 0.18, 0.31], dtype=np.float32)

    manifest = build_manifest(
        model_type="340c",
        checkpoint="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt",
        data_path="data/vol_surface_with_ret.npz",
        history_start_index=5792,
        history_end_index=5822,
        history_len=30,
        future_len=30,
        samples=3,
        candidate_count=192,
        seed=568,
        scenario_shape=(3, 30, 5, 5),
        path_mean_iv=path_mean_iv,
    )

    json.dumps(manifest)
    assert manifest["policy"] == "severity_stratified_selection"
    assert manifest["probability_interpretation"] == "stress_scenario_set_not_calibrated_law"
    assert manifest["bucket_counts"] == {"calm": 1, "central": 1, "stress": 1}
    assert manifest["path_mean_iv"]["min"] == 0.12
    assert manifest["path_mean_iv"]["max"] == 0.31
