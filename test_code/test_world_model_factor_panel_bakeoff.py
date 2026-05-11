from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.analyze_factor_panel_probe_bakeoff import (  # noqa: E402
    run_factor_panel_probe_bakeoff,
)


def test_factor_panel_probe_bakeoff_scores_raw_factor_baselines(tmp_path):
    data_path = tmp_path / "multi_factor_data.npz"
    levels = np.stack(
        [
            np.arange(18, dtype=np.float32),
            np.arange(18, dtype=np.float32) * 2.0,
        ],
        axis=1,
    )
    returns = (np.arange(18, dtype=np.float32) * 0.1)[:, None]
    np.savez(
        data_path,
        levels=levels,
        level_columns=np.asarray(["spx", "vix"]),
        returns=returns,
        return_columns=np.asarray(["spx_ret"]),
    )

    result = run_factor_panel_probe_bakeoff(
        SimpleNamespace(
            factor_data_path=data_path,
            device="cpu",
            history_len=4,
            future_len=3,
            test_start=16,
            val_size=3,
            max_train_windows=4,
            max_val_windows=2,
            batch_size=4,
            ridge_alpha=1.0,
            scale_checkpoint="",
            output_json=tmp_path / "factor_bakeoff.json",
            report_md=tmp_path / "factor_bakeoff.md",
        )
    )

    assert result["analysis"] == "world_model_factor_panel_probe_bakeoff"
    assert result["objective_family"] == "downstream_probe_factor_panel_bakeoff"
    assert result["uses_future_targets_as_pretraining"] is False
    assert result["feature_surfaces"] == ["raw_factor_last", "raw_factor_flat"]
    assert "scale_barlow_last" in result["skipped_feature_surfaces"]
    assert result["summary_counts"]["raw_factor_plus_scale_best_raw_wins"] is None
    assert (
        result["decision_summary"]["next_step"]
        == "factor_family_normalized_probe_audit_before_promotion"
    )
    assert set(result["future_summary"]) == {
        "factor_future_mean_delta",
        "factor_future_range",
        "factor_future_terminal_delta",
        "factor_future_max_abs_step",
    }
    assert math.isfinite(
        result["future_probe_rows"]["raw_factor_last"]["targets"][
            "factor_future_mean_delta"
        ]["mse"]
    )
    assert result["part_b_blocked"] is True
    assert (tmp_path / "factor_bakeoff.json").exists()
    assert (tmp_path / "factor_bakeoff.md").exists()
