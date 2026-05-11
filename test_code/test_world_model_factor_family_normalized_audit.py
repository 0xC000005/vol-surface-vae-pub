from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.analyze_factor_family_normalized_probe_audit import (  # noqa: E402
    run_factor_family_normalized_probe_audit,
)


def test_factor_family_normalized_audit_splits_level_and_return_targets(tmp_path):
    data_path = tmp_path / "multi_factor_data.npz"
    base = np.arange(22, dtype=np.float32)
    levels = np.stack([1000.0 + base * 10.0, 5.0 + base], axis=1)
    returns = np.stack([base * 0.01, base * -0.02], axis=1)
    np.savez(
        data_path,
        levels=levels,
        level_columns=np.asarray(["spx", "vix"]),
        returns=returns,
        return_columns=np.asarray(["spx_ret", "vix_ret"]),
    )

    result = run_factor_family_normalized_probe_audit(
        SimpleNamespace(
            factor_data_path=data_path,
            device="cpu",
            history_len=5,
            future_len=3,
            test_start=19,
            val_size=3,
            max_train_windows=5,
            max_val_windows=2,
            batch_size=4,
            ridge_alpha=1.0,
            scale_checkpoint="",
            output_json=tmp_path / "factor_family_audit.json",
            report_md=tmp_path / "factor_family_audit.md",
        )
    )

    assert result["analysis"] == "world_model_factor_family_normalized_probe_audit"
    assert result["objective_family"] == "downstream_probe_factor_family_normalized"
    assert result["uses_future_targets_as_pretraining"] is False
    assert result["target_metric_units"] == "train_standardized_per_factor_family"
    assert result["feature_metric_units"] == "train_standardized_per_feature_surface"
    assert set(result["family_summary"]) == {"factor_level", "factor_return"}
    assert result["family_column_counts"] == {"factor_level": 2, "factor_return": 2}
    assert result["summary_counts"]["target_family_cells"] == 8
    assert result["summary_counts"]["raw_factor_plus_scale_best_raw_wins"] is None
    assert math.isfinite(
        result["family_summary"]["factor_level"]["factor_future_mean_delta"][
            "raw_factor_last_normalized_mse"
        ]
    )
    assert result["promotion_decision"] == "PROBE_ONLY_DO_NOT_PROMOTE"
    assert result["part_b_blocked"] is True
    assert (tmp_path / "factor_family_audit.json").exists()
    assert (tmp_path / "factor_family_audit.md").exists()
