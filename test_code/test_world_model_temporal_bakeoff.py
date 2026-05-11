from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.analyze_temporal_jepa_bakeoff import (  # noqa: E402
    run_temporal_jepa_bakeoff,
)


def test_temporal_jepa_bakeoff_compares_raw_random_and_temporal(tmp_path):
    result = run_temporal_jepa_bakeoff(
        SimpleNamespace(
            device="cpu",
            history_len=8,
            future_len=4,
            target_len=2,
            max_train_windows=8,
            max_val_windows=4,
            epochs=1,
            batch_size=4,
            seed=2173,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
            barlow_weight=0.05,
            offdiag_weight=0.005,
            hidden_dim=12,
            latent_dim=6,
            predictor_hidden_dim=8,
            ridge_alpha=10.0,
            scale_checkpoint="",
            output_json=tmp_path / "bakeoff.json",
            report_md=tmp_path / "bakeoff.md",
        )
    )

    assert result["analysis"] == "world_model_temporal_jepa_bakeoff"
    assert result["objective_family"] == "downstream_probe_frozen_bakeoff"
    assert result["uses_future_targets_as_pretraining"] is False
    assert result["part_b_blocked"] is True
    assert result["promotion_decision"] == "DO_NOT_PROMOTE"
    assert result["decision"]["part_b_blocked"] is True
    assert result["decision"]["temporal_context_target_status"] in {
        "do_not_promote",
        "needs_larger_confirmatory_run",
    }
    assert "future_utility_status" in result["decision"]
    assert result["feature_surfaces"] == [
        "raw_surface_last",
        "raw_surface_flat",
        "temporal_jepa_last",
        "raw_surface_last_plus_temporal_jepa_last",
        "random_temporal_last",
        "raw_surface_last_plus_random_temporal_last",
    ]
    assert "scale_barlow_last" in result["skipped_feature_surfaces"]
    assert math.isfinite(
        result["current_state_rows"]["temporal_jepa_last"]["iv_surface"]["mse"]
    )
    assert set(result["future_summary"].keys()) == {
        "future_mean_delta",
        "future_range",
        "future_terminal_delta",
        "future_max_abs_step",
        "future_drawdown",
    }
    assert result["summary_counts"]["temporal_raw_plus_future_improvements"] >= 0
    assert (tmp_path / "bakeoff.json").exists()
    assert (tmp_path / "bakeoff.md").exists()
