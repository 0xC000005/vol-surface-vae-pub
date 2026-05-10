from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.surface_local_jepa_smoke import (  # noqa: E402
    run_surface_local_jepa_smoke,
)


def test_surface_local_jepa_smoke_reports_target_token_health(tmp_path):
    result = run_surface_local_jepa_smoke(
        SimpleNamespace(
            device="cpu",
            history_len=6,
            future_len=3,
            max_train_windows=4,
            max_val_windows=4,
            epochs=1,
            seed=2154,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
            barlow_weight=0.05,
            offdiag_weight=0.005,
            token_hidden_dim=8,
            hidden_dim=12,
            latent_dim=6,
            predictor_hidden_dim=10,
            retrieval_eval_rows=32,
            output_json=tmp_path / "surface_local_smoke.json",
            checkpoint="",
        )
    )

    assert result["objective_family"] == "token_geometry_level_context_to_target_jepa"
    assert result["uses_future_targets"] is False
    assert result["uses_value_reconstruction"] is False
    assert result["promotion_decision"] == "SMOKE_ONLY_DO_NOT_PROMOTE"
    assert result["val_target_token_rows"] > 1
    assert result["initial_val_loss"] >= 0.0
    assert result["final_val_loss"] >= 0.0
    assert result["val_alignment"] >= 0.0
    assert result["val_retrieval_subset_rows"] <= 32
    assert result["val_predicted_health"]["effective_rank"] >= 0.0
    assert result["val_target_health"]["effective_rank"] >= 0.0
    assert (tmp_path / "surface_local_smoke.json").exists()
