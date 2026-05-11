from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.temporal_block_jepa_smoke import (  # noqa: E402
    run_temporal_block_jepa_smoke,
)


def test_temporal_block_jepa_smoke_reports_probe_guardrails(tmp_path):
    result = run_temporal_block_jepa_smoke(
        SimpleNamespace(
            device="cpu",
            history_len=8,
            future_len=4,
            target_len=2,
            max_train_windows=8,
            max_val_windows=4,
            epochs=1,
            batch_size=4,
            seed=2172,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
            barlow_weight=0.05,
            offdiag_weight=0.005,
            hidden_dim=12,
            latent_dim=6,
            predictor_hidden_dim=8,
            ridge_alpha=10.0,
            retrieval_eval_rows=16,
            output_json=tmp_path / "temporal_block_jepa.json",
            checkpoint="",
            report_md=tmp_path / "temporal_block_jepa.md",
        )
    )

    assert result["objective_family"] == "context_to_target_jepa_temporal_diagnostic"
    assert result["objective_family_base"] == "context_to_target_jepa"
    assert result["uses_future_targets"] is False
    assert result["uses_value_reconstruction"] is False
    assert result["uses_decoder"] is False
    assert result["promotion_decision"] == "SMOKE_ONLY_DO_NOT_PROMOTE"
    assert result["part_b_blocked"] is True
    assert result["train_target_time_rows"] == 16
    assert result["val_target_time_rows"] == 8
    assert math.isfinite(result["final_val_loss"])
    assert result["val_context_health"]["effective_rank"] >= 0.0
    assert result["val_target_health"]["effective_rank"] >= 0.0
    assert result["current_state_guardrail"]["raw_plus_to_raw_ratio"] > 0.0
    assert set(result["future_probe_summary"]) == {
        "future_mean_delta",
        "future_range",
    }
    assert (tmp_path / "temporal_block_jepa.json").exists()
    assert (tmp_path / "temporal_block_jepa.md").exists()
