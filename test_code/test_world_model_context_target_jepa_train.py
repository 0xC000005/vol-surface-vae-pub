from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.context_target_jepa_smoke import (  # noqa: E402
    train_context_target_smoke,
)


def test_context_target_jepa_train_smoke_writes_artifacts(tmp_path):
    result_json = tmp_path / "context_target_smoke.json"
    checkpoint = tmp_path / "context_target_smoke.pt"
    args = argparse.Namespace(
        epochs=1,
        batch_size=4,
        history_len=8,
        future_len=4,
        hidden_dim=12,
        latent_dim=6,
        predictor_hidden_dim=8,
        max_train_windows=8,
        max_val_windows=4,
        lr=1e-3,
        weight_decay=1e-4,
        barlow_weight=0.05,
        barlow_offdiag_weight=0.005,
        ema_decay=0.99,
        grad_clip=1.0,
        seed=2140,
        device="cpu",
        output_json=result_json,
        checkpoint=checkpoint,
    )

    result = train_context_target_smoke(args)

    assert result_json.exists()
    assert checkpoint.exists()
    assert result["objective_family"] == "context_to_target_jepa"
    assert result["uses_future_targets"] is False
    assert result["train_shape"] == [8, 8, 58]
    assert result["val_shape"] == [4, 8, 58]
    assert np.isfinite(result["val_metrics"]["loss"])
