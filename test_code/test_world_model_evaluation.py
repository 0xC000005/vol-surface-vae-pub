from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.part1_metrics import (
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.evaluation.part2_metrics import compact_path_sample_metrics
from experiments.world.evaluation.world_data import build_iv_world_windows


def _write_surface_npz(path, n_days: int = 20) -> np.ndarray:
    surface = np.arange(n_days * 25, dtype=np.float32).reshape(n_days, 5, 5)
    np.savez(
        path,
        surface=surface,
        ret=np.zeros(n_days, dtype=np.float32),
        price=np.ones(n_days, dtype=np.float32),
        slopes=np.zeros(n_days, dtype=np.float32),
        skews=np.zeros(n_days, dtype=np.float32),
        levels=np.ones(n_days, dtype=np.float32),
    )
    return surface


def test_build_iv_world_windows_uses_manifest_style_split(tmp_path):
    data_path = tmp_path / "surface.npz"
    surface = _write_surface_npz(data_path)
    regime_path = tmp_path / "regimes.npz"
    labels = np.arange(surface.shape[0] - 3 - 2 + 1, dtype=np.int32)
    np.savez(regime_path, labels=labels, history_len=3, future_len=2, n_regimes=5)

    train = build_iv_world_windows(
        data_path=data_path,
        split="train",
        history_len=3,
        future_len=2,
        test_start=16,
        val_size=3,
        max_windows=2,
        normalize=False,
        regime_path=regime_path,
    )
    val = build_iv_world_windows(
        data_path=data_path,
        split="val",
        history_len=3,
        future_len=2,
        test_start=16,
        val_size=3,
        normalize=False,
        regime_path=regime_path,
    )

    assert train.past_window.shape == (2, 3, 25)
    assert train.future_window.shape == (2, 2, 25)
    assert train.start_index.tolist() == [0, 1]
    np.testing.assert_array_equal(train.past_window[0, 0], surface[0].reshape(25))
    np.testing.assert_array_equal(train.future_window[0, 0], surface[3].reshape(25))
    assert train.regime_label.tolist() == [0, 1]

    assert val.past_window.shape == (3, 3, 25)
    assert val.future_window.shape == (3, 2, 25)
    assert val.start_index.tolist() == [8, 9, 10]
    assert val.regime_label.tolist() == [8, 9, 10]


def test_part1_metrics_detect_prediction_retrieval_and_rank():
    target = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    predicted = target + np.array(
        [
            [0.01, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.00, 0.00, 0.03],
            [0.01, 0.01, 0.00],
        ],
        dtype=np.float32,
    )

    pred_metrics = latent_prediction_metrics(predicted, target)
    retrieval = retrieval_metrics(predicted, target, top_k=(1, 2))
    health = representation_health_metrics(target)

    assert pred_metrics["mse"] < 0.001
    assert pred_metrics["cosine_mean"] > 0.99
    assert retrieval["top1"] == 1.0
    assert retrieval["mrr"] == 1.0
    assert health["effective_rank"] > 2.0
    assert health["variance_min"] > 0.0


def test_compact_path_sample_metrics_score_decoder_samples():
    truth = np.array(
        [
            [[0.10, 0.20], [0.11, 0.22], [0.12, 0.23]],
            [[0.30, 0.40], [0.33, 0.39], [0.35, 0.38]],
        ],
        dtype=np.float32,
    )
    offsets = np.array([-0.02, 0.0, 0.02], dtype=np.float32)
    samples = truth[:, None, :, :] + offsets[None, :, None, None]

    metrics = compact_path_sample_metrics(samples, truth)

    assert 0.0 <= metrics["coverage_90"] <= 1.0
    assert metrics["pairwise_distance_mean"] > 0.0
    assert metrics["variance_ratio"] > 0.0
    assert metrics["effective_rank"] > 0.0
    assert metrics["corr_frobenius"] >= 0.0
