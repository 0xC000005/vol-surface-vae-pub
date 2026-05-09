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
from experiments.world.part1_jepa_latent.jepa_smoke import (
    JEPAConfig,
    JEPAWorldModel,
    jepa_loss,
    retrieval_contrastive_loss,
    update_ema,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import (
    HorizonJEPAConfig,
    HorizonJEPAWorldModel,
    horizon_jepa_loss,
    select_horizon_prefix,
    select_horizon_target,
)
from experiments.world.part1_jepa_latent.supervised_horizon_frame import (
    SupervisedHorizonConfig,
    SupervisedHorizonFrameModel,
    checkpoint_selection_score,
    decode_horizon_prediction,
    make_horizon_frame_targets,
    supervised_horizon_loss,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (
    ridge_probe_metrics,
    ridge_probe_predict,
)


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


def test_jepa_smoke_model_forward_loss_and_ema_update():
    import torch

    torch.manual_seed(7)
    cfg = JEPAConfig(input_dim=5, hidden_dim=8, latent_dim=4, predictor_hidden_dim=8)
    model = JEPAWorldModel(cfg)
    past = torch.randn(6, 3, 5)
    future = torch.randn(6, 3, 5)

    out = model(past, future)
    loss, parts = jepa_loss(
        out,
        variance_weight=0.1,
        covariance_weight=0.1,
        retrieval_weight=0.1,
    )
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    update_ema(model.context_encoder, model.target_encoder, decay=0.5)
    after = list(model.target_encoder.parameters())

    assert out["context"].shape == (6, 4)
    assert out["target"].shape == (6, 4)
    assert out["predicted"].shape == (6, 4)
    assert torch.isfinite(loss)
    assert parts["prediction"] >= 0.0
    assert parts["retrieval"] > 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_retrieval_contrastive_loss_prefers_matching_pairs():
    import torch

    predicted = torch.eye(4)
    target = torch.eye(4)
    shuffled = target[[1, 0, 3, 2]]

    matching = retrieval_contrastive_loss(predicted, target, temperature=0.1)
    mismatched = retrieval_contrastive_loss(predicted, shuffled, temperature=0.1)

    assert matching < mismatched


def test_horizon_jepa_selects_prefixes_and_scores_loss():
    import torch

    torch.manual_seed(11)
    cfg = HorizonJEPAConfig(
        input_dim=5,
        hidden_dim=8,
        latent_dim=4,
        predictor_hidden_dim=8,
        horizons=(1, 3),
    )
    model = HorizonJEPAWorldModel(cfg)
    past = torch.randn(6, 4, 5)
    future = torch.randn(6, 5, 5)

    prefix = select_horizon_prefix(future, horizon=3)
    frame = select_horizon_target(future, horizon=3, target_mode="frame")
    prefix_target = select_horizon_target(future, horizon=3, target_mode="prefix")
    out = model(past, future)
    loss, parts = horizon_jepa_loss(
        out,
        variance_weight=0.1,
        covariance_weight=0.1,
        retrieval_weight=0.1,
    )

    assert prefix.shape == (6, 3, 5)
    assert frame.shape == (6, 1, 5)
    assert prefix_target.shape == (6, 3, 5)
    assert torch.equal(frame[:, 0, :], future[:, 2, :])
    assert out["context"].shape == (6, 4)
    assert out["target"].shape == (6, 2, 4)
    assert out["predicted"].shape == (6, 2, 4)
    assert torch.isfinite(loss)
    assert parts["prediction"] >= 0.0
    assert parts["retrieval"] > 0.0


def test_horizon_jepa_trainable_target_gets_regularization_gradients():
    import torch

    torch.manual_seed(13)
    cfg = HorizonJEPAConfig(
        input_dim=5,
        hidden_dim=8,
        latent_dim=4,
        predictor_hidden_dim=8,
        horizons=(1, 3),
    )
    model = HorizonJEPAWorldModel(cfg, target_encoder_mode="trainable")
    past = torch.randn(6, 4, 5)
    future = torch.randn(6, 5, 5)

    out = model(past, future)
    loss, _parts = horizon_jepa_loss(
        out,
        variance_weight=0.1,
        covariance_weight=0.1,
        retrieval_weight=0.1,
        target_regularization_grad=True,
    )
    loss.backward()

    target_grad = next(model.target_encoder.parameters()).grad
    assert target_grad is not None
    assert torch.isfinite(target_grad).all()


def test_supervised_horizon_delta_targets_reconstruct_frames():
    import torch

    torch.manual_seed(17)
    cfg = SupervisedHorizonConfig(
        input_dim=5,
        hidden_dim=8,
        context_dim=4,
        predictor_hidden_dim=8,
        horizons=(1, 3),
        target_mode="delta",
    )
    model = SupervisedHorizonFrameModel(cfg)
    past = torch.randn(6, 4, 5)
    future = torch.randn(6, 5, 5)

    target, frame = make_horizon_frame_targets(
        past,
        future,
        horizons=cfg.horizons,
        target_mode="delta",
    )
    pred_target = model(past)
    pred_frame = decode_horizon_prediction(
        past,
        pred_target,
        target_mode="delta",
    )
    loss, parts = supervised_horizon_loss(pred_target, target, frame, past)
    contrastive_loss, contrastive_parts = supervised_horizon_loss(
        pred_target,
        target,
        frame,
        past,
        retrieval_weight=0.1,
    )

    assert target.shape == (6, 2, 5)
    assert frame.shape == (6, 2, 5)
    assert pred_target.shape == (6, 2, 5)
    assert pred_frame.shape == (6, 2, 5)
    torch.testing.assert_close(target[:, 1, :], future[:, 2, :] - past[:, -1, :])
    assert torch.isfinite(loss)
    assert torch.isfinite(contrastive_loss)
    assert parts["target_mse"] >= 0.0
    assert contrastive_parts["retrieval"] > 0.0


def test_supervised_horizon_loss_adds_context_regularizers():
    import torch

    past = torch.randn(8, 4, 3)
    predicted_target = torch.zeros(8, 2, 3)
    target = torch.zeros(8, 2, 3)
    frame = decode_horizon_prediction(past, target, target_mode="delta")
    context = torch.randn(8, 5)

    base_loss, base_parts = supervised_horizon_loss(
        predicted_target,
        target,
        frame,
        past,
        target_mode="delta",
        frame_weight=0.0,
    )
    regularized_loss, parts = supervised_horizon_loss(
        predicted_target,
        target,
        frame,
        past,
        target_mode="delta",
        frame_weight=0.0,
        context=context,
        context_variance_weight=0.1,
        context_covariance_weight=0.1,
        context_variance_gamma=0.5,
    )

    assert base_parts["context_variance"] == 0.0
    assert base_parts["context_covariance"] == 0.0
    assert parts["context_variance"] >= 0.0
    assert parts["context_covariance"] >= 0.0
    assert regularized_loss > base_loss


def test_checkpoint_selection_score_prefers_requested_metric():
    low_mse = {"val_mse": 0.01, "val_mrr_mean": 0.02, "val_top5_mean": 0.03}
    high_mrr = {"val_mse": 0.02, "val_mrr_mean": 0.05, "val_top5_mean": 0.02}

    assert checkpoint_selection_score(low_mse, "mse") > checkpoint_selection_score(high_mrr, "mse")
    assert checkpoint_selection_score(high_mrr, "mrr") > checkpoint_selection_score(low_mse, "mrr")
    assert checkpoint_selection_score(low_mse, "top5") > checkpoint_selection_score(high_mrr, "top5")


def test_ridge_probe_predict_recovers_linear_multivariate_targets():
    train_x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    weights = np.array([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=np.float64)
    bias = np.array([0.5, -0.25, 0.75], dtype=np.float64)
    train_y = train_x @ weights + bias
    val_x = np.array([[0.5, 0.5], [3.0, -2.0]], dtype=np.float64)
    val_y = val_x @ weights + bias

    pred = ridge_probe_predict(train_x, train_y, val_x, alpha=1e-9)

    assert np.mean((pred - val_y) ** 2) < 1e-10


def test_ridge_probe_metrics_reports_horizon_metrics():
    train_context = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    val_context = np.array([[0.5, 0.5], [3.0, -2.0]], dtype=np.float64)
    weights_h1 = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float64)
    weights_h5 = np.array([[0.25, 1.5], [-1.0, 0.75]], dtype=np.float64)
    train_targets = np.stack(
        [train_context @ weights_h1, train_context @ weights_h5],
        axis=1,
    )
    val_targets = np.stack(
        [val_context @ weights_h1, val_context @ weights_h5],
        axis=1,
    )

    metrics = ridge_probe_metrics(
        train_context,
        val_context,
        train_targets,
        val_targets,
        horizons=(1, 5),
        alpha=1e-9,
    )

    assert metrics["overall_prediction"]["mse"] < 1e-10
    assert metrics["per_horizon"]["1"]["prediction"]["mse"] < 1e-10
    assert metrics["per_horizon"]["5"]["prediction"]["mse"] < 1e-10
