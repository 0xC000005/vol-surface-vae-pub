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
    select_horizon_delta_target,
    select_horizon_prefix,
    select_horizon_target,
)
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (
    fit_delta_pca_target,
    inverse_transform_delta_targets,
    make_horizon_delta_matrix,
    transform_delta_targets,
)
from experiments.world.part1_jepa_latent.target_encoder_distill import (
    DeltaTargetEncoder,
    TargetEncoderDistillConfig,
    evaluate_distilled_targets,
    target_encoder_distill_loss,
)
from experiments.world.part1_jepa_latent.frozen_target_jepa import (
    FrozenTargetJEPAConfig,
    FrozenTargetJEPAWorldModel,
    freeze_module,
    frozen_target_jepa_loss,
)
from experiments.world.part1_jepa_latent.direct_delta_pca_predictor import (
    DirectDeltaPCAConfig,
    DirectDeltaPCAPredictor,
    flatten_past_window,
)
from experiments.world.part1_jepa_latent.supervised_horizon_frame import (
    SupervisedHorizonConfig,
    SupervisedHorizonFrameModel,
    checkpoint_selection_score,
    context_correlation_loss,
    decode_horizon_prediction,
    epoch_checkpoint_path,
    make_horizon_frame_targets,
    save_epoch_checkpoint,
    soft_neighborhood_contrastive_loss,
    supervised_horizon_loss,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (
    ridge_probe_metrics,
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.score_context_runs import composite_part1_score


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


def test_horizon_jepa_delta_target_subtracts_last_past_frame():
    import torch

    past = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [2.0, -3.0]],
        ]
    )
    future = torch.tensor(
        [
            [[4.0, 5.0], [7.0, 9.0], [10.0, 12.0]],
            [[3.0, -1.0], [4.0, -2.0], [8.0, 1.0]],
        ]
    )

    delta_frame = select_horizon_delta_target(
        past,
        future,
        horizon=2,
        target_mode="frame",
    )
    delta_prefix = select_horizon_delta_target(
        past,
        future,
        horizon=2,
        target_mode="prefix",
    )

    torch.testing.assert_close(delta_frame[:, 0, :], future[:, 1, :] - past[:, -1, :])
    torch.testing.assert_close(delta_prefix, future[:, :2, :] - past[:, -1:, :])


def test_fixed_delta_pca_target_round_trips_rank_two_deltas():
    train_deltas = np.array(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]],
            [[2.0, 0.0, 4.0], [0.0, 2.0, -2.0]],
            [[-1.0, 0.0, -2.0], [0.0, -1.0, 1.0]],
            [[-2.0, 0.0, -4.0], [0.0, -2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    target = fit_delta_pca_target(train_deltas, target_dim=2)
    z = transform_delta_targets(train_deltas, target)
    reconstructed = inverse_transform_delta_targets(z, target)

    assert z.shape == (4, 2, 2)
    np.testing.assert_allclose(reconstructed, train_deltas, atol=1e-5)


def test_make_horizon_delta_matrix_uses_requested_horizons():
    past = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ],
        dtype=np.float32,
    )
    future = np.array(
        [
            [[4.0, 6.0], [5.0, 9.0], [7.0, 10.0]],
            [[3.0, 5.0], [4.0, 8.0], [8.0, 11.0]],
        ],
        dtype=np.float32,
    )

    deltas = make_horizon_delta_matrix(past, future, horizons=(1, 3))

    assert deltas.shape == (2, 2, 2)
    np.testing.assert_allclose(deltas[:, 0, :], future[:, 0, :] - past[:, -1, :])
    np.testing.assert_allclose(deltas[:, 1, :], future[:, 2, :] - past[:, -1, :])


def test_target_encoder_distill_model_outputs_horizon_codes():
    import torch

    torch.manual_seed(29)
    cfg = TargetEncoderDistillConfig(
        input_dim=5,
        hidden_dim=12,
        target_dim=3,
        horizons=(1, 3),
    )
    model = DeltaTargetEncoder(cfg)
    deltas = torch.randn(7, 2, 5)
    target = torch.randn(7, 2, 3)

    predicted = model(deltas)
    loss, parts = target_encoder_distill_loss(predicted, target)

    assert predicted.shape == (7, 2, 3)
    assert torch.isfinite(loss)
    assert parts["target_mse"] >= 0.0


def test_evaluate_distilled_targets_scores_perfect_fixed_pca_codes():
    train_deltas = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [-1.0, 1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    pca = fit_delta_pca_target(train_deltas, target_dim=3)
    target_z = transform_delta_targets(train_deltas, pca)

    metrics = evaluate_distilled_targets(
        predicted_z=target_z,
        target_z=target_z,
        truth_delta=train_deltas,
        pca_target=pca,
        horizons=(1, 3),
    )

    assert metrics["overall_prediction"]["mse"] == 0.0
    assert metrics["overall_retrieval"]["mrr_mean"] == 1.0
    assert metrics["overall_delta_decode"]["mse_mean"] < 1e-10
    assert metrics["predicted_health"]["effective_rank"] > 1.0


def test_frozen_target_jepa_predicts_horizon_codes_and_freezes_target_encoder():
    import torch

    torch.manual_seed(31)
    cfg = FrozenTargetJEPAConfig(
        input_dim=5,
        hidden_dim=12,
        context_dim=4,
        target_dim=3,
        predictor_hidden_dim=10,
        horizons=(1, 3),
    )
    model = FrozenTargetJEPAWorldModel(cfg)
    target_encoder = DeltaTargetEncoder(
        TargetEncoderDistillConfig(input_dim=5, hidden_dim=12, target_dim=3, horizons=(1, 3))
    )
    freeze_module(target_encoder)
    past = torch.randn(7, 4, 5)
    target = torch.randn(7, 2, 3)

    predicted = model(past)
    loss, parts = frozen_target_jepa_loss(predicted, target)

    assert predicted.shape == (7, 2, 3)
    assert torch.isfinite(loss)
    assert parts["prediction"] >= 0.0
    assert all(not param.requires_grad for param in target_encoder.parameters())


def test_direct_delta_pca_predictor_flattens_past_and_predicts_horizon_codes():
    import torch

    torch.manual_seed(37)
    cfg = DirectDeltaPCAConfig(
        input_dim=6,
        hidden_dim=10,
        target_dim=3,
        horizons=(1, 3),
    )
    model = DirectDeltaPCAPredictor(cfg)
    past = torch.randn(5, 3, 2)

    flat = flatten_past_window(past)
    predicted = model(past)

    assert flat.shape == (5, 6)
    assert predicted.shape == (5, 2, 3)


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


def test_context_correlation_loss_penalizes_duplicate_dimensions():
    import torch

    torch.manual_seed(23)
    context = torch.randn(64, 4)
    duplicated = context.clone()
    duplicated[:, 1] = duplicated[:, 0]

    assert context_correlation_loss(duplicated) > context_correlation_loss(context)


def test_soft_neighborhood_contrastive_loss_prefers_aligned_predictions():
    import torch

    target = torch.eye(6)
    aligned = target.clone()
    permuted = torch.roll(target, shifts=1, dims=0)

    aligned_loss = soft_neighborhood_contrastive_loss(
        aligned,
        target,
        temperature=0.1,
        target_temperature=0.2,
    )
    permuted_loss = soft_neighborhood_contrastive_loss(
        permuted,
        target,
        temperature=0.1,
        target_temperature=0.2,
    )

    assert aligned_loss < permuted_loss


def test_checkpoint_selection_score_prefers_requested_metric():
    low_mse = {"val_mse": 0.01, "val_mrr_mean": 0.02, "val_top5_mean": 0.03}
    high_mrr = {"val_mse": 0.02, "val_mrr_mean": 0.05, "val_top5_mean": 0.02}

    assert checkpoint_selection_score(low_mse, "mse") > checkpoint_selection_score(high_mrr, "mse")
    assert checkpoint_selection_score(high_mrr, "mrr") > checkpoint_selection_score(low_mse, "mrr")
    assert checkpoint_selection_score(low_mse, "top5") > checkpoint_selection_score(high_mrr, "top5")


def test_save_epoch_checkpoint_writes_numbered_checkpoint(tmp_path):
    import torch

    cfg = SupervisedHorizonConfig(input_dim=3, hidden_dim=4, context_dim=2, predictor_hidden_dim=5)
    model = SupervisedHorizonFrameModel(cfg)
    row = {"epoch": 7, "val_mse": 0.1, "val_mrr_mean": 0.2}

    checkpoint_path = save_epoch_checkpoint(
        tmp_path,
        epoch=7,
        model=model,
        cfg=cfg,
        epoch_summary=row,
        args={"selection_metric": "mrr"},
    )

    assert checkpoint_path == epoch_checkpoint_path(tmp_path, 7)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 7
    assert payload["epoch_summary"] == row
    assert payload["config"]["input_dim"] == 3


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


def test_composite_part1_score_rewards_rank_probe_and_retrieval():
    weaker_train = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.019},
            "overall_retrieval": {"mrr_mean": 0.04, "top5_mean": 0.05, "top10_mean": 0.10},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }
    weaker_audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 2.0,
            "offdiag_abs_mean": 0.6,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.08},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }
    stronger_train = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.07, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }
    stronger_audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 5.0,
            "offdiag_abs_mean": 0.3,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.11},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }

    assert composite_part1_score(stronger_train, stronger_audit)["score"] > composite_part1_score(
        weaker_train,
        weaker_audit,
    )["score"]


def test_composite_part1_score_penalizes_frame_mse_worse_than_persistence():
    audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 5.0,
            "offdiag_abs_mean": 0.3,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.11},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }
    beats_persistence = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.07, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }
    misses_persistence = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.022},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.07, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }

    assert composite_part1_score(beats_persistence, audit)["score"] > composite_part1_score(
        misses_persistence,
        audit,
    )["score"]


def test_composite_part1_score_penalizes_topk_below_persistence():
    audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 5.0,
            "offdiag_abs_mean": 0.3,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.11},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }
    stronger_topk = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.09, "top10_mean": 0.14},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.08, "top10_mean": 0.13},
        },
    }
    weaker_topk = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.06, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.08, "top10_mean": 0.13},
        },
    }

    assert composite_part1_score(stronger_topk, audit)["score"] > composite_part1_score(
        weaker_topk,
        audit,
    )["score"]
