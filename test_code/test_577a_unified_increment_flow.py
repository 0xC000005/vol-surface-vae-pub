import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.train_577a_unified_increment_flow import (
    UnifiedIncrementFlow,
    UnifiedIncrementFlowConfig,
    fit_path_gaussian,
    fit_increment_normal_score,
    normal_score_inverse,
    normal_score_transform,
)


def test_unified_increment_flow_forward_shape_matches_full_path() -> None:
    cfg = UnifiedIncrementFlowConfig(
        history_len=4,
        future_len=3,
        n_vars=5,
        hidden_dim=16,
        time_embed_dim=8,
        depth=2,
        dropout=0.0,
    )
    model = UnifiedIncrementFlow(cfg)
    history = torch.randn(2, 4, 5)
    x_t = torch.randn(2, 3, 5)
    t = torch.rand(2)

    out = model(history, x_t, t)

    assert out.shape == x_t.shape


def test_unified_increment_flow_training_loss_and_sampling_are_finite() -> None:
    cfg = UnifiedIncrementFlowConfig(
        history_len=4,
        future_len=3,
        n_vars=5,
        hidden_dim=16,
        time_embed_dim=8,
        depth=2,
        dropout=0.0,
    )
    model = UnifiedIncrementFlow(cfg)
    history = torch.randn(2, 4, 5)
    target = torch.randn(2, 3, 5)

    loss, metrics = model.training_loss(history, target)
    samples = model.sample(history, n_samples=3, n_steps=2)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["pred_velocity_std"])
    assert samples.shape == (2, 3, 3, 5)
    assert torch.isfinite(samples).all()


def test_unified_increment_flow_supports_cumulative_state_loss() -> None:
    cfg = UnifiedIncrementFlowConfig(
        history_len=4,
        future_len=3,
        n_vars=5,
        hidden_dim=16,
        time_embed_dim=8,
        depth=2,
        dropout=0.0,
    )
    model = UnifiedIncrementFlow(cfg)
    history = torch.randn(2, 4, 5)
    target = torch.randn(2, 3, 5)

    loss, metrics = model.training_loss(history, target, fm_loss_mode="cumulative_state")

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["cumulative_mse"])
    assert torch.isfinite(metrics["velocity_mse"])
    assert metrics["cumulative_mse"] >= 0


def test_unified_increment_flow_accepts_path_gaussian_source() -> None:
    cfg = UnifiedIncrementFlowConfig(
        history_len=4,
        future_len=3,
        n_vars=5,
        hidden_dim=16,
        time_embed_dim=8,
        depth=2,
        dropout=0.0,
        source_mode="path_gaussian",
    )
    model = UnifiedIncrementFlow(cfg)
    target = torch.randn(8, 3, 5).numpy()
    mean, chol = fit_path_gaussian(target, shrinkage=0.2, jitter=1e-4)
    model.set_source_gaussian(torch.from_numpy(mean), torch.from_numpy(chol))

    source = model.draw_source(4, device=torch.device("cpu"), dtype=torch.float32)

    assert source.shape == (4, 3, 5)
    assert torch.isfinite(source).all()


def test_unified_increment_flow_accepts_empirical_path_source() -> None:
    cfg = UnifiedIncrementFlowConfig(
        history_len=4,
        future_len=3,
        n_vars=5,
        hidden_dim=16,
        time_embed_dim=8,
        depth=2,
        dropout=0.0,
        source_mode="empirical_path",
    )
    model = UnifiedIncrementFlow(cfg)
    bank = torch.randn(7, 15)
    model.set_source_bank(bank)

    source = model.draw_source(4, device=torch.device("cpu"), dtype=torch.float32)

    assert source.shape == (4, 3, 5)
    assert torch.isfinite(source).all()


def test_unified_increment_flow_accepts_conditional_empirical_source() -> None:
    cfg = UnifiedIncrementFlowConfig(
        history_len=4,
        future_len=3,
        n_vars=5,
        hidden_dim=16,
        time_embed_dim=8,
        depth=2,
        dropout=0.0,
        source_mode="empirical_conditional",
        conditional_source_topk=3,
    )
    model = UnifiedIncrementFlow(cfg)
    bank = torch.randn(7, 15)
    keys = torch.randn(7, 5)
    model.set_source_bank(bank, keys)
    history = torch.randn(4, 4, 5)

    source = model.draw_source(4, device=torch.device("cpu"), dtype=torch.float32, history=history)

    assert source.shape == (4, 3, 5)
    assert torch.isfinite(source).all()


def test_unified_increment_flow_accepts_conditional_affine_source() -> None:
    cfg = UnifiedIncrementFlowConfig(
        history_len=4,
        future_len=3,
        n_vars=5,
        hidden_dim=16,
        time_embed_dim=8,
        depth=2,
        dropout=0.0,
        source_mode="conditional_affine",
    )
    model = UnifiedIncrementFlow(cfg)
    history = torch.randn(4, 4, 5)
    target = torch.randn(4, 3, 5)

    source = model.draw_source(4, device=torch.device("cpu"), dtype=torch.float32, history=history)
    loss, metrics = model.training_loss(history, target, source_prior_nll_weight=0.05)

    assert source.shape == (4, 3, 5)
    assert torch.isfinite(source).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["source_prior_nll"])
    assert metrics["source_prior_nll"] > 0


def test_normal_score_increment_transform_is_bounded_and_invertible_on_quantiles() -> None:
    target = torch.linspace(-2.0, 2.0, steps=40).reshape(4, 2, 5).numpy()
    quantiles, _levels, normal_levels = fit_increment_normal_score(
        target,
        n_quantiles=21,
        cdf_eps=1e-3,
    )

    scores = normal_score_transform(target, quantiles, normal_levels)
    recovered = normal_score_inverse(scores, quantiles, normal_levels)
    extreme = normal_score_inverse(scores * 100.0, quantiles, normal_levels)

    assert recovered.shape == target.shape
    assert abs(float(recovered.mean() - target.mean())) < 1e-4
    assert extreme.max() <= quantiles[-1].max() + 1e-6
    assert extreme.min() >= quantiles[0].min() - 1e-6
