import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_multihead_mixed_coordinate_path_flow_matching import (  # noqa: E402
    GenericMultiHeadMixedCoordinatePathFMConfig,
    GenericMultiHeadMixedCoordinatePathFlowMatching,
)


def _model(
    n_cells: int, iv_count: int
) -> GenericMultiHeadMixedCoordinatePathFlowMatching:
    cfg = GenericMultiHeadMixedCoordinatePathFMConfig(
        history_len=4,
        future_len=3,
        n_cells=n_cells,
        iv_count=iv_count,
        memory_dim=16,
        memory_layers=1,
        memory_heads=2,
        memory_ff=32,
        token_dim=16,
        token_layers=1,
        token_heads=2,
        token_ff=32,
        time_dim=8,
        flow_steps=2,
        n_quantiles=31,
        conditioning_mode="prefix",
        prefix_feature_mode="scale",
        level_score_channels=list(range(iv_count)),
        head_hidden=16,
    )
    model = GenericMultiHeadMixedCoordinatePathFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    quantile_levels = (
        torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5
    ) / cfg.n_quantiles
    model.set_empirical_quantiles(level_quantiles, increment_quantiles, quantile_levels)
    return model


def test_multihead_joint_loss_and_sampling_are_finite() -> None:
    torch.manual_seed(652)
    model = _model(n_cells=6, iv_count=4)
    cfg = model.cfg
    history_state = 0.2 * torch.randn(2, cfg.history_len, cfg.n_cells)
    history_increment = 0.05 * torch.randn(2, cfg.history_len, cfg.n_cells)
    future_increment = 0.05 * torch.randn(2, cfg.future_len, cfg.n_cells)
    future_state = history_state[:, -1:, :] + torch.cumsum(future_increment, dim=1)

    loss, metrics = model.training_loss(
        history_state,
        history_increment,
        future_state,
        future_increment,
    )
    samples = model.sample_batched(
        history_state,
        history_increment,
        n_samples=3,
        n_steps=cfg.future_len,
        chunk_size=2,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["fm_loss"])
    assert samples.shape == (2, 3, cfg.future_len, cfg.n_cells)
    assert torch.isfinite(samples).all()


def test_multihead_iv_only_scope_has_no_factor_head() -> None:
    torch.manual_seed(653)
    model = _model(n_cells=5, iv_count=5)
    cfg = model.cfg
    history_state = 0.2 * torch.randn(2, cfg.history_len, cfg.n_cells)
    history_increment = 0.05 * torch.randn(2, cfg.history_len, cfg.n_cells)
    context, _current_score = model.encode_history(history_state, history_increment)
    x_t = torch.randn(2, cfg.future_len, cfg.n_cells)
    t = torch.rand(2)

    velocity = model.predict_velocity(x_t, context, t)

    assert model.factor_count == 0
    assert model.factor_head is None
    assert velocity.shape == x_t.shape
    assert torch.isfinite(velocity).all()


def test_multihead_rejects_invalid_iv_count() -> None:
    cfg = GenericMultiHeadMixedCoordinatePathFMConfig(
        history_len=2,
        future_len=1,
        n_cells=3,
        iv_count=4,
    )

    try:
        GenericMultiHeadMixedCoordinatePathFlowMatching(cfg)
    except ValueError as exc:
        assert "iv_count" in str(exc)
    else:
        raise AssertionError("expected iv_count validation failure")
