import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_mixed_coordinate_path_flow_matching import (  # noqa: E402
    GenericMixedCoordinatePathFMConfig,
    GenericMixedCoordinatePathFlowMatching,
)


def _small_model() -> GenericMixedCoordinatePathFlowMatching:
    cfg = GenericMixedCoordinatePathFMConfig(
        history_len=4,
        future_len=3,
        n_cells=5,
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
        level_score_channels=[0, 1],
    )
    model = GenericMixedCoordinatePathFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    quantile_levels = (
        torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5
    ) / cfg.n_quantiles
    model.set_empirical_quantiles(level_quantiles, increment_quantiles, quantile_levels)
    return model


def test_path_flow_loss_and_sampling_are_finite() -> None:
    torch.manual_seed(647)
    model = _small_model()
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


def test_path_flow_level_score_channels_respect_level_support() -> None:
    torch.manual_seed(648)
    model = _small_model()
    cfg = model.cfg
    history_state = torch.zeros(2, cfg.history_len, cfg.n_cells)
    history_increment = torch.zeros(2, cfg.history_len, cfg.n_cells)

    increments = model.sample_batched(
        history_state,
        history_increment,
        n_samples=2,
        n_steps=cfg.future_len,
        chunk_size=1,
    )
    levels = history_state[:, None, -1:, :] + torch.cumsum(increments, dim=2)
    masked_levels = levels[..., [0, 1]]

    assert masked_levels.min() >= -1.0001
    assert masked_levels.max() <= 1.0001


def test_path_flow_terminal_path_loss_is_finite() -> None:
    torch.manual_seed(654)
    cfg = GenericMixedCoordinatePathFMConfig(
        history_len=4,
        future_len=3,
        n_cells=5,
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
        level_score_channels=[0, 1],
        terminal_path_loss_weight=0.25,
        terminal_tail_weight=1.0,
        terminal_tail_threshold=0.5,
    )
    model = GenericMixedCoordinatePathFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    quantile_levels = (
        torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5
    ) / cfg.n_quantiles
    model.set_empirical_quantiles(level_quantiles, increment_quantiles, quantile_levels)
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

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["terminal_path_loss"])
    assert torch.isfinite(metrics["terminal_tail_rate"])


def test_path_flow_rejects_out_of_range_level_score_channels() -> None:
    cfg = GenericMixedCoordinatePathFMConfig(
        history_len=2,
        future_len=1,
        n_cells=3,
        level_score_channels=[3],
    )

    try:
        GenericMixedCoordinatePathFlowMatching(cfg)
    except ValueError as exc:
        assert "out-of-range" in str(exc)
    else:
        raise AssertionError("expected level_score_channels validation failure")


def test_path_flow_conditional_source_affine_is_finite() -> None:
    torch.manual_seed(650)
    cfg = GenericMixedCoordinatePathFMConfig(
        history_len=4,
        future_len=3,
        n_cells=5,
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
        level_score_channels=[0, 1],
        conditional_source_affine=True,
    )
    model = GenericMixedCoordinatePathFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    quantile_levels = (
        torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5
    ) / cfg.n_quantiles
    model.set_empirical_quantiles(level_quantiles, increment_quantiles, quantile_levels)
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
        n_samples=2,
        n_steps=cfg.future_len,
        chunk_size=1,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["source_scale_mean"])
    assert torch.isfinite(metrics["source_loc_abs"])
    assert samples.shape == (2, 2, cfg.future_len, cfg.n_cells)
    assert torch.isfinite(samples).all()


def test_path_flow_horizon_scalar_source_affine_preserves_shapes() -> None:
    torch.manual_seed(651)
    cfg = GenericMixedCoordinatePathFMConfig(
        history_len=4,
        future_len=3,
        n_cells=5,
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
        level_score_channels=[0, 1],
        conditional_source_affine=True,
        source_affine_mode="horizon_scalar",
    )
    model = GenericMixedCoordinatePathFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    quantile_levels = (
        torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5
    ) / cfg.n_quantiles
    model.set_empirical_quantiles(level_quantiles, increment_quantiles, quantile_levels)
    history_state = 0.2 * torch.randn(2, cfg.history_len, cfg.n_cells)
    history_increment = 0.05 * torch.randn(2, cfg.history_len, cfg.n_cells)
    context, _current_score = model.encode_history(history_state, history_increment)

    loc, scale = model.conditional_source_affine(context, cfg.future_len)

    assert loc is not None
    assert scale is not None
    assert loc.shape == (2, cfg.future_len, cfg.n_cells)
    assert scale.shape == (2, cfg.future_len, cfg.n_cells)
    assert torch.isfinite(loc).all()
    assert torch.isfinite(scale).all()
