import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_multihead_state_conditioned_mixed_coordinate_flow_matching import (  # noqa: E402
    GenericMultiHeadStateConditionedMixedCoordinateFMConfig,
    GenericMultiHeadStateConditionedMixedCoordinateFlowMatching,
)


def _model(
    n_cells: int = 6,
    iv_count: int = 4,
) -> GenericMultiHeadStateConditionedMixedCoordinateFlowMatching:
    cfg = GenericMultiHeadStateConditionedMixedCoordinateFMConfig(
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
    model = GenericMultiHeadStateConditionedMixedCoordinateFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    quantile_levels = (
        torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5
    ) / cfg.n_quantiles
    model.set_empirical_quantiles(level_quantiles, increment_quantiles, quantile_levels)
    return model


def test_multihead_ar_loss_and_sampling_are_finite() -> None:
    torch.manual_seed(658)
    model = _model()
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


def test_multihead_ar_uses_shared_source_but_typed_readouts() -> None:
    torch.manual_seed(659)
    model = _model(n_cells=5, iv_count=3)
    cfg = model.cfg
    x_t = torch.randn(2, cfg.n_cells)
    current_score = torch.randn(2, cfg.n_cells)
    memory_state = torch.randn(2, cfg.memory_dim)
    t = torch.rand(2)

    velocity = model.velocity(x_t, current_score, memory_state, t)

    assert model.velocity.iv_count == 3
    assert model.velocity.factor_count == 2
    assert velocity.shape == x_t.shape
    assert torch.isfinite(velocity).all()


def test_multihead_ar_rejects_invalid_iv_count() -> None:
    cfg = GenericMultiHeadStateConditionedMixedCoordinateFMConfig(
        history_len=2,
        future_len=1,
        n_cells=3,
        iv_count=4,
    )

    try:
        GenericMultiHeadStateConditionedMixedCoordinateFlowMatching(cfg)
    except ValueError as exc:
        assert "iv_count" in str(exc)
    else:
        raise AssertionError("expected iv_count validation failure")
