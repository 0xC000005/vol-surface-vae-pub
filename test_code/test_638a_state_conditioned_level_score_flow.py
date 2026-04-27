import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_state_conditioned_level_score_flow_matching import (  # noqa: E402
    GenericStateConditionedLevelScoreFMConfig,
    GenericStateConditionedLevelScoreFlowMatching,
)


def test_state_conditioned_level_score_loss_and_sampling_are_finite() -> None:
    torch.manual_seed(638)
    cfg = GenericStateConditionedLevelScoreFMConfig(
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
    )
    model = GenericStateConditionedLevelScoreFlowMatching(cfg)
    quantiles = torch.linspace(-2.0, 2.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    levels = (
        torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5
    ) / cfg.n_quantiles
    model.set_empirical_quantiles(quantiles, quantiles, levels)

    history_state = torch.randn(2, cfg.history_len, cfg.n_cells)
    history_increment = 0.1 * torch.randn(2, cfg.history_len, cfg.n_cells)
    future_increment = 0.1 * torch.randn(2, cfg.future_len, cfg.n_cells)
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


def test_state_conditioned_level_score_sampling_respects_level_quantile_support() -> (
    None
):
    torch.manual_seed(639)
    cfg = GenericStateConditionedLevelScoreFMConfig(
        history_len=3,
        future_len=2,
        n_cells=2,
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
        n_quantiles=21,
        conditioning_mode="prefix",
    )
    model = GenericStateConditionedLevelScoreFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    model.set_empirical_quantiles(level_quantiles, increment_quantiles)
    history_state = torch.zeros(2, cfg.history_len, cfg.n_cells)
    history_increment = torch.zeros(2, cfg.history_len, cfg.n_cells)

    increments = model.sample_batched(
        history_state, history_increment, n_samples=2, n_steps=cfg.future_len
    )
    levels = history_state[:, None, -1:, :] + torch.cumsum(increments, dim=2)

    assert levels.min() >= -1.0001
    assert levels.max() <= 1.0001
