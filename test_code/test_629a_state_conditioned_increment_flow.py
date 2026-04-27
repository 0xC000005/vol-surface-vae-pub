import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_state_conditioned_increment_flow_matching import (  # noqa: E402
    GenericStateConditionedIncrementFMConfig,
    GenericStateConditionedIncrementFlowMatching,
)


def test_state_conditioned_increment_loss_and_sampling_are_finite() -> None:
    torch.manual_seed(629)
    cfg = GenericStateConditionedIncrementFMConfig(
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
    )
    model = GenericStateConditionedIncrementFlowMatching(cfg)
    quantiles = torch.linspace(-2.0, 2.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / cfg.n_quantiles
    model.set_empirical_quantiles(quantiles, quantiles, levels)

    history_state = torch.randn(2, cfg.history_len, cfg.n_cells)
    history_increment = 0.1 * torch.randn(2, cfg.history_len, cfg.n_cells)
    future_state = torch.randn(2, cfg.future_len, cfg.n_cells)
    future_increment = 0.1 * torch.randn(2, cfg.future_len, cfg.n_cells)

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


def test_state_conditioned_increment_quantiles_validate_shape() -> None:
    cfg = GenericStateConditionedIncrementFMConfig(history_len=2, future_len=1, n_cells=3, n_quantiles=11)
    model = GenericStateConditionedIncrementFlowMatching(cfg)
    bad_quantiles = torch.zeros(cfg.n_cells + 1, cfg.n_quantiles)

    try:
        model.set_empirical_quantiles(bad_quantiles, bad_quantiles)
    except ValueError as exc:
        assert "level_quantiles" in str(exc)
    else:
        raise AssertionError("expected quantile shape validation failure")
