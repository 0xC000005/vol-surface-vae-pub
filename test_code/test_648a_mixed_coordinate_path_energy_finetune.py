import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_mixed_coordinate_path_flow_matching import (  # noqa: E402
    GenericMixedCoordinatePathFMConfig,
    GenericMixedCoordinatePathFlowMatching,
)
from experiments.backfill.block_ar.train_648a_mixed_coordinate_path_energy_finetune import (  # noqa: E402
    differentiable_mixed_path_samples,
    mixed_path_energy_loss,
)


def _tiny_model() -> GenericMixedCoordinatePathFlowMatching:
    cfg = GenericMixedCoordinatePathFMConfig(
        history_len=4,
        future_len=3,
        n_cells=4,
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


def test_differentiable_mixed_path_sampler_propagates_gradients() -> None:
    torch.manual_seed(648)
    model = _tiny_model()
    history_level = 0.2 * torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    history_increment = 0.05 * torch.randn(2, model.cfg.history_len, model.cfg.n_cells)

    samples = differentiable_mixed_path_samples(
        model,
        history_level,
        history_increment,
        n_samples=2,
        n_steps=model.cfg.future_len,
        flow_steps=2,
        temperature=1.0,
    )
    loss = samples.mean()
    loss.backward()
    grad_norm = sum(
        float(param.grad.detach().abs().sum())
        for param in model.parameters()
        if param.grad is not None
    )

    assert samples.shape == (2, 2, model.cfg.future_len, model.cfg.n_cells)
    assert torch.isfinite(samples).all()
    assert grad_norm > 0.0


def test_mixed_path_energy_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(649)
    model = _tiny_model()
    history_level = 0.2 * torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    history_increment = 0.05 * torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    future_increment = 0.05 * torch.randn(2, model.cfg.future_len, model.cfg.n_cells)
    future_level = history_level[:, -1:, :] + torch.cumsum(future_increment, dim=1)

    loss, metrics = mixed_path_energy_loss(
        model,
        history_level,
        history_increment,
        future_level,
        future_increment,
        train_sample_count=2,
        rollout_flow_steps=2,
        energy_weight=0.1,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.5,
        energy_eps=1e-6,
        temperature=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["energy"])
    assert torch.isfinite(metrics["fm_loss"])
    assert torch.isfinite(metrics["sample_mixed_std"])
