import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_state_conditioned_mixed_coordinate_flow_matching import (  # noqa: E402
    GenericStateConditionedMixedCoordinateFMConfig,
    GenericStateConditionedMixedCoordinateFlowMatching,
)
from experiments.backfill.block_ar.train_645a_local_conditional_distribution_finetune import (  # noqa: E402
    local_distribution_loss,
    local_neighbor_indices,
    run_loss_epoch,
)


def _tiny_model() -> GenericStateConditionedMixedCoordinateFlowMatching:
    cfg = GenericStateConditionedMixedCoordinateFMConfig(
        history_len=3,
        future_len=2,
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
        n_quantiles=21,
        conditioning_mode="prefix",
        prefix_feature_mode="scale",
        level_score_channels=[0, 1],
    )
    model = GenericStateConditionedMixedCoordinateFlowMatching(cfg)
    level_quantiles = torch.linspace(-1.0, 1.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    increment_quantiles = torch.linspace(-0.5, 0.5, cfg.n_quantiles).repeat(
        cfg.n_cells, 1
    )
    model.set_empirical_quantiles(level_quantiles, increment_quantiles)
    return model


def test_local_distribution_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(645)
    model = _tiny_model()
    cfg = model.cfg
    history_level = 0.1 * torch.randn(4, cfg.history_len, cfg.n_cells)
    history_increment = 0.05 * torch.randn(4, cfg.history_len, cfg.n_cells)
    future_increment = 0.05 * torch.randn(4, cfg.future_len, cfg.n_cells)
    future_level = history_level[:, -1:, :] + torch.cumsum(future_increment, dim=1)

    loss, metrics = local_distribution_loss(
        model,
        history_level,
        history_increment,
        future_level,
        future_increment,
        train_sample_count=2,
        rollout_flow_steps=1,
        rollout_steps=cfg.future_len,
        sample_temperature=1.0,
        k_neighbors=3,
        recent_steps=2,
        local_weight=0.05,
        fm_anchor_weight=1.0,
        n_projections=4,
        n_quantiles=4,
        horizon_end_weight=1.5,
    )
    loss.backward()

    grad_norm = sum(
        parameter.grad.abs().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["local_sw"])
    assert grad_norm > 0


def test_local_neighbor_indices_respects_requested_cap() -> None:
    torch.manual_seed(646)
    model = _tiny_model()
    cfg = model.cfg
    history_level = 0.1 * torch.randn(5, cfg.history_len, cfg.n_cells)
    history_increment = 0.05 * torch.randn(5, cfg.history_len, cfg.n_cells)

    idx = local_neighbor_indices(
        model,
        history_level,
        history_increment,
        k_neighbors=3,
        recent_steps=2,
    )

    assert idx.shape == (5, 3)
    assert idx.min() >= 0
    assert idx.max() < 5


def test_run_loss_epoch_smoke() -> None:
    torch.manual_seed(647)
    model = _tiny_model()
    cfg = model.cfg
    history_level = 0.1 * torch.randn(4, cfg.history_len, cfg.n_cells)
    history_increment = 0.05 * torch.randn(4, cfg.history_len, cfg.n_cells)
    future_increment = 0.05 * torch.randn(4, cfg.future_len, cfg.n_cells)
    future_level = history_level[:, -1:, :] + torch.cumsum(future_increment, dim=1)
    loader = DataLoader(
        TensorDataset(
            history_level,
            history_increment,
            future_level,
            future_increment,
        ),
        batch_size=4,
    )
    args = SimpleNamespace(
        train_sample_count=2,
        rollout_flow_steps=1,
        rollout_steps=cfg.future_len,
        train_sample_temperature=1.0,
        k_neighbors=3,
        recent_steps=2,
        local_weight=0.05,
        fm_anchor_weight=1.0,
        n_projections=4,
        n_quantiles=4,
        horizon_end_weight=1.5,
        clip_grad=1.0,
    )

    metrics = run_loss_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        max_batches=1,
        args=args,
    )

    assert metrics["total"] > 0
    assert metrics["local_sw"] > 0
