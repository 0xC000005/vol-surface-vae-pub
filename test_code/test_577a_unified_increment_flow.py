import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.train_577a_unified_increment_flow import (
    UnifiedIncrementFlow,
    UnifiedIncrementFlowConfig,
    fit_path_gaussian,
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
