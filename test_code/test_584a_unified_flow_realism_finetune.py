import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.train_577a_unified_increment_flow import (
    UnifiedIncrementFlow,
    UnifiedIncrementFlowConfig,
)
from experiments.backfill.block_ar.train_584a_unified_flow_realism_finetune import (
    differentiable_sample,
    range_and_coverage_loss,
)


def test_range_and_coverage_loss_is_finite_and_differentiable() -> None:
    samples = torch.randn(2, 4, 3, 5, requires_grad=True)
    target = torch.zeros(2, 3, 5)
    encoded_min = torch.full((5,), -1.0)
    encoded_max = torch.full((5,), 1.0)

    loss, metrics = range_and_coverage_loss(
        samples,
        target,
        encoded_min=encoded_min,
        encoded_max=encoded_max,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["encoded_coverage"])
    assert samples.grad is not None


def test_differentiable_sample_preserves_shape() -> None:
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

    samples = differentiable_sample(model, history, n_samples=3, n_steps=2)

    assert samples.shape == (2, 3, 3, 5)
    assert torch.isfinite(samples).all()
