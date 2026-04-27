import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_state_conditioned_increment_flow_matching import (  # noqa: E402
    GenericStateConditionedIncrementFMConfig,
    GenericStateConditionedIncrementFlowMatching,
)
from experiments.backfill.block_ar.train_634a_native_joint_rollout_proper_score import (  # noqa: E402
    rollout_level_increment_scores_with_grad,
    rollout_proper_score_loss,
)


def _tiny_model() -> GenericStateConditionedIncrementFlowMatching:
    cfg = GenericStateConditionedIncrementFMConfig(
        history_len=4,
        future_len=2,
        n_cells=3,
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
    model = GenericStateConditionedIncrementFlowMatching(cfg)
    quantiles = torch.linspace(-3.0, 3.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    model.set_empirical_quantiles(quantiles, quantiles)
    return model


def test_differentiable_rollout_helper_propagates_gradients() -> None:
    torch.manual_seed(634)
    model = _tiny_model()
    history_level = torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    history_increment = 0.1 * torch.randn(2, model.cfg.history_len, model.cfg.n_cells)

    level_scores, increment_scores = rollout_level_increment_scores_with_grad(
        model,
        history_level,
        history_increment,
        n_samples=2,
        n_steps=model.cfg.future_len,
        flow_steps=2,
    )
    loss = level_scores.mean() + increment_scores.mean()
    loss.backward()

    grad_norm = sum(
        float(param.grad.detach().abs().sum())
        for param in model.parameters()
        if param.grad is not None
    )
    assert level_scores.shape == (2, 2, model.cfg.future_len, model.cfg.n_cells)
    assert increment_scores.shape == (2, 2, model.cfg.future_len, model.cfg.n_cells)
    assert torch.isfinite(level_scores).all()
    assert torch.isfinite(increment_scores).all()
    assert grad_norm > 0.0


def test_rollout_proper_score_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(635)
    model = _tiny_model()
    history_level = torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    history_increment = 0.1 * torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    future_increment = 0.1 * torch.randn(2, model.cfg.future_len, model.cfg.n_cells)
    base = history_level[:, -1:, :]
    future_level = base + torch.cumsum(future_increment, dim=1)

    loss, metrics = rollout_proper_score_loss(
        model,
        history_level,
        history_increment,
        future_level,
        future_increment,
        train_sample_count=2,
        rollout_flow_steps=2,
        rollout_weight=0.03,
        sw_weight=0.5,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.5,
        n_projections=4,
        n_quantiles=4,
        energy_eps=1e-6,
        temperature=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["energy"])
    assert torch.isfinite(metrics["sw"])
    assert torch.isfinite(metrics["fm_loss"])
