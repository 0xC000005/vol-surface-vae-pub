import torch
import sys

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
)
from experiments.backfill.block_ar.train_666a_normalized_innovation_rollout_energy_finetune import (
    differentiable_normalized_rollout_samples,
    normalized_rollout_energy_loss,
)


def _tiny_model() -> GenericStateAwareNormalizedInnovationFlowMatching:
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
        history_len=4,
        future_len=3,
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
        n_quantiles=17,
        prefix_feature_mode="scale",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    level_quantiles = torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells)
    model.set_level_quantiles(level_quantiles, levels)
    return model


def _batch(model: GenericStateAwareNormalizedInnovationFlowMatching):
    cfg = model.cfg
    history_level = torch.randn(5, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(5, cfg.history_len, cfg.n_cells)
    center = torch.randn(5, cfg.n_cells) * 0.01
    scale = torch.rand(5, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(5, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)
    return history_level, history_norm, future_level, future_norm, center, scale


def test_differentiable_normalized_rollout_samples_backpropagates():
    torch.manual_seed(17)
    model = _tiny_model()
    history_level, history_norm, _future_level, _future_norm, center, scale = _batch(model)

    samples = differentiable_normalized_rollout_samples(
        model,
        history_level,
        history_norm,
        center,
        scale,
        n_samples=2,
        n_steps=3,
        flow_steps=2,
        temperature=1.0,
    )
    loss = samples.mean()
    loss.backward()

    assert samples.shape == (5, 2, 3, 2)
    assert torch.isfinite(samples).all()
    assert any(param.grad is not None for param in model.parameters())


def test_normalized_rollout_energy_loss_is_finite():
    torch.manual_seed(19)
    model = _tiny_model()
    history_level, history_norm, future_level, future_norm, center, scale = _batch(model)

    loss, metrics = normalized_rollout_energy_loss(
        model,
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        train_sample_count=2,
        rollout_flow_steps=2,
        energy_weight=0.2,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=1.0,
    )

    assert torch.isfinite(loss)
    assert metrics["energy"] >= 0.0
    assert metrics["sample_norm_std"] > 0.0


def test_normalized_rollout_energy_loss_can_score_level_paths():
    torch.manual_seed(23)
    model = _tiny_model()
    history_level, history_norm, future_level, future_norm, center, scale = _batch(model)

    torch.manual_seed(29)
    loss_without_level, metrics_without_level = normalized_rollout_energy_loss(
        model,
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        train_sample_count=2,
        rollout_flow_steps=2,
        energy_weight=0.2,
        level_energy_weight=0.0,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=1.0,
    )
    torch.manual_seed(29)
    loss_with_level, metrics_with_level = normalized_rollout_energy_loss(
        model,
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        train_sample_count=2,
        rollout_flow_steps=2,
        energy_weight=0.2,
        level_energy_weight=0.1,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=1.0,
    )

    assert torch.isfinite(loss_with_level)
    assert metrics_without_level["level_energy"].item() == 0.0
    assert metrics_with_level["level_energy"].item() > 0.0
    assert loss_with_level > loss_without_level
