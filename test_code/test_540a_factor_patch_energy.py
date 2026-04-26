import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_540a_factor_patch_energy_finetune import (
    combined_factor_patch_loss,
    sample_rollout_scores_with_grad_factor,
)


def _toy_factor_model() -> EmpiricalNormalScoreCausalMemoryTransitionFlowMatching:
    cfg = EmpiricalNormalScoreCausalMemoryTransitionFMConfig(
        history_len=3,
        future_len=4,
        n_cells=5,
        n_quantiles=19,
        factor_dim=2,
        memory_dim=16,
        memory_layers=1,
        memory_heads=2,
        memory_ff=32,
        token_dim=16,
        token_layers=1,
        token_heads=2,
        token_ff=32,
        time_dim=8,
        model_dropout=0.0,
        flow_steps=2,
    )
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(cfg)
    levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / cfg.n_quantiles
    quantiles = torch.stack(
        [torch.linspace(0.05 + 0.01 * i, 0.95 - 0.01 * i, cfg.n_quantiles) for i in range(cfg.n_cells)],
        dim=0,
    )
    model.set_empirical_quantiles(quantiles, levels)
    return model


def test_factor_rollout_scores_are_differentiable_and_shape_stable() -> None:
    torch.manual_seed(0)
    model = _toy_factor_model()
    history_01 = torch.rand(3, 3, 5)
    history_norm = normalize_iv(history_01)
    factor_history = torch.randn(3, 3, 2)

    samples = sample_rollout_scores_with_grad_factor(
        model=model,
        history_norm=history_norm,
        factor_history=factor_history,
        n_samples=2,
        n_steps=4,
        flow_steps=2,
    )

    assert samples.shape == (3, 2, 4, 5)
    assert torch.isfinite(samples).all()
    assert samples.requires_grad


def test_combined_factor_patch_loss_is_finite() -> None:
    torch.manual_seed(1)
    model = _toy_factor_model()
    history_01 = torch.rand(4, 3, 5)
    future_01 = torch.rand(4, 4, 5)
    factor_history = torch.randn(4, 3, 2)

    loss, metrics = combined_factor_patch_loss(
        model=model,
        history_norm=normalize_iv(history_01),
        future_norm=normalize_iv(future_01),
        factor_history=factor_history,
        train_sample_count=2,
        rollout_flow_steps=2,
        patch_len=2,
        patch_energy_weight=0.05,
        fm_anchor_weight=1.0,
        energy_eps=1e-6,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["fm_loss"])
    assert torch.isfinite(metrics["patch_energy"])
    assert metrics["sample_score_std"] > 0

