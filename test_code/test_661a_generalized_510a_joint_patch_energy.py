import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_empirical_score_transition_flow_matching import (  # noqa: E402
    GenericEmpiricalScoreTransitionFMConfig,
    GenericEmpiricalScoreTransitionFlowMatching,
)
from experiments.backfill.block_ar.train_661a_generalized_510a_joint_patch_energy import (  # noqa: E402
    combined_loss,
    patch_energy_score,
    sample_rollout_scores_with_grad,
)


def _tiny_model(n_vars: int = 6) -> GenericEmpiricalScoreTransitionFlowMatching:
    cfg = GenericEmpiricalScoreTransitionFMConfig(
        history_len=4,
        future_len=3,
        n_cells=n_vars,
        n_quantiles=31,
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
        model_dropout=0.0,
        prefix_feature_mode="scale",
        conditioning_mode="prefix",
    )
    model = GenericEmpiricalScoreTransitionFlowMatching(cfg)
    base = torch.linspace(-2.0, 2.0, cfg.n_quantiles)
    quantiles = torch.stack([base + 0.05 * i for i in range(n_vars)], dim=0)
    model.set_empirical_quantiles(quantiles)
    return model


def test_patch_energy_score_prefers_aligned_samples() -> None:
    target = torch.randn(3, 4, 5)
    good = target[:, None] + 0.01 * torch.randn(3, 6, 4, 5)
    bad = target[:, None] + 1.0 + 0.01 * torch.randn(3, 6, 4, 5)

    good_score, *_ = patch_energy_score(good, target, patch_len=2, eps=1e-6)
    bad_score, *_ = patch_energy_score(bad, target, patch_len=2, eps=1e-6)

    assert good_score < bad_score


def test_score_rollout_with_grad_shape_and_finiteness() -> None:
    model = _tiny_model(n_vars=5)
    history = torch.randn(2, 4, 5).clamp(-1.5, 1.5)

    samples = sample_rollout_scores_with_grad(
        model,
        history,
        n_samples=3,
        n_steps=3,
        flow_steps=2,
        temperature=1.0,
    )

    assert samples.shape == (2, 3, 3, 5)
    assert torch.isfinite(samples).all()


def test_combined_loss_is_finite_and_differentiable() -> None:
    model = _tiny_model(n_vars=6)
    history = torch.randn(2, 4, 6).clamp(-1.5, 1.5)
    future = torch.randn(2, 3, 6).clamp(-1.5, 1.5)

    loss, metrics = combined_loss(
        model,
        history,
        future,
        train_sample_count=3,
        rollout_flow_steps=2,
        patch_len=2,
        patch_energy_weight=0.05,
        fm_anchor_weight=1.0,
        energy_eps=1e-6,
        temperature=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["patch_energy"])
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
