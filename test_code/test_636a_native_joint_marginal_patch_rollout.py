import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.generic_state_conditioned_increment_flow_matching import (  # noqa: E402
    GenericStateConditionedIncrementFMConfig,
    GenericStateConditionedIncrementFlowMatching,
)
from experiments.backfill.block_ar.train_636a_native_joint_marginal_patch_rollout import (  # noqa: E402
    marginal_crps_score,
    rollout_marginal_patch_loss,
    short_patch_energy_score,
)


def _tiny_model() -> GenericStateConditionedIncrementFlowMatching:
    cfg = GenericStateConditionedIncrementFMConfig(
        history_len=4,
        future_len=3,
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


def test_marginal_crps_and_patch_energy_prefer_aligned_samples() -> None:
    torch.manual_seed(636)
    target = torch.randn(4, 3, 2)
    good = target[:, None] + 0.05 * torch.randn(4, 5, 3, 2)
    bad = good + 1.0

    good_crps, *_ = marginal_crps_score(good, target)
    bad_crps, *_ = marginal_crps_score(bad, target)
    good_patch, *_ = short_patch_energy_score(good, target, patch_len=2, eps=1e-6)
    bad_patch, *_ = short_patch_energy_score(bad, target, patch_len=2, eps=1e-6)

    assert torch.isfinite(good_crps)
    assert torch.isfinite(good_patch)
    assert good_crps < bad_crps
    assert good_patch < bad_patch


def test_rollout_marginal_patch_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(637)
    model = _tiny_model()
    history_level = torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    history_increment = 0.1 * torch.randn(2, model.cfg.history_len, model.cfg.n_cells)
    future_increment = 0.1 * torch.randn(2, model.cfg.future_len, model.cfg.n_cells)
    future_level = history_level[:, -1:, :] + torch.cumsum(future_increment, dim=1)

    loss, metrics = rollout_marginal_patch_loss(
        model,
        history_level,
        history_increment,
        future_level,
        future_increment,
        train_sample_count=2,
        rollout_flow_steps=2,
        marginal_weight=0.05,
        increment_weight=0.5,
        patch_weight=0.01,
        fm_anchor_weight=1.0,
        patch_len=2,
        energy_eps=1e-6,
        temperature=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["level_crps"])
    assert torch.isfinite(metrics["increment_crps"])
    assert torch.isfinite(metrics["patch_energy"])
