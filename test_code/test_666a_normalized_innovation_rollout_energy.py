import math
import torch
import sys

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    enable_conditional_base_noise_scale,
)
from experiments.backfill.block_ar.train_666a_normalized_innovation_rollout_energy_finetune import (
    dispersion_calibration_loss,
    differentiable_normalized_rollout_samples,
    effective_readout_iv_count,
    marginal_crps_path_score,
    normalized_rollout_energy_loss,
    standardized_level_delta_paths,
    structured_variogram_path_score,
)


def _tiny_model(risk_state_dim: int = 0) -> GenericStateAwareNormalizedInnovationFlowMatching:
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
        risk_state_dim=int(risk_state_dim),
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


def test_effective_readout_iv_count_uses_scope_semantics():
    assert effective_readout_iv_count("iv_only", n_cells=25, iv_count=25) == 25
    assert effective_readout_iv_count("anchor_only", n_cells=13, iv_count=25) == 0
    assert effective_readout_iv_count("joint38", n_cells=38, iv_count=25) == 25


def test_standardized_level_delta_paths_are_unit_free_from_last_history_level():
    history_level = torch.tensor([[[10.0, 100.0], [11.0, 98.0]]])
    sampled_level = torch.tensor([[[[12.0, 101.0], [14.0, 92.0]]]])
    target_level = torch.tensor([[[13.0, 96.0], [15.0, 88.0]]])
    scale = torch.tensor([[0.5, 2.0]])

    sampled_delta, target_delta = standardized_level_delta_paths(
        sampled_level,
        target_level,
        history_level,
        scale,
    )

    torch.testing.assert_close(sampled_delta, torch.tensor([[[[2.0, 1.5], [6.0, -3.0]]]]))
    torch.testing.assert_close(target_delta, torch.tensor([[[4.0, -1.0], [8.0, -5.0]]]))


def test_marginal_crps_path_score_matches_two_member_crps():
    samples = torch.tensor([[[[-1.0]], [[1.0]]]])
    target = torch.tensor([[[0.0]]])

    score, target_dist, pair_dist = marginal_crps_path_score(samples, target)

    torch.testing.assert_close(score, torch.tensor(0.5))
    torch.testing.assert_close(target_dist, torch.tensor(1.0))
    torch.testing.assert_close(pair_dist, torch.tensor(1.0))


def test_structured_variogram_path_score_matches_single_pair():
    samples = torch.tensor([[[[0.0, 2.0]]]])
    target = torch.tensor([[[0.0, 1.0]]])

    score = structured_variogram_path_score(samples, target, power=1.0)

    torch.testing.assert_close(score, torch.tensor(1.0))


def test_dispersion_calibration_penalizes_flat_underdispersed_spread():
    target = torch.tensor(
        [
            [[2.0], [2.0]],
            [[0.5], [0.5]],
        ]
    )
    matched_samples = torch.tensor(
        [
            [[[2.0], [2.0]], [[-2.0], [-2.0]]],
            [[[0.5], [0.5]], [[-0.5], [-0.5]]],
        ]
    )
    flat_samples = torch.zeros_like(matched_samples) + 0.1

    matched_loss, *_ = dispersion_calibration_loss(matched_samples, target)
    flat_loss, *_ = dispersion_calibration_loss(flat_samples, target)

    assert matched_loss < flat_loss


def test_dispersion_calibration_supports_channel_mode():
    target = torch.tensor(
        [
            [[2.0, 0.5], [2.0, 0.5]],
            [[0.5, 2.0], [0.5, 2.0]],
        ]
    )
    matched_samples = torch.tensor(
        [
            [[[2.0, 0.5], [2.0, 0.5]], [[-2.0, -0.5], [-2.0, -0.5]]],
            [[[0.5, 2.0], [0.5, 2.0]], [[-0.5, -2.0], [-0.5, -2.0]]],
        ]
    )

    loss, rank_mse, global_log_mse, spread_mean, corr = dispersion_calibration_loss(
        matched_samples,
        target,
        mode="channel",
    )

    assert torch.isfinite(loss)
    assert rank_mse >= 0.0
    assert global_log_mse >= 0.0
    assert spread_mean > 0.0
    assert corr > 0.0


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


def test_differentiable_rollout_uses_conditional_base_noise_scale():
    torch.manual_seed(89)
    model = _tiny_model()
    history_level, history_norm, _future_level, _future_norm, center, scale = _batch(model)

    torch.manual_seed(97)
    unit_samples = differentiable_normalized_rollout_samples(
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
    enable_conditional_base_noise_scale(model, scale_min=0.25, scale_max=4.0)
    with torch.no_grad():
        model.base_noise_log_scale[-1].bias.fill_(math.log(0.5))

    torch.manual_seed(97)
    scaled_samples = differentiable_normalized_rollout_samples(
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

    assert not torch.allclose(scaled_samples, unit_samples)


def test_differentiable_rollout_uses_risk_context():
    torch.manual_seed(91)
    model = _tiny_model(risk_state_dim=2)
    history_level, history_norm, _future_level, _future_norm, center, scale = _batch(model)

    torch.manual_seed(103)
    base_samples = differentiable_normalized_rollout_samples(
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
    with torch.no_grad():
        model.risk_context_proj[-1].bias.fill_(0.5)

    torch.manual_seed(103)
    risk_samples = differentiable_normalized_rollout_samples(
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

    assert not torch.allclose(risk_samples, base_samples)


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


def test_normalized_rollout_energy_loss_can_score_channel_balanced_level_paths():
    torch.manual_seed(31)
    model = _tiny_model()
    history_level, history_norm, future_level, future_norm, center, scale = _batch(model)

    torch.manual_seed(37)
    loss_without_channel, metrics_without_channel = normalized_rollout_energy_loss(
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
        channel_level_energy_weight=0.0,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=1.0,
    )
    torch.manual_seed(37)
    loss_with_channel, metrics_with_channel = normalized_rollout_energy_loss(
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
        channel_level_energy_weight=0.1,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=1.0,
    )

    assert torch.isfinite(loss_with_channel)
    assert metrics_without_channel["channel_level_energy"].item() == 0.0
    assert metrics_with_channel["channel_level_energy"].item() > 0.0
    assert loss_with_channel > loss_without_channel


def test_normalized_rollout_energy_loss_can_contrast_shuffled_history_rollouts():
    torch.manual_seed(41)
    model = _tiny_model()
    history_level, history_norm, future_level, future_norm, center, scale = _batch(model)

    torch.manual_seed(43)
    loss_without_contrast, metrics_without_contrast = normalized_rollout_energy_loss(
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
    torch.manual_seed(43)
    loss_with_contrast, metrics_with_contrast = normalized_rollout_energy_loss(
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
        condition_rollout_contrast_weight=0.1,
        condition_rollout_contrast_margin=0.0,
    )

    assert torch.isfinite(loss_with_contrast)
    assert metrics_without_contrast["condition_rollout_contrast"].item() == 0.0
    assert metrics_with_contrast["condition_rollout_contrast"].item() > 0.0
    assert loss_with_contrast > loss_without_contrast


def test_normalized_rollout_energy_loss_can_use_nearest_history_hard_negatives():
    torch.manual_seed(47)
    model = _tiny_model()
    history_level, history_norm, future_level, future_norm, center, scale = _batch(model)

    torch.manual_seed(53)
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
        condition_rollout_contrast_weight=0.1,
        condition_rollout_contrast_margin=0.0,
        condition_rollout_negative_mode="nearest_history",
    )

    assert torch.isfinite(loss)
    assert metrics["condition_rollout_contrast"].item() > 0.0


def test_normalized_rollout_energy_loss_can_use_dispersion_calibration():
    torch.manual_seed(59)
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
        dispersion_calibration_weight=0.1,
        dispersion_calibration_mode="channel",
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=1.0,
    )

    assert torch.isfinite(loss)
    assert metrics["dispersion_calibration"].item() >= 0.0
    assert metrics["dispersion_spread_target_ratio"].item() >= 0.0
