import numpy as np
import sys
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    enable_group_residual_velocity_readout,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (
    UnifiedVariableSpec,
    encode_state,
)
from experiments.backfill.block_ar.normalized_innovation_662_utils import (
    normalize_increment_windows,
    reconstruct_state_from_normalized_increments,
)


def test_normalize_increment_windows_roundtrips_future_state():
    specs = [
        UnifiedVariableSpec("factor:eq", "factor:eq", 0, "log_level"),
        UnifiedVariableSpec("factor:rate", "factor:rate", 1, "diff_level"),
    ]
    history_raw = np.array(
        [
            [
                [100.0, 1.0],
                [101.0, 1.1],
                [103.0, 1.0],
                [102.0, 1.2],
            ]
        ],
        dtype=np.float32,
    )
    future_raw = np.array(
        [
            [
                [104.0, 1.3],
                [106.0, 1.1],
            ]
        ],
        dtype=np.float32,
    )
    encoded_path = encode_state(np.concatenate([history_raw, future_raw], axis=1), specs)
    increments = np.diff(encoded_path, axis=1).astype(np.float32)
    history_increment = increments[:, : history_raw.shape[1]]
    history_increment[:, 0] = 0.0
    future_increment = increments[:, history_raw.shape[1] - 1 :]

    history_norm, future_norm, center, scale = normalize_increment_windows(
        history_increment,
        future_increment,
        scale_floor=1e-6,
    )
    reconstructed = reconstruct_state_from_normalized_increments(
        history_raw[:, -1],
        future_norm,
        center,
        scale,
        specs,
    )

    assert history_norm.shape == history_increment.shape
    assert np.all(scale > 0.0)
    np.testing.assert_allclose(reconstructed, future_raw, rtol=1e-5, atol=1e-5)


def test_ewma_mean_center_uses_history_only_signed_drift():
    history_increment = np.array(
        [
            [
                [0.0, 0.0],
                [1.0, -2.0],
                [3.0, -4.0],
            ]
        ],
        dtype=np.float32,
    )
    future_increment = np.array([[[5.0, -6.0]]], dtype=np.float32)

    history_norm, future_norm, center, scale = normalize_increment_windows(
        history_increment,
        future_increment,
        half_life=1.0,
        scale_floor=1e-6,
        center_mode="ewma_mean",
    )

    ages = np.array([2.0, 1.0, 0.0])
    weights = np.power(0.5, ages)
    weights = weights / weights.sum()
    expected_center = np.sum(weights[None, :, None] * history_increment, axis=1)
    expected_scale = np.sqrt(np.sum(weights[None, :, None] * history_increment * history_increment, axis=1))

    np.testing.assert_allclose(center, expected_center.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scale, expected_scale.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(history_norm, (history_increment - center[:, None, :]) / scale[:, None, :])
    np.testing.assert_allclose(future_norm, (future_increment - center[:, None, :]) / scale[:, None, :])


def test_state_aware_normalized_innovation_flow_loss_and_sampling():
    torch.manual_seed(7)
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
    model.eval()
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    level_quantiles = torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells)
    model.set_level_quantiles(level_quantiles, levels)

    history_level = torch.randn(5, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(5, cfg.history_len, cfg.n_cells)
    center = torch.randn(5, cfg.n_cells) * 0.01
    scale = torch.rand(5, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(5, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)

    loss, metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
    )
    loss.backward()
    samples = model.sample_batched(
        history_level,
        history_norm,
        center,
        scale,
        n_samples=3,
        n_steps=cfg.future_len,
        chunk_size=2,
    )

    assert torch.isfinite(loss)
    assert metrics["target_norm_std"] > 0.0
    assert samples.shape == (5, 3, cfg.future_len, cfg.n_cells)
    assert torch.isfinite(samples).all()


def test_scale_drift_prefix_conditions_on_drift_without_centering_increment():
    torch.manual_seed(17)
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
        prefix_feature_mode="scale_drift",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    model.eval()
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    level_quantiles = torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells)
    model.set_level_quantiles(level_quantiles, levels)

    history_level = torch.randn(5, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(5, cfg.history_len, cfg.n_cells)
    center = torch.zeros(5, cfg.n_cells)
    drift_feature = torch.randn(5, cfg.n_cells) * 0.01
    scale = torch.rand(5, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(5, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)

    loss, metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        drift_feature=drift_feature,
    )
    samples = model.sample_batched(
        history_level,
        history_norm,
        center,
        scale,
        drift_feature=drift_feature,
        n_samples=2,
        n_steps=cfg.future_len,
        chunk_size=2,
    )

    assert torch.isfinite(loss)
    assert metrics["drift_feature_abs"] > 0.0
    assert samples.shape == (5, 2, cfg.future_len, cfg.n_cells)


def test_condition_contrast_loss_adds_ranking_penalty():
    torch.manual_seed(11)
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

    history_level = torch.randn(5, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(5, cfg.history_len, cfg.n_cells)
    center = torch.randn(5, cfg.n_cells) * 0.01
    scale = torch.rand(5, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(5, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)

    loss, metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        condition_contrast_weight=0.3,
        condition_contrast_margin=0.1,
    )

    assert torch.isfinite(loss)
    assert metrics["condition_contrast_loss"] >= 0.0
    assert metrics["condition_contrast_weight"] == 0.3
    assert torch.isclose(loss.detach(), metrics["fm_loss"] + 0.3 * metrics["condition_contrast_loss"])


def test_group_residual_velocity_readout_starts_as_noop():
    torch.manual_seed(13)
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
        history_len=4,
        future_len=3,
        n_cells=5,
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
    model.eval()
    x_t = torch.randn(4, cfg.n_cells)
    current_level = torch.randn(4, cfg.n_cells)
    memory_state = torch.randn(4, cfg.memory_dim)
    t = torch.rand(4)

    before = model.velocity(x_t, current_level, memory_state, t)
    enable_group_residual_velocity_readout(model, iv_count=3)
    after = model.velocity(x_t, current_level, memory_state, t)

    assert model.cfg.velocity_readout_mode == "group_residual"
    assert model.cfg.readout_iv_count == 3
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)
