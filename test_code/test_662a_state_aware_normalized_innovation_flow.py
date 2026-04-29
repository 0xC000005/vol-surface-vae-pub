import numpy as np
import pytest
import sys
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    enable_conditional_base_noise_scale,
    enable_group_head_velocity_readout,
    enable_group_residual_velocity_readout,
    enable_prefix_feature_mode,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (
    UnifiedVariableSpec,
    encode_state,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (
    IncrementCoordinateBlock,
)
from experiments.backfill.block_ar.normalized_innovation_662_utils import (
    normalize_increment_windows,
    reconstruct_state_from_normalized_increments,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (
    tail_asinh_scale_from_future_norm,
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


def test_future_element_weight_masks_flow_loss_terms():
    torch.manual_seed(724)
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
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    model.set_level_quantiles(torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells), levels)

    history_level = torch.randn(4, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(4, cfg.history_len, cfg.n_cells)
    center = torch.randn(4, cfg.n_cells) * 0.01
    scale = torch.rand(4, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(4, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)
    future_weight = torch.ones_like(future_norm)
    future_weight[..., 1] = 0.0

    loss, metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        future_element_weight=future_weight,
    )

    assert torch.isfinite(loss)
    assert torch.isclose(metrics["future_element_weight_mean"], torch.tensor(0.5))
    with pytest.raises(ValueError, match="future_element_weight must have shape"):
        model.training_loss(
            history_level,
            history_norm,
            future_level,
            future_norm,
            center,
            scale,
            future_element_weight=future_weight[:, :, :1],
        )


def test_mixed_support_no_update_loss_uses_selected_channels_only():
    torch.manual_seed(735)
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
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
        n_quantiles=17,
        mixed_support_observation="bernoulli_no_update",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    model.set_level_quantiles(torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells), levels)
    model.set_mixed_support_no_update(
        torch.tensor([False, True, False]),
        torch.tensor([0.01, 0.75, 0.01]),
    )

    history_level = torch.randn(4, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(4, cfg.history_len, cfg.n_cells)
    center = torch.randn(4, cfg.n_cells) * 0.01
    scale = torch.rand(4, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(4, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)
    no_update_target = torch.zeros_like(future_norm)
    no_update_target[:, ::2, 1] = 1.0
    no_update_target[..., 0] = 1.0

    loss, metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        future_no_update_target=no_update_target,
        mixed_support_weight=0.7,
    )

    assert torch.isfinite(loss)
    assert metrics["mixed_support_enabled"].item() == 1.0
    assert metrics["mixed_support_selected_count"].item() == 1.0
    assert torch.isclose(metrics["mixed_support_target_rate"], torch.tensor(2.0 / 3.0))
    assert metrics["mixed_support_bce_loss"] > 0.0
    torch.testing.assert_close(
        loss.detach(),
        metrics["fm_loss"] + 0.7 * metrics["mixed_support_bce_loss"],
    )


def test_mixed_support_sampling_emits_exact_no_update_for_selected_channels():
    torch.manual_seed(736)
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
        flow_steps=1,
        n_quantiles=17,
        mixed_support_observation="bernoulli_no_update",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    model.set_level_quantiles(torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells), levels)
    model.set_mixed_support_no_update(
        torch.tensor([False, True]),
        torch.tensor([0.01, 1.0]),
    )

    history_level = torch.randn(3, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(3, cfg.history_len, cfg.n_cells)
    center = torch.randn(3, cfg.n_cells) * 0.01
    scale = torch.rand(3, cfg.n_cells) * 0.05 + 0.01
    samples = model.sample_batched(
        history_level,
        history_norm,
        center,
        scale,
        n_samples=4,
        n_steps=cfg.future_len,
        chunk_size=2,
    )

    assert samples.shape == (3, 4, cfg.future_len, cfg.n_cells)
    torch.testing.assert_close(samples[..., 1], torch.zeros_like(samples[..., 1]))


def test_innovation_score_coordinate_roundtrips_normalized_innovations():
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
        innovation_coordinate="score",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    quantiles = torch.stack(
        [
            torch.linspace(-4.0, 4.0, cfg.n_quantiles),
            torch.linspace(-2.0, 2.0, cfg.n_quantiles),
        ]
    )
    model.set_innovation_quantiles(quantiles, levels)

    values = torch.tensor(
        [
            [[-2.0, -1.0], [0.0, 0.0], [2.0, 1.0]],
            [[-1.5, -0.5], [1.5, 0.5], [3.0, 1.5]],
        ]
    )
    scores = model.normalized_innovations_to_scores(values)
    recovered = model.scores_to_normalized_innovations(scores)

    assert scores.shape == values.shape
    torch.testing.assert_close(recovered, values, atol=1e-5, rtol=1e-5)


def test_hybrid_sticky_score_coordinate_only_scores_masked_channels():
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
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
        n_quantiles=17,
        innovation_coordinate="hybrid_sticky_score",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    quantiles = torch.stack(
        [
            torch.linspace(-4.0, 4.0, cfg.n_quantiles),
            torch.linspace(-2.0, 2.0, cfg.n_quantiles),
            torch.linspace(-1.0, 1.0, cfg.n_quantiles),
        ]
    )
    model.set_innovation_quantiles(quantiles, levels)
    model.set_innovation_score_mask(torch.tensor([False, True, False]))

    values = torch.tensor([[[0.25, 0.50, -0.25], [1.0, -0.5, 0.0]]])
    flow = model._to_flow_coordinate(values)
    recovered = model._from_flow_coordinate(flow)

    torch.testing.assert_close(flow[..., 0], values[..., 0])
    torch.testing.assert_close(flow[..., 2], values[..., 2])
    assert not torch.allclose(flow[..., 1], values[..., 1])
    torch.testing.assert_close(recovered, values, atol=1e-5, rtol=1e-5)


def test_hybrid_tail_asinh_coordinate_only_compresses_masked_channels():
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
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
        innovation_coordinate="hybrid_tail_asinh",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    model.set_innovation_tail_asinh(
        torch.tensor([False, True, False]),
        torch.tensor([1.0, 2.0, 1.0]),
    )

    values = torch.tensor([[[0.25, 4.00, -0.25], [1.0, -2.0, 0.0]]])
    flow = model._to_flow_coordinate(values)
    recovered = model._from_flow_coordinate(flow)

    torch.testing.assert_close(flow[..., 0], values[..., 0])
    torch.testing.assert_close(flow[..., 2], values[..., 2])
    torch.testing.assert_close(flow[..., 1], torch.asinh(values[..., 1] / 2.0))
    torch.testing.assert_close(recovered, values, atol=1e-5, rtol=1e-5)


def test_tail_asinh_scale_from_future_norm_uses_generic_nonzero_updates():
    specs = [
        UnifiedVariableSpec("factor:a", "factor:a", 0, "diff_level"),
        UnifiedVariableSpec("factor:b", "factor:b", 1, "diff_level"),
        UnifiedVariableSpec("factor:c", "factor:c", 2, "diff_level"),
    ]
    history = np.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    future = np.array(
        [
            [[1.0, 3.0, 1.5], [1.0, 3.0, 2.0], [1.0, 5.0, 2.5]],
            [[2.0, 4.0, 2.5], [2.0, 4.0, 3.0], [2.0, 6.0, 3.5]],
        ],
        dtype=np.float32,
    )
    block = IncrementCoordinateBlock(
        history_increment=np.zeros_like(history),
        future_increment=np.zeros_like(future),
        history_state=history,
        future_state=future,
        indices=np.arange(history.shape[0]),
        specs=specs,
    )
    future_norm = np.array(
        [
            [[0.0, 4.0, 1.0], [0.0, 0.0, 1.5], [0.0, 2.0, 2.0]],
            [[0.0, 3.0, 1.0], [0.0, 0.0, 1.5], [0.0, 1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([True, True, False])

    scale, report = tail_asinh_scale_from_future_norm(
        block,
        "anchor_only",
        0,
        sticky_mask=mask,
        future_norm=future_norm,
        zero_eps=1e-10,
        scale_quantile=0.5,
        min_scale=0.5,
    )

    np.testing.assert_allclose(scale, np.array([1.0, 2.5, 1.0], dtype=np.float32))
    assert report["selected_names"] == ["factor:a", "factor:b"]
    assert report["rows"][0]["nonzero_count"] == 0
    assert report["rows"][1]["nonzero_count"] == 4
    assert report["rows"][2]["selected"] is False


def test_innovation_score_flow_loss_and_sampling_reconstructs_normalized_increments():
    torch.manual_seed(107)
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
        innovation_coordinate="score",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    model.set_level_quantiles(torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells), levels)
    model.set_innovation_quantiles(torch.stack([torch.linspace(-4.0, 4.0, cfg.n_quantiles)] * cfg.n_cells), levels)

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
    assert metrics["target_flow_std"] > 0.0
    assert samples.shape == (5, 3, cfg.future_len, cfg.n_cells)
    assert torch.isfinite(samples).all()


def test_risk_state_allocation_loss_conditions_flow_context():
    torch.manual_seed(701)
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
        risk_state_dim=4,
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    model.set_level_quantiles(torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells), levels)

    history_level = torch.randn(6, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(6, cfg.history_len, cfg.n_cells)
    center = torch.randn(6, cfg.n_cells) * 0.01
    scale = torch.rand(6, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(6, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)

    loss, metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        risk_state_weight=0.1,
        risk_state_rank_weight=0.1,
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
    assert metrics["risk_state_enabled"].item() == 1.0
    assert metrics["risk_state_loss"].item() >= 0.0
    assert metrics["risk_state_rank_loss"].item() >= 0.0
    assert samples.shape == (6, 3, cfg.future_len, cfg.n_cells)
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


def test_scale_local_prefix_uses_current_level_geometry():
    torch.manual_seed(744)
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
        prefix_feature_mode="scale_local",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    model.eval()
    levels = torch.linspace(0.05, 0.95, cfg.n_quantiles)
    level_quantiles = torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles)] * cfg.n_cells)
    model.set_level_quantiles(level_quantiles, levels)

    history_level = torch.randn(5, cfg.history_len, cfg.n_cells) * 0.2
    history_norm = torch.randn(5, cfg.history_len, cfg.n_cells)
    center = torch.zeros(5, cfg.n_cells)
    scale = torch.rand(5, cfg.n_cells) * 0.05 + 0.01
    future_norm = torch.randn(5, cfg.future_len, cfg.n_cells)
    increments = future_norm * scale[:, None, :] + center[:, None, :]
    future_level = history_level[:, -1:, :] + torch.cumsum(increments, dim=1)

    loss, _metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
    )
    samples = model.sample_batched(
        history_level,
        history_norm,
        center,
        scale,
        n_samples=2,
        n_steps=cfg.future_len,
        chunk_size=2,
    )

    assert model.feature_proj.in_features == 8 * cfg.n_cells
    assert torch.isfinite(loss)
    assert samples.shape == (5, 2, cfg.future_len, cfg.n_cells)


def test_prefix_feature_mode_upgrade_preserves_existing_projection_columns():
    torch.manual_seed(745)
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
    old_weight = model.feature_proj.weight.detach().clone()
    old_bias = model.feature_proj.bias.detach().clone()
    old_in = model.feature_proj.in_features

    enable_prefix_feature_mode(model, prefix_feature_mode="scale_local")

    assert model.cfg.prefix_feature_mode == "scale_local"
    assert model.feature_proj.in_features == 8 * cfg.n_cells
    torch.testing.assert_close(model.feature_proj.weight[:, :old_in], old_weight)
    torch.testing.assert_close(model.feature_proj.bias, old_bias)
    assert torch.count_nonzero(model.feature_proj.weight[:, old_in:]) == 0


def test_ar1_base_noise_like_has_temporal_correlation():
    torch.manual_seed(71)
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
        history_len=4,
        future_len=8,
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
        base_noise_rho=0.8,
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    ref = torch.zeros(1024, cfg.future_len, cfg.n_cells)

    noise = model._base_noise_like(ref)
    lag_corr = torch.corrcoef(torch.stack([noise[:, :-1].reshape(-1), noise[:, 1:].reshape(-1)]))[0, 1]

    assert lag_corr > 0.55


def test_conditional_base_noise_scale_starts_at_unit_scale():
    torch.manual_seed(83)
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
    enable_conditional_base_noise_scale(model, scale_min=0.5, scale_max=2.0)
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
    samples = model.sample_batched(
        history_level,
        history_norm,
        center,
        scale,
        n_samples=2,
        n_steps=cfg.future_len,
        chunk_size=2,
    )

    assert torch.isfinite(loss)
    assert samples.shape == (5, 2, cfg.future_len, cfg.n_cells)
    assert metrics["base_noise_scale_enabled"] == 1.0
    assert torch.isclose(metrics["base_noise_scale_mean"], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(metrics["base_noise_scale_min"], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(metrics["base_noise_scale_max"], torch.tensor(1.0), atol=1e-6)


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


def test_group_head_velocity_readout_starts_as_noop():
    torch.manual_seed(97)
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
    enable_group_head_velocity_readout(model, iv_count=3)
    after = model.velocity(x_t, current_level, memory_state, t)

    assert model.cfg.velocity_readout_mode == "group_head"
    assert model.cfg.readout_iv_count == 3
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)
