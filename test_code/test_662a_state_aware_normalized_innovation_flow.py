import numpy as np
import sys
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
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
