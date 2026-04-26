import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.local_shift_normalized_empirical_score_transition_flow_matching import (
    LocalShiftNormalizedEmpiricalScoreTransitionFMConfig,
    LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching,
)


def _tiny_model() -> LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching:
    cfg = LocalShiftNormalizedEmpiricalScoreTransitionFMConfig(
        history_len=3,
        future_len=2,
        n_cells=2,
        model_hidden=16,
        model_layers=1,
        token_dim=8,
        token_layers=1,
        token_heads=1,
        token_ff=16,
        memory_dim=8,
        memory_layers=1,
        memory_heads=1,
        memory_ff=16,
        time_dim=4,
        n_quantiles=17,
        flow_steps=2,
        local_scale_floor=0.05,
    )
    model = LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching(cfg)
    levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / cfg.n_quantiles
    quantiles = torch.stack(
        [
            torch.linspace(-3.0, 3.0, cfg.n_quantiles),
            torch.linspace(-2.0, 2.0, cfg.n_quantiles),
        ],
        dim=0,
    )
    model.set_empirical_quantiles(quantiles, levels)
    return model


def test_causal_local_stats_use_only_history_last_and_trailing_moves() -> None:
    model = _tiny_model()
    history_01 = torch.tensor(
        [
            [
                [0.10, 0.50],
                [0.15, 0.45],
                [0.25, 0.47],
            ]
        ],
        dtype=torch.float32,
    )

    center, scale = model.causal_local_stats(history_01)

    expected_scale = torch.sqrt(
        torch.tensor([[0.05**2 + 0.10**2, 0.05**2 + 0.02**2]]) / 2.0
    ).clamp_min(0.05)
    assert torch.allclose(center, torch.tensor([[0.25, 0.47]]))
    assert torch.allclose(scale, expected_scale)


def test_local_transform_round_trips_before_iv_clamp() -> None:
    model = _tiny_model()
    history_01 = torch.tensor(
        [
            [
                [0.20, 0.30],
                [0.25, 0.40],
                [0.35, 0.45],
            ]
        ],
        dtype=torch.float32,
    )
    future_01 = torch.tensor(
        [
            [
                [0.40, 0.50],
                [0.30, 0.35],
            ]
        ],
        dtype=torch.float32,
    )

    center, scale = model.causal_local_stats(history_01)
    local = model.to_local_values(future_01, center, scale)
    restored = model.from_local_values(local, center, scale)

    assert torch.allclose(restored, future_01, atol=1e-6)


def test_training_loss_and_sampling_support_native_11_suite_api() -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    history_01 = torch.tensor(
        [
            [[0.20, 0.40], [0.21, 0.42], [0.22, 0.41]],
            [[0.60, 0.30], [0.58, 0.31], [0.57, 0.33]],
        ],
        dtype=torch.float32,
    )
    future_01 = torch.tensor(
        [
            [[0.24, 0.43], [0.23, 0.44]],
            [[0.55, 0.34], [0.56, 0.35]],
        ],
        dtype=torch.float32,
    )

    loss, metrics = model.training_loss(history_01 * 2.0 - 1.0, future_01 * 2.0 - 1.0)
    samples = model.sample_batched(
        history_01 * 2.0 - 1.0,
        n_samples=3,
        n_steps=2,
        chunk_size=2,
    )

    assert loss.ndim == 0
    assert {"total", "fm_loss", "local_scale_mean", "transition_std"} <= set(metrics)
    assert samples.shape == (2, 3, 2, 2)
    assert torch.isfinite(samples).all()
    assert float(samples.min()) >= 0.0
    assert float(samples.max()) <= 1.0
