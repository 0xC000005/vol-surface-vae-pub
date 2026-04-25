import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.coherent_panel_score_path_model import (
    CoherentPanelScorePathConfig,
    CoherentPanelScorePathModel,
)


def _toy_model() -> CoherentPanelScorePathModel:
    cfg = CoherentPanelScorePathConfig(
        history_len=3,
        future_len=4,
        n_vars=5,
        n_quantiles=19,
        history_hidden=12,
        token_hidden=20,
    )
    model = CoherentPanelScorePathModel(cfg)
    levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / cfg.n_quantiles
    quantiles = torch.stack(
        [
            torch.linspace(0.05, 0.95, cfg.n_quantiles),
            torch.linspace(-0.10, 0.10, cfg.n_quantiles),
            torch.linspace(10.0, 20.0, cfg.n_quantiles),
            torch.linspace(-2.0, 3.0, cfg.n_quantiles),
            torch.linspace(100.0, 110.0, cfg.n_quantiles),
        ],
        dim=0,
    )
    model.set_empirical_quantiles(quantiles, levels)
    chol = torch.eye(cfg.future_len * cfg.n_vars)
    chol[1, 0] = 0.40
    model.set_residual_cholesky(chol)
    return model


def test_panel_training_loss_is_finite_for_raw_mixed_scale_values() -> None:
    torch.manual_seed(0)
    model = _toy_model()
    history = torch.rand(6, 3, 5)
    future = torch.rand(6, 4, 5)
    history[..., 1] = history[..., 1] * 0.20 - 0.10
    future[..., 1] = future[..., 1] * 0.20 - 0.10
    history[..., 2] = history[..., 2] * 10.0 + 10.0
    future[..., 2] = future[..., 2] * 10.0 + 10.0

    loss, metrics = model.training_loss(history, future)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["nll"])
    assert metrics["target_std"] > 0


def test_panel_sampling_returns_raw_future_panel_values() -> None:
    torch.manual_seed(1)
    model = _toy_model()
    history = torch.zeros(2, 3, 5)

    samples = model.sample_batched(history, n_samples=9, n_steps=4, chunk_size=4)

    assert samples.shape == (2, 9, 4, 5)
    assert torch.all(samples[..., 0] >= 0.05)
    assert torch.all(samples[..., 0] <= 0.95)
    assert torch.all(samples[..., 2] >= 10.0)
    assert torch.all(samples[..., 2] <= 20.0)
