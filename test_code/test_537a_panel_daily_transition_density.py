import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.panel_daily_cholesky_transition_model import (
    PanelDailyCholeskyTransitionConfig,
    PanelDailyCholeskyTransitionModel,
)


def _toy_model() -> PanelDailyCholeskyTransitionModel:
    cfg = PanelDailyCholeskyTransitionConfig(
        history_len=3,
        future_len=4,
        n_vars=5,
        n_quantiles=19,
        history_hidden=12,
        context_hidden=20,
        dropout=0.0,
    )
    model = PanelDailyCholeskyTransitionModel(cfg)
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
    return model


def test_daily_transition_loss_is_finite_for_raw_mixed_scale_values() -> None:
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
    assert metrics["innovation_std"] > 0
    assert metrics["diag_mean"] > 0


def test_daily_transition_params_are_full_lower_triangular() -> None:
    torch.manual_seed(1)
    model = _toy_model()
    history = torch.rand(2, 3, 5)
    future_prefix = torch.rand(2, 2, 5)

    mean, chol = model.teacher_forced_params(history, future_prefix)

    assert mean.shape == (2, 3, 5)
    assert chol.shape == (2, 3, 5, 5)
    assert torch.allclose(chol, torch.tril(chol))
    assert torch.all(torch.diagonal(chol, dim1=-2, dim2=-1) > 0)


def test_daily_transition_sampling_rolls_forward_raw_panel_values() -> None:
    torch.manual_seed(2)
    model = _toy_model()
    history = torch.zeros(2, 3, 5)

    samples = model.sample_batched(history, n_samples=7, n_steps=4, chunk_size=3)

    assert samples.shape == (2, 7, 4, 5)
    assert torch.all(samples[..., 0] >= 0.05)
    assert torch.all(samples[..., 0] <= 0.95)
    assert torch.all(samples[..., 2] >= 10.0)
    assert torch.all(samples[..., 2] <= 20.0)

