import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.coherent_gaussian_score_path_model import (
    CoherentGaussianScorePathConfig,
    CoherentGaussianScorePathModel,
)


def _toy_model() -> CoherentGaussianScorePathModel:
    cfg = CoherentGaussianScorePathConfig(
        history_len=3,
        future_len=4,
        n_cells=3,
        n_quantiles=21,
        history_hidden=16,
        token_hidden=24,
    )
    model = CoherentGaussianScorePathModel(cfg)
    levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / cfg.n_quantiles
    quantiles = torch.stack(
        [
            torch.linspace(0.05, 0.95, cfg.n_quantiles),
            torch.linspace(0.10, 0.90, cfg.n_quantiles),
            torch.linspace(0.15, 0.85, cfg.n_quantiles),
        ],
        dim=0,
    )
    model.set_empirical_quantiles(quantiles, levels)
    chol = torch.eye(cfg.future_len * cfg.n_cells)
    chol[1, 0] = 0.35
    chol[5, 2] = -0.20
    model.set_residual_cholesky(chol)
    return model


def test_training_loss_is_finite_with_full_path_cholesky() -> None:
    torch.manual_seed(0)
    model = _toy_model()
    history = torch.rand(5, 3, 3) * 2.0 - 1.0
    future = torch.rand(5, 4, 3) * 2.0 - 1.0

    loss, metrics = model.training_loss(history, future)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["nll"])
    assert metrics["residual_cholesky_logdet"].shape == ()


def test_sample_batched_returns_one_shot_future_levels() -> None:
    torch.manual_seed(1)
    model = _toy_model()
    history = torch.rand(2, 3, 3) * 2.0 - 1.0

    samples = model.sample_batched(
        history,
        n_samples=7,
        n_steps=4,
        chunk_size=3,
        history_is_normalized=True,
    )

    assert samples.shape == (2, 7, 4, 3)
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)


def test_residual_cholesky_controls_token_correlation() -> None:
    torch.manual_seed(2)
    model = _toy_model()
    history = torch.zeros(1, 3, 3)
    samples = model.sample_scores_batched(history, n_samples=512, chunk_size=128)
    flat = samples[0].reshape(samples.shape[1], -1)
    corr = torch.corrcoef(flat[:, :2].T)[0, 1]

    assert corr > 0.15
