import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_gaussian_transition_law import (
    GenericGaussianTransitionConfig,
    GenericGaussianTransitionLaw,
)


def _toy_model(n_cells: int = 4) -> GenericGaussianTransitionLaw:
    cfg = GenericGaussianTransitionConfig(
        history_len=5,
        future_len=3,
        n_cells=n_cells,
        n_quantiles=21,
        memory_dim=16,
        memory_layers=1,
        memory_heads=4,
        memory_ff=32,
        head_hidden=32,
        model_dropout=0.0,
    )
    model = GenericGaussianTransitionLaw(cfg)
    levels = torch.linspace(0.01, 0.99, cfg.n_quantiles)
    quantiles = torch.stack(
        [torch.linspace(0.05 + 0.01 * i, 0.95 + 0.01 * i, cfg.n_quantiles) for i in range(n_cells)]
    )
    model.set_empirical_quantiles(quantiles, levels)
    return model


def test_gaussian_transition_loss_and_sampling_are_finite():
    torch.manual_seed(614)
    model = _toy_model()
    history = torch.rand(6, 5, 4) * 0.7 + 0.1
    future = torch.rand(6, 3, 4) * 0.7 + 0.1
    loss, metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)
    assert metrics["diag_mean"].item() > 0
    samples = model.sample_batched(history[:2], n_samples=3, n_steps=3, chunk_size=2)
    assert samples.shape == (2, 3, 3, 4)
    assert torch.isfinite(samples).all()


def test_gaussian_transition_scale_feature_mode():
    torch.manual_seed(615)
    cfg = GenericGaussianTransitionConfig(
        history_len=4,
        future_len=2,
        n_cells=3,
        n_quantiles=17,
        prefix_feature_mode="scale",
        memory_dim=12,
        memory_layers=1,
        memory_heads=3,
        memory_ff=24,
        head_hidden=24,
        model_dropout=0.0,
    )
    model = GenericGaussianTransitionLaw(cfg)
    levels = torch.linspace(0.01, 0.99, cfg.n_quantiles)
    quantiles = torch.stack([torch.linspace(0.0, 1.0, cfg.n_quantiles) for _ in range(cfg.n_cells)])
    model.set_empirical_quantiles(quantiles, levels)
    history = torch.rand(3, 4, 3)
    future = torch.rand(3, 2, 3)
    loss, _metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)


def test_student_t_transition_loss_and_sampling_are_finite():
    torch.manual_seed(616)
    cfg = GenericGaussianTransitionConfig(
        history_len=4,
        future_len=2,
        n_cells=3,
        n_quantiles=17,
        memory_dim=12,
        memory_layers=1,
        memory_heads=3,
        memory_ff=24,
        head_hidden=24,
        model_dropout=0.0,
        distribution_family="student_t",
        student_t_df=5.0,
    )
    model = GenericGaussianTransitionLaw(cfg)
    levels = torch.linspace(0.01, 0.99, cfg.n_quantiles)
    quantiles = torch.stack([torch.linspace(0.0, 1.0, cfg.n_quantiles) for _ in range(cfg.n_cells)])
    model.set_empirical_quantiles(quantiles, levels)
    history = torch.rand(3, 4, 3)
    future = torch.rand(3, 2, 3)
    loss, metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)
    assert metrics["diag_mean"].item() > 0
    samples = model.sample_batched(history[:1], n_samples=2, n_steps=2, chunk_size=1)
    assert samples.shape == (1, 2, 2, 3)
    assert torch.isfinite(samples).all()


def test_mean_loss_weight_path_is_finite():
    torch.manual_seed(618)
    cfg = GenericGaussianTransitionConfig(
        history_len=4,
        future_len=2,
        n_cells=3,
        n_quantiles=17,
        memory_dim=12,
        memory_layers=1,
        memory_heads=3,
        memory_ff=24,
        head_hidden=24,
        model_dropout=0.0,
        distribution_family="student_t",
        student_t_df=5.0,
        mean_loss_weight=1.0,
    )
    model = GenericGaussianTransitionLaw(cfg)
    levels = torch.linspace(0.01, 0.99, cfg.n_quantiles)
    quantiles = torch.stack([torch.linspace(0.0, 1.0, cfg.n_quantiles) for _ in range(cfg.n_cells)])
    model.set_empirical_quantiles(quantiles, levels)
    history = torch.rand(3, 4, 3)
    future = torch.rand(3, 2, 3)
    loss, metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)
    assert metrics["mean_loss"].item() >= 0
