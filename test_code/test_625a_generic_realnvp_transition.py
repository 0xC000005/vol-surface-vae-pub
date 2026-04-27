import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_realnvp_transition_law import (  # noqa: E402
    GenericRealNVPTransitionConfig,
    GenericRealNVPTransitionLaw,
)


def _toy_model(n_cells: int = 4) -> GenericRealNVPTransitionLaw:
    cfg = GenericRealNVPTransitionConfig(
        history_len=5,
        future_len=3,
        n_cells=n_cells,
        n_quantiles=21,
        memory_dim=16,
        memory_layers=1,
        memory_heads=4,
        memory_ff=32,
        coupling_layers=4,
        coupling_hidden=32,
        coupling_depth=1,
        model_dropout=0.0,
        scale_clip=1.0,
    )
    model = GenericRealNVPTransitionLaw(cfg)
    levels = torch.linspace(0.01, 0.99, cfg.n_quantiles)
    quantiles = torch.stack(
        [torch.linspace(0.05 + 0.01 * i, 0.95 + 0.01 * i, cfg.n_quantiles) for i in range(n_cells)]
    )
    model.set_empirical_quantiles(quantiles, levels)
    return model


def test_realnvp_transition_loss_and_sampling_are_finite() -> None:
    torch.manual_seed(625)
    model = _toy_model()
    history = torch.rand(6, 5, 4) * 0.7 + 0.1
    future = torch.rand(6, 3, 4) * 0.7 + 0.1
    loss, metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)
    assert metrics["z_std"].item() > 0
    samples = model.sample_batched(history[:2], n_samples=3, n_steps=3, chunk_size=2)
    assert samples.shape == (2, 3, 3, 4)
    assert torch.isfinite(samples).all()


def test_realnvp_forward_inverse_are_consistent_at_initialization() -> None:
    torch.manual_seed(626)
    model = _toy_model(n_cells=6)
    z = torch.randn(5, 6)
    current = torch.randn(5, 6)
    memory = torch.randn(5, model.cfg.memory_dim)

    y, log_det = model._flow_forward(z, current, memory)
    z_back, log_det_inv = model._flow_inverse(y, current, memory)

    assert torch.allclose(z_back, z, atol=1e-5)
    assert torch.allclose(log_det + log_det_inv, torch.zeros_like(log_det), atol=1e-5)


def test_realnvp_scale_feature_mode() -> None:
    torch.manual_seed(627)
    cfg = GenericRealNVPTransitionConfig(
        history_len=4,
        future_len=2,
        n_cells=3,
        n_quantiles=17,
        prefix_feature_mode="scale",
        memory_dim=12,
        memory_layers=1,
        memory_heads=3,
        memory_ff=24,
        coupling_layers=3,
        coupling_hidden=24,
        coupling_depth=1,
        model_dropout=0.0,
    )
    model = GenericRealNVPTransitionLaw(cfg)
    levels = torch.linspace(0.01, 0.99, cfg.n_quantiles)
    quantiles = torch.stack([torch.linspace(0.0, 1.0, cfg.n_quantiles) for _ in range(cfg.n_cells)])
    model.set_empirical_quantiles(quantiles, levels)
    history = torch.rand(3, 4, 3)
    future = torch.rand(3, 2, 3)
    loss, _metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)

