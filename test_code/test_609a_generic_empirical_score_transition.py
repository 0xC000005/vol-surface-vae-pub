import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_empirical_score_transition_flow_matching import (
    GenericEmpiricalScoreTransitionFMConfig,
    GenericEmpiricalScoreTransitionFlowMatching,
)


def _tiny_model(n_vars: int = 6) -> GenericEmpiricalScoreTransitionFlowMatching:
    cfg = GenericEmpiricalScoreTransitionFMConfig(
        history_len=4,
        future_len=3,
        n_cells=n_vars,
        n_quantiles=31,
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
        model_dropout=0.0,
        conditioning_mode="prefix",
    )
    model = GenericEmpiricalScoreTransitionFlowMatching(cfg)
    base = torch.linspace(-2.0, 2.0, cfg.n_quantiles)
    quantiles = torch.stack([base + 0.1 * i for i in range(n_vars)], dim=0)
    model.set_empirical_quantiles(quantiles)
    return model


def test_generic_empirical_score_roundtrip_shape():
    model = _tiny_model(n_vars=5)
    values = torch.randn(2, 4, 5).clamp(-1.5, 1.5)
    scores = model.values_to_scores(values)
    decoded = model.scores_to_values(scores)
    assert scores.shape == values.shape
    assert decoded.shape == values.shape
    assert torch.isfinite(decoded).all()


def test_generic_transition_loss_and_sample_shapes():
    model = _tiny_model(n_vars=6)
    history = torch.randn(3, 4, 6).clamp(-1.5, 1.5)
    future = torch.randn(3, 3, 6).clamp(-1.5, 1.5)
    loss, metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)
    assert metrics["transition_std"].item() > 0.0
    samples = model.sample_batched(history, n_samples=4, n_steps=3, chunk_size=2)
    assert samples.shape == (3, 4, 3, 6)
    assert torch.isfinite(samples).all()


def test_conditional_source_scale_path():
    cfg = GenericEmpiricalScoreTransitionFMConfig(
        history_len=4,
        future_len=3,
        n_cells=4,
        n_quantiles=31,
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
        model_dropout=0.0,
        conditioning_mode="prefix",
        conditional_source_scale=True,
    )
    model = GenericEmpiricalScoreTransitionFlowMatching(cfg)
    quantiles = torch.stack([torch.linspace(-2.0, 2.0, cfg.n_quantiles) for _ in range(cfg.n_cells)])
    model.set_empirical_quantiles(quantiles)
    history = torch.randn(2, 4, 4).clamp(-1.5, 1.5)
    future = torch.randn(2, 3, 4).clamp(-1.5, 1.5)
    loss, metrics = model.training_loss(history, future)
    assert torch.isfinite(loss)
    assert "source_scale_mean" in metrics
    samples = model.sample_batched(history, n_samples=2, n_steps=2, chunk_size=1)
    assert samples.shape == (2, 2, 2, 4)
    assert torch.isfinite(samples).all()
