import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


def _small_cfg(factor_dim: int = 0):
    return EmpiricalNormalScoreCausalMemoryTransitionFMConfig(
        history_len=4,
        future_len=3,
        n_cells=25,
        hidden_dim=16,
        gru_layers=1,
        gru_dropout=0.0,
        model_hidden=32,
        model_layers=1,
        model_dropout=0.0,
        time_dim=8,
        token_dim=16,
        token_layers=1,
        token_heads=2,
        token_ff=32,
        memory_dim=16,
        memory_layers=1,
        memory_heads=2,
        memory_ff=32,
        flow_steps=2,
        n_quantiles=21,
        factor_dim=factor_dim,
    )


def _model(factor_dim: int = 0):
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(_small_cfg(factor_dim))
    quantiles = torch.linspace(0.05, 0.95, model.cfg.n_quantiles).repeat(model.cfg.n_cells, 1)
    levels = (torch.arange(model.cfg.n_quantiles, dtype=torch.float32) + 0.5) / model.cfg.n_quantiles
    model.set_empirical_quantiles(quantiles, levels)
    return model


def test_factor_history_conditions_training_loss():
    torch.manual_seed(0)
    model = _model(factor_dim=6)
    history = normalize_iv(torch.rand(2, 4, 5, 5) * 0.4 + 0.25)
    future = normalize_iv(torch.rand(2, 3, 5, 5) * 0.4 + 0.25)
    factor_history = torch.randn(2, 4, 6)

    loss, metrics = model.training_loss(history, future, factor_history=factor_history)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["factor_context_abs"] > 0
    loss.backward()


def test_factor_history_conditions_sampling_shape():
    torch.manual_seed(0)
    model = _model(factor_dim=6).eval()
    history = normalize_iv(torch.rand(2, 4, 5, 5) * 0.4 + 0.25)
    factor_history = torch.randn(2, 4, 6)

    samples = model.sample_batched(
        history,
        n_samples=3,
        n_steps=2,
        chunk_size=2,
        factor_history=factor_history,
    )

    assert samples.shape == (2, 3, 2, 5, 5)
    assert torch.isfinite(samples).all()


def test_no_factor_configuration_keeps_legacy_training_path():
    torch.manual_seed(0)
    model = _model(factor_dim=0)
    history = normalize_iv(torch.rand(2, 4, 5, 5) * 0.4 + 0.25)
    future = normalize_iv(torch.rand(2, 3, 5, 5) * 0.4 + 0.25)

    loss, metrics = model.training_loss(history, future)

    assert torch.isfinite(loss)
    assert "factor_context_abs" not in metrics
