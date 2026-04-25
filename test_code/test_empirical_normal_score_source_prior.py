import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
)


def _lag1_corr(noise: torch.Tensor) -> float:
    x = noise[:, :-1].reshape(-1)
    y = noise[:, 1:].reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).mean() / (x.std(unbiased=False) * y.std(unbiased=False)))


def _tiny_cfg(**overrides):
    base = dict(
        history_len=4,
        future_len=30,
        n_cells=3,
        memory_dim=8,
        memory_layers=1,
        memory_heads=1,
        memory_ff=16,
        token_dim=8,
        token_layers=1,
        token_heads=1,
        token_ff=16,
        time_dim=8,
        flow_steps=2,
        model_dropout=0.0,
    )
    base.update(overrides)
    return EmpiricalNormalScoreCausalMemoryTransitionFMConfig(**base)


def test_path_source_ar_creates_temporally_correlated_source_noise():
    torch.manual_seed(123)
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(
        _tiny_cfg(path_source_ar=0.85)
    )
    target = torch.zeros(4096, 30, 3)

    noise = model._source_noise_like(target)

    assert noise.shape == target.shape
    assert _lag1_corr(noise) > 0.75


def test_default_source_noise_remains_temporally_white():
    torch.manual_seed(123)
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(_tiny_cfg())
    target = torch.zeros(4096, 30, 3)

    noise = model._source_noise_like(target)

    assert abs(_lag1_corr(noise)) < 0.05
