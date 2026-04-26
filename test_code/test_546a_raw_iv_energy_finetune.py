import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.local_shift_normalized_empirical_score_transition_flow_matching import (
    LocalShiftNormalizedEmpiricalScoreTransitionFMConfig,
    LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching,
)
from experiments.backfill.block_ar.train_546a_local_shift_raw_iv_energy_finetune import (
    combined_loss,
    sample_rollout_iv_with_grad,
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


def test_raw_iv_rollout_with_grad_returns_bounded_paths_and_allows_gradients() -> None:
    torch.manual_seed(11)
    model = _tiny_model()
    history_01 = torch.tensor(
        [
            [[0.20, 0.40], [0.21, 0.42], [0.22, 0.41]],
            [[0.60, 0.30], [0.58, 0.31], [0.57, 0.33]],
        ],
        dtype=torch.float32,
    )

    samples = sample_rollout_iv_with_grad(
        model=model,
        history_norm=history_01 * 2.0 - 1.0,
        n_samples=3,
        n_steps=2,
        flow_steps=2,
    )
    objective = samples.mean()
    objective.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert samples.shape == (2, 3, 2, 2)
    assert float(samples.detach().min()) >= 0.0
    assert float(samples.detach().max()) <= 1.0
    assert grads
    assert any(torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0 for grad in grads)


def test_combined_loss_includes_raw_iv_energy_term() -> None:
    torch.manual_seed(13)
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

    loss, metrics = combined_loss(
        model=model,
        history_norm=history_01 * 2.0 - 1.0,
        future_norm=future_01 * 2.0 - 1.0,
        train_sample_count=2,
        rollout_flow_steps=2,
        energy_eps=1e-6,
        raw_energy_weight=1.0,
        fm_anchor_weight=1.0,
    )

    assert loss.ndim == 0
    assert {"total", "fm_loss", "raw_iv_energy", "raw_target_dist", "raw_pair_dist"} <= set(
        metrics
    )
    assert float(metrics["raw_iv_energy"]) > 0.0
