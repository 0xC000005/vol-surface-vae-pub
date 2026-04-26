import sys

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.local_shift_normalized_empirical_score_transition_flow_matching import (
    LocalShiftNormalizedEmpiricalScoreTransitionFMConfig,
    LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching,
)
from experiments.backfill.block_ar.train_548a_local_shift_unpaired_raw_iv_sw_finetune import (
    combined_loss,
    joint_sliced_wasserstein_loss,
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


def test_joint_sliced_wasserstein_is_smaller_for_matching_batch_law() -> None:
    torch.manual_seed(17)
    history = torch.randn(4, 3, 2)
    target = torch.randn(4, 2, 2)
    matching = target[:, None].repeat(1, 2, 1, 1)
    shifted = matching + 3.0

    match_loss, _, _ = joint_sliced_wasserstein_loss(
        history_iv=history,
        generated_iv=matching,
        target_iv=target,
        n_projections=16,
        history_weight=0.5,
        future_weight=1.0,
        p=1.0,
        eps=1e-6,
    )
    shifted_loss, _, _ = joint_sliced_wasserstein_loss(
        history_iv=history,
        generated_iv=shifted,
        target_iv=target,
        n_projections=16,
        history_weight=0.5,
        future_weight=1.0,
        p=1.0,
        eps=1e-6,
    )

    assert float(match_loss) < float(shifted_loss)


def test_unpaired_sw_combined_loss_keeps_fm_anchor_and_backpropagates() -> None:
    torch.manual_seed(19)
    model = _tiny_model()
    history_01 = torch.tensor(
        [
            [[0.20, 0.40], [0.21, 0.42], [0.22, 0.41]],
            [[0.60, 0.30], [0.58, 0.31], [0.57, 0.33]],
            [[0.30, 0.20], [0.32, 0.22], [0.34, 0.25]],
            [[0.70, 0.50], [0.69, 0.48], [0.68, 0.47]],
        ],
        dtype=torch.float32,
    )
    future_01 = torch.tensor(
        [
            [[0.24, 0.43], [0.23, 0.44]],
            [[0.55, 0.34], [0.56, 0.35]],
            [[0.36, 0.26], [0.37, 0.28]],
            [[0.67, 0.46], [0.66, 0.45]],
        ],
        dtype=torch.float32,
    )

    loss, metrics = combined_loss(
        model=model,
        history_norm=history_01 * 2.0 - 1.0,
        future_norm=future_01 * 2.0 - 1.0,
        train_sample_count=2,
        rollout_flow_steps=2,
        sw_weight=0.05,
        fm_anchor_weight=1.0,
        n_projections=8,
        history_weight=0.5,
        future_weight=1.0,
        wasserstein_p=1.0,
        eps=1e-6,
    )
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert {"total", "fm_loss", "joint_sw", "center_gap", "sample_spread"} <= set(metrics)
    assert float(metrics["joint_sw"]) >= 0.0
    assert grads
    assert any(torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0 for grad in grads)
