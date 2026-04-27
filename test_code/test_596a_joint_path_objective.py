import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.train_596a_final_path_joint_objective import (  # noqa: E402
    full_path_energy_score,
    sliced_projection_wasserstein_score,
)


def test_full_path_energy_score_prefers_target_aligned_samples() -> None:
    target = torch.zeros(3, 4, 2)
    good = 0.01 * torch.randn(3, 5, 4, 2)
    bad = good + 2.0

    good_score, *_ = full_path_energy_score(good, target, eps=1e-8)
    bad_score, *_ = full_path_energy_score(bad, target, eps=1e-8)

    assert torch.isfinite(good_score)
    assert good_score < bad_score


def test_sliced_projection_wasserstein_score_is_differentiable_and_shift_sensitive() -> None:
    target = torch.randn(4, 3, 2)
    samples = (target[:, None] + 0.05 * torch.randn(4, 6, 3, 2)).requires_grad_(True)
    shifted = samples + 1.0

    generator = torch.Generator().manual_seed(7)
    aligned = sliced_projection_wasserstein_score(
        samples,
        target,
        n_projections=8,
        n_quantiles=8,
        generator=generator,
    )
    generator = torch.Generator().manual_seed(7)
    shifted_score = sliced_projection_wasserstein_score(
        shifted,
        target,
        n_projections=8,
        n_quantiles=8,
        generator=generator,
    )

    assert torch.isfinite(aligned)
    assert aligned < shifted_score
    aligned.backward()
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()
