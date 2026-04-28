import numpy as np
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.audit_664a_condition_diversity_diagnostics import (
    condition_diversity_summary,
    make_deranged_permutation,
)


def test_make_deranged_permutation_has_no_fixed_points():
    perm = make_deranged_permutation(9, seed=123)

    assert sorted(perm.tolist()) == list(range(9))
    assert np.all(perm != np.arange(9))


def test_condition_diversity_summary_detects_condition_shuffle_penalty():
    target = np.zeros((2, 2, 1), dtype=np.float32)
    original = np.array(
        [
            [[[-0.1], [0.1]], [[0.0], [0.0]], [[0.1], [-0.1]]],
            [[[0.0], [0.0]], [[0.1], [-0.1]], [[-0.1], [0.1]]],
        ],
        dtype=np.float32,
    )
    shuffled = original + 2.0

    summary = condition_diversity_summary(target, original, shuffled)

    assert summary["original_mean_mae"] < 0.05
    assert summary["shuffled_mean_mae"] > 1.9
    assert summary["shuffle_mae_ratio"] > 30.0
    assert summary["within_sample_std"] > 0.0
    assert summary["paired_condition_effect_over_std"] > 10.0
