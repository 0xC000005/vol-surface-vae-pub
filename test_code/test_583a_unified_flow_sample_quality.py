import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_583a_unified_flow_sample_quality import (
    effective_rank,
    summarize_samples,
)


def test_effective_rank_is_positive_for_nonconstant_paths() -> None:
    rng = np.random.default_rng(7)
    paths = rng.normal(size=(4, 3, 2, 5)).astype(np.float32)

    rank = effective_rank(paths)

    assert rank > 1.0
    assert np.isfinite(rank)


def test_summarize_samples_reports_range_coverage_and_condition_metrics() -> None:
    rng = np.random.default_rng(11)
    history = np.ones((3, 2, 30), dtype=np.float32)
    gt = np.ones((3, 2, 30), dtype=np.float32)
    samples = np.repeat(gt[:, None], 5, axis=1) + rng.normal(scale=0.01, size=(3, 5, 2, 30))
    increments = rng.normal(size=(3, 5, 2, 30)).astype(np.float32)
    train = np.ones((6, 4, 30), dtype=np.float32)

    summary = summarize_samples(
        samples,
        increments,
        gt_state=gt,
        history_state=history,
        train_state=train,
        label="demo",
    )

    assert summary["label"] == "demo"
    assert summary["finite_state_rate"] == 1.0
    assert summary["iv_90_interval_coverage"] >= 0.0
    assert summary["increment_effective_rank"] > 1.0
