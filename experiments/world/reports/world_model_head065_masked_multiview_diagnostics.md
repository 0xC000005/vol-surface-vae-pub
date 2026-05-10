# World Model HEAD065: Masked Multiview Diagnostics

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: before training a masked-multiview encoder, we can define reusable
diagnostics for same-state alignment, same-state retrieval, Barlow-style
cross-correlation, representation health, and geometry-stratified visibility.

Falsifier: diagnostics require a trained model, cannot score paired views, do
not catch mismatched positives, or cannot summarize geometry/mask visibility.

## Implementation

Added `experiments/world/evaluation/masked_multiview_metrics.py`.

The module provides:

- `same_state_multiview_metrics`;
- `barlow_cross_correlation_metrics`;
- `mask_visibility_summary`;
- `flattened_time_rows`.

Updated `test_code/test_world_model_evaluation.py` with focused tests for:

- alignment/retrieval/Barlow metrics on known matching embeddings;
- Barlow degradation under mismatched views;
- visibility summaries by geometry and factor family.

Updated `experiments/world/evaluation/README.md` to include the diagnostics.

## Validation

Focused tests:

```bash
pytest test_code/test_world_model_evaluation.py::test_masked_multiview_metrics_score_alignment_and_visibility \
  test_code/test_world_model_evaluation.py::test_barlow_cross_correlation_penalizes_mismatched_views -q
```

Result: `2 passed in 0.76s`.

Full world-model evaluation test slice:

```bash
pytest test_code/test_world_model_evaluation.py -q
```

Result: `36 passed in 0.77s`.

Compile check:

```bash
python -m py_compile experiments/world/evaluation/masked_multiview_metrics.py \
  test_code/test_world_model_evaluation.py
```

## Real-Data Metric Smoke

Command:

```bash
python - <<'PY'
from experiments.world.evaluation.masked_multiview_data import build_masked_multiview_batch
from experiments.world.evaluation.masked_multiview_metrics import (
    flattened_time_rows, same_state_multiview_metrics
)
b = build_masked_multiview_batch(split='train', max_windows=16, seed=650, normalize=True)
metrics = same_state_multiview_metrics(
    flattened_time_rows(b.view_a_values),
    flattened_time_rows(b.view_b_values),
)
print(metrics['alignment'], metrics['retrieval'], metrics['barlow'])
PY
```

Saved summary: `results/world/masked_multiview_head065_metric_smoke.json`.

Raw masked-view baseline metrics, before any encoder:

| metric | value |
| --- | ---: |
| alignment MSE | 0.046740 |
| alignment cosine mean | 0.958239 |
| retrieval MRR | 0.246423 |
| retrieval top1 | 0.085417 |
| retrieval top5 | 0.383333 |
| retrieval top10 | 0.762500 |
| Barlow diag mean | 0.407648 |
| Barlow diag loss | 0.536486 |
| Barlow offdiag abs mean | 0.086700 |
| view A effective rank | 11.981477 |
| view B effective rank | 10.913984 |

This is a baseline for raw masked tensors, not a model result. It is useful
because future encoders should improve same-state retrieval and Barlow diagonal
agreement without collapsing rank or encoding mask artifacts.

## Decision / Next Step

The data and diagnostic layers are ready for a first encoder smoke.

Next iteration should implement the smallest masked-multiview encoder smoke:

- encode view A and view B with shared or EMA encoders;
- pool/score same relative positions;
- optimize alignment plus Barlow-style redundancy control;
- report the HEAD065 diagnostics by train/val split and mask family.

## Artifacts

- `experiments/world/evaluation/masked_multiview_metrics.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/evaluation/README.md`
- `experiments/world/reports/world_model_head065_masked_multiview_diagnostics.md`
- `results/world/masked_multiview_head065_metric_smoke.json`
