# World Model HEAD076: Geometry-Aware Barlow Smoke

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: a minimal token-aware encoder can use geometry descriptors to
improve Part 1 over the flattened daily GRU while keeping the same canonical
direct Barlow objective.

Falsifier: same-state retrieval, rank, or redundancy are worse than HEAD070.

## Implementation

Added `experiments/world/part1_jepa_latent/masked_multiview_geometry_barlow_smoke.py`.

The model uses:

- per-token value, observed mask, synthetic mask;
- token descriptors built from geometry coordinates, geometry id, and factor
  family;
- shared token MLP;
- mean pooling over tokens per day;
- temporal GRU over daily pooled states;
- canonical direct Barlow loss on per-time-row embeddings.

Added a unit test for token descriptor construction, model shape, and finite
loss.

## Validation

Focused test:

```bash
pytest test_code/test_world_model_evaluation.py::test_geometry_aware_barlow_encoder_uses_token_descriptors -q
```

Result: `1 passed in 0.74s`.

Real-data smoke:

```bash
python experiments/world/part1_jepa_latent/masked_multiview_geometry_barlow_smoke.py --device cpu
```

Training loss decreased from `0.164367` to `0.068565`.

## Result

Validation comparison:

| metric | HEAD076 geometry-aware | HEAD070 flat canonical |
| --- | ---: | ---: |
| alignment MSE | 0.001153 | 0.007052 |
| cosine mean | 0.998920 | 0.992878 |
| retrieval top1 | 0.138802 | 0.321354 |
| retrieval top5 | 0.257292 | 0.662500 |
| retrieval top10 | 0.333073 | 0.841927 |
| Barlow diag mean | 0.886596 | 0.930924 |
| Barlow diag loss | 0.019703 | 0.005442 |
| offdiag abs mean | 0.402751 | 0.216527 |
| offdiag loss | 0.223741 | 0.070609 |
| view A effective rank | 3.322565 | 14.501471 |
| view B effective rank | 3.321497 | 14.593816 |

## Decision

The falsifier fired. The minimal geometry-aware encoder improves pointwise
alignment/cosine, but it collapses rank and loses same-state retrieval. Mean
pooling over token embeddings is likely too lossy, and the token descriptors
alone do not solve the geometry problem.

HEAD070 remains the current Part 1 reference candidate. The next iteration
should analyze whether any geometry-aware architecture is worth pursuing now,
or whether Part 1 should instead stabilize around HEAD070 and proceed with
probe/report consolidation.

## Artifacts

- `experiments/world/part1_jepa_latent/masked_multiview_geometry_barlow_smoke.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/evaluation/README.md`
- `experiments/world/reports/world_model_head076_geometry_barlow_smoke.md`
- `results/world/masked_multiview_geometry_barlow_head076.json`
- `models/world/checkpoints/part1_jepa_latent/masked_multiview_geometry_barlow_head076.pt`
