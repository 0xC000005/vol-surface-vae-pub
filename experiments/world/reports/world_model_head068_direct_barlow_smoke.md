# World Model HEAD068: Direct Barlow Masked-Multiview Smoke

Date: 2026-05-09

Iteration type: `experiment`

## Literature Status

`supported_adjacent`: the objective is the Barlow Twins two-view
cross-correlation objective applied to structured masked views of the same
market window and same relative time row.

This is intentionally not a future-prediction objective and not an EMA/predictor
hybrid.

## Hypothesis / Falsifier

Hypothesis: for the current corruption-based Part 1 objective, a direct
two-view Barlow baseline is more principled and healthier than the HEAD066
EMA/predictor hybrid because the redundancy-reduction objective is applied to
the embeddings we evaluate.

Falsifier: direct Barlow still fails same-state retrieval/rank versus HEAD066
and the raw masked-view baseline.

## Implementation

Added `experiments/world/part1_jepa_latent/masked_multiview_barlow_smoke.py`.

The script uses:

- the same geometry-aware masked multiview data builder;
- the same value plus observed/synthetic mask input channels;
- one shared GRU encoder for both masked views;
- direct Barlow cross-correlation loss on per-time-row embeddings;
- the same HEAD065 alignment, retrieval, Barlow, rank, and visibility
  diagnostics.

Added `test_direct_masked_multiview_barlow_scores_encoder_embeddings` to cover
shared-encoder outputs and finite direct Barlow loss.

## Validation

Red test first:

```bash
pytest test_code/test_world_model_evaluation.py::test_direct_masked_multiview_barlow_scores_encoder_embeddings -q
```

Initial result: failed with `ModuleNotFoundError` for the new module.

After implementation:

```bash
pytest test_code/test_world_model_evaluation.py::test_direct_masked_multiview_barlow_scores_encoder_embeddings -q
```

Result: `1 passed in 0.71s`.

## Real-Data Smoke

Command:

```bash
python experiments/world/part1_jepa_latent/masked_multiview_barlow_smoke.py --device cpu
```

Training loss decreased from `0.037550` to `0.001835` over 8 epochs.

Validation metrics:

| metric | direct Barlow | raw masked-view baseline | HEAD066 hybrid |
| --- | ---: | ---: | ---: |
| alignment MSE | 0.004775 | 0.056997 | 0.064487 |
| cosine mean | 0.995122 | 0.908502 | 0.939162 |
| retrieval top1 | 0.394271 | 0.042969 | 0.000260 |
| retrieval top5 | 0.680990 | 0.204427 | 0.002344 |
| retrieval top10 | 0.853906 | 0.373177 | 0.006510 |
| Barlow diag mean | 0.952146 | 0.561084 | 0.378211 |
| Barlow diag loss | 0.003628 | 0.354536 | 0.439239 |
| Barlow offdiag abs mean | 0.468407 | 0.112105 | 0.204527 |
| view A effective rank | 4.484127 | 12.820380 | 4.134431 |
| view B effective rank | 4.525738 | 12.797703 | 9.047730 |

Saved result: `results/world/masked_multiview_barlow_head068.json`.

## Decision / Next Step

The direct Barlow baseline strongly improves same-state retrieval and alignment,
so HEAD067's diagnosis was useful. However, it is not yet a complete Part 1
solution: effective rank remains low and off-diagonal correlations are high.

The next iteration should be post-experiment analysis. The specific question is
whether the retrieval gain is coming from a small number of dominant latent
dimensions, and whether the off-diagonal redundancy is a loss-weight issue, a
projection-dimensionality issue, or an encoder/data geometry issue. Do not add
multiple knobs until that is understood.

## Artifacts

- `experiments/world/part1_jepa_latent/masked_multiview_barlow_smoke.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/evaluation/README.md`
- `experiments/world/reports/world_model_head068_direct_barlow_smoke.md`
- `results/world/masked_multiview_barlow_head068.json`
- `models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head068.pt`
