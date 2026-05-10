# World Model HEAD066: Masked Multiview JEPA Smoke

Date: 2026-05-09

Iteration type: `experiment`

## Literature Status

`canonical_jepa` for the EMA/stop-gradient target encoder and latent
same-state alignment. `supported_adjacent` for Barlow-style redundancy control,
which this branch already restricts to true masked-multiview positive pairs.

No future prediction or future range target is used as a pretraining objective.

## Hypothesis / Falsifier

Hypothesis: a small sequence encoder can learn same-state masked-view
embeddings from value plus observed/synthetic mask channels using alignment and
Barlow-style redundancy control.

Falsifier: the smoke improves only pointwise alignment while same-state
retrieval, rank, or off-diagonal redundancy become worse than the raw masked
view baseline.

## Implementation

Added `experiments/world/part1_jepa_latent/masked_multiview_jepa_smoke.py`.

The smoke uses:

- feature channels: masked value, real observed mask, synthetic SSL mask;
- context encoder: GRU over the history window, returning a latent for each
  relative time row;
- target encoder: EMA/stop-gradient copy of the context encoder;
- predictor: small MLP over each context latent;
- loss: predicted-target MSE plus Barlow diagonal/off-diagonal loss;
- diagnostics: HEAD065 same-state alignment, retrieval, Barlow
  cross-correlation, representation health, and visibility summaries.

Added a unit test covering feature construction, per-time-row output shapes,
and finite loss.

## Validation

Red test first:

```bash
pytest test_code/test_world_model_evaluation.py::test_masked_multiview_jepa_uses_mask_channels_and_scores_time_rows -q
```

Initial result: failed with `ModuleNotFoundError` for the new module.

After implementation:

```bash
pytest test_code/test_world_model_evaluation.py::test_masked_multiview_jepa_uses_mask_channels_and_scores_time_rows -q
```

Result: `1 passed in 0.71s`.

## Real-Data Smoke

Command:

```bash
python experiments/world/part1_jepa_latent/masked_multiview_jepa_smoke.py \
  --epochs 6 \
  --batch_size 64 \
  --max_train_windows 512 \
  --max_val_windows 128 \
  --device cpu \
  --output_json results/world/masked_multiview_jepa_head066.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/masked_multiview_jepa_head066.pt
```

Training loss decreased from `0.327894` to `0.044515`.

Validation predicted-target metrics:

| metric | model | raw masked-view baseline |
| --- | ---: | ---: |
| alignment MSE | 0.064487 | 0.067611 |
| alignment cosine mean | 0.939162 | 0.891644 |
| retrieval top1 | 0.000260 | 0.035156 |
| retrieval top5 | 0.002344 | 0.190885 |
| retrieval top10 | 0.006510 | 0.357031 |
| Barlow diag mean | 0.378211 | 0.521182 |
| Barlow diag loss | 0.439239 | 0.404900 |
| Barlow offdiag abs mean | 0.204527 | 0.106873 |
| view A effective rank | 4.134431 | 12.916789 |
| view B effective rank | 9.047730 | 12.771882 |

Saved result: `results/world/masked_multiview_jepa_head066.json`.

## Decision / Next Step

The first smoke is useful but not yet a viable Part 1 model. It improves
pointwise alignment and cosine, but retrieval is far worse than the raw masked
view baseline, off-diagonal correlation is higher, and predicted embeddings are
low-rank.

Next iteration should be `post_experiment_analysis`, not a new knob. The main
question is whether this architecture/loss is optimizing an easy smoothing
solution: high cosine to the EMA target without preserving enough instance/date
identity for same-state retrieval.

## Artifacts

- `experiments/world/part1_jepa_latent/masked_multiview_jepa_smoke.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/evaluation/README.md`
- `experiments/world/reports/world_model_head066_masked_multiview_jepa_smoke.md`
- `results/world/masked_multiview_jepa_head066.json`
- `models/world/checkpoints/part1_jepa_latent/masked_multiview_jepa_head066.pt`
