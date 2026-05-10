# World Model HEAD070: Canonical Direct Barlow Scaling

Date: 2026-05-09

Iteration type: `experiment`

## Literature Status

`supported_adjacent`: this keeps the Barlow Twins two-view objective and fixes
the loss scaling to match the paper's summed off-diagonal term while retaining
mean reductions in code.

No future-prediction objective was added.

## Hypothesis / Falsifier

Hypothesis: HEAD068's low-rank/high-redundancy failure was partly caused by
underweighting off-diagonal terms. A canonical mean-scaled Barlow loss should
improve rank and redundancy while preserving most same-state retrieval.

Falsifier: rank/redundancy do not improve, or retrieval collapses back toward
the HEAD066 hybrid.

## Implementation

Updated `torch_barlow_cross_correlation_loss` with
`canonical_mean_scale=True`, which uses:

```text
diag_mean + lambda * (D - 1) * offdiag_mean
```

Updated the direct Barlow smoke to use canonical mean scaling by default. The
model architecture, masks, training budget, optimizer, and diagnostics stayed
unchanged from HEAD068.

Added a unit test that verifies canonical mean-scaled loss equals the expected
`lambda * (D - 1)` off-diagonal scaling.

## Validation

Focused tests:

```bash
pytest test_code/test_world_model_evaluation.py::test_barlow_loss_supports_canonical_mean_scaled_offdiag \
  test_code/test_world_model_evaluation.py::test_direct_masked_multiview_barlow_scores_encoder_embeddings -q
```

Result: `2 passed in 0.73s`.

Real-data smoke:

```bash
python experiments/world/part1_jepa_latent/masked_multiview_barlow_smoke.py --device cpu
```

Training loss decreased from `0.067116` to `0.013533` over 8 epochs.

## Result

Validation comparison:

| metric | HEAD070 canonical | HEAD068 weak offdiag | raw masked-view baseline | HEAD066 hybrid |
| --- | ---: | ---: | ---: | ---: |
| alignment MSE | 0.007052 | 0.004775 | 0.056997 | 0.064487 |
| cosine mean | 0.992878 | 0.995122 | 0.908502 | 0.939162 |
| retrieval top1 | 0.321354 | 0.394271 | 0.042969 | 0.000260 |
| retrieval top5 | 0.662500 | 0.680990 | 0.204427 | 0.002344 |
| retrieval top10 | 0.841927 | 0.853906 | 0.373177 | 0.006510 |
| Barlow diag mean | 0.930924 | 0.952146 | 0.561084 | 0.378211 |
| Barlow diag loss | 0.005442 | 0.003628 | 0.354536 | 0.439239 |
| offdiag abs mean | 0.216527 | 0.468407 | 0.112105 | 0.204527 |
| offdiag loss | 0.070609 | 0.281904 | 0.036859 | n/a |
| view A effective rank | 14.501471 | 4.484127 | 12.820380 | 4.134431 |
| view B effective rank | 14.593816 | 4.525738 | 12.797703 | 9.047730 |

The falsifier did not fire. Canonical scaling sharply improved rank and
off-diagonal redundancy while keeping retrieval far above the raw baseline and
the HEAD066 hybrid. Retrieval is lower than HEAD068, but the representation is
much healthier.

## Decision / Next Step

HEAD070 is the current Part 1 reference candidate for masked-multiview
pretraining. It is not final: off-diagonal redundancy is still above the raw
baseline, and the next iteration should analyze whether that residual
redundancy is acceptable or whether a single architecture-level change, such as
separating representation and projection, is warranted.

Do not move to the flow decoder yet.

## Artifacts

- `experiments/world/part1_jepa_latent/masked_multiview_jepa_smoke.py`
- `experiments/world/part1_jepa_latent/masked_multiview_barlow_smoke.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/evaluation/README.md`
- `experiments/world/reports/world_model_head070_canonical_barlow_scaling.md`
- `results/world/masked_multiview_barlow_head070.json`
- `models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt`
