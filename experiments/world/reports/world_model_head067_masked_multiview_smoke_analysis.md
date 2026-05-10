# World Model HEAD067: Masked Multiview Smoke Analysis

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

Why did HEAD066 improve alignment/cosine but fail same-state retrieval and rank?

## Evidence

HEAD066 validation metrics show a classic low-rank smoothing failure:

| metric | predicted-target | context-target | raw masked-view baseline |
| --- | ---: | ---: | ---: |
| alignment MSE | 0.064487 | 1.059583 | 0.067611 |
| cosine mean | 0.939162 | 0.489508 | 0.891644 |
| retrieval top1 | 0.000260 | 0.001302 | 0.035156 |
| retrieval top5 | 0.002344 | 0.005729 | 0.190885 |
| retrieval top10 | 0.006510 | 0.011198 | 0.357031 |
| Barlow diag mean | 0.378211 | 0.438019 | 0.521182 |
| Barlow offdiag abs mean | 0.204527 | 0.198248 | 0.106873 |
| view A effective rank | 4.134431 | 5.761601 | 12.916789 |

The singular-value concentration is also worse than the raw baseline:

| representation | top1 share | top3 share | top5 share | effective rank |
| --- | ---: | ---: | ---: | ---: |
| validation predicted | 0.285542 | 0.615311 | 0.782305 | 4.134431 |
| validation context | 0.231095 | 0.480394 | 0.637851 | 5.761601 |
| validation target | 0.153529 | 0.387828 | 0.559758 | 9.047730 |
| raw validation view A | 0.113825 | 0.266199 | 0.371825 | 12.916789 |

The predictor produces very high cosine to the target, but it compresses many
dates into a small part of latent space. Context embeddings are not merely
being harmed by the predictor; context-target retrieval is also near zero.

## Literature Gate

Primary sources:

- Barlow Twins, arXiv:2103.03230:
  https://arxiv.org/abs/2103.03230
- I-JEPA, arXiv:2301.08243:
  https://arxiv.org/abs/2301.08243

Classification:

- EMA/stop-gradient target prediction is `canonical_jepa` for a masked
  context-to-target objective.
- Barlow-style cross-correlation on two corrupted views is
  `supported_adjacent` for the current same-state masked-multiview objective.
- The exact HEAD066 hybrid, where Barlow is applied mainly to
  predictor-to-EMA-target outputs, is less cleanly supported than either
  source family on its own.

Barlow Twins is especially relevant to the current objective because it is
defined on two distorted views of the same sample and its redundancy-reduction
term is intended to avoid collapse without requiring a predictor, stop-gradient,
or EMA target. I-JEPA is relevant when the task is context-to-target latent
prediction under masks; it does not by itself justify using future forecasting
or a low-rank predictor as the pretraining goal.

## Diagnosis

The most likely failure is objective mismatch:

- The task we now want is same-state multiview invariance.
- The HEAD066 architecture still used a JEPA-style predictor and EMA target.
- The alignment term rewarded the predictor for matching a moving target
  embedding.
- The Barlow term did not sufficiently protect the actual context
  representation used downstream.
- High cosine therefore became achievable without preserving instance/date
  identity, which is exactly what same-state retrieval measures.

This is not evidence that masked multiview pretraining is wrong. It is evidence
that the first hybrid implementation optimized the wrong surface of the model.

## Decision

The next experiment should remove, not add, moving parts:

- train a direct two-view Barlow baseline over the same masked multiview data;
- use one shared encoder for both views;
- apply the Barlow objective directly to the encoder embeddings that will be
  evaluated;
- keep the same HEAD065 diagnostics and raw masked-view baseline;
- do not add future prediction, range prediction, neighborhood losses, or a new
  collection of research knobs.

Falsifier for the next experiment: if direct two-view Barlow still fails to
improve same-state retrieval/rank over HEAD066 and the raw masked-view baseline,
then the issue is likely in the encoder architecture or mask/data geometry, not
just the EMA/predictor hybrid.

## Artifacts

- `results/world/masked_multiview_jepa_head066.json`
- `experiments/world/reports/world_model_head066_masked_multiview_jepa_smoke.md`
- `experiments/world/reports/world_model_head067_masked_multiview_smoke_analysis.md`
