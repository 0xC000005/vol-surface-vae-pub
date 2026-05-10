# World Model HEAD069: Direct Barlow Loss-Scaling Analysis

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

Why did HEAD068 dramatically improve same-state retrieval while still leaving
low effective rank and high off-diagonal redundancy?

## Finding

The direct objective is directionally right, but the implemented Barlow loss is
not scaled like the canonical Barlow Twins loss.

HEAD068 used:

```text
mean((diag(C) - 1)^2) + 0.005 * mean(offdiag(C)^2)
```

The Barlow Twins paper uses summed diagonal and off-diagonal terms:

```text
sum_i (1 - C_ii)^2 + lambda * sum_{i != j} C_ij^2
```

If we keep mean reductions for stable reporting, the paper-style equivalent for
latent dimension `D` is:

```text
mean_diag + lambda * (D - 1) * mean_offdiag
```

For HEAD068, `D = 64`, so the canonical-average equivalent of `lambda = 0.005`
is an effective mean off-diagonal weight of `0.315`, not `0.005`.

## Quantitative Check

From `results/world/masked_multiview_barlow_head068.json` validation metrics:

| quantity | value |
| --- | ---: |
| latent dimension | 64 |
| configured offdiag weight | 0.005 |
| diagonal mean loss | 0.003628 |
| offdiag mean loss | 0.281904 |
| current loss formula | 0.005037 |
| canonical-average equivalent | 0.092427 |
| current offdiag share | 0.279822 |
| canonical offdiag share | 0.960751 |
| mean-style weight matching canonical scaling | 0.315 |

This explains the HEAD068 pattern: the model learned excellent diagonal
agreement and retrieval, but the off-diagonal penalty was too weak to force
decorrelation at the canonical Barlow scale.

## Decision

The next experiment should make one principled correction, not add a new family
of knobs:

- change the direct Barlow loss to support canonical mean-scaled reduction:
  `diag_mean + lambda * (D - 1) * offdiag_mean`;
- keep the default paper lambda `0.005`;
- keep the same architecture, masks, training budget, diagnostics, and raw
  baseline;
- compare directly against HEAD068.

Falsifier: if canonical scaling materially improves off-diagonal redundancy and
effective rank but destroys same-state retrieval, then the direct Barlow path
needs a representation/projection separation or encoder change. If it improves
redundancy while preserving retrieval, this becomes the Part 1 reference
candidate.

## Artifacts

- `results/world/masked_multiview_barlow_head068.json`
- `experiments/world/reports/world_model_head068_direct_barlow_smoke.md`
- `experiments/world/reports/world_model_head069_direct_barlow_loss_scaling_analysis.md`
