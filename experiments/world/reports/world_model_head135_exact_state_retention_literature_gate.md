# World Model HEAD135: Exact-State Retention Literature Gate

Date: 2026-05-10

## Iteration Type

`research_ideation`

## Objective Family

Design gate for `masked_multiview_invariance` versus possible
`context_to_target_jepa` or masked-reconstruction branches.

## Local Evidence

The scaled flat Barlow candidate is no longer failing through the easy failure
classes:

- representation health is stable across seeds `680/681/682`;
- corruption robustness passes on mask-family and stratified-mask audits;
- regime probes show some balanced minority-regime signal;
- downstream utility is structured by target family, not uniformly bad.

The active blocker is exact-state retention:

- current-IV MSE is `0.013756` for scaled Barlow versus `0.005630` for raw
  last-surface features;
- scaled Barlow is worse on `20/25` IV cells;
- persistence/exact-state targets are the downstream family where Barlow loses.

## Literature Check

- I-JEPA is a non-generative context-to-target objective: a context block
  predicts representations of target blocks in the same image. Its masking
  strategy is deliberately semantic: large target blocks plus informative,
  spatially distributed context. Source: <https://arxiv.org/abs/2301.08243>.
- V-JEPA uses feature prediction as the standalone self-supervised objective
  for video, evaluating frozen representations on image/video tasks rather than
  relying on pixel reconstruction. Source: <https://arxiv.org/abs/2404.08471>.
- Barlow Twins aligns embeddings from distorted views and reduces redundancy;
  it is a strong collapse/redundancy control, but the objective is view
  invariance and does not by itself guarantee retention of every state detail.
  Source: <https://arxiv.org/abs/2103.03230>.
- VICReg makes the invariance/variance/covariance decomposition explicit. It is
  useful collapse control, but the invariance term still says two views should
  agree, so view-specific details can be suppressed unless the view/target
  design makes them necessary. Source: <https://arxiv.org/abs/2105.04906>.
- TS2Vec is relevant because it learns timestamp-level time-series
  representations and then aggregates them for subsequences; this supports
  auditing representation surface and temporal pooling before changing the
  objective. Source: <https://arxiv.org/abs/2106.10466>.
- MAE-style masked reconstruction is a principled alternative when exact value
  recovery matters, but it is a different generative/reconstruction branch from
  JEPA-style latent prediction. Sources:
  <https://arxiv.org/abs/2111.06377> and <https://arxiv.org/abs/2301.08871>.

## Candidate Routes

| route | literature status | what it would test | risk |
| --- | --- | --- | --- |
| Representation-surface audit | supported by time-series timestamp-level representation work | Whether exact-state information exists in sequence/time surfaces before last pooling | No training change, but may reveal current reporting is using the wrong surface |
| Same-window context-to-target JEPA | canonical JEPA | Predict latent targets for masked current/history blocks, not future values | Objective-family change; needs explicit target branch and collapse checks |
| Masked value reconstruction | MAE-style adjacent branch | Recover masked current values directly where exact state matters | Can become low-level reconstruction and fight the semantic/invariance goal |
| Keep pure global Barlow | current supported-adjacent branch | Further scale and audit the existing invariant representation | Already diagnosed as compressing exact IV state |

## Decision

Do not add an ad hoc exact-value auxiliary loss as the next move.

The most conservative next step is a representation-surface audit: evaluate
whether the frozen scaled encoder's per-time or flattened sequence embeddings
retain exact current state better than the current last-state probe. This tests
whether the failure is the learned representation itself or the chosen
downstream readout surface.

If that audit fails, the principled objective change is not future prediction.
It is a same-window, same-history context-to-target JEPA branch for masked
current/history state blocks, with Barlow/VICReg health checks on the evaluated
encoder surface. MAE-style value reconstruction should remain a separate
diagnostic branch, not the default JEPA route.
