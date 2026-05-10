# World Model HEAD155: Surface-Local Smoke Failure Diagnosis

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`token_geometry_level_context_to_target_jepa` smoke diagnosis.

## Hypothesis

HEAD154's loss drop is insufficient if target-token retrieval remains weak,
predicted variance shrinks, or the target latent itself is low-rank.

## Findings

- Loss improved: `True`.
- Loss delta: `-0.184330`.
- Retrieval top10: `0.054688`.
- Random top10 for subset: `0.019531`.
- Top10/random: `2.800000`.
- Median retrieval rank: `187.500000`.
- Predicted effective rank fraction: `0.164412`.
- Target effective rank fraction: `0.197918`.
- Predicted/target rank ratio: `0.830708`.
- Predicted/target variance ratio: `0.136648`.

## Decision

Promotion decision: `DO_NOT_PROMOTE`.

- Predictor variance shrinkage: `True`.
- Target latent also low-rank: `True`.
- Retrieval only weakly above random: `True`.

The smoke reduces loss, but both the target and predicted token latents are low-rank; the predictor additionally shrinks variance. This is not a mask-coverage problem and should be diagnosed before any architecture knob tuning.

Next: Audit selected target-token latents by geometry/family and compare predictor retrieval against target-latent intrinsic separability.
