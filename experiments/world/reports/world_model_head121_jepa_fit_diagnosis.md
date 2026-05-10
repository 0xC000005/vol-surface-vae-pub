# World Model HEAD121: JEPA Fit Diagnosis

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`masked_multiview_invariance` failure diagnosis against JEPA literature.

## Hypothesis

HEAD070 is not beating simple market-state baselines because the current
masked-view task is too easy and too close to raw-state preservation, not
because the representation has collapsed.

## Falsifier

This diagnosis would be wrong if the default masks were already aggressive,
the two views had low overlap, raw baselines were weak, or JEPA literature
showed that mild two-view alignment should reliably beat raw-state baselines.

## Literature Anchors

- I-JEPA frames masking as a core design choice: target blocks need semantic
  scale and the context must be informative/distributed.
- V-JEPA uses masked feature prediction, a predictor, stop-gradient/EMA
  target encoder, and large continuous spatio-temporal masks; it explicitly
  compares downstream frozen representations and reports data-scale effects.
- Barlow Twins supports direct two-view redundancy reduction, but it is an
  invariance method over distorted views and benefits from high-dimensional
  outputs; it does not by itself guarantee superiority over raw features on
  low-level persistence-like tasks.
- Recent time-series JEPA work points to multi-resolution dynamics, regime
  structure, and task-aligned latent prediction/control as important when
  precursor signals live across multiple temporal scales.

Sources:

- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2404.08471
- Barlow Twins: https://arxiv.org/abs/2103.03230
- MTS-JEPA: https://arxiv.org/abs/2602.04643
- LaT-PFN: https://arxiv.org/abs/2405.10093
- TS-JEPA: https://arxiv.org/abs/2406.04853

## Local Mask Difficulty

| split | view A hidden | view B hidden | both visible | both hidden | view disagreement | union hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6.95% | 7.41% | 86.44% | 0.79% | 12.77% | 13.56% |
| val | 6.63% | 7.76% | 86.60% | 1.00% | 12.40% | 13.40% |

The default validation views keep most observed entries visible in both
views. That makes same-state alignment a useful sanity check, but not a
hard semantic missing-information problem.

## Local Representation And Probe Evidence

- Same-state retrieval is healthy: top10 `0.841927` versus raw `0.373177`, MRR `0.478177` versus raw `0.149323`.
- Effective rank is non-collapsed: `14.501471`.
- Validation view A visible rates: overall `93.37%`, IV surface `88.97%`, factor levels `97.20%`.
- Barlow beats the mean baseline on `5/5` targets, is best standalone on `2/5`, and improves raw last-surface features on `3/5` targets.

## Diagnosis

- The current branch is not wrong as a collapse-controlled masked multiview invariance smoke test, but it is weaker than canonical JEPA as a market-state learning recipe.

Most likely reasons:

- The mask is too mild: validation views keep about 92-93% of observed entries, leaving most raw state visible in both views.
- The objective aligns two heavily overlapping corrupted full windows instead of predicting large missing target regions from a distributed context.
- Raw last-surface baselines are strong for persistence-like targets because they preserve exact level information that an abstract embedding may compress away.
- The current model is smoke scale: 384 training windows, 8 epochs, latent_dim 64, hidden_dim 128.
- The probes mix targets that reward exact low-level state copying with targets that reward path-shape abstraction; Barlow only wins the latter family today.

Not supported by current evidence:

- Representation collapse is not the main failure class.
- Mask-artifact leakage is not the main failure class under the audited default masks.
- There is not enough evidence to conclude the Barlow objective is intrinsically wrong.

## Decision

We are not ready to change the objective yet. First run a controlled
hard-mask diagnostic and stronger frozen probes. If the same encoder/loss
improves under harder, lower-overlap masks, then the issue was mask
difficulty. If it does not, the next likely bottleneck is objective family
or capacity/scale.

Next experiment shape:

- Do not add many knobs. Add one controlled hard-mask diagnostic preset before changing architecture.
- Compare current masks against a hard structured preset with much lower overlap, e.g. larger surface blocks, longer time blocks, whole geometry groups, and cross-family stress masks.
- Keep the same encoder/loss for the first hard-mask diagnostic so the failure class is mask difficulty rather than architecture churn.
- Score not only retrieval but also frozen present-state probes, factor-panel probes, and incremental value over raw/PCA/persistence baselines.
