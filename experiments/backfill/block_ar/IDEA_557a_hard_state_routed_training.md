# 557a Hard-State Routed Training Ideation

## Local Evidence

The current risk candidates are not failing because of global width or persistent scenario collapse.

556a showed:

- 510a and 555a both score `8/11`.
- Both pass layer3 persistent severe undercoverage.
- Both fail regime layer2 through localized regime/horizon/cell occupancy.
- Stable undercovered cells across both include:
  - turbulent h30 cell `[2,3]`
  - turbulent h7 cell `[4,3]`
  - calm h1 cell `[1,0]`
- 555a improves conditionality but worsens the worst risk-regime cell, so factor conditioning alone is not the missing piece.

This rules out the next obvious bad moves:

- no global interval widening;
- no post-hoc factor score policy;
- no per-cell evaluator-time calibration table;
- no larger factor side-channel just because factors have signal.

## Recent Literature Signals

The recent time-series papers that are relevant to this failure mode point toward routed/specialized modeling of rare regimes, not a monolithic global correction.

- M2FMoE, accepted by AAAI 2026, frames extreme events as sparse but high-impact states and uses multi-resolution / multi-view frequency mixture-of-experts to learn regular and extreme patterns without explicit extreme-event labels. Source: https://arxiv.org/abs/2601.08631
- Super-Linear uses lightweight frequency-specialized linear experts with a spectral router and emphasizes efficiency, robustness to sampling rates, and interpretability. Source: https://arxiv.org/abs/2509.15105
- MoLE shows that a router over simple linear experts can adapt linear-centric forecasters to different temporal patterns. Source: https://arxiv.org/abs/2312.06786
- GCGNet targets exogenous-variable forecasting by modeling temporal and channel correlations jointly and aligning generated correlation structure to data-derived graph structure. Source: https://arxiv.org/abs/2603.08032

The common transferable idea is not "add a complicated conference architecture." It is simpler:

> keep the base generative law clean, but make training pay disproportionate attention to sparse hard conditional states, and if needed route a small number of simple source/path experts rather than globally widening every scenario.

## Candidate Directions

### Recommended Next Experiment: Training-Only Hard-State Replay

Use the current 510a frontier as the source model.

1. Sample the pre-validation calibration block with the frozen source model.
2. Compute hard-state scores from calibration only:
   - window/cell undercoverage,
   - upper stress misses,
   - path jump miss,
   - optionally future-level miss.
3. Fine-tune the existing empirical-normal-score core with the same patch-energy style objective, but reweight minibatches/windows by hard-state score.
4. Do not store per-cell tables for inference.
5. Do not change the evaluator.
6. Evaluate on the official validation suite and the risk-readiness audit.

This is defensible as hard-example replay / importance-weighted empirical risk minimization. It uses realized futures only in the training block, not in validation or inference. It directly targets the observed pathology while keeping the deployed sampler a learned generator.

### Backup Direction: Small Routed Source-Prior Experts

If hard-state replay fails, add a minimal source-prior mixture:

- one regular source expert;
- one extreme/path-jump source expert;
- one level-shift source expert;
- a history-only router.

This borrows the routed-expert idea from M2FMoE, Super-Linear, and MoLE, but keeps it much smaller than a new architecture. It should only be attempted after hard-state replay, because it adds deployed model complexity.

### Not Recommended Now: Graph-Consistency Core

GCGNet-style graph alignment is relevant because risk scenarios need temporal/channel coherence, but our cross-cell correlation already passes. Graph consistency may help later for multi-factor scenario generation, but it is not the immediate bottleneck for IV-only risk deployability.

## Decision

The next HEAD experiment should be a training-only hard-state replay patch-energy fine-tune from 510a. It is the smallest clean move that directly addresses localized regime/cell under-inclusion without adding post-hoc calibration or evaluator-specific inference logic.
