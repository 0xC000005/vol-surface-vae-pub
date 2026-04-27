# 644a Objective Ideation After 641a

## Context

641a solved the native-joint construction problem but not IV conditional calibration. 643a then falsified scalar sample temperature. The next move must improve state/cell/horizon-specific probability allocation without separating IV and anchor factors.

## Recent Literature Signals

Sources checked:

- DistDF, ICLR 2026: https://openreview.net/forum?id=VrdLwUmzBy
- TSFlow, ICLR 2025: https://openreview.net/forum?id=uxVBbSlKQ4
- Local Geometry Attention, ICLR 2026: https://openreview.net/forum?id=NCQPCxN7ds
- MixLinear, ICLR 2026: https://openreview.net/forum?id=QUj0KuCumD

The relevant signal is not "use a bigger Transformer." The useful theme is that time-series models need objectives and priors that reflect the future path distribution and local temporal geometry:

- DistDF argues for joint-distribution Wasserstein alignment because direct forecast losses can be biased under label autocorrelation.
- TSFlow argues that flow matching benefits when the source/prior is closer to time-series temporal structure than simple white noise.
- Local Geometry Attention argues that local time-series geometry is important under noise, anomalies, and shifts.
- MixLinear is a warning against architectural overbuild: simple temporal/frequency structure can be competitive in low-resource settings.

## Existing Repo Evidence

Prior experiments already falsify several tempting interpretations:

- Global sample temperature is not enough: 611a and 643a both damage structure while only partly improving coverage.
- Naive learned source scale is not enough: 612a improved IV count but collapsed to the lower scale clamp and had poor anchor-factor realism.
- One-realization final-path losses are not enough: 596a worsened coverage and median placement despite a literature-aligned path objective.
- Likelihood-family changes alone are not enough: 616a Student-t improved coverage but damaged level KS, tail scale, and mean reversion.

The missing piece is not more width or a different marginal tail family. It is estimating a conditional future-path distribution from sparse historical realizations.

## Candidate Directions

### A. Local Conditional Distribution Alignment

Use local history neighborhoods during training to form an empirical conditional target distribution, then align generated future paths to that local target with a sliced Wasserstein or energy-style discrepancy.

This borrows the right parts of DistDF and Local Geometry Attention:

- DistDF: align joint future distributions, not pointwise MSE.
- LGA: use local temporal geometry, not global unconditional batches.

This is not retrieval at inference. It only supplies a training objective from nearby histories so the model sees more than one plausible future per condition.

### B. Structured Source Prior For 641a

Add a TSFlow-inspired temporally structured source to the mixed-coordinate model, such as AR/GP-like source noise over the future path.

This may improve pathwise coherence, but past source-prior experiments and the 643 temperature result warn that source shaping can become another sampling knob unless it is learned or strongly justified.

### C. Direct Sequence-Level Likelihood

Return to a likelihood model but train a full future-path law rather than teacher-forced one-step transitions.

This is principled, but prior Gaussian/Student-t likelihood runs showed large trade-offs, and implementing a good full-path likelihood would be a larger paradigm shift.

## Decision

The next clean experiment should be A: local conditional distribution alignment as a finetune objective for the existing 641a model.

One causal statement:

> Because each history has only one realized future, train the generator against the empirical future distribution of nearby histories instead of one realized future or a global unconditional batch.

Guardrails:

- one shared IV-plus-anchor-factor model;
- no retrieval or neighbor lookup at inference;
- no separate IV/factor losses;
- one local-neighborhood objective plus an FM anchor;
- evaluate with the same IV full suite and joint-panel audit.

Falsifier:

- If the objective does not improve IV coverage/level placement without damaging 641a's joint factor audit, close this direction and move to a larger sequence-level probabilistic model.
