# 575a Unified Joint Increment Flow

## Context

574a is the current best risk-manager joint stress product: it combines the accepted
IV stress deck with anchor-factor overlays matched by selected severity quantile.
That is operationally useful, but it is not a calibrated joint conditional law.

The user corrected the external research target: the relevant venue class is time
series forecasting literature and time-series journals/conferences, not
cybersecurity. The methodological requirement is also stricter now:

- do not treat IV and anchor factors as separate model families;
- do not keep improving a policy-composed stress deck as the publication path;
- learn one conditional distribution over future financial-factor paths;
- keep the architecture clean enough to generalize to more factors and longer
  horizons.

## Recent Literature Signal

The useful time-series signal is consistent with the failure pattern in our own
experiments:

- Sundial / TimeFlow uses flow matching as a native continuous-valued forecasting
  loss and explicitly frames it as a way to mitigate mode collapse in
  probabilistic forecasting.
- CW-Gen argues that multivariate probabilistic forecasting is hard because of
  non-stationarity, inter-variable dependence, and distribution shift, and that
  conditional mean/covariance information can improve diffusion or flow models.
- TSFlow supports data-dependent priors for time-series flow matching rather than
  a fixed standard normal prior.
- AdaPTS and related foundation-model-adapter work support a unified multivariate
  representation layer for heterogeneous variables rather than separate bespoke
  models for each factor type.
- TabPFN-TS is a reminder that data framing and lightweight temporal features can
  matter as much as architecture; this matters for our factor panel before we add
  another large model.

The common implication is not "add another IV correction" or "add another anchor
factor postprocessor." The common implication is:

> model the future path as one continuous multivariate conditional law, with
> reversible factor-specific data transforms and a shared generative core.

## Proposed Model Class

Name:

- `575a = Unified Joint Increment Flow (UJIF)`

The model's sample is one future tensor:

- shape: `horizon x variable`;
- variables: all IV cells plus all anchor-factor states in one list;
- target: transformed future increments, not duplicate independent future levels.

Canonical variable treatment:

- IV cells: transform levels to a stable continuous coordinate, then model future
  increments in that coordinate and reconstruct levels afterward.
- equity, FX, commodity levels: model log-return increments and reconstruct levels.
- rates and spreads: model arithmetic differences and reconstruct levels.

The transform metadata is allowed because it defines units and reversibility, not a
separate architecture. A variable metadata embedding is also allowed if it only
describes coordinate type, surface location, and factor identity. The generative
core must remain shared.

## Core Architecture

Minimal V0:

- one shared history encoder over the full factor panel;
- one shared future-variable/time representation;
- one vanilla continuous generative core, preferably conditional flow matching;
- one output tensor over all future increments;
- one reversible reconstruction path for all variables.

What V0 should not contain:

- separate IV model plus factor model;
- separate stress selector as part of the probability model;
- duplicated factor level and factor increment targets;
- hard low-rank readout;
- hand-coded bounded idiosyncratic or error-correction side paths;
- evaluator-specific post-hoc corrections.

This stays aligned with the bitter-lesson reset: make the data representation
correct, give the model the full joint tensor, and evaluate the law directly before
adding specialized structure.

## Objective

The first trainable version should use a small objective set:

- flow-matching loss on the full future increment tensor;
- optional joint proper-score term such as energy or sliced-Wasserstein over full
  future paths if flow-only samples are too narrow;
- no separate risk-deck calibration during base-law training.

Policy stress selection can exist downstream, but it must be reported separately:

- base learned law metrics;
- optional risk-manager stress-deck metrics derived from the learned law.

## Evaluation Contract

The model must be evaluated as one joint scenario generator:

- existing IV 11-suite on reconstructed IV surfaces;
- factor marginal and path realism on reconstructed anchor-factor levels;
- cross-factor dependence and projected-portfolio coverage;
- severity-conditional behavior across the combined panel;
- longer-horizon smoke tests at 60, 90, and 152 or 252 days once 30-day framing is
  verified.

Level-frequency KS should remain diagnostic rather than a hard risk-deployability
blocker, but it is still useful for the publication-grade conditional-law report.

## Most Principled Next Experiment

Do not train the model first. Build the data framing and audit first.

Next iteration should implement:

- `576a_unified_increment_panel_dataset`;
- split-safe construction of one variable list containing IV cells and anchor
  factors;
- reversible transform/inverse-transform checks;
- no duplicated future-level target channels;
- shape checks for 30-day and longer horizons;
- a small oracle/replay audit showing that reconstructed IV and factor levels are
  exactly recoverable from canonical increments.

Only after this passes should we train a small V0 conditional flow model.

## Decision

574a remains the preferred risk-manager stress product. It should not be the
publication path.

The publication path should pivot to a native joint increment-law model. The
current evidence says the bottleneck is not a missing IV-specific trick; it is that
we still do not have one unified conditional probability law over all factors.

## Sources

- Sundial / TimeFlow: https://arxiv.org/abs/2502.00816
- CW-Gen: https://arxiv.org/abs/2509.20928
- TSFlow: https://arxiv.org/abs/2410.03024
- AdaPTS: https://arxiv.org/abs/2502.10235
- TabPFN-TS: https://arxiv.org/abs/2501.02945
