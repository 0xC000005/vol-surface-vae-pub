# 585a Recent Time-Series DNN Reset: Learned Conditional Prior, Not Retrieval

## Context

The 582c/583a/584a sequence clarified the current bottleneck. The unified
increment-flow data framing is useful, but the best prototype is source-dominated:
conditional empirical top-k source selection determines most sample quality, and
the learned flow changes the source paths only marginally. Tightening top-k would
turn the model into a retrieval stress deck. Small reconstructed-state penalties
do not create meaningful learned transport.

That means the next step should not be another local realism knob. The model needs
a prior/source distribution that is conditional and learnable, but not a historical
future-path lookup.

## External Literature Signal

Recent time-series work points to the same direction:

- Sundial / TimeFlow treats flow matching as a native continuous-valued
  probabilistic forecasting objective and explicitly motivates it as a way to avoid
  discrete tokenization and mode collapse.
- CW-Gen argues that prior-free diffusion/flow models ignore useful conditional
  mean and covariance information; it replaces the standard terminal Gaussian with
  a conditional mean/covariance prior and reports better multivariate probabilistic
  forecasts under distribution shift.
- vLinear's WFMLoss argues that final-series-oriented flow training can outperform
  velocity-oriented flow training, and that path/horizon weighting is useful when
  some future horizons are more reliable than others.
- MixLinear is a warning against assuming that bigger attention is always the
  missing ingredient; compact temporal/frequency structure can be enough if the
  problem framing is right.
- GCGNet reinforces that time and channel dependence should be modeled jointly,
  not as separated temporal and factor modules.
- FreqCycle and M2FMoE reinforce the importance of mid/high-frequency and extreme
  dynamics, but importing their full module stacks would violate the clean
  pathology guard at this stage.

## Local Implication

The cleanest synthesis is not a new large Transformer and not more source
selection. It is:

1. keep the unified IV-plus-anchor-factor increment panel;
2. keep one shared continuous generative core;
3. replace empirical path retrieval with a learned conditional source prior;
4. audit whether the learned flow now changes samples materially beyond the prior.

This is close to CW-Gen in spirit but minimal enough for this repo: a history
encoder predicts a conditional Gaussian prior over the full future increment
tensor, then the same flow transports that prior to the target path.

## Proposed Falsifier

Implement `586a = conditional-affine-source unified increment flow`.

Core:

- one GRU history encoder already present in `UnifiedIncrementFlow`;
- one conditional affine Gaussian source head over the flattened future increment
  path;
- one shared flow velocity network;
- one unified 38-variable target panel;
- no IV/factor-specific heads;
- no empirical future-path source bank;
- no evaluator-specific calibration.

Training:

- flow matching loss as before;
- small conditional-prior NLL term only to teach the source head a valid
  conditional mean/scale before transport;
- no top-k selection.

Acceptance read:

- If conditional source samples are realistic and post-flow deltas become
  materially nonzero, the branch is alive.
- If the prior is poor and the flow cannot recover, then the current MLP flow
  capacity/source design is insufficient and the next paradigm should be a
  final-series-oriented path-token model rather than more source tricks.

## Sources

- Sundial / TimeFlow: https://arxiv.org/abs/2502.00816
- CW-Gen: https://arxiv.org/abs/2509.20928
- vLinear / WFMLoss: https://arxiv.org/abs/2601.13768
- MixLinear: https://openreview.net/forum?id=QUj0KuCumD
- GCGNet: https://arxiv.org/abs/2603.08032
- FreqCycle: https://arxiv.org/abs/2603.09661
- M2FMoE: https://arxiv.org/abs/2601.08631
