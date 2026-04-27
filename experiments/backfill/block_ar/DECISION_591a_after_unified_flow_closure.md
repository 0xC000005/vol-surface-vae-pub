# 591a Paradigm Shift After Unified Path-Flow Closure

## Context

Iterations 575-590 tested a clean unified IV-plus-anchor-factor future-increment
program:

- 576 built the canonical reversible 38-variable increment panel.
- 577-584 tested independent, path-Gaussian, empirical, conditional empirical,
  and auxiliary-realism unified flows.
- 585 refreshed recent time-series DNN evidence and selected learned conditional
  priors over retrieval.
- 586-588 tested learned conditional-affine priors, cumulative/final-series loss,
  and official scoring.
- 589-590 tested a narrow learned latent source and source-temperature frontier.

The branch is now scientifically useful but below frontier.

## Closure Evidence

Best official scores inside this branch:

- 587a conditional-affine cumulative flow: `1/11`;
- 589a conditional-latent cumulative flow: `3/11`;
- 590a latent temperature `2.0`: `2/11`;
- 590a latent temperature `4.0`: `1/11`.

The branch taught two useful lessons:

- cumulative/final-series-oriented loss is necessary for stable reconstructed
  levels;
- a narrow common stochastic bottleneck is necessary for cross-cell dependence.

But it also exposed the blocker:

- full-dimensional source noise gives rank/correlation failure;
- latent source noise gives underdispersion;
- scalar temperature improves coverage/tails only by breaking dependence or
  surface validity;
- one-shot MLP path decoders do not recover local jump law, h1 mean reversion,
  coverage, regime behavior, and cross-cell structure simultaneously.

## Why Not Keep Tuning 575-590?

The remaining obvious knobs are not principled:

- latent dimension sweep;
- temperature sweep;
- source log-scale caps;
- MLP depth/width;
- IV-specific clamps;
- post-hoc path filters.

These would optimize symptoms while the mechanism is already clear. The one-shot
path-flow decoder is the wrong dependency class for this data and suite.

## Return Target

The only learned-law frontier remains the AR/state-space transition family:

- 392a: empirical-score AR transition flow with rollout energy, `8/11`;
- 510a: patch-energy final checkpoint, `8/11`;
- 526/527/530 factor-conditioned branch: below frontier but shows broader factor
  conditioning is mechanically safe;
- 564/567: risk-manager stress deck is presentable as policy overlay, not a
  calibrated learned law.

The next research branch should therefore start from the AR/state-space frontier,
not from one-shot unified path flow.

## Next Candidate

Name:

- `592a = AR common-latent transition flow`

Principle:

- keep the empirically successful AR transition/state-space geometry of
  `392a/510a`;
- add one scenario-level common stochastic latent to the rollout, shared across
  days and cells;
- feed that latent into the existing transition law at each rollout step;
- train/fine-tune with a final-series/patch objective so the latent affects
  long-horizon level occupancy and regime coverage;
- preserve the existing support-valid empirical-score coordinate and generated
  state feedback.

Why this is not a low-rank hack:

- the latent is a stochastic input, not a hard low-rank readout;
- the decoder remains the learned AR transition network;
- local daily moves are still produced stepwise from generated state;
- the common latent only supplies persistent scenario-level shock information.

## Acceptance Gate

Do not accept a model below the frontier.

The first 592a falsifier must recover at least the structural frontier:

- official score at least `8/11`, or
- if below `8/11`, it must improve one of the three frontier failures without
  losing conditionality, cointegration, mean reversion, cross-cell correlation, or
  pathwise jump realism.

If 592a cannot do that, the program should stop treating new neural architectures
as the likely route to `11/11` on this data and should return to the separated
risk-policy framing around `510a/564a`.
