# 646a Post-645 Paradigm Decision

## Context

The 641a line solved one major scientific issue: native IV plus anchor-factor generation without post-hoc gluing. But the follow-up attempts did not make the model deployable:

- 641a: clean native joint baseline, but IV score `4/11`.
- 643a: scalar temperature widened coverage/pathwise but damaged conditionality, kurtosis, tail scale, and cross-cell structure.
- 645a: local conditional distribution alignment improved pathwise KS and correlations but worsened IV coverage, conditionality, level placement, median bias, and factor marginal quality.

## Failure Mechanism

The active family is now scientifically legible but capped:

1. One-step/free-running transition training is good at local daily changes, support validity, and co-movement.
2. It is bad at allocating full-path conditional probability mass over 30 days.
3. Post-hoc or finetune objectives can move one metric group, but they damage another because the base object is still a one-step transition law rolled forward.

The repeated pattern is not random:

- widening samples helps coverage/pathwise but destroys tail/correlation balance;
- local distribution objectives help path geometry but worsen conditional location;
- likelihood-family changes improve coverage but damage level KS/tails/mean reversion;
- final-path one-realization objectives narrow or bias the scenario deck.

## Decision

Do not keep stacking auxiliary losses, source priors, temperatures, or local-neighborhood objectives onto 641a.

The next publishable path should be a paradigm shift to a direct sequence-level conditional path law:

- input: history panel in the same generic state representation;
- output law: the entire future path `[horizon, channels]` in a support-valid mixed coordinate;
- generative core: vanilla flow matching or diffusion over the full future tensor;
- architecture: one shared temporal/channel transformer or local-attention denoiser;
- no separate IV/factor treatment;
- no low-rank decoder;
- no retrieval at inference;
- no hand-coded stress deck.

This is closer to the first-principles target: train the same object that is sampled and evaluated, the full conditional future path.

## Why This Is Not Repeating Old One-Shot Failures

The old 575-590 one-shot family failed mostly as shallow unified-flow/MLP variants with weak future structure and poor IV gates. The proposed reset differs in the parts now justified by evidence:

- use the 641a mixed coordinate so IV support and factor increments are represented correctly;
- use a path-native objective from the start, not a one-step AR objective plus later repair;
- use temporal/channel attention over future tokens rather than a compressed MLP bottleneck;
- train as one generic 38-channel path law that can also run IV-only by changing `D`.

## Next Experiment

647a should be a minimal full-path mixed-coordinate flow:

- start with `joint38`, horizon 30, same train/val framing as 641a;
- train a full-future flow target:
  - `iv:*`: future level-score path deltas from current score;
  - `factor:*`: future encoded-increment path;
- condition on history level/increment scores through one causal history encoder;
- denoise/generate all future `[T,D]` tokens jointly with a simple transformer;
- evaluate IV 11-suite and joint-panel audit.

Falsifier:

- If it cannot beat 641a on IV coverage/location while preserving joint factor audit, direct path-law modeling is not an immediate rescue, and 641a should be frozen as the clean native joint risk-stress baseline.
