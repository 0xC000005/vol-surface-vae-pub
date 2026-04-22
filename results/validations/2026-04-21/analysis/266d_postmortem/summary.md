# 266d Postmortem

- checkpoint: `models/backfill/266d_v0_s42/best_model.pt`
- best epoch: `30`
- suite score: `1/11`
- passes: `block_ar`

## Headline Read

`266d-v0` made the cleanest possible generative-core swap inside the `266` family:
- keep the temporal bottleneck
- keep the decoder
- replace latent diffusion with vanilla conditional flow matching

It failed harder than `266c`.

The failure is clean:
the latent flow-matching prior collapsed to an almost deterministic point forecaster.

Coverage, conditional width, jump incidence, and scenario diversity all effectively vanished.

## Full Sample Metrics

- coverage90: `0.001`
- calibration error: `0.500`
- turb/calm width ratio: `0.925`
- ACF corr: `0.803`
- kurtosis ratio: `0.331`
- corr ratio: `1.557`
- rank ratio: `0.477`
- cointegration ratio: `0.705`
- MR ratio: `1.959`
- MR h30 ratio: `0.812`
- max-jump KS: `1.000`
- max-jump q99 ratio: `0.009`
- very-small-move ratio: `1.724`
- small-move ratio: `1.367`
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`

## Relative To 266c

Relative to `266c-v0`:
- coverage collapsed: `0.204 -> 0.001`
- calibration worsened: `0.405 -> 0.500`
- jump q99 collapsed: `0.439 -> 0.009`
- tiny-move oversmoothing worsened: `1.312 -> 1.724`

But several deterministic-structure metrics improved:
- corr ratio: `2.113 -> 1.557`
- rank ratio: `0.242 -> 0.477`
- cointegration ratio: `0.318 -> 0.705`
- ACF corr: `0.718 -> 0.803`

So `266d` did not become noisy junk. It became a **too-deterministic center-path generator**.

## Mechanism Conclusion

This is the important conclusion:

1. The temporal bottleneck and decoder can support a cleaner center path than `266c` showed.
2. The vanilla FM prior over deterministic encoded future tokens collapses stochasticity instead of learning conditional scenario spread.
3. So the deeper issue is no longer diffusion versus FM by itself.

The deeper issue is the modeling assumption that the future token path is a **deterministic target code** to be generated from history.

That assumption makes it too easy for the model to become a point forecaster.

## Decision

Treat the deterministic-target `266` family as closed.

Do **not** keep tuning diffusion versus FM inside the same latent-target setup.

The next principled move is a paradigm shift:

- keep the narrow bottleneck idea
- keep the architecture elegant
- but move to a **probabilistic latent-token model**
  where scenario variability is part of the model specification,
  not something the prior is asked to recover from deterministic future codes

## Next Question

What is the smallest first-principles probabilistic latent-token model that:
- keeps the clean bottleneck doctrine,
- stays publishable and generalizable,
- and makes scenario diversity explicit from the start?
