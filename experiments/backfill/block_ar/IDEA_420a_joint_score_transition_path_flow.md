# 420a: Joint Normal-Score Transition-Path Flow

## Context

The active frontier remains `392a` at `8/11`. The latest 419a student-forced AR fine-tune made the mechanism sharper:

- it removed most undercoverage and preserved cointegration / correlation / mean reversion,
- but it over-widened high-coverage cells, weakened conditionality, made small moves too rare, and worsened level KS from `10/25` to `7/25`.

That means off-policy AR transition fine-tuning is acting mostly as a learned width actuator. A weaker weight could interpolate between 392a and 419a, but the endpoint already moved level occupancy in the wrong direction. Running a weight sweep would be knob accumulation.

## Hypothesis

The clean missing object is the joint law of the whole future transition path:

```text
history -> flow over (z_1 - z_0, z_2 - z_1, ..., z_30 - z_29)
```

where `z_t` is the empirical normal-score surface and `z_0` is the last history surface.

This sits between the two capped families:

- 392a AR transition FM learns excellent local one-day mechanics but relies on recursive rollout to imply the 30-day level law.
- 413/417 direct future-level path FM makes the 30-day level path native, but loses transition geometry and structural coupling.

A joint transition-path flow makes the entire 30-day stochastic transition sequence native while reconstructing levels by cumulative summation from the observed last history state. It avoids recursive exposure bias without asking a direct level model to rediscover transition geometry from scratch.

## Model

Minimal implementation:

- empirical normal-score coordinate, same recent framing line as 340/385/392;
- target variable: 30 x 25 score-transition tensor;
- source noise: IID Gaussian tensor of the same shape;
- velocity input: history scores plus the noised cumulative future score path derived from the noised transition tensor;
- token features: noised level and noised transition at each future step;
- sampler: draw one full transition path, integrate the flow, cumulative-sum transitions from the last history score, then decode through the empirical quantile table.

No post-hoc scaling, retrieval, low-rank readout, bounded idio/EC path, or regime table is introduced.

## Why This Is Not Another Direct-Path Retry

The failed direct path branch modeled future levels as the flow target. That improved level occupancy in 413a but broke structural coupling and mean-reversion geometry.

The proposed target is future score transitions. The path is still generated jointly, but the learned object is closer to the stationary daily move law that 392a already models well. Level occupancy becomes native through cumulative integration of the sampled transition path rather than through recursive one-day sampling or direct level denoising.

## Decisive Test

Run one first implementation, `421a`, with the same efficient axial/token path mixer scale as the existing direct-path family. Evaluate unchanged on the official full 11-suite.

Success criterion: improve beyond the 392a frontier or at least combine 392a's structural passes with a clear level-KS/regime-layer2 gain. If it reproduces the old direct-transition failures, close this paradigm and move to a true path-level latent/world-model architecture.
