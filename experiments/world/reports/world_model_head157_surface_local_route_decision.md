# World Model HEAD157: Surface-Local Route Decision

Date: 2026-05-10

## Iteration Type

`paradigm_shift`

## Objective Family

`token_geometry_level_context_to_target_jepa` route decision.

## Hypothesis

If the surface-local token JEPA route fails because targets are not covering the
right regions, then richer target coverage or simple mask changes might be the
next principled step. If it fails because the target latent surface itself is
low-rank and geometry-dominated, then tuning masks, EMA, hidden size, or
predictor depth is not principled.

## Evidence Chain

- HEAD149 showed the scaled Barlow exact-state gap is concentrated in wing
  moneyness and edge maturities.
- HEAD151-152 verified the surface-local data contract and target coverage,
  including `iv_m0_t0`, wings, and edge maturities.
- HEAD153 added a minimal token/geometry context-to-target scaffold with clean
  target-encoder outputs selected by `(window, relative_time, token)`.
- HEAD154 showed the route is runnable, but target-token retrieval is weak:
  top10 is `0.054688` on a 512-row subset and predicted effective rank is
  `3.945894` in a 24-dimensional latent.
- HEAD155 showed the failure is not just predictor retrieval: the target latent
  is also low-rank, and predicted/target variance ratio is `0.136648`.
- HEAD156 showed the target latent clusters strongly by token/factor:
  factor-neighbor share is `10.542023x` random, while predictor exact-row top10
  is only `0.042969`.

## Decision

Demote the current surface-local token context-to-target route as implemented.
The failure is representation geometry, not target coverage.

Do not tune:

- target mask coverage;
- hidden size;
- predictor depth;
- EMA decay;
- training epochs;
- Barlow/off-diagonal weights.

If this route is revisited, it needs a new design gate where target latents are
required to carry state variation before predictor training. Intrinsic target
separability must be checked before spending another cycle on prediction.

## Status

The active learned candidate remains the scaled Barlow reference, but it is still
`DO_NOT_PROMOTE`. Part B remains blocked.
