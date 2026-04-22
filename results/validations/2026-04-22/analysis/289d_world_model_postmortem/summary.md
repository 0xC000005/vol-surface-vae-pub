## 289d-v0 Postmortem

### Result
- `289d-v0` scored `2/11`
- passes:
  - `surface`
  - `block_ar`

### Context
`289c` was the best world-model Stage A result so far:
- `3/11`
- cross-cell structure back in gate
- improved MR, ACF, level KS, and jump scale

The open question was whether the remaining bottleneck was simply too little state capacity under rollout.

`289d` tested the smallest clean capacity increase:
- replace the single latent vector with `4` latent tokens
- update them with a small sequence-aware Transformer block
- keep the same first-principles deterministic autoregressive world-model setup

### High-signal metrics
- ACF corr: `0.544`
- kurtosis ratio: `4.043`
- cointegration ratio: `1.192`
- worst-cell cointegration ratio: `0.118`
- corr ratio: `0.212`
- rank ratio: `2.168`
- MR ratio: `0.781`
- active-cell MR pass rate: `41.7%`
- active-cell MR corr: `0.719`
- change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- pathwise q90 ratio: `0.086`
- pathwise q99 ratio: `0.135`

### What improved vs 289c
Only one major deterministic structural metric improved materially:
- aggregate cointegration ratio: `0.292 -> 1.192`

That shows the token-state family can express some longer-run equilibrium structure.

### What regressed vs 289c
The main useful gains from `289c` were given back:
- `n_pass: 3 -> 2`
- corr ratio: `0.596 -> 0.212`
- rank ratio: `1.356 -> 2.168`
- MR ratio: `0.912 -> 0.781`
- active-cell MR rate: `45.8% -> 41.7%`
- level KS pass cells: `5 -> 0`
- ACF corr: `0.634 -> 0.544`

Jump scale did not materially improve beyond `289c`:
- q90: `0.094 -> 0.086`
- q99: `0.131 -> 0.135`

### Teacher-vs-rollout probe
Validation probe on 256 windows:
- teacher-change MAE: `0.0359`
- rollout-change MAE: `0.0333`
- teacher-level MAE: `0.0358`
- rollout-level MAE: `0.0618`
- teacher-change std: `0.0293`
- rollout-change std: `0.0116`
- target-change std: `0.1076`

Interpretation:
- unlike `289c`, the one-step model itself got weaker
- the token-state parameterization fragmented the state rather than improving useful capacity
- rollout variance is still compressed, but now the teacher-forced path is also much too smooth

### Mechanistic diagnosis
The clean read is:
- `289c` global latent state preserved common structure but remained too compressive
- `289d` token-state capacity loosened that compression in the wrong way
- the model lost global shared structure before it learned sharp local dynamics

So the next problem is not “more latent slots.”
It is “how to preserve access to long-horizon history/context during rollout without fragmenting the shared state.”

### Decision
Do not continue local token-state tweaks.

Keep the world-model Stage A paradigm, but treat the naive token-state branch as falsified.

Next step: `289e-v0`
- return to the stronger `289c`-style global latent state
- add explicit read-only history memory access during rollout
- use current observation plus history memory to update the recurrent latent state

### Kill criteria for 289e
`289e` is only alive if it preserves the `289c` common-structure gains and improves at least one of the still-dead deterministic bottlenecks:
- change KS above `0/25`
- cointegration above `0.292` without losing corr ratio
- jump q90/q99 materially above `0.094 / 0.131`
