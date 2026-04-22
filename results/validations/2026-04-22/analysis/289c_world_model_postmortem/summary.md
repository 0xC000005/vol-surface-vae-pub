## 289c-v0 Postmortem

### Result
- `289c-v0` scored `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### Context
`289b` kept the latent world-model line alive, but rollout variance collapsed:
- one-step latent prediction was usable
- generated-state rollout became too smooth
- structural suites improved, but jump realism and change-law fidelity stayed dead

`289c` changed exactly one thing:
- replace the shallow `[curr_level, prev_change] -> latent input` update with an explicit observation encoder
- feed that observation embedding into the recurrent latent update and decoder at every step

### High-signal metrics
- coverage90: `0.0`
- calibration error: `0.500`
- ACF corr: `0.634`
- kurtosis ratio: `9.158`
- cointegration ratio: `0.292`
- corr ratio: `0.596`
- rank ratio: `1.356`
- MR ratio: `0.912`
- active-cell MR pass rate: `45.8%`
- active-cell MR corr: `0.743`
- change KS pass cells: `0/25`
- level KS pass cells: `5/25`
- cell MAE pass cells: `24/25`
- pathwise max-jump KS: `1.000`
- pathwise q90 ratio: `0.094`
- pathwise q99 ratio: `0.131`

### What improved vs 289b
`289c` was a real gain:
- `n_pass: 2 -> 3`
- ACF corr: `0.404 -> 0.634`
- corr ratio: `0.440 -> 0.596` into gate
- rank ratio: `1.751 -> 1.356`
- MR ratio: `0.719 -> 0.912`
- active-cell MR corr: `0.676 -> 0.743`
- level KS pass cells: `2 -> 5`
- jump q90 ratio: `0.045 -> 0.094`
- jump q99 ratio: `0.070 -> 0.131`

So the explicit observation encoder improved generated-state reuse and recovered a usable common-structure backbone.

### What is still broken
The same dominant failure remains:
- deterministic coverage is still zero, which is expected for Stage A
- cointegration is still below gate
- change KS is still `0/25`
- pathwise jump realism is still dead
- rollout is still much lower-variance than the teacher-forced path

### Teacher-vs-rollout probe
Validation probe on 256 windows:
- teacher-change MAE: `0.0334`
- rollout-change MAE: `0.0331`
- teacher-level MAE: `0.0334`
- rollout-level MAE: `0.0564`
- teacher-change std: `0.0603`
- rollout-change std: `0.0130`
- target-change std: `0.1076`

Interpretation:
- the one-step predictor is slightly better than `289b`
- rollout variance is still heavily compressed
- the world-model line is now bottlenecked by **state capacity under rollout**, not by direct observation modeling anymore

### Mechanistic diagnosis
The explicit observation encoder helped, but a single latent vector still compresses the evolving market state too aggressively.

That is why:
- common structure enters gate
- mean-reversion improves
- but sharp local change-law and jump scale remain too weak

### Decision
Keep the deterministic world-model Stage A line active.

Next step: `289d-v0`
- keep the same first-principles world-model framing
- keep deterministic autoregressive rollout
- keep normalized-change target
- replace the single latent vector with a **small latent token/state set**
- let a sequence-aware latent transition carry more state capacity without adding retrieval or hand-crafted factor structure

### Kill criteria for 289d
`289d` is only alive if it improves the still-live rollout-capacity symptoms together:
- change KS strictly above `0/25`
- cointegration materially above `0.292`
- jump q90/q99 materially above `0.094 / 0.131`
- cross-cell correlation stays in gate
