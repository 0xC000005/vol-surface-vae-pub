## 289b-v0 Postmortem

### Result
- `289b-v0` scored `2/11`
- passes:
  - `surface`
  - `block_ar`

### Context
`289b` was the first latent-state follow-up to `289a`.

`289a` falsified direct observation-space world modeling:
- `1/11`
- direct panel-space transition collapsed cross-cell structure and jumps

`289b` kept the same deterministic Stage A world-model paradigm, but inserted a learned latent state:
- history GRU encoder
- latent GRU transition
- decoder from latent state + current panel level to next normalized change

### High-signal metrics
- coverage90: `0.0`
- calibration error: `0.500`
- corr ratio: `0.440`
- rank ratio: `1.751`
- cointegration ratio: `0.340`
- MR ratio: `0.719`
- active-cell MR pass rate: `45.8%`
- active-cell MR corr: `0.676`
- change KS pass cells: `0/25`
- level KS pass cells: `2/25`
- cell MAE pass cells: `24/25`
- pathwise max-jump KS: `1.000`
- pathwise q90 ratio: `0.045`
- pathwise q99 ratio: `0.070`

### What improved vs 289a
The latent state materially improved deterministic structure:
- `n_pass`: `1 -> 2`
- corr ratio: `0.056 -> 0.440`
- rank ratio: `2.871 -> 1.751`
- MR ratio: `0.686 -> 0.719`
- active-cell MR pass rate: `33.3% -> 45.8%`
- cell MAE pass cells: `23 -> 24`

This means the world-model paradigm is still alive and the latent state was the right direction.

### What got worse
The model became even more rollout-smooth on sharp dynamics:
- cointegration ratio: `0.476 -> 0.340`
- change KS pass cells: `7 -> 0`
- jump q90 ratio: `0.077 -> 0.045`
- jump q99 ratio: `0.125 -> 0.070`

So `289b` traded direct observation-space collapse for low-variance latent rollout collapse.

### Teacher-vs-rollout probe
A quick validation probe on 256 val windows shows the issue is specifically rollout:

- teacher-change MAE: `0.0343`
- rollout-change MAE: `0.0332`
- teacher-level MAE: `0.0343`
- rollout-level MAE: `0.0630`
- teacher-change std: `0.0596`
- rollout-change std: `0.0119`
- target-change std: `0.1076`

Interpretation:
- the one-step latent predictor is not dead
- but autoregressive rollout rapidly collapses variance
- the latent state update is not preserving enough generated-state information over time

### Mechanistic diagnosis
The main pathology is now clean:

- `289a`: direct observation-space transition was too hard
- `289b`: latent state helps, but a single latent vector with the current update/decoder parameterization collapses under rollout

This is now an exposure-bias / state-update problem, not a support-object problem and not a retrieval problem.

### Decision
Keep the world-model Stage A paradigm.

Next step: `289c-v0`
- keep deterministic Stage A
- keep autoregressive rollout
- keep normalized-change target
- keep learned latent state
- change the state update from a single latent vector transition to an explicit observation-encoded recurrent state update so generated panel states are re-encoded more richly at each step

### Kill criteria for 289c
`289c` is only alive if it improves the rollout-collapse metrics together:
- corr ratio into gate or near-gate
- cointegration materially above `289b`
- jump q90/q99 materially above `289b`
- deterministic change KS materially above `0/25`
