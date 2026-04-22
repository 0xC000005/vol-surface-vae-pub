## 290a Stage A World-Model Formulation Shift

### Context
`289e` showed that the current world-model backbone can preserve most shared deterministic structure:
- cross-cell correlation back near GT
- active mean-reversion support now strong
- history-memory access improved one-step state quality

But the same local-law failure stayed dead:
- change KS still `0/25`
- jump q90/q99 still far too low
- cointegration still below gate

This suggests the next bottleneck is no longer mainly state architecture.
It is the **deterministic smooth-regression target**.

### Decision
Next step: `290a-v0`

### Family
Keep the `289e` history-memory world-model backbone, but change the next-step output formulation.

### Core idea
Keep:
- deterministic Stage A world model
- autoregressive rollout
- history-memory recurrent backbone
- no retrieval bank
- no hand-coded factor head

Change:
- replace pure smooth regression on next normalized change with a sharper discrete output target

### Minimal 290a-v0 design
- backbone:
  - same as `289e`
- output head:
  - predict discretized normalized-change bins for each panel cell
- decoding:
  - deterministic decode by argmax or expected-bin center
- training:
  - cross-entropy on discretized next-change targets
  - rollout still in continuous space using decoded bin centers

### Why this is still first-principles
This is still a learned parametric world model.
The change is only in how the next-step target is represented.

The motivation is empirical:
- the current regression target smooths local move-size structure too aggressively
- a discrete target may better preserve sharp but deterministic next-step behavior

### Why this is the smallest justified shift
The `289b -> 289e` line progressively fixed state representation:
- latent state
- explicit observation encoder
- explicit history-memory access

Shared structure now largely works.
The local law does not.

So the next clean step is to change the target formulation, not add more recurrent state machinery.

### Kill criteria
`290a` is only alive if it keeps the `289e` structural gains and moves at least one dead local-law metric:
- change KS > `0/25`
- jump q90/q99 > `0.071 / 0.111`
- cointegration > `0.277` without losing corr ratio or active MR support
