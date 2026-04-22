## 289e-v0 Postmortem

### Result
- `289e-v0` scored `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### Context
`289d` showed that naive token-state capacity was the wrong next move.

`289e` returned to the stronger `289c` global latent state, but added explicit read-only history-memory access during rollout:
- history GRU memory
- observation encoder
- attention read over history memory at each generated step
- recurrent latent state update from `[observation, history-context]`

### High-signal metrics
- ACF corr: `0.503`
- kurtosis ratio: `3.888`
- cointegration ratio: `0.277`
- worst-cell cointegration ratio: `0.039`
- corr ratio: `0.952`
- rank ratio: `1.232`
- MR ratio: `0.970`
- active-cell MR pass rate: `83.3%`
- active-cell MR corr: `0.763`
- change KS pass cells: `0/25`
- level KS pass cells: `5/25`
- pathwise q90 ratio: `0.071`
- pathwise q99 ratio: `0.111`

### What improved vs 289c
`289e` substantially improved shared deterministic structure:
- corr ratio: `0.596 -> 0.952`
- rank ratio: `1.356 -> 1.232`
- MR ratio: `0.912 -> 0.970`
- active-cell MR pass rate: `45.8% -> 83.3%`
- active-cell MR corr: `0.743 -> 0.763`

Teacher-vs-rollout probe also improved the one-step state quality:
- teacher-change std: `0.0603 -> 0.0687`
- rollout-change std: `0.0130 -> 0.0134`

So explicit history-memory access is directionally right.

### What did not improve
The remaining deterministic failures did not move enough:
- `n_pass` stayed `3/11`
- change KS stayed `0/25`
- cointegration stayed below gate: `0.292 -> 0.277`
- jump q90/q99 regressed:
  - `0.094 -> 0.071`
  - `0.131 -> 0.111`

### Mechanistic diagnosis
The world-model Stage A line is now in a different regime:
- shared structure is mostly under control
- active mean reversion is mostly under control
- the dominant remaining failure is **oversmoothed local change-law / tail behavior**

This now looks less like a state-architecture problem and more like a **deterministic regression-target problem**:
- the model learns a smooth center path
- but not a sharp enough local move-size law to pass change KS or jump realism

### Decision
Keep the world-model Stage A family active, but stop treating the main blocker as state architecture alone.

Next step: `290a-v0`
- keep the stronger `289e` history-memory world-model backbone
- change the deterministic output formulation away from pure smooth regression
- first candidate: joint discretized normalized-change target driven by the world-model hidden state

### Kill criteria for 290a
`290a` is only alive if it preserves the `289e` common-structure gains and finally moves at least one of the local law metrics:
- change KS above `0/25`
- jump q90/q99 above `0.071 / 0.111`
- cointegration above `0.277` without giving back corr ratio / MR
