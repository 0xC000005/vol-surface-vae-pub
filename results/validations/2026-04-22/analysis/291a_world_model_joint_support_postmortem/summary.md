## 291a-v0 Postmortem

### Result
- `291a-v0` scored `1/11`
- pass:
  - `block_ar`

### Context
`290a/290b` showed that the `289e` world-model backbone was still viable, but the output head could not stay factorized per cell:
- hard per-cell discrete decode was too brittle
- soft per-cell discrete decode was too smooth

`291a` kept the same `289e` history-memory backbone and replaced only the output formulation:
- learned **joint** next-change support encoder
- learned joint support decoder back to the full 25-cell next normalized-change panel
- world-model hidden state predicted the next joint support code

### Training read
- training was stable
- best epoch: `1`
- best val total: `0.02197`

So this was not a numerical collapse. It was a modeling failure.

### High-signal metrics
- ACF corr: `0.409`
- kurtosis ratio: `0.262`
- change KS pass cells: `0/25`
- level KS pass cells: `7/25`
- corr ratio: `0.021`
- rank ratio: `0.590`
- cointegration ratio: below gate
- MR ratio: `0.033`
- active MR cells: `0/24`
- pathwise q90 ratio: `0.009`
- pathwise q99 ratio: `0.015`

### Mechanistic diagnosis
`291a` removed the per-cell factorization problem, but the replacement was still wrong.

The continuous joint support code appears too entangled for the world-model head to predict reliably:
- shared structure collapsed almost completely (`corr_ratio = 0.021`)
- mean reversion collapsed (`0.033`)
- jump scale nearly vanished

This is a different failure mode than `290b`:
- `290b` preserved some global structure but stayed too smooth
- `291a` lost the current-state anchoring needed to keep the joint law coherent at all

The clean read is:
- a learned joint support object is still the right direction
- but a **continuous** joint support code is too hard to predict directly with the current deterministic head

### Decision
Close the naive continuous joint-support branch.

If this line stays alive, the next support object should be:
- learned from data
- still joint across the 25-cell panel
- but **discrete / codebook-like**, so the world-model predicts stable support states rather than unconstrained continuous support codes

### Next step
`291b-v0`
- keep the `289e` history-memory world-model backbone
- replace the continuous joint support code with a learned discrete support codebook for next normalized-change panels
- predict code logits or code indices from the world-model state
