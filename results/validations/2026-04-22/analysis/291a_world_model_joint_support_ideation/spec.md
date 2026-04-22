## 291a Stage A Joint-Support World-Model Shift

### Context
The `289` world-model line established a viable deterministic Stage A backbone:
- `289e` history-memory access recovered shared structure and active mean reversion

The `290` formulation bracket then isolated the next bottleneck:
- `290a` hard per-cell discrete decode was too brittle
- `290b` soft per-cell discrete decode was too smooth

So the remaining problem is not just "continuous vs discrete."
It is that the output formulation is still **factorized per cell**, while the missing deterministic suites depend on a **joint next-change law**.

### Decision
Next step: `291a-v0`

### Family
Keep the deterministic Stage A world-model backbone from `289e`, but change the next-step output representation again.

### Core idea
Instead of predicting `25` cellwise change bins independently, predict a **small learned joint support code** for the next normalized-change panel.

Then decode that code back into the full `25`-cell next normalized change.

This keeps the model:
- learned
- parametric
- world-model-based
- first-principles

while removing the hand-factorized per-cell output bottleneck.

### Minimal 291a-v0 design
Backbone:
- same history-memory recurrent backbone as `289e`

Support object:
- learned encoder over one-step normalized-change panels
- small bottleneck code for the whole next-step panel
- deterministic decoder from code to full next normalized change

Prediction head:
- world-model hidden state predicts the next support code
- rollout decodes that code into the next normalized change panel

Training:
- reconstruction loss for the support autoencoder
- world-model loss on predicted next support code / decoded next panel
- no retrieval bank
- no low-rank head
- no bounded side paths

### Why this is the smallest justified shift
The evidence now says:
- state architecture is good enough to keep
- per-cell output factorization is not

So the smallest principled change is:
- keep the world-model
- keep deterministic Stage A
- replace only the output support object with a learned **joint** one

### Why this is still first-principles
This is still a learned parametric model of dynamics.
It does not assume:
- low-rank factors
- retrieval support objects
- hand-coded error-correction
- hand-coded idiosyncratic paths

It only introduces a learned bottleneck for the **joint next-step panel**, which is consistent with the repo-wide "learn the support from data" reset.

### Kill criteria
`291a` is only alive if it improves at least one local-law suite without giving back the shared-structure gains from `289e`:
- change KS > `0/25`
- jump q90/q99 > `0.071 / 0.111`
- cointegration > `0.277`
- while keeping corr ratio near `289e` and MR ratio above `0.90`
