## 290b-v0 Postmortem

### Result
- `290b-v0` scored `1/11`
- pass:
  - `block_ar`

### Context
`290a` kept the stronger `289e` history-memory world-model backbone and changed only the next-step output formulation:
- discretized normalized-change bins
- teacher-forced cross-entropy target
- continuous rollout during training via expected-bin decode

`290a` then failed at full evaluation because hard argmax decode produced degenerate cross-cell correlation slices. `290b` was the smallest follow-up:
- same trained discrete-output checkpoint class
- same backbone
- same discrete target
- only change: inference/sample decode switched from hard argmax to expected-bin-center (`soft`) to match training rollout

### High-signal metrics
- ACF corr: `0.402`
- kurtosis ratio: `7.926`
- change KS pass cells: `0/25`
- level KS pass cells: `12/25`
- cointegration ratio: `0.202`
- corr ratio: `1.293`
- rank ratio: `0.498`
- MR ratio: `0.418`
- pathwise q90 ratio: `0.052`
- pathwise q99 ratio: `0.098`

### What 290b fixed
`290b` did fix the exact `290a` evaluator pathology:
- the full `11`-suite completed cleanly
- cross-cell correlation no longer crashed
- mean correlation stayed in-range
- rank was close to the lower gate instead of numerically invalid

So the hard argmax deployment mismatch in `290a` was real.

### What 290b made clear
Once inference matched training, the deeper formulation problem became obvious:
- the discrete head remained far too smooth on local move-size law
- change KS stayed `0/25`
- jump realism stayed essentially dead
- MR collapsed from `0.970` in `289e` to `0.418`
- cointegration stayed weak

So the discrete head did not recover the missing local-law sharpness without paying for it elsewhere.

### 289e vs 290a vs 290b read
- `289e`:
  - structurally viable world-model backbone
  - shared structure and active MR mostly near gate
  - local change-law still oversmoothed
- `290a`:
  - discrete target improved some local-law proxies and level KS
  - but hard decode was too brittle to evaluate reliably
- `290b`:
  - soft decode restored evaluator stability
  - but exposed that the **factorized per-cell discrete formulation is itself too smooth / too weakly joint**

### Mechanistic diagnosis
The live bottleneck is no longer just "regression is too smooth."

It is now narrower:
- a **factorized per-cell output head** cannot simultaneously preserve
  - sharp local move-size law
  - stable cross-cell structure
  - strong active mean reversion

Hard decode buys sharpness by becoming brittle.
Soft decode buys stability by smoothing away the law we were trying to recover.

### Decision
Keep the `289e` history-memory world-model Stage A backbone alive, but close the naive per-cell discrete-head branch.

Next step should not be:
- another hard/soft decode interpolation
- temperature tweaking
- more per-cell bin engineering

Next step should be:
- a **joint next-change support formulation** learned from data, so the world-model predicts a small joint change code/state rather than independent per-cell bins

### Next step
`291a-v0`
- keep the `289e` history-memory backbone
- replace the factorized per-cell discrete head with a learned joint next-change support object
- keep the model deterministic for Stage A
- let the support be learned, not retrieved and not hand-factorized
