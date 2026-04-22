## 293h ideation

### Context
`293g` showed that support use is live:
- the scaffold was no longer being ignored
- cointegration improved
- level-side metrics improved slightly

But the first residual formulation over-anchored to the scaffold because it treated the scaffold as a **daily level pull target**:
- short-horizon MR overshot
- regime width timing inverted
- daily change KS collapsed

So the next move should not change the support object. It should only change how the support enters the daily baseline.

### Decision
Next step: `293h-v0`

Keep:
- the `293d` monolithic coarse support object
- the same residual daily token law from `293g`

Change:
- use the scaffold as a **low-frequency drift schedule**
- not as a direct daily level attractor

### Mechanism
For each future day:
- `293g` baseline:
  - change implied by moving from current state directly toward that day's scaffold level
- `293h` baseline:
  - change implied by the scaffold's own day-to-day increment schedule

So the residual law becomes:
- deviation around the scaffold's planned increment
- not deviation around a direct snap-to-scaffold move

### Why this is the right next move
This is the smallest direct fix to the `293g` pathology.

It preserves:
- explicit support
- residual daily law
- fixed-horizon joint-law framing

It changes only:
- the baseline support-use geometry

### Expected effect
If the diagnosis is right, `293h` should:
- retain some of `293g`'s structural gains
- reduce over-mean-reversion
- improve regime-width timing
- recover some daily change-law sharpness

### Kill criteria
`293h` is alive only if it materially improves at least one of:
- change KS
- MR ratio / active support
- regime differentiation

while preserving:
- cointegration
- cross-cell structure
- surface validity

If it does not, then the support-use subfamily is likely near a local cap too.
