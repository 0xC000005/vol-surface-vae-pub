## 293h / 294a comparison

### Context
`293h` was the strongest end-state of the explicit support branch:
- support representation + scaffold-increment residual law
- score `4/11`
- but with dead level KS and severe over-reversion

`294a` was the first continuous scaffold follow-up:
- low-frequency basis path over the full 30-day window
- residual daily token law around scaffold increments
- score `3/11`

The question is whether `294a` is just another local tradeoff, or a genuinely new
path-shape mechanism class worth continuing.

### Score and pass pattern
- `293h`: `4/11`
  - passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`
- `294a`: `3/11`
  - passes: `surface`, `block_ar`, `cross_cell_correlation`

So `294a` lost one suite on score, but the suite pattern alone hides the more
important mechanism change.

### What 294a improved materially
Relative to `293h`, `294a` improved:
- level KS: `0/25 -> 9/25`
- coverage90: `0.980 -> 0.782`
- calibration error: `0.143 -> 0.087`
- rank ratio: `1.234 -> 1.043`
- active MR pass count: `1/24 -> 4/24`

And it preserved:
- change KS: `25/25`
- strong cross-cell structure
- surface validity

That is a real path-shape gain, not evaluator noise.

### What 294a broke
Relative to `293h`, `294a` worsened:
- cointegration ratio: `0.811 -> 2.247`
- worst-cell cointegration ratio: `0.263 -> 0.211`
- aggregate MR ratio: `2.321 -> 0.163`
- active-cell slope corr: `0.632 -> -0.291`

So the failure mode flipped:
- `293h` was too strongly anchored / over-reverting
- `294a` is too smooth / under-reverting

### Mechanism conclusion
`294a` is not another support-branch tradeoff.

The explicit continuous scaffold is a live new mechanism class because it:
- materially improves level-law fidelity
- reduces the old over-reversion pathology
- preserves the local daily law and cross-cell dependence

But the current low-frequency basis is too globally smooth.
It cannot express enough localized curvature or mid-window reversion to keep:
- MR profile
- worst-cell cointegration robustness
- jump timing

So the bottleneck is now narrower than before:
- not support representation
- not support use
- not local daily-law capacity
- specifically the **global smoothness of the current continuous scaffold basis**

### Decision
Keep the `294` family alive.

The next step should be **research ideation for `294b`** with one narrow goal:
- keep the continuous scaffold paradigm
- replace the overly smooth global basis with a sharper continuous scaffold object
- do not return to support codebooks or support-use geometry search
