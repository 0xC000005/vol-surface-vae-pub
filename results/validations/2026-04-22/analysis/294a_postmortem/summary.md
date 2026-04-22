## 294a-v0 postmortem

### Result
- model: `294a`
- checkpoint: `models/backfill/294a_v0_s42/best_model.pt`
- eval: `results/block_ar/294a_v0_s42/full11.json`
- score: `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.782`
- calibration error: `0.087`
- h1 / h30 coverage90: `0.603 / 0.865`
- change KS pass: `25/25`
- level KS pass: `9/25`
- corr ratio: `1.177`
- rank ratio: `1.043`
- cointegration ratio: `2.247`
- worst-cell cointegration ratio: `0.211`
- MR ratio: `0.163`
- active MR pass count: `4/24`
- active-cell slope corr: `-0.291`
- mean turbulent/calm width ratio: `1.106`
- max-jump KS: `0.662`
- jump q90 / q99 ratio: `0.828 / 1.141`
- ACF corr: `0.944`

### Training read
- best validation epoch: `2`
- best val total: `5.226`
- basis coefficient regression trained stably from the start
- residual token accuracy stayed low but nonzero throughout
- optimization was cleaner than the late `293g/293h` support-use runs

### Relative to 293h
What improved:
- level KS: `0/25 -> 9/25`
- coverage90: `0.980 -> 0.782` moved much closer to the 0.90 target
- calibration error: `0.143 -> 0.087`
- rank ratio: `1.234 -> 1.043`
- mean turbulent/calm width ratio: `0.876 -> 1.106`
- active MR pass count: `1/24 -> 4/24`

What got worse:
- score: `4/11 -> 3/11`
- cointegration ratio: `0.811 -> 2.247`
- worst-cell cointegration ratio: `0.263 -> 0.211`
- aggregate MR ratio: `2.321 -> 0.163`
- active-cell slope corr: `0.632 -> -0.291`

What stayed wrong:
- conditionality is still weak
- pathwise jump realism is still far below gate
- regime coverage is better behaved but still below pass

### Mechanism read
`294a` is the first clean evidence that the `293` family was missing an explicit
continuous path-shape controller, not just another support codebook or scaffold-use
variant.

The low-frequency basis scaffold changed the regime of failure materially:
- it preserved the local daily law (`25/25` change KS)
- it kept cross-cell structure alive
- it improved level-law fidelity sharply
- and it removed the severe over-reversion pathology of `293g/293h`

But the scaffold is now too smooth and too weakly mean-reverting:
- aggregate MR collapsed below gate
- worst-cell cointegration robustness failed harder
- jump realism remained weak

So this is not the old support-branch tradeoff. It is a new tradeoff:
- explicit continuous scaffold improves path-shape fit
- current basis scaffold under-controls sharpness and reversion

### Decision
Keep the `294` continuous scaffold family alive for one narrow follow-up.

The next principled step is **post-experiment analysis / ideation for `294b`**:
- stay in the continuous scaffold family
- do not return to support codebooks
- add one small mechanism that makes the scaffold less overly smooth while preserving
  the `294a` gains in change KS, level KS, and cross-cell structure
