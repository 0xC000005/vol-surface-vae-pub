## 287d-v0 Postmortem

### Result
- Model: `287d-v0`
- Eval: `results/block_ar/287d_v0_s42/full11.json`
- Score: `4/11`
- Passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### Comparison vs 287c
`287d` kept the `287c` support primitive and fast/slow key split fixed, and changed only one thing:
- build the history/query channels in the same local-history z-coordinate used by the replay support

That produced a very clean bracket:

Improvements vs `287c`:
- `level KS pass cells`: `10/25 -> 14/25`
- `max-jump KS`: `0.359 -> 0.354` (small)

Regressions vs `287c`:
- `cointegration ratio`: `1.337 -> 1.273`
- `corr_ratio`: `0.971 -> 0.907`
- `rank_ratio`: `1.298 -> 1.449`
- `aggregate MR ratio`: `1.056 -> 0.168`
- `active_cell_slope_corr`: `0.504 -> 0.394`
- `acf_correlation`: `0.958 -> 0.934`
- `kurtosis ratio`: `1.223 -> 1.364`

### Mechanism Read
The pure local-history query coordinate is too local.

It helps the retrieval key line up with near-term level fidelity:
- better level KS
- slightly better jump-shape fit

But it removes too much slow absolute-state information:
- aggregate MR collapses
- active-cell MR support weakens further
- slow structural alignment softens

So the `287c -> 287d` bracket clarifies the real requirement:
- fast query channel should live in the local-history coordinate
- slow query channel should **not** be purely local-history

The model still needs a slow absolute anchor to preserve mean reversion and structural robustness.

### Decision
Keep the `287` family alive.

Next step: `287e-v0`
- keep normalized-change replay fixed
- keep future fast/slow keys fixed
- use a **mixed query key**:
  - fast history channel from local-history z
  - slow history channel from globally normalized level displacement

This is now the smallest principled follow-up because `287c` and `287d` together show that the fast channel wants local-history alignment, while the slow channel wants an absolute-state anchor.
