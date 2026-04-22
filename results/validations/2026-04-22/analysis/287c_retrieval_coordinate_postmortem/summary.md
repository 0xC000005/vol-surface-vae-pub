## 287c-v0 Postmortem

### Result
- Model: `287c-v0`
- Eval: `results/block_ar/287c_v0_s42/full11.json`
- Score: `4/11`
- Passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### Comparison vs 287b
- `287b`: `3/11`
- `287c`: `4/11`

Material improvements from splitting the retrieval key into fast and slow channels while keeping normalized-change replay fixed:
- `cointegration ratio`: `0.742 -> 1.337`
- `corr_ratio`: `1.001 -> 0.971`
- `rank_ratio`: `1.153 -> 1.298`
- `aggregate MR ratio`: `0.927 -> 1.056`
- `level KS pass cells`: `3/25 -> 10/25`
- `window-floor failure rate`: `100% -> 100%` unchanged because the model is still deterministic
- `change KS pass cells`: `25/25 -> 24/25` essentially preserved

Tradeoffs / unresolved misses:
- deterministic coverage remains exactly `0%`, which is expected for Stage A
- mean-reversion suite still fails because:
  - `active_pass_rate = 66.7%`
  - `active_cell_slope_corr = 0.504`
- pathwise jump realism still fails:
  - `max_jump_ks = 0.359`
- time-series suite still fails on skew / tail-shape details even though:
  - `acf_correlation = 0.958`
  - `kurtosis_ratio = 1.223`

### Comparison vs 277d
`277d` remains the stronger deterministic Stage A frontier at `5/11`.

Relative to `277d`, `287c` gives:
- better local level fidelity:
  - `level KS pass cells: 10/25` vs `0/25`
- stronger local cointegration:
  - `cointegration ratio: 1.337` vs `0.766`

But `287c` gives back too much active-cell structure:
- `active_pass_rate: 66.7%` vs `87.5%`
- `active_cell_slope_corr: 0.504` vs `0.750`
- `max_jump_ks: 0.359` vs `0.391` only modestly better

### Mechanism Read
The `287b` diagnosis was directionally right: one mixed retrieval embedding was washing out slow structure.

`287c` fixed that partially:
- the split fast/slow future key materially restored slow global structure
- the normalized-change replay support primitive remains the right support object

But the remaining miss is now cleaner:
- the **history/query side is still not fully aligned with the local-history replay coordinate**
- future keys live in the local-history `delta_z / cumulative_disp_z` space
- query keys are still built from globally normalized history levels and their transforms

So the family is still alive, but the next bottleneck is not the support primitive or the fast/slow split anymore.
It is the **query coordinate mismatch**.

### Decision
Next step: `287d-v0`

Keep fixed:
- same deterministic Stage A retrieval family
- same normalized-change replay support primitive
- same fast/slow split retrieval logic

Change exactly one thing:
- build the history/query fast and slow channels in the same local-history z-coordinate used by the replay support

This is the smallest principled follow-up to test whether the remaining active-cell MR miss is due to query/support coordinate mismatch rather than lack of backbone capacity.
