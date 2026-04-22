## 287e-v0 Postmortem

### Result
- Model: `287e-v0`
- Eval: `results/block_ar/287e_v0_s42/full11.json`
- Score: `4/11`
- Passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### Comparison vs 287c and 287d
`287e` kept the replay support and future fast/slow keys fixed, and used the mixed query design suggested by the `287c -> 287d` bracket:
- fast history channel from local-history z
- slow history channel from globally normalized displacement

Relative to `287c`:
- `change KS pass cells`: `24/25 -> 25/25`
- `level KS pass cells`: `10/25 -> 12/25`
- `max-jump KS`: `0.359 -> 0.349`
- `mr_active_corr`: `0.504 -> 0.731`

But:
- `mr_active_rate`: `66.7% -> 8.3%`
- `kurtosis ratio`: `1.223 -> 1.463`

Relative to `287d`:
- aggregate MR recovers strongly:
  - `mr_gt_ratio`: `0.168 -> 0.928`
- active-cell slope correlation recovers:
  - `0.394 -> 0.731`
- level KS gives back a little:
  - `14/25 -> 12/25`

### Mechanism Read
The mixed query key confirms the `287c/287d` diagnosis:
- local-history fast alignment is useful
- the slow query channel does need a global absolute anchor

But the bracket also shows that **query-key design alone is no longer enough**.

What `287e` does:
- preserves the better aggregate structural behavior from the globally anchored slow channel
- keeps the local level/change support mostly intact

What it still cannot do:
- activate the right set of strong mean-reverting cells
- keep time-series tail shape inside gate

So the remaining miss is no longer a simple query-coordinate mismatch.
The single-nearest deterministic retrieval itself is now the more likely cap.

### Decision
Close the local query-key sub-branch as locally exhausted.

Next step:
- `post_experiment_analysis` over `287c / 287d / 287e`
- then select the next smallest Stage A move from that bracket

Current hypothesis:
- the family now needs a **support/use change**, not another query-key tweak
- most likely direction is a deterministic top-k / soft retrieval replay rather than a different single-key embedding
