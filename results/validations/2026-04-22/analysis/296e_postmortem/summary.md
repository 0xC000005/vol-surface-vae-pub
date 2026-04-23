## 296e-v0 postmortem

### Result
- model: `296e`
- checkpoint: `models/backfill/296e_v0_s42/best_model.pt`
- eval: `results/block_ar/296e_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`
  - `mean_reversion`

### High-signal metrics
- coverage90: `0.930`
- calibration error: `0.082`
- h1 / h30 coverage90: `0.859 / 0.960`
- change KS pass: `16/25`
- level KS pass: `0/25`
- corr ratio: `0.750`
- rank ratio: `2.006`
- cointegration ratio: `0.734`
- worst-cell cointegration ratio: `0.278`
- MR ratio: `1.094`
- active-cell slope corr: `0.791`
- turb/calm width ratio: `1.165`
- max-jump KS: `0.394`
- q99 ratio: `1.136`

### Training read
- best validation epoch: `19`
- best val total: `-0.036`
- the fast early-horizon basis was heavily used:
  - best-epoch h1 profile component `~1.80`
- scale predictions stayed broad (`pred_scale_mean ~ 0.65` at best epoch)

So the multiresolution basis really did shift shell mass toward the front of the path.

### Relative to 296c
What improved:
- h1 coverage90: `0.593 -> 0.859`
- turb/calm width ratio: `1.110 -> 1.165`
- max-jump KS: `0.453 -> 0.394`
- q99 ratio: `0.900 -> 1.136`

What got worse:
- score stayed below the frontier (`5/11 -> 4/11`)
- overall coverage overexpanded: `0.882 -> 0.930`
- calibration error worsened: `0.031 -> 0.082`
- change KS: `19/25 -> 16/25`
- corr ratio: `0.904 -> 0.750`
- rank ratio: `1.491 -> 2.006`
- surface validity failed on calendar arbitrage

### Mechanism read
`296e` proves the early-horizon geometry was a real bottleneck.

The fast basis achieved exactly the targeted gains:
- short-horizon coverage
- regime-sensitive width
- stronger jump scale

But it overpaid for them:
- too much global width
- weaker local daily-law fidelity
- weaker structural carryover

So the next bottleneck is not "can the shell move early-horizon mass?"
It can.

The bottleneck is:
- how to add that fast local mass **without** globally over-dispersing the shell or
  degrading the backbone-derived structure.

### Decision
Do post-experiment analysis comparing `296c`, `296d`, and `296e`.

Key question:
- did `296e` validate the multiresolution shell as the right direction,
- or did it just trade one local failure for another?
