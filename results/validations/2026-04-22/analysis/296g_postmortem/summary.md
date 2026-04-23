## 296g postmortem

### Context
`296g-v0` tested the narrowest remaining mechanism inside the `296` hybrid family:

- keep frozen `277d`
- keep the zero-mean Gaussian shell
- keep the `296f` budget-plus-redistribution multiresolution basis
- add one learned scalar fast-shell gate before budget renormalization

The hypothesis was that `296f` remained too globally active, and a window-level gate could turn the fast shell on only when needed.

### Result
`296g-v0` scored `4/11`.

Passes:
- `block_ar`
- `cointegration`
- `cross_cell_correlation`
- `mean_reversion`

Failed suites:
- `surface`
- `coverage`
- `conditionality`
- `time_series`
- `regime_coverage`
- `distributional_fidelity`
- `pathwise_jump_realism`

### Key metrics
Against the local hybrid bracket:

| model | score | surface | cov90 | h1 cov90 | cal err | turb/calm | chg KS | lvl KS | corr | rank | MR | jump KS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `296c` | 5 | pass | 0.882 | 0.593 | 0.031 | 1.110 | 19 | 0 | 0.904 | 1.491 | 1.086 | 0.453 |
| `296e` | 4 | fail | 0.930 | 0.859 | 0.082 | 1.165 | 16 | 0 | 0.750 | 2.006 | 1.094 | 0.394 |
| `296f` | 4 | fail | 0.928 | 0.862 | 0.076 | 1.117 | 17 | 0 | 0.772 | 1.912 | 1.113 | 0.389 |
| `296g` | 4 | fail | 0.926 | 0.845 | 0.075 | 1.106 | 17 | 0 | 0.771 | 1.913 | 1.105 | 0.388 |

The learned gate did not become a meaningful conditional activation mechanism:

- gate mean: `0.778`
- gate std: `0.004`
- gate p10/p90: `0.773 / 0.783`
- Spearman(gate, history realized variance): `-0.005`
- calm gate mean: `0.777`
- turbulent gate mean: `0.778`

### Mechanism read
`296g` is a clean negative.

The scalar gate trained, but it became nearly constant and did not correlate with the regime proxy. As a result, it mostly reproduced `296f` with a mild static fast-shell rescaling:

- h1 coverage remained high but slightly worse than `296f`
- regime width did not recover
- surface validity stayed failed
- level KS stayed dead at `0/25`
- jump KS improved only marginally relative to `296f`

So the remaining bottleneck is not solved by a global learned gate.

### Decision
The local `296e -> 296f -> 296g` fast-shell branch is near a cap.

The next principled step is post-experiment analysis of the broader `296` hybrid family, with special attention to whether the fixed `277d` backbone plus zero-mean shell can ever repair level-law fidelity. If the answer is no, the next move should be a paradigm/interface shift rather than another shell knob.
