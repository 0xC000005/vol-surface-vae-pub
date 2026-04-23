## 296 hybrid family postmortem

### Context
The `296` family tested a hybrid decomposition:

- frozen `277d` structural backbone owns the center path
- stochastic shell owns scenario dispersion
- shell is zero-mean to avoid dragging the backbone

This was motivated by the `277d` vs `295a` split:

- `277d` preserves structure but has no stochastic adequacy
- `295a` learns a stochastic shell but loses center-path structure

### Evidence
The key local bracket is:

| model | score | passes | cov90 | h1 cov90 | cal err | turb/calm | chg KS | lvl KS | corr | rank | MR | jump KS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `277d` | 5 | surface, block_ar, coint, xcell, MR | 0.000 | 0.000 | 0.500 | 1.000 | 23 | 0 | 1.058 | 1.027 | 1.065 | 0.391 |
| `296c` | 5 | surface, block_ar, coint, xcell, MR | 0.882 | 0.593 | 0.031 | 1.110 | 19 | 0 | 0.904 | 1.491 | 1.086 | 0.453 |
| `296e` | 4 | block_ar, coint, xcell, MR | 0.930 | 0.859 | 0.082 | 1.165 | 16 | 0 | 0.750 | 2.006 | 1.094 | 0.394 |
| `296f` | 4 | block_ar, coint, xcell, MR | 0.928 | 0.862 | 0.076 | 1.117 | 17 | 0 | 0.772 | 1.912 | 1.113 | 0.389 |
| `296g` | 4 | block_ar, coint, xcell, MR | 0.926 | 0.845 | 0.075 | 1.106 | 17 | 0 | 0.771 | 1.913 | 1.105 | 0.388 |

### Mechanism read
The hybrid decomposition is partly valid:

- frozen `277d` structural suites are mostly preserved
- zero-mean shells can add broad coverage/calibration
- multiresolution shells can restore h1 coverage and improve jump scale

But the current interface is capped:

- all `296` variants have level KS `0/25`
- the shell cannot repair the unconditional level law if the fixed center support is wrong
- fast-shell geometry improves local coverage but overpays through global overdispersion, surface failures, and weak regime differentiation
- the `296g` gate did not become conditional, so another scalar activation knob is not justified

The local branch has exhausted three clean mechanisms:

- global profile scale (`296c`)
- fast multiresolution shell (`296e`)
- budget and scalar activation controls (`296f/296g`)

### Decision
Do not add another shell knob inside the same fixed-center zero-mean interface.

The next principled step is a paradigm/interface shift. The model needs stochasticity in the **structural center support**, not only a zero-mean shell around one fixed center path.

### Next direction
Move to `297a` ideation:

- preserve the hybrid insight that structure and stochastic adequacy are different jobs
- replace the single fixed center path with a small learned structural center-support distribution
- keep the stochastic shell secondary and mean-controlled
- do not return to broad daily residual correction or per-cell gating
