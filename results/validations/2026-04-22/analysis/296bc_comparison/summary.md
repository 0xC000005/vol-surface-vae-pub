## 296b vs 296c

### Context
Both `296b` and `296c` keep the same validated hybrid decomposition:
- frozen `277d` backbone
- zero-mean coarse residual shell
- paired symmetric sampling

The only difference is shell scale allocation:
- `296b`: per-knot-per-cell coarse scales
- `296c`: same plus a shared query-conditioned knot-profile multiplier

This comparison asks whether the scale-allocation subfamily is still moving the real
bottleneck, or only changing secondary metrics.

### Headline comparison
| model | score | passes | main read |
| --- | --- | --- | --- |
| `296b` | `4/11` | block_ar, cointegration, cross_cell_correlation, mean_reversion | first live hybrid shell geometry |
| `296c` | `5/11` | surface, block_ar, cointegration, cross_cell_correlation, mean_reversion | better support behavior, same stochastic bottleneck |

### High-signal metrics
| metric | `296b` | `296c` | read |
| --- | ---: | ---: | --- |
| score | `4/11` | `5/11` | frontier tie recovered |
| surface | fail | pass | improved support behavior |
| coverage90 | `0.892` | `0.882` | essentially flat |
| calibration error | `0.039` | `0.031` | slight improvement |
| h1 coverage90 | `0.605` | `0.593` | no improvement |
| turb/calm width ratio | `1.141` | `1.110` | slightly worse |
| change KS pass | `20/25` | `19/25` | flat to slightly worse |
| level KS pass | `0/25` | `0/25` | unchanged |
| cointegration ratio | `0.723` | `0.744` | slight improvement |
| MR ratio | `1.088` | `1.086` | unchanged |
| max-jump KS | `0.446` | `0.453` | unchanged to slightly worse |

### What this proves
The scale-allocation refinement is **not dead**, but it is close to a local cap.

Why:
- it can still move support/surface quality
- it can slightly improve calibration and retain structure

But it is **not** moving the actual open bottleneck:
- short-horizon coverage
- regime-sensitive width
- level-law fidelity
- pathwise jump realism

So another small scale-head tweak is unlikely to break the frontier.

### Mechanism read
`296c` improved how shell variability sits inside support boundaries, not how that
variability is allocated where the suite still fails.

That suggests the next missing ingredient is probably **not** another scale factor.
It is more likely one of:
- a different shell support object
- a different shell training objective
- or an explicit short-horizon deviation family inside the same zero-mean geometry

### Decision
Treat the pure scale-allocation subfamily as **near a local cap**.

Next step:
- `296d` ideation

Constraint:
- keep the hybrid decomposition and the zero-mean shell geometry
- but change something deeper than another scale head
