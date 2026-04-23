## 277d vs 296a vs 296b

### Context
The hybrid question was:
- can a structural backbone plus stochastic shell hold both types of behavior at once?

`296a` failed the first interface test. `296b` changed only the shell geometry:
- zero-mean
- coarse knot controls
- paired symmetric sampling

So the comparison is now whether that geometry change actually validates the hybrid
program.

### Headline comparison
| model | score | main passes | headline read |
| --- | --- | --- | --- |
| `277d` | `5/11` | surface, block_ar, cointegration, cross_cell_correlation, mean_reversion | strongest structural backbone, deterministic undercoverage |
| `296a` | `2/11` | block_ar, cross_cell_correlation | bad shell interface destroyed both structure and stochastic adequacy |
| `296b` | `4/11` | block_ar, cointegration, cross_cell_correlation, mean_reversion | hybrid direction restored; shell geometry now broadly right |

### High-signal metrics
| metric | `277d` | `296a` | `296b` |
| --- | ---: | ---: | ---: |
| score | `5/11` | `2/11` | `4/11` |
| coverage90 | `0.000` | `0.483` | `0.892` |
| calibration error | deterministic miss | `0.299` | `0.039` |
| change KS pass | `23/25` | `1/25` | `20/25` |
| level KS pass | `0/25` | `0/25` | `0/25` |
| cointegration ratio | `0.766` | `0.440` | `0.723` |
| worst-cell cointegration ratio | `0.281` | `0.083` | `0.281` |
| MR ratio | `1.065` | `1.605` | `1.088` |
| active MR corr | `0.750` | `0.748` | `0.783` |
| max-jump KS | `0.391` | `0.723` | `0.446` |

### What this proves
`296b` is not just better than `296a`; it answers the main hybrid question.

The hybrid split is now **materially validated** in this narrower sense:
- a shell can add broad stochastic width
- without destroying the structural backbone

The evidence is:
- coverage and calibration recovered sharply from `277d`
- cointegration and MR stayed near the backbone frontier
- `296a`'s collapse was not proof the split was wrong, only proof that the daily
  residual-token interface was wrong

### Remaining bottleneck
The misses are now much narrower and more operational:
- short-horizon coverage is still too low (`h1 = 60.5%`)
- regime differentiation is still slightly weak (`turb/calm = 1.141`)
- level-law fidelity is still dead
- pathwise jump realism is still below gate
- surface validity missed narrowly on calendar arbitrage

That means the next step should **not** revisit the overall decomposition.

### Decision
Keep the hybrid paradigm alive.

Next step:
- `296c` ideation as a **narrow shell refinement**

Most principled target:
- improve short-horizon and regime-sensitive width allocation
- while preserving the `296b` structural carryover

Concretely, `296c` should modify shell scale allocation only, not the shell mean:
- horizon-sensitive scale profile
- or regime-sensitive scale modulation
- but still zero-mean and coarse-control based
