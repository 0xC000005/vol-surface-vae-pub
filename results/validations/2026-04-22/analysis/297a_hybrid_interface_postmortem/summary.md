## 297a fixed-center shell oracle diagnostic

### Context
`296` left a critical interface question:

- Is the fixed `277d` center plus shell interface intrinsically capped?
- Or have the learned shells simply failed to approximate the right residual law?

`297a` tested this with an evaluation-only oracle diagnostic:

- keep the frozen `277d` center path
- build an empirical residual bank from training windows only
- mean-center residual raw-change paths
- sample those residual paths around the fixed center
- evaluate full 11-suite on validation windows

Two variants were run:

- `297a`: independent residual draws
- `297a_paired`: paired symmetric residual draws

### Result
Both variants scored `4/11`.

| model | score | passes | cov90 | h1 cov90 | turb/calm | chg KS | lvl KS | corr | rank | MR | jump KS | q99 cells |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `297a` | 4 | surface, block_ar, coint, xcell | 0.830 | 0.895 | 0.973 | 4 | 0 | 0.964 | 1.285 | 1.063 | 0.342 | 3 |
| `297a_paired` | 4 | surface, block_ar, coint, xcell | 0.832 | 0.895 | 0.954 | 4 | 0 | 0.971 | 1.273 | 1.069 | 0.336 | 4 |

### Mechanism read
The diagnostic falsifies the fixed-center shell interface as the main path forward.

Even a strong empirical training residual shell:

- does not improve level KS (`0/25`)
- destroys change KS (`4/25`)
- fails regime sensitivity
- fails time-series tail/move-size profile
- fails pathwise jump realism
- weakens full-horizon mean reversion

This is stronger than the `296` learned-shell evidence. The issue is not just that the learned shell is weak. The interface itself is wrong for the remaining objective.

The fixed center path sets the level-support geometry too strongly. Adding residual paths around it can add width, but it does not correct the unconditional level law, and empirical residual replay introduces the wrong local change geometry.

### Decision
Close the fixed-`277d` center plus shell program.

Do not run more:

- scalar/per-cell shell gates
- shell budget tweaks
- heavier residual tails
- empirical residual replay around the same fixed center

The next paradigm must put stochasticity into the structural center/support object itself, not only into a residual shell around one fixed path.

### Next step
Run paradigm-shift ideation for `298a`.

Required constraint:
- preserve the lesson that structural validity and stochastic adequacy are distinct jobs
- abandon the one-fixed-center interface
- do not return to plain retrieval weighting or daily residual correction
