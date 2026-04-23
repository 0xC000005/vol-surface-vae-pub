## 296e vs 296f

### Context
Both runs test the same hybrid direction:
- frozen `277d` backbone
- zero-mean Gaussian shell
- multiresolution fast shell for early horizons

The only difference is whether the fast shell is budgeted:
- `296e`: unconstrained multiresolution shell
- `296f`: explicit per-cell budget plus normalized redistribution

### Headline comparison
| model | score | headline read |
| --- | --- | --- |
| `296e` | `4/11` | early-horizon geometry works, but over-disperses globally |
| `296f` | `4/11` | budgeted shell partially restores balance, but not enough |

### High-signal metrics
| metric | `296e` | `296f` | read |
| --- | ---: | ---: | --- |
| h1 coverage90 | `0.859` | `0.862` | local early-horizon gain preserved |
| overall coverage90 | `0.930` | `0.928` | slight pullback only |
| calibration error | `0.082` | `0.076` | modest improvement |
| turb/calm width ratio | `1.165` | `1.117` | some targeted width was lost |
| MAE reduction | `-1.0%` | `0.5%` | modest improvement |
| change KS pass | `16/25` | `17/25` | modest improvement |
| corr ratio | `0.750` | `0.772` | modest improvement |
| rank ratio | `2.006` | `1.912` | modest improvement |
| cointegration ratio | `0.734` | `0.739` | modest improvement |
| max-jump KS | `0.394` | `0.389` | modest improvement |
| q99 ratio | `1.136` | `1.188` | slightly stronger jump scale |

### What this proves
The constrained multiresolution family is still alive.

Why:
- `296f` preserved the best `296e` local gain: high h1 coverage
- while moving several global metrics back in the right direction

But it is not yet enough:
- surface validity still fails
- regime-sensitive width fell back below gate
- overall coverage is still too high
- level-law fidelity is still dead

### Mechanism read
The budget helped, but it is too global.

The fast shell now needs a cleaner answer to:
- **when** should it turn on strongly?

That suggests the next mechanism should be an activation/gating term on the fast shell,
not another global budget refinement.

### Decision
Keep the multiresolution hybrid family alive.

Next step:
- `296g` ideation

Constraint:
- keep the `296f` budget-plus-redistribution factorization
- add a minimal learned activation/gating mechanism for the fast shell
- do not add another global support or budget factor
