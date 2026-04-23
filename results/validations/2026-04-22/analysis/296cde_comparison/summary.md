## 296c vs 296d vs 296e

### Context
These three runs are the clean hybrid bracket around the same structural backbone:
- `296c`: strongest balanced hybrid frontier
- `296d`: same geometry, but heavier-tailed shell support
- `296e`: same Gaussian shell law, but multiresolution early-horizon shell geometry

The question is which direction is actually aligned with the remaining bottleneck.

### Headline comparison
| model | score | headline read |
| --- | --- | --- |
| `296c` | `5/11` | strongest balanced hybrid baseline |
| `296d` | `4/11` | heavier tails help global calibration, not local allocation |
| `296e` | `4/11` | early-horizon geometry is real, but unconstrained |

### High-signal metrics
| metric | `296c` | `296d` | `296e` | read |
| --- | ---: | ---: | ---: | --- |
| score | `5/11` | `4/11` | `4/11` | baseline still strongest overall |
| h1 coverage90 | `0.593` | `0.576` | `0.859` | early-horizon geometry matters, tails alone do not |
| overall coverage90 | `0.882` | `0.890` | `0.930` | `296e` over-disperses globally |
| calibration error | `0.031` | `0.024` | `0.082` | `296e` pays for local gains with global miscalibration |
| turb/calm width ratio | `1.110` | `1.049` | `1.165` | regime-sensitive width also needs geometry, not global tails |
| change KS pass | `19/25` | `19/25` | `16/25` | `296e` weakens daily-law fidelity |
| corr ratio | `0.904` | `0.849` | `0.750` | `296e` gives back more structure |
| rank ratio | `1.491` | `1.678` | `2.006` | `296e` over-disperses the joint law |
| cointegration ratio | `0.744` | `0.745` | `0.734` | mostly flat |
| MR ratio | `1.086` | `1.092` | `1.094` | mostly flat |
| max-jump KS | `0.453` | `0.439` | `0.394` | both changes help jump realism a bit |
| surface | pass | fail | fail | both local variants lose support discipline |

### What this proves
`296e` is the important result.

Why:
- `296d` showed the bottleneck is not just global tail heaviness
- `296e` showed the bottleneck **is** early-horizon shell allocation

So the hybrid line should not move toward:
- another global tail factor
- another global scale factor

It should move toward:
- a constrained multiresolution shell that can **redistribute** shell mass early
  without increasing total shell budget or degrading structure

### Mechanism read
The clean reading is:
- `296c` = best balanced baseline
- `296d` = global support-law changes are too blunt
- `296e` = early-horizon basis is directionally right, but too unconstrained

So the next missing ingredient is not more expressivity.
It is a **budget / normalization constraint** on the fast shell.

That should make the fast basis reallocate shell mass, not simply add more shell mass.

### Decision
Keep the hybrid paradigm alive.

Next step:
- `296f` ideation

Constraint:
- keep the frozen `277d` backbone
- keep the zero-mean shell
- keep the multiresolution direction from `296e`
- add a variance-budget / redistribution constraint so early-horizon gains do not
  become global overdispersion
