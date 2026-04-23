## 296c vs 296d

### Context
Both models keep the same validated hybrid decomposition:
- frozen `277d` backbone
- zero-mean coarse shell
- paired symmetric sampling

The only meaningful change is the shell law:
- `296c`: Gaussian coarse controls with shared query-conditioned knot profile
- `296d`: same geometry, but Student-t coarse controls with a learned query-conditioned
  degrees-of-freedom scalar

This comparison asks whether the remaining hybrid bottleneck is really a support-law
problem or something deeper about shell allocation.

### Headline comparison
| model | score | main read |
| --- | --- | --- |
| `296c` | `5/11` | strongest current hybrid frontier |
| `296d` | `4/11` | support law changed the tails, but not the real allocation bottleneck |

### High-signal metrics
| metric | `296c` | `296d` | read |
| --- | ---: | ---: | --- |
| score | `5/11` | `4/11` | support-law swap lost a suite |
| overall coverage90 | `0.882` | `0.890` | slight improvement |
| calibration error | `0.031` | `0.024` | slight improvement |
| h1 coverage90 | `0.593` | `0.576` | worse where the suite is still tightest |
| turb/calm width ratio | `1.110` | `1.049` | worse regime differentiation |
| change KS pass | `19/25` | `19/25` | unchanged |
| level KS pass | `0/25` | `0/25` | unchanged |
| corr ratio | `0.904` | `0.849` | structural carryover weakened |
| cointegration ratio | `0.744` | `0.745` | flat |
| MR ratio | `1.086` | `1.092` | flat |
| max-jump KS | `0.453` | `0.439` | slight improvement |
| q99 ratio | `0.900` | `0.928` | slight improvement |

### What this proves
The support-law branch is **alive in a narrow sense**, but simple support changes are
already near a local cap.

Why:
- heavier tails did affect the learned shell law
- broad calibration and pathwise tail scale improved

But the real open suites did not move in the right direction:
- h1 coverage worsened
- regime-sensitive width worsened
- level-law fidelity did not move
- structural carryover weakened

So the hybrid line is still not asking for "fatter tails everywhere".
It is asking for **better conditional allocation** of shell mass.

### Mechanism read
`296d` shows that global or query-wide support heaviness is too blunt.

The remaining miss is more specific:
- the shell needs the right width in the right windows and horizons
- without giving back the structural validity inherited from `277d`

That means the next clean move should not be another simple support swap.
It should change how the shell is **trained to allocate** mass, or change the support
in a more targeted way than one global tail parameter.

### Decision
Treat:
- the scale-allocation branch as near a local cap
- the simple support-law branch as also near a local cap

Next step:
- `296e` ideation

Constraint:
- keep the hybrid decomposition
- keep the zero-mean shell geometry
- change shell objective or targeted short-horizon allocation mechanism, not another
  global scale or global tail factor
