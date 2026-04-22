## 293d / 293e comparison

### Context
The current `293` branch is no longer about latent conditioning. It is now about the form of the explicit support object.

`293d`:
- one monolithic coarse path code over the full knot panel

`293e`:
- a compositional sequence of coarse knot tokens

The question is whether the branch is now monotonic toward a better support object, or whether it has entered a structural tradeoff.

### Score trajectory
- `293d`: `4/11`
- `293e`: `3/11`

So the compositional support object did not improve the overall frontier.

### What 293d does better
`293d` recovered the strongest structural anchoring in the branch:
- change KS: `25/25`
- cointegration pass
- worst-cell cointegration ratio back above gate
- MR ratio: `0.093`
- active-cell MR corr: `0.773`

This is the closest the branch has gotten to recovering path-structure-adjacent suites without breaking the local law.

### What 293e does better
`293e` improved the softer path-shape metrics:
- calibration error: `0.049 -> 0.038`
- level KS: `1/25 -> 3/25`
- rank ratio: `0.849 -> 0.914`
- regime width ratio: `0.984 -> 1.029`
- max-jump KS: `0.647 -> 0.625`

So the compositional support representation did help with:
- learnability
- level-side softness
- jump-side softness

### What 293e gives back
But `293e` weakened the structural anchor:
- score: `4/11 -> 3/11`
- change KS: `25/25 -> 24/25`
- worst-cell cointegration ratio fell below gate again
- MR ratio: `0.093 -> 0.075`
- active-cell MR corr: `0.773 -> 0.670`

This is not noise. It is a coherent tradeoff.

### Mechanism conclusion
The support-object branch is now in a **representation tradeoff**:

- monolithic support:
  - harder to predict
  - stronger structural anchor

- compositional support:
  - easier to predict
  - weaker structural anchor

So the branch is still alive, but the next move should not be:
- another pure monolithic variant
- another pure compositional variant

The obvious next move, if the family continues, is a **hybrid support object**:
- preserve a global coarse anchor
- while allowing knotwise compositional refinement

### Decision implication
The next step should be **research ideation**.

The question is:
- is there a clean hybrid support representation that keeps `293d`'s anchoring and `293e`'s learnability,
- without turning into a knob soup model?

If not, the branch should be considered near a local cap.
