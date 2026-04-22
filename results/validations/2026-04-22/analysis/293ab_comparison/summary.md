## 293a vs 293b comparison

### Context
`293a-v0` established that the new fixed-horizon joint-law family is alive:
- strong daily change KS
- live stochastic spread
- usable cross-cell structure

`293b-v0` then added one shared global path latent to test whether a single future-window-level variable could fix the missing long-horizon path shape.

### What changed

#### Improved in 293b
- score: `3/11 -> 4/11`
- calibration error: `0.041 -> 0.032`
- token accuracy at best checkpoint: `0.0229 -> 0.0264`
- cointegration pass recovered
- jump KS improved slightly: `0.700 -> 0.680`

#### Essentially unchanged
- change KS stayed strong: `24/25 -> 23/25`
- cross-cell structure stayed in gate:
  - corr ratio: `1.245 -> 1.256`
  - rank ratio: `0.880 -> 0.879`
- level KS stayed dead: `2/25 -> 2/25`
- coverage stayed alive but did not improve materially: `0.901 -> 0.895`
- token entropy at best checkpoint stayed almost identical: `4.845 -> 4.854`

#### Regressed
- regime differentiation: `1.010 -> 0.980`
- mean reversion ratio: `0.082 -> 0.034`
- active MR pass rate stayed `0`
- kurtosis ratio worsened: `1.735 -> 1.823`

### What this means
The shared latent did **not** become a path-shape controller.

Evidence:
- if it were controlling long-horizon path shape, the first things to move should have been:
  - level KS
  - MR
  - regime-sensitive width
- none of those improved
- MR got worse

What actually moved:
- calibration
- token accuracy
- cointegration stability

That is a different pattern. It says the latent is acting like:
- a mild global context / entropy regularizer
- or a better window-level correlation prior

not like:
- a mechanism that constrains the actual low-frequency trajectory shape

### Stable invariants across both runs
Across `293a` and `293b`, the following are now stable:
- local stochastic move law is alive
- daily change fidelity is strong
- cross-cell dependence is preserved
- long-horizon level-law fidelity is weak
- mean reversion is almost absent
- regime width allocation is weak
- pathwise jump ordering is weak

So the family diagnosis is no longer ambiguous:
- **local-law solved enough for the family to be live**
- **path-shape unsolved**

### Decision implication
The next mechanism should target **structured path shape**, not generic latent capacity.

Specifically, the next move should provide an explicit low-frequency trajectory scaffold that daily tokens must respect.

That points toward:
- coarse-to-fine future anchor paths / horizon knots

not:
- another generic global latent
- more token capacity
- more sampler entropy tweaks
