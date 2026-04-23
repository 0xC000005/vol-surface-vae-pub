## 296c ideation

### Context
`296b` restored the hybrid line:
- structural backbone passes stayed alive
- broad coverage and calibration came back

But the remaining miss is now narrow:
- h1 coverage still under target
- regime differentiation only slightly below gate
- shell width allocation appears too flat across horizons and windows

So `296c` should not change shell mean or shell support geometry.
It should change only **scale allocation**.

### Alternatives considered
1. **Global scalar temperature on top of 296b**
- too weak
- would widen all horizons together and likely give back long-horizon calibration
- rejected

2. **Query-conditioned horizon profile shared across cells**
- directly targets the observed problem: too little early-horizon width and too-flat
  regime allocation
- preserves zero-mean coarse shell geometry
- recommended

3. **Explicit regime classifier / hand-coded vol-of-vol gate**
- more hand-engineered than necessary at this stage
- keep as fallback only if the cleaner query-conditioned profile fails

### Decision
Next step: `296c-v0`

Keep everything from `296b` except the scale-allocation head.

### Concrete design
1. **Backbone**
- frozen `277d`
- unchanged

2. **Shell geometry**
- zero-mean coarse residual controls at the same knot horizons
- paired symmetric sampling
- unchanged

3. **New scale allocation**
- predict two factors from the shell encoder:
  - per-knot-per-cell base scales
  - a **shared query-conditioned knot profile** multiplier
- final knot scale:
  - `base_scale * profile_scale`
- profile is shared across cells so horizon width moves coherently

4. **Expected effect**
- more width at h1 when the query calls for it
- stronger turb/calm differentiation through query-dependent profile scaling
- less risk of destroying structure than a fully free daily shell

### Why this is the cleanest next move
It changes only one thing:
- how shell variance is allocated across knot horizons

It does **not** change:
- shell mean
- shell support object
- backbone path
- retrieval/decomposition logic

### Kill criteria
`296c` is alive only if it improves at least one of:
- h1 coverage
- turb/calm width ratio
- regime Layer-1 pass rate

while preserving:
- cointegration pass
- cross-cell correlation pass
- mean reversion pass

If it fails that, then the next step should likely be a different shell objective or a
different shell support family, not more scale-head tweaks.
