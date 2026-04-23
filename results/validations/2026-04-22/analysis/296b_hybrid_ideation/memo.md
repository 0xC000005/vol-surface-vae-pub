## 296b ideation

### Context
`296a` proved the hybrid split is still plausible, but the shell interface was wrong:
- daily residual-token corrections were too local
- exact token reconstruction made the shell near-deterministic
- the shell was free to drag the frozen `277d` center path away from its structural
  validity

So `296b` must satisfy two hard constraints:
1. **mean-preserving relative to the backbone**
2. **coarser pathwise freedom than direct daily correction**

### Alternatives considered
1. **Daily residual shell again, but add entropy / calibration regularization**
- smallest code change
- rejected for `v0`
- does not solve the deeper issue that daily local freedom can still distort the
  center path

2. **Latent pathwise shell with coarse zero-mean controls**
- residual freedom moves to knot horizons instead of daily steps
- mean-preserving sampling can be enforced explicitly
- recommended

3. **Return to retrieval-style Stage B around `277d`**
- rejected
- would revive the older support-weighting collapse path

### Decision
Next step: `296b-v0`

Use:
- frozen `277d` backbone
- learned shell that predicts **coarse residual control scales** at knot horizons
- zero-mean stochastic residual controls around those scales

### Concrete design
1. **Backbone**
- load the best `277d` checkpoint
- produce the deterministic center path
- keep it frozen

2. **Shell target**
- compute residual deviations from the center path at a short knot set
- operate on a coarse pathwise control object, not direct daily residual tokens

3. **Shell output**
- conditional **scale** (and optionally correlation-free diagonal covariance) for the
  residual control sequence
- mean fixed at zero

4. **Sampling rule**
- sample residual controls around zero
- linearly interpolate them across the 30-day window
- convert them into residual path deviations around the center path
- use **paired symmetric sampling** (`r` and `-r`) so the sample ensemble stays
  centered on the frozen backbone path

### Why this is cleaner than 296a
`296b` removes both failure points from `296a`:
- no freely learned daily corrective path
- no shell mean drift away from the backbone

The shell only controls:
- how much deviation is plausible
- and what coarse pathwise shape that deviation can take

The backbone still owns the center path.

### Expected effect
If this is right, `296b` should:
- preserve most of the `277d` structural passes
- improve coverage and calibration over `277d`
- avoid the `296a` failure mode where the shell itself becomes the de facto center path

### Kill criteria
`296b` is alive only if it keeps the backbone-centered suites:
- cointegration
- cross-cell correlation
- mean reversion

while materially improving at least one stochastic suite:
- coverage
- conditionality
- regime coverage
- jump realism

If it cannot do that, then the hybrid split itself is close to capped.
