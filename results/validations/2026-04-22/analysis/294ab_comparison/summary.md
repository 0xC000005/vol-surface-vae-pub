## 294a / 294b comparison

### Context
The `294` branch was the first explicit continuous path-shape family inside the
fixed-horizon joint-law program.

It tested two scaffold objects:
- `294a`: global low-frequency basis scaffold
- `294b`: piecewise-linear knot scaffold

Both keep the same residual daily token law and differ only in the scaffold object.

### Score trajectory
- `294a`: `3/11`
- `294b`: `3/11`

So the scaffold-object swap did not move the frontier.

### What 294a is better at
`294a` is the better path-fidelity member of the pair:
- coverage90: `0.782` vs `0.707`
- calibration error: `0.087` vs `0.124`
- level KS: `9/25` vs `6/25`
- max-jump KS: `0.662` vs `0.740`

So the smoother basis scaffold is better at keeping the long-window path shape and
distributional level law reasonable.

### What 294b is better at
`294b` is slightly better at long-run structural anchoring:
- cointegration ratio: `2.247 -> 1.402`
- active MR pass count: `4 -> 9`

But the improvement is narrow, because:
- worst-cell cointegration ratio stayed unchanged at `0.211`
- aggregate MR ratio got even worse: `0.163 -> 0.082`
- day-1 coverage collapsed to zero

So the knot scaffold improved some coarse structural averages while making the
short-horizon law worse.

### Mechanism conclusion
The `294` branch is now close to a local cap.

Why:
- changing the scaffold object does change behavior
- but both variants land in the same overall failure family:
  - perfect local change KS
  - decent cross-cell structure
  - but weak short-horizon uncertainty / reversion allocation

This means the bottleneck is deeper than the scaffold object itself.

The current explicit scaffold + residual daily token decomposition still cannot place
enough:
- day-1 width
- short-horizon reversion
- jump timing mass

while keeping the improved level-law fit.

### Decision
Treat the `294` continuous scaffold branch as **near a local cap**.

The next principled step should be **research ideation for the next mechanism class**,
not another local scaffold-object tweak.

The open design question is:
- what fixed-horizon path-shape mechanism can improve short-horizon uncertainty and
  reversion allocation,
- without giving up the `294a` level-law gains and without returning to support
  codebooks?
