## 293a / 293b / 293c comparison

### Context
The `293` family is the current clean fixed-horizon joint-law reset.

The branch so far:
- `293a`: local joint-token law baseline
- `293b`: add one shared global path latent
- `293c`: replace that with a time-structured knot latent scaffold

The question is now narrower:
- is the family still missing generic latent capacity,
- or is the latent-conditioning subfamily locally capped?

### Score trajectory
- `293a`: `3/11`
- `293b`: `4/11`
- `293c`: `3/11`

So the branch improved once with a global latent, then regressed when that latent was made more structured in time.

### Stable invariants across all three
These are now robust family properties, not noise:

- coverage is alive
  - `0.901 -> 0.895 -> 0.892`
- calibration is good and improved monotonically
  - `0.041 -> 0.032 -> 0.023`
- change KS is strong
  - `24/25 -> 23/25 -> 23/25`
- cross-cell structure is stable and in gate
  - corr ratio: `1.245 -> 1.256 -> 1.244`
  - rank ratio: `0.880 -> 0.879 -> 0.881`

These are the stable wins of the family:
- live local stochastic move law
- live spread
- non-degenerate joint dependence

### Stable failures across all three
These also stayed essentially unchanged:

- level KS is dead
  - `2/25 -> 2/25 -> 1/25`
- mean reversion is effectively absent
  - `0.082 -> 0.034 -> 0.057`
- regime-sensitive width allocation is dead
  - turb/calm: `1.010 -> 0.980 -> 0.990`
- pathwise jump realism is still far from gate
  - `0.700 -> 0.680 -> 0.674`

So the unresolved object is stable too:
- **long-horizon path shape**

### What the latent variants actually changed
`293b`:
- helped calibration
- recovered cointegration
- slightly improved jump KS
- did not improve level KS or MR

`293c`:
- improved calibration again
- marginally improved jump KS again
- slightly recovered MR from the `293b` trough, but still far below gate
- worsened level KS and cointegration

That is a very specific pattern:
- latent changes help **window-level distribution quality**
- latent changes do **not** control the actual low-frequency trajectory

### Mechanism conclusion
The latent-conditioning subfamily is now close to capped.

Why:
- one constant latent (`293b`) did not solve path shape
- one time-structured latent scaffold (`293c`) also did not solve path shape
- the family's wins and failures stayed almost invariant

So the current branch no longer looks like it needs:
- more latent size
- more latent time structure
- more entropy

It looks like it needs a **different representation class** for path shape.

### Decision implication
The next move should not be:
- `293d` with a bigger latent
- another KL tweak
- another conditioning-depth tweak

It should be a **research ideation step** for a more explicit coarse path representation.

The most likely clean candidate is:
- a small coarse future path support object at knot horizons
- predicted jointly with the daily token law
- so the model conditions on an explicit coarse trajectory, not only on an unstructured latent scaffold

### Bottom line
`293` remains alive.

But the live part is now specific:
- local stochastic joint law is alive

The currently capped part is also specific:
- latent conditioning is not turning into path-shape control

So the family should continue only if the next mechanism is an **explicit coarse path representation**, not another latent variant.
