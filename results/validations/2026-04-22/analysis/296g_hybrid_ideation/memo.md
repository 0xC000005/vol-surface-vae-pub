## 296g hybrid ideation

### Context
`296f` validated the constrained multiresolution family:
- the fast shell is still useful
- the global budget helps

But the comparison also showed the remaining miss:
- the fast shell is still too globally active
- what is missing is a cleaner answer to **when** the fast shell should turn on

### Candidate options
1. **Window-level fast-shell gate**
- keep the `296f` budget-plus-redistribution factorization
- add one learned scalar gate per query/window
- apply it only to the fast shell knots

2. **Per-cell fast-shell gate**
- same as above
- but gate the fast shell per cell

3. **Explicit regime classifier auxiliary task**
- keep `296f`
- add a side head that classifies stressed vs calm windows

### Recommendation
Choose **Option 1**.

Reason:
- it is the smallest mechanism that targets the actual remaining question
- it is more elegant than a per-cell gate
- it is more first-principles than adding a regime label side task
- it should preserve shared structure better than finer-grained gating

### 296g-v0
Keep fixed:
- frozen `277d` backbone
- zero-mean Gaussian shell
- multiresolution knot basis from `296f`
- explicit per-cell shell budget
- normalized redistribution

Change only:
- split knots into:
  - fast: early-horizon knots
  - slow: later knots
- add one learned scalar gate `g in (0, 1)` from the pooled query state
- multiply the fast-knot raw scales by `g` before budget renormalization

Expected read:
- if `296g` preserves the h1 gain while recovering some regime balance and surface
  discipline, the multiresolution hybrid line stays alive
- if not, the fast-shell branch is near a local cap and the hybrid line needs a
  broader rethink
