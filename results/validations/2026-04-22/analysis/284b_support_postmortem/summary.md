# 284b Support Postmortem

## Result
- Full 11-suite score: `4/11`
- Passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`

## What Was Tested
`284b` evaluated the existing `283a` reweighting checkpoint with a raw-future sampler,
so the sampled support finally matched the raw future support that `283a` had been
trained on.

## Mechanism Read
- This is the cleanest read so far on the raw-future support object.
- Relative to the anchored-delta family:
  - level fidelity improves materially (`level KS = 3/25`)
  - per-window coverage floor passes
  - floor/ceiling explosion behavior is clean
- But the cost is severe:
  - mean reversion collapses hard
  - active-cell slope correlation fails badly
  - regime differentiation remains wrong-way

So raw future support does improve the missing level-side behavior, but by itself it
breaks too much of the deterministic statistical-validity object.

## Decision
- Keep the support-family rethink active.
- Next step: `285a-v0`, a minimal support-transport object between raw future support
  and anchored-delta support using a horizon-decayed initial-level offset.
