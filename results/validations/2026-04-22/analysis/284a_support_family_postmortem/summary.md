# 284a Support Family Postmortem

## Question
After `282b`, `283a`, `283b`, and `283c`, is the live bottleneck still the Stage B
objective / weighting law, or has it moved upstream into the `277d` support family
itself?

## Comparison Read
- `282b` is the cleanest fixed-center residual result:
  - `5/11`
  - full-horizon mean reversion passes
  - but `level KS = 0/25`, `change KS = 5/25`
- `283b` is the cleanest reopened weighting result:
  - `4/11`
  - `change KS = 23/25`, `level KS = 1/25`
  - but mean-reversion active support still fails
- `283c` shows that local support-geometry interpolation does not create a new regime:
  - score drops back to `3/11`
  - level fidelity still does not recover materially
  - statistical-validity support does not come back enough either

## Mechanism Conclusion
The bottleneck has moved upstream.

The `277d` support family is built around replaying retrieved **delta paths** from the
query last level:
- deterministic center path: nearest retrieved delta path anchored to the query last
  level
- hierarchical Stage B variants: weighted or transformed versions of the same anchored
  future support

That family now appears structurally unable to satisfy both sides at once:
- preserve Stage A statistical validity
- and recover the missing level-side distributional fidelity

The recent bracket is important because it rules out two tempting explanations:
- it is **not** just the old expected-distance objective
- it is **not** just a train/inference support mismatch

## Decision
- Close the local `282b` to `283c` objective/support-geometry branch.
- Next step should be **research ideation for a new Stage A support object**, not
  another weighting tweak inside the existing `277d` anchored-delta family.

## Constraint For The Next Family
The next support family should:
- remain elegant and two-level
- keep retrieval/nonparametric support if possible
- stop assuming that replaying retrieved delta paths from the query last level is the
  right future support object
