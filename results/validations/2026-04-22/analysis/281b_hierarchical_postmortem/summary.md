# 281b Hierarchical Postmortem

## Result
- Full 11-suite score: 5/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation, mean_reversion

## What Was Tested
`281b` kept the same fixed `277d` Stage A center path and the same residual bank as
`281a`, but added the minimal joint Stage B control:
- learned residual temperature
- plus one learned residual scale

The training objective still minimized expected residual distance to the realized future
under the weighted-centered residual bank.

## Mechanism Read
- This is a clean negative on the **training objective**, not necessarily on the
  architecture class.
- The joint amplitude+shape control did not increase expressive power in the useful way.
- Instead it learned to collapse the residual law:
  - coverage fell catastrophically
  - regime coverage collapsed
  - coverage floor became nearly degenerate
- But the deterministic center-path suites remained intact, which means the fixed Stage A
  center path is still doing its job.

The likely cause is clear:
- the expected-distance objective rewards finding one narrow good-match residual
  configuration
- it does not reward preserving a usable scenario distribution
- once temperature and scale are both free, the model can minimize that objective by
  shrinking the distribution too aggressively

## Decision
- Keep the two-level hierarchy.
- Keep the fixed Stage A center path and fixed residual bank.
- Do not continue the current expected-distance training objective unchanged.
- Next step: post-experiment analysis of the Stage B training objective itself before
  another experiment.
