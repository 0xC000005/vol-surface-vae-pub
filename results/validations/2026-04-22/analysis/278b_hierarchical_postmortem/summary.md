# 278b Postmortem

## Result
- Full 11-suite score: 4/11
- Passes: surface, coverage, cointegration, cross_cell_correlation

## What Was Tested
`278b` kept the `278a` two-level hierarchy fixed and changed only the Stage B sampling
rule:
- same top-k retrieved future set
- same deterministic Stage A center family
- adaptive per-query sampling temperature derived from retrieval ambiguity

## Mechanism Read
- This did **not** improve the live Stage B bottleneck.
- Coverage stayed solved, but:
  - conditionality stayed below gate
  - regime width allocation did not improve enough
  - active MR support weakened further
- So simple retrieval-geometry heuristics are not enough.
- The Stage B family is alive, but it now needs a **learned reweighting mechanism**
  over the retrieved futures rather than another hand-designed temperature rule.

## Decision
- Keep the two-level hierarchy.
- Keep `277d` as Stage A deterministic frontier.
- Keep `278a` as the first live Stage B baseline.
- Close `278b` as a negative heuristic-allocation variant.
- Next step: `278c-v0`, learned query-conditioned reweighting over the top-k retrieved
  future set.
