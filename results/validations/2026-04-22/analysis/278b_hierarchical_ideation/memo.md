# 278b Hierarchical Ideation

## Context
`278a` validated the two-level retrieval hierarchy:
- Stage A center path from `277d`
- Stage B scenario spread from the top-k retrieved futures

Coverage is now solved at the headline level.
The live Stage B miss is more specific:
- width exists
- but it is not allocated strongly enough by regime and forecast difficulty
- turb/calm width ratio is 1.123, just under the 1.15 gate
- regime coverage still fails on turbulent longer horizons

## Decision
Next step: `278b-v0`

### Family
Adaptive retrieval-temperature scenario sampling.

### Core idea
Keep the same top-k retrieved future set as `278a`, but replace the fixed sampling
temperature with a per-query temperature derived from retrieval uncertainty:
- more ambiguous query => wider sampling
- sharper query => narrower sampling

## Why This Is The Smallest Principled Step
- no new neural generator
- no explicit vol-of-vol feature engineering
- no extra decoder
- no side path

Only the sampling rule changes, and it changes using the retrieval geometry itself.

## Pre-Registered Success Criteria
Relative to `278a`, `278b` should improve at least two of:
- conditionality MAE reduction
- turb/calm width ratio
- regime coverage layer-1 turbulent horizons
- persistent severe undercoverage rate

while preserving:
- overall coverage pass
- cointegration pass
- cross-cell correlation pass
- surface pass

## Immediate Next Action
Implement `278b-v0` with per-query adaptive temperature from the top-k retrieval score
dispersion and rerun the full 11-suite.
