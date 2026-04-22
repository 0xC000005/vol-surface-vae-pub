# 278a Postmortem

## Result
- Full 11-suite score: 4/11
- Passes: surface, coverage, cointegration, cross_cell_correlation

## Why This Matters
`278a` is the first explicit two-level hierarchical model in the reset line:
- Stage A center-path family: `277d` learned retrieval
- Stage B scenario layer: sample from the retrieved top-k future paths

This immediately validates the hierarchy:
- coverage jumped from 0% to 74.5%
- all horizon-level coverage gates passed
- calibration error dropped to 0.100

## Key Metrics
- coverage:
  - overall cov90: 74.5%
  - h1/h7/h14/h30: 83.5% / 78.7% / 77.0% / 70.2%
  - calibration error: 0.100
- conditionality:
  - MAE reduction: 4.6%
  - turb/calm width ratio: 1.123
- regime coverage:
  - layer-1 calm windows: all 4 horizons pass
  - layer-1 turb windows: 2/4 horizons pass
  - persistent severe undercoverage: 6.9%
- deterministic carryover:
  - cointegration: pass
  - cross-cell correlation: pass
  - surface: pass
  - mean reversion: aggregate profile pass, but active support no longer passes

## Mechanism Read
- The two-level split is now empirically justified.
- The retrieval-based scenario layer solves the pure undercoverage problem without
  needing a separate stochastic neural generator.
- The remaining Stage B bottleneck is now narrow:
  - width exists
  - but width is not allocated strongly enough across regimes and windows
- The remaining Stage A carryover bottleneck also remains visible:
  - level KS is still dead
  - jump-shape realism is still weak
  - active mean-reversion support weakened under ensemble sampling

## Decision
- Keep the hierarchical retrieval family alive.
- `277d` remains the deterministic Stage A frontier.
- `278a` becomes the first live Stage B baseline.
- Next step: `278b-v0`, adaptive retrieval-temperature scenario sampling to improve
  regime-sensitive width allocation without changing the center-path family.
