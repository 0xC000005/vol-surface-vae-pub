### Context

720a analyzed the residual 719a failure before changing the model again. The question was whether BBB OAS remained above the KS gate because the sticky-zero threshold was too conservative, or because the continuous law does not separate atom and non-atom dynamics cleanly.

### Findings

Generated once per scope and swept train-derived sticky thresholds:

- Identity baseline has zero generated exact no-change mass for OAS, while GT no-change mass is high: AAA `0.474`, BBB `0.383` in validation.
- q10/q50 sticky readout improves factor KS from `11/13` to `12/13` in both anchor and joint, but overproduces zeros: anchor BBB generated zero `0.638` versus GT `0.383`; joint BBB generated zero `0.663` versus GT `0.383`.
- Despite overproducing zeros, BBB still fails: anchor BBB KS `0.225`, joint BBB KS `0.221`.
- q90 threshold over-snaps the series and worsens BBB to `0.383` while reducing factor correlation amplitude.
- Joint factor-correlation amplitude is fragile: identity ratio `0.357`, q10/q50 `0.346`, q90 `0.312`.

Artifacts:

- `experiments/backfill/block_ar/analyze_720a_sticky_residual_sweep.py`
- `results/block_ar/720a_sticky_residual_sweep/analysis.json`
- `results/block_ar/720a_sticky_residual_sweep/analysis.md`

### Mechanism Read

The residual BBB failure is not because the sticky threshold is too weak. The readout already creates too many exact zero moves, yet BBB remains above the KS gate. The real problem is the mixture structure: the model needs to separate no-change probability from the conditional distribution of nonzero moves. A deterministic threshold on a continuous sample entangles those two pieces and also attenuates correlation amplitude.

### Decision

Do not tune the sticky threshold further. The next decisive step should be a mixed/hurdle innovation experiment: either an empirical atom-gate diagnostic to establish feasibility or a learned atom gate plus continuous nonzero innovation path, applied by the same variable-type rule across IV, anchor, and joint scopes.
