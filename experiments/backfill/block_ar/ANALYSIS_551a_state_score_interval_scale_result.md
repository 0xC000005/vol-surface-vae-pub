# 551a State-Score Interval-Scale Result

## Context

550a showed a clean causal-state signal: `history_abs_move_q90` predicts realized
future path jumps, but the frontier model's dispersion is not reliably aligned with
that state. 551a tested the smallest deployable intervention that follows from that
audit:

- keep the frozen `392a` / `340c` learned generator unchanged;
- use only one history-observable state score: `history_abs_move_q90`;
- fit three calibration bins on the pre-validation block only;
- apply one median-preserving residual scale per state bin and horizon;
- avoid per-cell calibration tables, validation futures, retrieval, new architecture,
  or evaluator-specific loss.

Artifacts:

- Script: `experiments/backfill/block_ar/evaluate_551a_state_score_interval_scale_calibrated_system.py`
- Tests: `test_code/test_551a_state_score_interval_scale.py`
- Result JSON: `results/block_ar/551a_state_score_interval_scale_392a/full11.json`
- Result report: `results/block_ar/551a_state_score_interval_scale_392a/full11.md`

## Result

551a scored `6/11`.

Failed suites:

- coverage
- conditionality
- cointegration
- regime_coverage
- distributional_fidelity

Key metrics versus the uncalibrated frontier:

| Metric | 392a frontier | 551a |
| --- | ---: | ---: |
| Suite score | `8/11` | `6/11` |
| Coverage90 | `0.8675` | `0.9020` |
| Calibration error | `0.0240` | `0.0139` |
| Conditional MAE reduction | `5.14%` | `4.99%` |
| Cointegration gen/GT ratio | `0.700` | `0.661` |
| Regime layer2 | `0/8` | `0/8` |
| Daily-change KS pass cells | `25/25` | `25/25` |
| Level KS pass cells | `10/25` | `13/25` |
| Pathwise max-jump KS | `0.373` | `0.208` |

551a improves average interval calibration, level KS, and pathwise jump KS, but it
does not improve the deployable suite score. The coverage suite still fails because
several cells exceed the 95% overcoverage cap at h1, h14, and h30. Regime coverage
remains `0/8` at layer2, and conditionality falls just below the gate.

## Mechanism Read

The result is a useful falsifier because it isolates the 550a signal. The state score
is real, but width-only calibration is not expressive enough:

- It moves aggregate coverage from undercoverage toward the target.
- It also creates localized overcoverage, especially at later horizons.
- It does not fix the cell/regime allocation problem.
- It slightly dilutes the conditionality margin and damages a cointegration worst cell.
- It improves level KS from `10/25` to `13/25`, but still misses the `15/25` gate.

The fitted scale means by state bin were `[1.147, 1.189, 1.129]`, so the calibration
did not learn a clean monotone "higher state equals wider law" policy. That is
important: even the best simple pre-validation fit does not map the causal risk score
into a stable deployable width rule.

## Decision

Do not tune alpha, bin thresholds, target coverage, or scale ranges inside this branch.
That would repeat the old calibration-table failure pattern with a new state feature.

551a closes the minimal state-score width-calibration idea. The next principled move,
if continuing the calibrated-system route, must change the object from post-hoc width
to a reliability objective that can learn conditional level allocation without
destroying the `392a` structural passes. If that cannot be specified cleanly, the
honest local conclusion remains that `392a` / `510a` are the learned frontier at
`8/11`, and further `11/11` progress needs either broader data or a separately reported
policy layer with explicit limitations.
