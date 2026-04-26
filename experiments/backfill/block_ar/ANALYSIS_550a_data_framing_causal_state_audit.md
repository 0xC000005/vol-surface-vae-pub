# 550a Data-Framing and Causal-State Audit

## Context

The most principled next step was a diagnostic iteration rather than another model
mutation. The current learned frontier already has useful conditional point skill, but
is not deployable because reliability failures remain. This audit tests whether those
failures are explainable by observable, history-only causal-state variables or by split
geometry before adding any new architecture.

Artifacts:

- Audit script: `experiments/backfill/block_ar/audit_550a_data_framing_causal_state.py`
- Focused tests: `test_code/test_550a_data_framing_causal_state_audit.py`
- Results JSON: `results/autoresearch/550a_data_framing_causal_state_audit_s550/audit.json`
- Results report: `results/autoresearch/550a_data_framing_causal_state_audit_s550/audit.md`

## Result

The audit ran the current frontier checkpoint
`models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt` as model type
`340c` on 192 validation windows with 48 samples per window.

Headline diagnostics:

- Mean 90% coverage: `0.867`
- Mean 90% width: `0.0904`
- Mean median MAE: `0.0259`
- Unconditional median MAE proxy: `0.0423`
- Mean conditional MAE reduction proxy: `38.2%`
- Generated mean path max jump: `0.549`
- Ground-truth mean path max jump: `0.623`

The model is not useless as a conditional generator: it materially beats an
unconditional median proxy on center accuracy. The deployability issue is that its
conditional law is too narrow and not reliably state-aware enough in risk-relevant
states.

## Split and Framing Read

The official validation block has 441 rolling windows, but adjacent 30-day futures
overlap by 29 days. The 192-window audit subset is only about `6.4` non-overlapping
30-day paths, and the full official validation block is about `14.7` non-overlapping
paths.

This does not invalidate the suite, but it means split-level gates, especially jump
and level-distribution gates, are statistically brittle. Any deployable result should
therefore report both:

- the base model law metrics, and
- the final calibrated risk-system metrics under a fixed, predeclared calibration
protocol.

## Causal-State Read

The strongest history-only associations show a clean failure mechanism:

- `history_abs_move_q90` strongly predicts realized future path max jump
  (`Spearman 0.609`).
- `history_abs_move_q90` is negatively associated with model width
  (`Spearman -0.312`).
- High `history_vov` buckets have lower coverage (`0.840`) than low `history_vov`
  buckets (`0.884`).
- High `history_abs_move_mean` buckets have lower coverage (`0.846`) than mid buckets
  (`0.893`).
- Higher `last_mean` reduces the model's MAE advantage: low `last_mean` has `43.4%`
  MAE reduction, while high `last_mean` has `31.9%`.

This is the key pathology: the model has learned conditional center skill, but its
sample dispersion is not monotonically aligned with the observable risk state. In
some high-risk histories, the realized future is jumpier while the generated law is
not wide enough.

## Decision

Do not reset architecture now. Also do not add another large architectural component.
The next principled experiment should be a minimal state-aware reliability objective
or calibration protocol that uses only history-observable state variables.

The cleanest candidate is a training/evaluation protocol that:

- keeps the same generative backbone,
- stratifies by a small fixed causal-state summary such as `history_abs_move_q90`,
  `history_vov`, and `last_mean`,
- penalizes undercoverage or insufficient path-jump support in high-risk strata,
- reports base model metrics separately from calibrated-system metrics.

This remains first-principled because the added structure is not a hand-coded market
formula or a bespoke decoder. It is a reliability constraint on conditional
distributions: histories that empirically imply larger future path risk should not
receive narrower generated distributions.
