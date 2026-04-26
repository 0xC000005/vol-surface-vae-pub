# 553a Factor-State Signal Audit

## Context

552a found that the closest current risk-lens candidate is still not fully presentable:
`510a` is realistic and conditionally useful, but it fails regime under-inclusion and
is IV-only. The next principled diagnostic was therefore to test whether existing
non-IV factors contain history-only signal for future stress states.

The local dataset includes:

- `ret`
- `price`
- `slopes`
- `skews`
- `levels`

553a computed simple history-only summaries for these factors and compared their
rank correlations against future IV and return stress targets.

Artifacts:

- Script: `experiments/backfill/block_ar/audit_553a_factor_state_signal.py`
- Tests: `test_code/test_553a_factor_state_signal_audit.py`
- Report: `results/autoresearch/553a_factor_state_signal/factor_signal.md`
- JSON: `results/autoresearch/553a_factor_state_signal/factor_signal.json`

## Result

The factor signal is material.

- best IV-history absolute Spearman: `0.576`;
- best factor-history absolute Spearman: `0.799`;
- factor signal beats IV-only signal: `true`;
- factor signal material: `true`.

Strong factor signals include:

- `price_hist_last` vs `future_ret_abs_mean`: Spearman `-0.799`;
- `levels_hist_last` vs `future_ret_abs_mean`: Spearman `0.757`;
- `levels_hist_last` vs `future_iv_level_mean`: Spearman `0.731`;
- `price_hist_last` vs `future_iv_level_mean`: Spearman `-0.724`;
- `slopes_hist_last` vs `future_iv_level_mean`: Spearman `-0.659`;
- `skews_hist_last` vs `future_iv_level_mean`: Spearman `-0.638`.

## Mechanism Read

The current IV-only generator is missing observable state information that is relevant
to risk use. This is not just a calibration problem:

- price, level, slope, skew, and return-history summaries carry stress-state signal;
- the signal targets future return stress and future IV level, both relevant to a risk
  manager;
- the factor signal is stronger than the best IV-only history signal in this audit.

This explains why IV-only wrappers struggle with regime inclusion. The generator sees
part of the state, but not enough of the state that risk managers care about.

## Decision

The next serious model direction should be a factor-conditioned scenario generator or
a joint IV+factor path model. The clean product framing is:

- base learned IV generator remains `392a` / `510a` frontier evidence;
- risk prototype can use `510a` as the current IV-only scenario core;
- next research should add observed factor-history conditioning and eventually generate
  factor-consistent scenarios, not tune overcoverage or post-hoc width tables.

This is aligned with recent time-series research trends toward conditional joint-law
models and compact state encoders: the next model should learn a richer state
representation, not add more hand calibration.
