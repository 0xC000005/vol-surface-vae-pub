# 11-Suite Data-Framing and Oracle Audit

## Scope

- validation framing: `192` windows, history `30`, future `30`, `test_start=4511`, `val_size=441`
- oracle samples per window: `48`
- official sample-array suites are run exactly; conditionality is a labeled proxy because the official gate requires a model/shuffled-history API.

## Oracle Scores

| Oracle | Official sample-array score | Conditionality proxy | Failed sample-array suites |
|---|---:|---:|---|
| `repeat_gt` | `7/10` | `False (tc=undefined, mae_red=100.0%)` | coverage, regime_coverage, distributional_fidelity |
| `train_marginal` | `4/10` | `False (tc=1.044, mae_red=0.0%)` | coverage, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| `val_marginal_oracle` | `8/10` | `False (tc=1.004, mae_red=42.0%)` | regime_coverage, mean_reversion |
| `history_knn` | `4/10` | `False (tc=0.939, mae_red=33.6%)` | coverage, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| `regime_bucket` | `4/10` | `False (tc=0.966, mae_red=12.1%)` | coverage, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| `persistence_residual` | `4/10` | `False (tc=0.996, mae_red=25.8%)` | coverage, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |

## Data Self-Consistency

- train-vs-val level KS: `1/25` cells pass D<0.15; median D `0.422`, worst `0.748`
- val split-half level KS: `5/25` cells pass D<0.15; median D `0.425`, worst `0.793`
- train-vs-val daily-change KS: `15/25`; val split-half daily-change KS: `20/25`
- train-vs-val path max-jump KS: `0.462`; val split-half path max-jump KS: `0.542`

## Conditional Signal

- `training` / `conditionality_full_surface_rv`: step_abs_mean turb/calm `1.419`, step_abs_q90 `1.581`, path_range `1.184`, h30_abs_dev `1.091`
- `training` / `regime_mean_iv_std`: step_abs_mean turb/calm `1.499`, step_abs_q90 `1.795`, path_range `1.321`, h30_abs_dev `1.131`
- `validation` / `conditionality_full_surface_rv`: step_abs_mean turb/calm `1.011`, step_abs_q90 `0.907`, path_range `0.865`, h30_abs_dev `0.745`
- `validation` / `regime_mean_iv_std`: step_abs_mean turb/calm `0.989`, step_abs_q90 `0.901`, path_range `1.034`, h30_abs_dev `1.223`

## Conditional Center Predictability

- unconditional train-median MAE: `0.04235`
- persistence MAE: `0.03207` (24.3% vs unconditional)
- kNN mean MAE: `0.03011` (28.9% vs unconditional)
- kNN median MAE: `0.02902` (31.5% vs unconditional)

## Gate Classification

- `surface`: valid sanity/economic-shape gate. Ground-truth-relative calendar margin and loose butterfly/explosion gates test generated-surface plausibility, not conditional signal strength.
- `coverage`: valid calibration gate, but incompatible with pure deterministic GT replay and with indiscriminate conservative widening. Per-cell upper coverage bound at 95% penalizes overbroad scenario clouds; passing requires calibrated diversity, not just safety.
- `conditionality`: partly policy-prior-like on this split. Validation realized movement has weak or negative turbulent/calm ratios under the suite's history-volatility splits (full-surface step_abs_mean=1.011, mean-IV step_abs_mean=0.989). A >1.15 width requirement may be defensible as conservative risk policy but is not strongly identified by conditional futures here.
- `time_series`: valid distributional-law gate. ACF, kurtosis, tail scale, and move-size shares are unconditional path-law requirements, not a regime prior. Persistent failures indicate wrong dynamic law.
- `block_ar`: mostly technical sanity gate. Boundary smoothness prevents block stitching artifacts. It is not central for one-shot models but is lenient and not a persistent blocker.
- `cointegration`: valid but weak economic consistency gate. The gate is relative to GT pass rates and can pass under broad path clouds; useful as a floor, not sufficient evidence of good conditional generation.
- `regime_coverage`: mixed calibration/policy gate. Layer 1/3 are risk-calibration checks; Layer 2 also has a 95% upper bound, so a conservative prior must be controlled rather than globally widening all intervals.
- `distributional_fidelity`: valid, but level-KS is a strict nonstationarity-sensitive requirement. Train-vs-val level KS passes 1/25 while val split-half passes 5/25. Both are far below the 15/25 gate, so the current slice has substantial level-marginal drift even inside validation. The requirement is not contradictory, but it is stricter than daily-change fidelity and can punish a train-only learned model for regime/time-period shift.
- `cross_cell_correlation`: valid joint-law gate. Cross-cell correlation/rank is required for multivariate risk scenarios and has been passable by 340c, so it is not contradictory.
- `mean_reversion`: valid dynamic/economic gate. This has been passable by 340c and 353a; failure is architecture/dynamic-law mismatch rather than an impossible requirement.
- `pathwise_jump_realism`: valid but very stringent under current split instability. Validation split-half path max-jump KS is 0.542 against a 0.20 gate. That is not a near miss; pathwise extreme behavior shifts materially within the validation period. The gate targets a real risk-manager property, but exact passing may need explicit tail-risk calibration rather than only learned average conditional dynamics.

## Bottom Line

The 11-suite is not globally contradictory, but it is not a pure likelihood test. The validation-marginal oracle reaches 8/10 sample-array suites, so most gates are mutually satisfiable when the target marginal law is known. But deterministic GT replay reaches only 7/10 because anti-overcoverage and median-bias gates require genuine diversity. Coverage, regime coverage, conditionality, and pathwise tails encode risk-manager calibration preferences; the regime-width portion is weakly supported by the current validation conditional signal and may require an explicit conservative risk prior. That prior is justifiable only if framed as policy calibration and kept constrained by the upper coverage gates, because broad unconditional widening is penalized.
