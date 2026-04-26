# 537a Panel Daily Cholesky Transition Law Result

## Setup
537a replaced the failed one-shot global Gaussian panel law with a causal autoregressive daily transition density over the aligned 51-variable panel:

- 25 IV surface cells;
- 13 factor levels;
- 13 factor returns/diffs;
- empirical normal-score coordinates;
- GRU prefix encoder;
- full conditional daily Cholesky innovation density;
- exact teacher-forced Gaussian likelihood;
- 30-day autoregressive rollout, returning the IV subpanel to the unchanged 11-suite.

Artifacts:

- Train script: `experiments/backfill/block_ar/train_537a_panel_daily_cholesky_transition.py`
- Eval script: `experiments/backfill/block_ar/evaluate_537a_panel_daily_cholesky_transition.py`
- Checkpoint: `models/backfill/537a_panel_daily_cholesky_transition_s537/best_model.pt`
- Result JSON: `results/autoresearch/537a_panel_daily_cholesky_transition_s537/full11.json`
- Result summary: `results/autoresearch/537a_panel_daily_cholesky_transition_s537/summary.md`

## Result
Score: `4/11`.

Passed:

- surface validity;
- block-AR boundary/growing-uncertainty checks;
- IV-EWMA cointegration;
- cross-cell correlation structure.

Failed:

- coverage;
- conditionality;
- time-series properties;
- regime coverage;
- distributional fidelity;
- mean reversion;
- pathwise jump realism.

Key metrics:

- overall 90% coverage: `0.957`, above the effective per-cell upper bound pressure;
- h30 90% coverage: `0.946`;
- conditionality MAE reduction: `3.83%`, below the `>5%` gate;
- daily-change KS cells: `4/25`;
- level KS cells: `1/25`;
- cross-cell correlation ratio: `0.827`, effective-rank ratio `1.719`;
- aggregate mean-reversion ratio: `0.814`, but active cells only `8/24`;
- pathwise max-jump KS: `0.270`, but per-cell extreme-jump scale cells only `7/25`.

## Mechanism Read
The autoregressive panel transition was a useful falsifier because it fixed two important failures of the global Gaussian panel family:

- cross-cell dependence became realistic enough to pass;
- path-level max-jump incidence/scale became realistic enough at the aggregate path level.

The remaining failure pattern is also clean:

- the daily full-covariance Gaussian transition is too broad and too symmetric in the wrong cells;
- it preserves aggregate path scale but not per-cell move-size and level occupancy;
- it has weak long-horizon conditional signal after the first week;
- mean reversion is right in aggregate but too weak and uneven cell-by-cell.

This is not a reason to stack paper modules. The issue is the daily innovation family: a single conditional Gaussian innovation cannot simultaneously match asymmetric/fat-tailed cell moves, level occupancy, and cellwise mean-reversion geometry.

## Decision
537a is below the old `392a`/`510a` frontier and below the user's deployability target. It is still scientifically useful:

- it shows that autoregressive panel conditioning is not enough by itself;
- it shows that learned conditional cross-channel covariance is useful;
- it points to the next minimal change: keep the causal panel transition frame, but replace the daily Gaussian innovation with a small likelihood-based non-Gaussian innovation law.

Next principled step: implement 538a as a daily conditional mixture/logistic-or-student innovation law in the same autoregressive panel transition frame, without adding a separate calibration layer or evaluator-specific correction.

