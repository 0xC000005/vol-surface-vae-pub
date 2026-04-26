# 538a Panel Daily Mixture Transition Law Result

## Setup
538a kept the 537a causal 51-variable panel transition frame but replaced the single conditional Gaussian innovation with a three-component conditional Gaussian mixture:

- same empirical normal-score coordinates;
- same GRU prefix encoder;
- same full shared daily Cholesky correlation;
- per-component means and diagonal scales;
- exact mixture likelihood under teacher forcing;
- 30-day autoregressive rollout into the unchanged IV 11-suite.

Artifacts:

- Model: `diffusion/block_ar/panel_daily_mixture_transition_model.py`
- Train script: `experiments/backfill/block_ar/train_538a_panel_daily_mixture_transition.py`
- Eval script: `experiments/backfill/block_ar/evaluate_538a_panel_daily_mixture_transition.py`
- Checkpoint: `models/backfill/538a_panel_daily_mixture_transition_s538/best_model.pt`
- Result JSON: `results/autoresearch/538a_panel_daily_mixture_transition_s538/full11.json`
- Result summary: `results/autoresearch/538a_panel_daily_mixture_transition_s538/summary.md`

## Result
Score: `3/11`.

Passed:

- surface validity;
- block-AR boundary/growing-uncertainty checks;
- cross-cell correlation structure.

Failed:

- coverage;
- conditionality;
- time-series properties;
- IV-EWMA cointegration;
- regime coverage;
- distributional fidelity;
- mean reversion;
- pathwise jump realism.

Key metrics:

- overall 90% coverage: `0.930`;
- h30 90% coverage: `0.912`;
- conditionality MAE reduction: `2.0%`;
- daily-change KS cells: `20/25`;
- level KS cells: `1/25`;
- kurtosis ratio: `0.818`;
- cross-cell correlation ratio: `0.856`, effective-rank ratio `1.548`;
- aggregate mean-reversion ratio: `0.908`, but active cells only `3/24`;
- pathwise max-jump KS: `0.231`, but per-cell extreme-jump scale cells only `14/25`;
- cointegration ratio: `0.774`, but worst cell ratio `0.214` missed the `0.25` gate.

## Mechanism Read
The mixture density did exactly what the hypothesis predicted locally:

- it fixed daily-change distribution much more than 537a (`20/25` daily KS cells vs `4/25`);
- it brought aggregate kurtosis into range;
- it preserved cross-cell correlation and aggregate pathwise max-jump realism.

But it did not solve the deployability bottleneck:

- level occupancy remained poor (`1/25` level KS);
- long-horizon conditionality remained weak;
- mean-reversion geometry was still too inactive cellwise;
- per-cell extreme scale remained uneven;
- cointegration became fragile in the weakest cell.

The clean read is that one-step teacher-forced innovation likelihood is not aligned enough with the 30-day free-rollout path law. Better local innovation density can match daily moves while still generating the wrong level distribution after rollout.

## Decision
Do not keep adding daily innovation variants. The remaining bottleneck is path-level training/framing, not another marginal likelihood head.

Next principled step: run a post-538 analysis/ideation cycle focused on horizon-aware proper scoring for the same AR panel transition family, or abandon one-step-likelihood AR if the path-level objective would collapse back into evaluator-specific tuning.

