# 618a joint38 AR Student-t mean-placement NLL result

## Context

617a showed that Student-t sampler temperature improves the likelihood path to `5/11`, but stable failures remain: level occupancy, mean reversion, coverage layer2, and per-cell tail-scale imbalance. The next hypothesis was that the likelihood model is using broad stochastic coverage instead of learning a sufficiently accurate conditional center.

618a therefore adds a small one-step mean-placement objective inside the same likelihood model:

- same joint38 state panel;
- same causal memory;
- same Student-t full-covariance transition density;
- same predicted transition mean;
- loss = Student-t NLL + `mean_loss_weight * smooth_l1(predicted_mean, realized_increment)`.

No separate IV/factor path and no post-hoc correction are introduced.

## Change

Updated:

- `diffusion/block_ar/generic_gaussian_transition_law.py`
- `experiments/backfill/block_ar/train_614a_unified_ar_gaussian_likelihood.py`
- `test_code/test_614a_generic_gaussian_transition.py`

Verification:

```bash
python -m py_compile \
  diffusion/block_ar/generic_gaussian_transition_law.py \
  experiments/backfill/block_ar/train_614a_unified_ar_gaussian_likelihood.py \
  experiments/backfill/block_ar/evaluate_614a_unified_ar_gaussian_likelihood.py

pytest test_code/test_614a_generic_gaussian_transition.py -q
```

Result: `4 passed`.

## Run

Training:

- state scope: `joint38`;
- epochs: `8`;
- recent train windows: `2048`;
- distribution family: `student_t`;
- degrees of freedom: `5`;
- mean loss weight: `1.0`;
- seed: `618`.

Training result:

- best epoch: `4`;
- best validation objective: `-8.938077`;
- final validation objective: `-4.028872`;
- finite sample rate: `1.0`.

## Full Suite Result

Official IV bridge on 441 windows / 48 samples:

- score: `4/11`;
- passed: conditionality, block-AR, cointegration, cross-cell correlation;
- failed: surface, coverage, time-series properties, regime coverage, distributional fidelity, mean reversion, pathwise jump realism.

Key metrics:

- cov90 overall: `92.0%`;
- conditional MAE reduction: `6.0%`;
- turbulent/calm width ratio: `0.889`;
- persistent severe undercoverage: `404/11025 = 3.7%`;
- daily-change KS cells: `14/25`;
- level KS cells: `0/25`;
- median-bias cells: `15/25`;
- kurtosis ratio: `0.460`;
- cointegration worst-cell ratio: `0.269`;
- cross-cell corr/rank: `0.695 / 1.945`;
- mean-reversion ratio: `0.404`;
- pathwise max-jump KS: `0.397`;
- per-cell q99 jump-scale cells: `6/25`.

## Comparison

| run | score | surface | cov90 | cond MAE | cat. undercov | daily KS | level KS | bias cells | kurtosis | coin worst | MR ratio | path KS | q99 cells |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 616a Student-t 1.00 | 4/11 | fail | 92.2% | 7.9% | 3.6% | 14/25 | 0/25 | 21/25 | 0.452 | 0.545 | 0.427 | 0.403 | 6/25 |
| 617a Student-t 0.90 | 5/11 | pass | 91.6% | 9.2% | 3.7% | 17/25 | 0/25 | 21/25 | 0.474 | 0.343 | 0.370 | 0.479 | 6/25 |
| 618a mean loss 1.0 | 4/11 | fail | 92.0% | 6.0% | 3.7% | 14/25 | 0/25 | 15/25 | 0.460 | 0.269 | 0.404 | 0.397 | 6/25 |

## Mechanism Read

The one-step mean-placement auxiliary loss does not solve the placement pathology. It weakens several important behaviors while leaving the core failures intact:

- surface validity regresses;
- daily-change KS regresses;
- median-bias cells regress from `21/25` to `15/25`;
- cointegration worst-cell weakens materially;
- cross-cell rank becomes less realistic;
- mean reversion remains far below target;
- level KS remains `0/25`;
- per-cell q99 tail balance remains `6/25`.

The likely reason is that a generic one-step mean penalty competes with the heavy-tailed likelihood without specifically addressing the multi-step conditional level occupancy problem. It improves neither the AR placement geometry nor the per-cell scale imbalance.

## Decision

Close mean-loss weight `1.0` as a deployable fix.

The likelihood path has produced a useful best current local result at 617a temp `0.90`, but further local knobs are becoming less principled. The next step should be post-experiment analysis / ideation focused on why level occupancy and mean reversion remain broken under the likelihood family before adding another training term.

## Artifacts

- `models/backfill/618a_joint38_ar_studentt_mean_nll_e8_w2048_s618/train_summary.json`
- `models/backfill/618a_joint38_ar_studentt_mean_nll_e8_w2048_s618/training_history.json`
- `results/autoresearch/618a_joint38_ar_studentt_mean_nll_e8_w2048/full11.json`
- `results/autoresearch/618a_joint38_ar_studentt_mean_nll_e8_w2048/full11.md`
