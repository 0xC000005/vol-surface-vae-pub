# 660a Train vs Validation Generation Gap

## Executive Read

The current 658a failure is not pure out-of-distribution failure. Train-tail generation improves materially versus validation, so distribution shift is real, but the in-training audit still scores only `5/11` and still fails level occupancy, median bias, per-cell conditionality, regime coverage, and some anchor support. The correct diagnosis is a mixed failure: OOD shift amplifies the problem, but the model has not fully learned the training conditional level/path law either.

## IV Full-Suite Comparison

| metric | train-tail | validation | read |
| --- | ---: | ---: | --- |
| score | 5/11 | 5/11 | same headline score, different failure severity |
| cov90 | 0.829 | 0.687 | validation undercoverage is much worse |
| h30 cov90 | 0.752 | 0.663 | train-tail clears horizon coverage, validation is weak |
| conditional MAE reduction | 5.791% | -0.452% | conditional signal works in-sample but not out-of-sample |
| turb/calm width ratio | 1.162 | 0.956 | train-tail learns risk-width ordering; validation reverses it |
| daily KS pass | 25/25 | 24/25 | local daily law works on both |
| level KS pass | 11/25 | 7/25 | level law remains bad even in-sample |
| level KS median | 0.184 | 0.226 | OOD worsens an existing level-placement failure |
| median-bias pass | 14/25 | 12/25 | median placement is not solved on train |
| bad window-floor rate | 0.039 | 0.186 | validation has many more severe undercoverage windows |
| kurtosis ratio | 1.164 | 0.742 | tail shape is better in-sample |
| corr ratio | 0.669 | 0.750 | both acceptable, not the main bottleneck |
| mean-reversion pass | yes | no | validation failure is mostly generalization/profile drift |
| path max-jump KS | 0.154 | 0.480 | path extremes generalize poorly but remain inside relaxed gate |
| regime layer2 | 1/8 | 0/8 | regime-cell coverage is not solved in-sample |

## Joint Anchor Comparison

| metric | train-tail | validation | read |
| --- | ---: | ---: | --- |
| factor delta KS mean | 0.092 | 0.109 | train is better, but not perfect |
| factor delta KS pass | 11/13 | 11/13 | two factors fail in both splits |
| q99 abs-delta pass | 13/13 | 13/13 | tail scale is broadly learned |
| factor corr shape | 0.951 | 0.892 | correlation shape generalizes reasonably |
| factor abs-corr ratio | 0.711 | 0.610 | shock amplitude is attenuated in both |
| IV-factor corr shape | 0.932 | 0.868 | shape is better on train, still alive on validation |
| IV-factor abs-corr ratio | 0.606 | 0.592 | joint shock amplitude is intrinsically too small |
| worst factor KS | factor:aaa_oas=0.306 | factor:aaa_oas=0.246 | credit-spread factors remain hard |

## Diagnosis

1. Not pure OOD: if the model were fundamentally fine and only validation were shifted, train-tail should be close to deployable. It is not; it still fails six suites.
2. Not pure in-sample failure either: validation is materially worse on coverage, conditionality, mean reversion, path jumps, and joint correlation amplitude. Distribution shift is a real amplifier.
3. The robust learned part is local movement: daily IV changes, factor deltas, q99 movement scale, and broad correlation shape are alive on both splits.
4. The unresolved learned part is conditional placement: level occupancy, median placement, per-cell coverage geometry, regime-cell coverage, and absolute joint shock amplitude are not reliable even on training windows.

## Implication

Existing validation results should not be read as simply 'the models cannot learn the data.' They show a split-generalization problem layered on top of an incomplete in-sample conditional-law fit. The next experiment should therefore test objective balance or level-placement loss on the train-tail audit first. If train-tail becomes strong but validation stays weak, the bottleneck moves to OOD calibration. If train-tail remains weak, more validation tuning is wasted because the model has not learned the training law.
