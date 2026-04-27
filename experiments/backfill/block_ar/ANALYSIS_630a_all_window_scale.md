# 630a All-Window Scale Test for 629a

## Hypothesis

629a established the clean state-conditioned encoded-increment architecture, but
still failed IV coverage, regime scaling, level occupancy, and jump-tail
calibration. 630a tested the Bitter-Lesson/data-scale explanation by training
the same architecture on all available training windows instead of the recent
2048-window subset.

## Result

Teacher-forced validation loss improved, but free-running IV quality worsened.

| metric | 629a recent 2048 | 630a all windows |
| --- | ---: | ---: |
| train windows | 2048 | 4010 |
| best val loss | 1.1323 | 1.0587 |
| IV suite score | 3/11 | 2/11 |
| surface explosion rate | 45.1% | 72.5% |
| overall 90% coverage | 76.7% | 81.8% |
| conditionality MAE reduction | 7.9% | 4.4% |
| daily-change KS pass | 25/25 | 22/25 |
| level KS pass | 5/25 | 1/25 |
| median-bias pass | 13/25 | 5/25 |
| cross-cell corr/rank ratio | 0.967 / 1.167 | 0.925 / 1.223 |
| h1 mean-reversion ratio | 0.976 | 0.708 |
| pathwise max-jump KS | 0.464 | 0.711 |
| per-cell q99 tail pass | 19/25 | 7/25 |

The joint-panel audit improved in dependence but not enough to offset IV damage:

| metric | 629a recent 2048 | 630a all windows |
| --- | ---: | ---: |
| factor KS mean | 0.095 | 0.097 |
| factor KS pass <0.20 | 12/13 | 12/13 |
| q99 ratio median | 1.164 | 1.087 |
| q99 pass [0.5,2.0] | 13/13 | 13/13 |
| factor-factor corr | 0.823 | 0.907 |
| factor-factor gen/gt mean abs | 0.135 / 0.225 | 0.150 / 0.225 |
| IV-factor corr | 0.883 | 0.891 |
| IV-factor gen/gt mean abs | 0.123 / 0.149 | 0.127 / 0.149 |

Artifacts:

- `models/backfill/630a_joint38_statecond_increment_e8_all_s630/best_model.pt`
- `results/autoresearch/630a_joint38_statecond_increment_e8_all/full11.md`
- `results/autoresearch/630a_joint38_statecond_increment_e8_all/joint_panel.md`

## Mechanism Read

630a falsifies the idea that the remaining failures are solved by naive data
scale alone. More historical windows improve teacher-forced loss and joint
factor dependence, but they worsen the current validation regime's IV free-run
behavior. This is consistent with nonstationary financial levels and regime
mixing: older data helps average factor co-movement but distorts the IV level
coordinate and jump scale needed for the validation period.

The important lesson is that validation loss on one-step increments is not a
sufficient proxy for deployable 30-day scenario quality. The model needs a better
generic representation of state-dependent scale, not just more samples.

## Decision

Keep the 629a architecture as the active family and keep the recent-window
training frame as the better IV base. Do not keep all-window training as the
frontier model.

The next most principled move is a small generic volatility-state feature in the
same state-conditioned increment law: let the causal memory see encoded levels,
increments, absolute increments, and squared increments. This is not IV-specific
and directly targets the remaining failure mode: weak turbulent widening and
state-dependent scale calibration.
