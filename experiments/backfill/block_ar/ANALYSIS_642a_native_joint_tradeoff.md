# 642a Native Joint Trade-Off Analysis

## Question

After 641a, decide whether the native joint mixed-coordinate line is still the right path or whether the 610/612 AR line should be restored as the active model.

## Evidence

IV broad-frame comparison:

| model | IV score | cov90 | cond MAE red. | turb/calm width | daily KS | level KS | median pass | q99 tail | path KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 510a same-frame | 5/11 | 0.696 | 7.61% | 1.093 | 25/25 | 3/25 | 10/25 | 24/25 | 0.463 |
| 596a final-path objective | 5/11 | 0.657 | 5.58% | 1.078 | 25/25 | 2/25 | 6/25 | 24/25 | 0.456 |
| 610a native joint AR | 5/11 | 0.757 | 4.98% | 1.009 | 24/25 | 2/25 | 11/25 | 21/25 | 0.446 |
| 612a conditional source scale | 6/11 | 0.774 | 11.6% | 0.910 | 24/25 | 13/25 | 18/25 | 22/25 | 0.638 |
| 631a state-conditioned increment | 4/11 | 0.781 | 8.88% | 1.008 | 24/25 | 8/25 | 16/25 | 19/25 | 0.531 |
| 638a state-conditioned level-score | 4/11 | 0.697 | 4.31% | 0.970 | 23/25 | 6/25 | 9/25 | 22/25 | 0.555 |
| 641a mixed coordinate | 4/11 | 0.647 | 4.51% | 0.997 | 24/25 | 4/25 | 8/25 | 23/25 | 0.610 |

Native joint anchor-factor comparison:

| model | factor KS mean | factor q99 pass | factor-factor corr | gen factor abs corr | IV-factor corr | gen IV-factor abs corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 610a | 0.133 | 2/13 | 0.565 | 0.054 | 0.664 | 0.075 |
| 612a | 0.135 | 5/13 | 0.559 | 0.037 | 0.698 | 0.053 |
| 631a | 0.115 | 13/13 | 0.865 | 0.133 | 0.889 | 0.119 |
| 638a | 0.156 | 6/13 | 0.514 | 0.052 | 0.663 | 0.073 |
| 641a | 0.098 | 13/13 | 0.866 | 0.137 | 0.852 | 0.102 |

## Mechanism Read

There are two different frontiers:

- The IV-score frontier is still 612a on the broad frame at `6/11`, but its learned source-scale mechanism collapsed to the lower clamp and its anchor-factor joint audit is poor.
- The native-joint scenario frontier is now 641a: it has strong anchor-factor marginal tails and realistic factor/factor and IV/factor co-movement in one shared scenario path.

This means the correct active object depends on the target:

- If the target is IV-only suite count, 641a is not the best model.
- If the target is a scientifically defensible IV-plus-anchor-factor scenario generator, 641a is the best clean baseline because 610/612 do not preserve anchor-factor realism.

641a's remaining IV issue is not tail realism in daily changes. Daily-change KS is `24/25`, q99 tail pass is `23/25`, and surfaces are support-valid. The issue is conditional level placement and interval width over the 30-day path: coverage is low, level KS is weak, median placement is biased, and pathwise max-jump KS is high.

## Decision

Do not abandon 641a. The joint requirement makes 610/612 insufficient despite their higher IV suite scores.

Do not add another coordinate branch. The mixed-coordinate rule already captures the clean data-coordinate distinction.

The next lowest-risk falsifier is an evaluation-only sample-temperature sweep on 641a. This is not a new architecture and not a claim that temperature is the learned law. It tests whether 641a's current IV failure is mostly underdispersion in a support-valid coordinate. If moderate temperature improves coverage/pathwise KS while preserving daily-change KS, cross-cell structure, and native joint factor audit, it can be framed as risk-policy calibration. If it fails, the next move must be objective-level conditional density/calibration, not another sampling knob.
