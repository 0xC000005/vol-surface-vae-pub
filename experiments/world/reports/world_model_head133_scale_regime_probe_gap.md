# World Model HEAD133: Scale Regime Probe Gap

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_regime_diagnostic` for frozen Part 1 features.

## Hypothesis

If the regime failure is purely no-signal, scaled Barlow should be worse
than raw features on both accuracy and balanced class recall. If it is
partly an imbalanced-label probe issue, scaled Barlow may lose majority
accuracy while improving macro recall or minority-class recall.

## Class Balance

| class | validation count | share |
| --- | ---: | ---: |
| 0 | 92 | 0.359375 |
| 3 | 153 | 0.597656 |
| 4 | 11 | 0.042969 |

## Scaled Regime Metrics

| feature | accuracy | majority | lift | macro recall | class 0 recall | class 3 recall | class 4 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| barlow_clean_last | 0.515625 | 0.597656 | -0.082031 | 0.521244 | 0.065217 | 0.771242 | 0.727273 |
| barlow_clean_mean | 0.011719 | 0.597656 | -0.585938 | 0.009425 | 0.021739 | 0.006536 | 0.000000 |
| raw_surface_last | 0.554688 | 0.597656 | -0.042969 | 0.352704 | 0.326087 | 0.732026 | 0.000000 |
| raw_surface_flat | 0.542969 | 0.597656 | -0.054688 | 0.330957 | 0.000000 | 0.901961 | 0.090909 |
| raw_surface_last_plus_barlow_clean_last | 0.453125 | 0.597656 | -0.144531 | 0.346618 | 0.706522 | 0.333333 | 0.000000 |

## Scaled Barlow Versus Raw Last Surface

- Accuracy delta: `-0.039062`.
- Macro-recall delta: `0.168540`.

| class | recall delta |
| --- | ---: |
| 0 | -0.260870 |
| 3 | 0.039216 |
| 4 | 0.727273 |

## Decision

- Accuracy gate passed: `False`.
- Balanced signal present: `True`.
- Promotion decision: `DO_NOT_PROMOTE`.

The scaled embedding does not pass the accuracy gate because the regime labels are majority-class dominated. It does contain some minority-regime signal, shown by higher macro recall and class-4 recall than raw surface features. Treat regime accuracy as a failed promotion layer, but use balanced accuracy/class recall for diagnosis before changing the representation objective.
