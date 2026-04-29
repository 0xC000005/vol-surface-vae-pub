# 741a IV Coverage Localization

| run | score | cov90 | cal err | cov under/over | regime under/over | regime combos | path KS | MR full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| temp1000_baseline | 6/11 | 0.836 | 0.046 | 5/0 | 24/12 | 0/8 | 0.442 | True |
| temp1025 | 5/11 | 0.853 | 0.030 | 5/2 | 17/26 | 2/8 | 0.514 | False |
| temp1050 | 5/11 | 0.868 | 0.015 | 4/6 | 15/35 | 1/8 | 0.579 | False |

## Stable Failures

- stable standard undercoverage cells across temperatures: `4`
- stable standard overcoverage cells across temperatures: `0`
- stable regime undercoverage cells across temperatures: `12`
- stable regime overcoverage cells across temperatures: `9`

## Baseline Cell Concentration

- standard undercoverage cell counts: `{'2,3': 2, '0,2': 1, '1,3': 1, '3,3': 1}`
- standard overcoverage cell counts: `{}`
- regime undercoverage cell counts: `{'2,3': 5, '3,3': 4, '0,2': 2, '1,2': 2, '2,2': 2, '3,2': 2, '1,0': 1, '1,3': 1, '1,4': 1, '3,4': 1, '4,0': 1, '4,3': 1, '4,4': 1}`
- regime overcoverage cell counts: `{'2,0': 3, '0,3': 2, '1,4': 2, '3,0': 2, '0,0': 1, '4,0': 1, '4,4': 1}`

## Worst Baseline Standard Undercoverage

| horizon | cell | cov90 | median above frac | mean bias IV pts |
| ---: | ---: | ---: | ---: | ---: |
| 30 | [2, 3] | 0.524 | 0.857 | 0.01 |
| 30 | [3, 3] | 0.578 | 0.811 | 0.01 |
| 30 | [0, 2] | 0.603 | 0.779 | 0.01 |
| 14 | [2, 3] | 0.673 | 0.857 | 0.01 |
| 30 | [1, 3] | 0.696 | 0.751 | 0.00 |
| 14 | [3, 3] | 0.717 | 0.811 | 0.01 |
| 30 | [1, 2] | 0.717 | 0.747 | 0.01 |
| 14 | [0, 2] | 0.744 | 0.779 | 0.01 |

## Mechanism Read

The remaining IV coverage defect is local and two-sided, not a global variance shortage. Temperature reduces aggregate calibration error but does not remove stable late-horizon undercoverage in specific cells, and it creates or preserves overcoverage in other cells. Regime layer-2 is the hardest gate because calm/turb splits expose the same local geometry with small per-regime sample sizes.

## Decision

Do not add another scalar sampler knob. The next model-side move, if any, must be a generic learned local uncertainty allocation mechanism tied to state/cell/horizon representation or a training objective that teaches local interval geometry without post-hoc calibration.
