# 715a Frozen-Framework Failure Attribution

## Read
- overall scorecard pass: `False`
- gate passes: `{'iv': False, 'anchor': False, 'joint': False, 'framework': True}`
- failure class: `localized coordinate/support and calibration failure, not framework inconsistency`

## IV Failure
- IV score: `7/11`
- effective failed suites: `['coverage', 'regime_coverage', 'distributional_fidelity']`
- cov90 overall / h30 worst cell: `0.800` / `0.490`
- level-KS pass cells: `12/25`
- regime layer2 pass cells: `2/8`
- regime layer3 catastrophic rate: `0.059`
- risk-state allocation pass: `True`

Worst IV coverage cells:

| horizon | cell | cov90 | gap to 0.90 |
| ---: | ---: | ---: | ---: |
| 30 | [3, 3] | 0.490 | 0.410 |
| 30 | [2, 3] | 0.506 | 0.394 |
| 30 | [0, 2] | 0.562 | 0.338 |
| 14 | [2, 3] | 0.587 | 0.313 |
| 14 | [3, 3] | 0.610 | 0.290 |

Worst IV regime cells:

| regime | horizon | cell | cov90 | gap to 0.90 |
| --- | ---: | ---: | ---: | ---: |
| turb | 30 | [2, 3] | 0.225 | 0.675 |
| turb | 30 | [3, 3] | 0.270 | 0.630 |
| turb | 30 | [0, 2] | 0.303 | 0.597 |
| turb | 30 | [2, 2] | 0.348 | 0.552 |
| turb | 30 | [3, 2] | 0.371 | 0.529 |

## Anchor And Joint Factor Failure
- anchor factor delta KS pass: `11/13`
- joint factor delta KS pass: `11/13`
- common failed factors: `['factor:aaa_oas', 'factor:bbb_oas']`
- joint IV-factor matrix corr: `0.909`
- anchor conditional-panel reduction: `4.52%`
- joint conditional-panel reduction: `4.06%`

| scope | factor | KS(delta) | q99 ratio | range ratio | gen max / gt max |
| --- | --- | ---: | ---: | ---: | ---: |
| anchor | factor:bbb_oas | 0.349 | 1.008 | 2.130 | 1.625 |
| anchor | factor:aaa_oas | 0.296 | 0.871 | 3.502 | 2.049 |
| anchor | factor:us2y | 0.151 | 0.900 | 1.557 | 1.155 |
| joint | factor:bbb_oas | 0.338 | 1.083 | 1.777 | 1.427 |
| joint | factor:aaa_oas | 0.278 | 0.859 | 5.017 | 2.628 |
| joint | factor:spx | 0.177 | 1.102 | 1.534 | 0.978 |

## Delta Versus 674a IV Control
- cov90 delta: `-0.037`
- h30 worst-cell delta: `-0.034`
- level-KS pass-cell delta: `-3`
- max-jump KS delta: `-0.100`

## Decision
Prefer a data-coordinate/support repair for positive spread-like factors and IV tail/regime calibration before adding another global loss or switching backend.
