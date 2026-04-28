# 719a Sticky-Zero Readout Joint-Panel Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.093`
- factor delta KS pass <0.20: `12/13`
- factor q99 abs-delta ratio median: `1.030`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.891`
- factor-factor corr MAE: `0.116`
- IV-factor corr matrix corr: `nan`
- IV-factor corr MAE: `nan`
- conditional median MAE reduction vs rolled deck: `4.50%`
- history-activity vs generated-width Spearman: `0.954`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.121 | 1.133 | [1829, 2648] | [1383, 2696] |
| factor:usdcad | 0.069 | 1.110 | [1.213, 1.408] | [1.113, 1.652] |
| factor:usdjpy | 0.060 | 1.008 | [100.1, 121.1] | [81.64, 153] |
| factor:dxy | 0.061 | 1.117 | [91.33, 103.3] | [85.36, 114.6] |
| factor:copper | 0.056 | 1.026 | [2.004, 3.224] | [1.476, 3.712] |
| factor:wheat | 0.040 | 1.022 | [361, 539.2] | [254.8, 879.9] |
| factor:crude_oil | 0.056 | 1.101 | [26.19, 58.94] | [4.54, 76.03] |
| factor:us2y | 0.149 | 0.905 | [0.56, 1.83] | [0.07338, 2.126] |
| factor:us10y | 0.072 | 1.081 | [1.37, 2.62] | [0.7047, 3.593] |
| factor:aaa_oas | 0.180 | 0.888 | [0.53, 0.99] | [0.4011, 2.76] |
| factor:bbb_oas | 0.225 | 1.030 | [1.29, 3.03] | [1.175, 5.384] |
| factor:nikkei | 0.056 | 1.174 | [1.495e+04, 2.294e+04] | [7588, 2.643e+04] |
| factor:gold | 0.068 | 0.993 | [1116, 1365] | [834.9, 1595] |

## Sticky-Zero Readout

- selected names: `['factor:aaa_oas', 'factor:bbb_oas']`
- zero-rate gate: `0.25`
- nonzero quantile: `0.1`

| channel | selected | zero rate | threshold | nonzero q10 | nonzero q50 | nonzero q90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| factor:us2y | `False` | `0.230` | `0` | `0.01` | `0.03` | `0.08` |
| factor:aaa_oas | `True` | `0.435` | `0.01` | `0.01` | `0.01` | `0.04` |
| factor:bbb_oas | `True` | `0.266` | `0.01` | `0.01` | `0.01` | `0.05` |
