# 719a Sticky-Zero Readout Joint-Panel Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.091`
- factor delta KS pass <0.20: `12/13`
- factor q99 abs-delta ratio median: `0.982`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.781`
- factor-factor corr MAE: `0.165`
- IV-factor corr matrix corr: `0.894`
- IV-factor corr MAE: `0.091`
- conditional median MAE reduction vs rolled deck: `4.26%`
- history-activity vs generated-width Spearman: `0.948`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.105 | 0.982 | [1829, 2648] | [1562, 2639] |
| factor:usdcad | 0.052 | 1.193 | [1.213, 1.408] | [1.06, 1.591] |
| factor:usdjpy | 0.046 | 0.973 | [100.1, 121.1] | [76.32, 138.1] |
| factor:dxy | 0.042 | 1.108 | [91.33, 103.3] | [83.7, 111] |
| factor:copper | 0.059 | 0.917 | [2.004, 3.224] | [1.647, 3.843] |
| factor:wheat | 0.056 | 1.076 | [361, 539.2] | [273.4, 881.6] |
| factor:crude_oil | 0.084 | 1.045 | [26.19, 58.94] | [1.177, 73.64] |
| factor:us2y | 0.142 | 0.877 | [0.56, 1.83] | [-0.06783, 2.062] |
| factor:us10y | 0.098 | 0.969 | [1.37, 2.62] | [0.871, 3.798] |
| factor:aaa_oas | 0.158 | 0.831 | [0.53, 0.99] | [0.2612, 1.541] |
| factor:bbb_oas | 0.221 | 1.030 | [1.29, 3.03] | [1.173, 4.661] |
| factor:nikkei | 0.086 | 1.049 | [1.495e+04, 2.294e+04] | [9318, 2.251e+04] |
| factor:gold | 0.037 | 0.908 | [1116, 1365] | [887.6, 1619] |

## Sticky-Zero Readout

- selected names: `['factor:aaa_oas', 'factor:bbb_oas']`
- zero-rate gate: `0.25`
- nonzero quantile: `0.1`

| channel | selected | zero rate | threshold | nonzero q10 | nonzero q50 | nonzero q90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| factor:us2y | `False` | `0.230` | `0` | `0.01` | `0.03` | `0.08` |
| factor:aaa_oas | `True` | `0.435` | `0.01` | `0.01` | `0.01` | `0.04` |
| factor:bbb_oas | `True` | `0.266` | `0.01` | `0.01` | `0.01` | `0.05` |
