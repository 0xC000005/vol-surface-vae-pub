# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.110`
- factor delta KS pass <0.20: `11/13`
- factor q99 abs-delta ratio median: `1.320`
- factor q99 abs-delta pass [0.5,2.0]: `12/13`
- factor-factor corr upper-triangle corr: `0.835`
- factor-factor corr MAE: `0.131`
- IV-factor corr matrix corr: `nan`
- IV-factor corr MAE: `nan`
- conditional median MAE reduction vs rolled deck: `4.13%`
- history-activity vs generated-width Spearman: `0.935`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.215 | 1.479 | [1829, 2648] | [1101, 2637] |
| factor:usdcad | 0.077 | 1.320 | [1.213, 1.408] | [1.028, 1.526] |
| factor:usdjpy | 0.100 | 1.350 | [100.1, 121.1] | [82.9, 146.6] |
| factor:dxy | 0.058 | 1.286 | [91.33, 103.3] | [82.85, 110.8] |
| factor:copper | 0.080 | 1.194 | [2.004, 3.224] | [1.544, 3.809] |
| factor:wheat | 0.112 | 1.239 | [361, 539.2] | [286.8, 841.5] |
| factor:crude_oil | 0.075 | 1.407 | [26.19, 58.94] | [6.103, 82.92] |
| factor:us2y | 0.125 | 1.179 | [0.56, 1.83] | [-0.1368, 2.162] |
| factor:us10y | 0.078 | 1.310 | [1.37, 2.62] | [0.8535, 3.673] |
| factor:aaa_oas | 0.202 | 4.959 | [0.53, 0.99] | [-0.1907, 4.252] |
| factor:bbb_oas | 0.186 | 1.918 | [1.29, 3.03] | [1.149, 6.562] |
| factor:nikkei | 0.078 | 1.470 | [1.495e+04, 2.294e+04] | [7058, 2.683e+04] |
| factor:gold | 0.044 | 1.189 | [1116, 1365] | [903.9, 1708] |
