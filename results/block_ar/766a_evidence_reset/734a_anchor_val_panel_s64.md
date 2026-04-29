# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.107`
- factor delta KS pass <0.20: `12/14`
- factor q99 abs-delta ratio median: `1.012`
- factor q99 abs-delta pass [0.5,2.0]: `14/14`
- factor-factor corr upper-triangle corr: `0.907`
- factor-factor corr MAE: `0.112`
- IV-factor corr matrix corr: `nan`
- IV-factor corr MAE: `nan`
- conditional median MAE reduction vs rolled deck: `5.11%`
- history-activity vs generated-width Spearman: `0.926`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.129 | 1.066 | [1829, 2648] | [1366, 2639] |
| factor:usdcad | 0.050 | 1.097 | [1.213, 1.408] | [1.113, 1.639] |
| factor:usdjpy | 0.046 | 1.002 | [100.1, 121.1] | [82.73, 142.9] |
| factor:dxy | 0.029 | 1.064 | [91.33, 103.3] | [84.58, 115.2] |
| factor:copper | 0.055 | 0.948 | [2.004, 3.224] | [1.416, 3.704] |
| factor:wheat | 0.053 | 0.999 | [361, 539.2] | [286.6, 843.3] |
| factor:crude_oil | 0.079 | 1.063 | [26.19, 58.94] | [4.902, 72.65] |
| factor:us2y | 0.189 | 0.850 | [0.56, 1.83] | [-0.02956, 1.969] |
| factor:us10y | 0.099 | 1.022 | [1.37, 2.62] | [0.8585, 3.44] |
| factor:aaa_oas | 0.261 | 0.831 | [0.53, 0.99] | [0.4053, 1.88] |
| factor:bbb_oas | 0.355 | 1.067 | [1.29, 3.03] | [1.211, 5.096] |
| factor:nikkei | 0.055 | 1.176 | [1.495e+04, 2.294e+04] | [6659, 2.547e+04] |
| factor:gold | 0.046 | 0.925 | [1116, 1365] | [823.8, 1681] |
| factor:vix | 0.050 | 0.926 | [9.14, 28.14] | [3.672, 326.1] |
