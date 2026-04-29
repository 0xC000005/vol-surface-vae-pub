# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.097`
- factor delta KS pass <0.20: `12/14`
- factor q99 abs-delta ratio median: `1.012`
- factor q99 abs-delta pass [0.5,2.0]: `14/14`
- factor-factor corr upper-triangle corr: `0.863`
- factor-factor corr MAE: `0.143`
- IV-factor corr matrix corr: `0.947`
- IV-factor corr MAE: `0.085`
- conditional median MAE reduction vs rolled deck: `5.01%`
- history-activity vs generated-width Spearman: `0.950`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.124 | 0.979 | [1829, 2648] | [1524, 2672] |
| factor:usdcad | 0.023 | 1.213 | [1.213, 1.408] | [0.956, 1.597] |
| factor:usdjpy | 0.044 | 1.013 | [100.1, 121.1] | [83.92, 138] |
| factor:dxy | 0.035 | 1.136 | [91.33, 103.3] | [83.8, 112.4] |
| factor:copper | 0.073 | 0.966 | [2.004, 3.224] | [1.606, 3.756] |
| factor:wheat | 0.052 | 1.011 | [361, 539.2] | [264.1, 832] |
| factor:crude_oil | 0.092 | 1.045 | [26.19, 58.94] | [5.768, 72.39] |
| factor:us2y | 0.163 | 0.949 | [0.56, 1.83] | [-0.3771, 2.108] |
| factor:us10y | 0.094 | 1.050 | [1.37, 2.62] | [1.017, 3.281] |
| factor:aaa_oas | 0.253 | 0.843 | [0.53, 0.99] | [0.2014, 1.342] |
| factor:bbb_oas | 0.262 | 1.084 | [1.29, 3.03] | [1.133, 3.778] |
| factor:nikkei | 0.055 | 1.126 | [1.495e+04, 2.294e+04] | [9884, 2.444e+04] |
| factor:gold | 0.067 | 0.979 | [1116, 1365] | [863.1, 1625] |
| factor:vix | 0.023 | 0.719 | [9.14, 28.14] | [3.716, 465.6] |
