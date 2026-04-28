# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.123`
- factor delta KS pass <0.20: `11/13`
- factor q99 abs-delta ratio median: `1.051`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.836`
- factor-factor corr MAE: `0.133`
- IV-factor corr matrix corr: `0.908`
- IV-factor corr MAE: `0.075`
- conditional median MAE reduction vs rolled deck: `3.84%`
- history-activity vs generated-width Spearman: `0.829`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.188 | 1.051 | [1829, 2648] | [1582, 2587] |
| factor:usdcad | 0.060 | 1.138 | [1.213, 1.408] | [1.058, 1.529] |
| factor:usdjpy | 0.055 | 0.909 | [100.1, 121.1] | [88.22, 137.7] |
| factor:dxy | 0.074 | 1.063 | [91.33, 103.3] | [85.82, 110.6] |
| factor:copper | 0.065 | 1.001 | [2.004, 3.224] | [1.489, 3.695] |
| factor:wheat | 0.073 | 1.201 | [361, 539.2] | [328, 885.6] |
| factor:crude_oil | 0.114 | 1.089 | [26.19, 58.94] | [13.61, 70.03] |
| factor:us2y | 0.132 | 0.902 | [0.56, 1.83] | [0.2274, 2.365] |
| factor:us10y | 0.089 | 1.080 | [1.37, 2.62] | [1.166, 3.715] |
| factor:aaa_oas | 0.285 | 0.797 | [0.53, 0.99] | [0.4497, 1.378] |
| factor:bbb_oas | 0.327 | 0.948 | [1.29, 3.03] | [1.244, 3.582] |
| factor:nikkei | 0.061 | 1.076 | [1.495e+04, 2.294e+04] | [7131, 2.279e+04] |
| factor:gold | 0.080 | 0.926 | [1116, 1365] | [931.3, 1557] |
