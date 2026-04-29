# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.109`
- factor delta KS pass <0.20: `12/14`
- factor q99 abs-delta ratio median: `0.943`
- factor q99 abs-delta pass [0.5,2.0]: `14/14`
- factor-factor corr upper-triangle corr: `0.903`
- factor-factor corr MAE: `0.113`
- IV-factor corr matrix corr: `nan`
- IV-factor corr MAE: `nan`
- conditional median MAE reduction vs rolled deck: `4.89%`
- history-activity vs generated-width Spearman: `0.928`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.142 | 0.979 | [1829, 2648] | [1477, 2652] |
| factor:usdcad | 0.044 | 1.038 | [1.213, 1.408] | [1.116, 1.634] |
| factor:usdjpy | 0.025 | 0.943 | [100.1, 121.1] | [82.65, 137] |
| factor:dxy | 0.046 | 1.011 | [91.33, 103.3] | [85.18, 112.2] |
| factor:copper | 0.070 | 0.928 | [2.004, 3.224] | [1.51, 3.828] |
| factor:wheat | 0.055 | 0.943 | [361, 539.2] | [271.9, 800.1] |
| factor:crude_oil | 0.091 | 1.039 | [26.19, 58.94] | [6.534, 72.13] |
| factor:us2y | 0.193 | 0.806 | [0.56, 1.83] | [-0.04432, 1.943] |
| factor:us10y | 0.110 | 0.973 | [1.37, 2.62] | [0.9703, 3.279] |
| factor:aaa_oas | 0.250 | 0.769 | [0.53, 0.99] | [0.4152, 1.996] |
| factor:bbb_oas | 0.333 | 0.940 | [1.29, 3.03] | [1.195, 4.382] |
| factor:nikkei | 0.067 | 1.062 | [1.495e+04, 2.294e+04] | [7719, 2.281e+04] |
| factor:gold | 0.052 | 0.895 | [1116, 1365] | [902.8, 1624] |
| factor:vix | 0.048 | 0.811 | [9.14, 28.14] | [3.707, 168.6] |
