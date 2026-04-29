# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.142`
- factor delta KS pass <0.20: `11/13`
- factor q99 abs-delta ratio median: `1.396`
- factor q99 abs-delta pass [0.5,2.0]: `11/13`
- factor-factor corr upper-triangle corr: `0.840`
- factor-factor corr MAE: `0.141`
- IV-factor corr matrix corr: `0.904`
- IV-factor corr MAE: `0.077`
- conditional median MAE reduction vs rolled deck: `3.43%`
- history-activity vs generated-width Spearman: `0.931`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.234 | 1.440 | [1829, 2648] | [1111, 2596] |
| factor:usdcad | 0.155 | 1.460 | [1.213, 1.408] | [1.106, 1.709] |
| factor:usdjpy | 0.109 | 1.342 | [100.1, 121.1] | [83.49, 140.4] |
| factor:dxy | 0.104 | 1.396 | [91.33, 103.3] | [85.67, 118.6] |
| factor:copper | 0.076 | 1.187 | [2.004, 3.224] | [1.506, 3.976] |
| factor:wheat | 0.110 | 1.350 | [361, 539.2] | [289.9, 877.1] |
| factor:crude_oil | 0.095 | 1.420 | [26.19, 58.94] | [5.193, 76.17] |
| factor:us2y | 0.174 | 1.086 | [0.56, 1.83] | [0.01649, 2.101] |
| factor:us10y | 0.071 | 1.261 | [1.37, 2.62] | [0.7653, 3.646] |
| factor:aaa_oas | 0.180 | 6.642 | [0.53, 0.99] | [-0.4091, 6.104] |
| factor:bbb_oas | 0.320 | 3.570 | [1.29, 3.03] | [1.235, 10.36] |
| factor:nikkei | 0.126 | 1.649 | [1.495e+04, 2.294e+04] | [6116, 2.39e+04] |
| factor:gold | 0.088 | 1.237 | [1116, 1365] | [845.5, 1681] |
