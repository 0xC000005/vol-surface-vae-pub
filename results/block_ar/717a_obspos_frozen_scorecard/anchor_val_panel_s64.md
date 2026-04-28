# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.130`
- factor delta KS pass <0.20: `11/13`
- factor q99 abs-delta ratio median: `1.025`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.903`
- factor-factor corr MAE: `0.103`
- IV-factor corr matrix corr: `nan`
- IV-factor corr MAE: `nan`
- conditional median MAE reduction vs rolled deck: `3.93%`
- history-activity vs generated-width Spearman: `0.907`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.198 | 1.111 | [1829, 2648] | [1479, 2620] |
| factor:usdcad | 0.037 | 1.087 | [1.213, 1.408] | [1.104, 1.571] |
| factor:usdjpy | 0.075 | 1.023 | [100.1, 121.1] | [73.91, 139.8] |
| factor:dxy | 0.031 | 1.093 | [91.33, 103.3] | [80.29, 111.6] |
| factor:copper | 0.095 | 0.967 | [2.004, 3.224] | [1.642, 3.549] |
| factor:wheat | 0.059 | 1.097 | [361, 539.2] | [297.1, 834.1] |
| factor:crude_oil | 0.113 | 1.025 | [26.19, 58.94] | [10.73, 72.55] |
| factor:us2y | 0.178 | 0.877 | [0.56, 1.83] | [0.1414, 2.169] |
| factor:us10y | 0.089 | 1.096 | [1.37, 2.62] | [1.006, 3.43] |
| factor:aaa_oas | 0.313 | 0.915 | [0.53, 0.99] | [0.4754, 6.135] |
| factor:bbb_oas | 0.407 | 1.017 | [1.29, 3.03] | [1.24, 6.988] |
| factor:nikkei | 0.068 | 1.144 | [1.495e+04, 2.294e+04] | [7353, 2.361e+04] |
| factor:gold | 0.032 | 0.892 | [1116, 1365] | [947.2, 1989] |
