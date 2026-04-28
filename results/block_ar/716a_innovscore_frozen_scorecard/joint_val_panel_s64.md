# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.114`
- factor delta KS pass <0.20: `11/13`
- factor q99 abs-delta ratio median: `1.073`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.877`
- factor-factor corr MAE: `0.123`
- IV-factor corr matrix corr: `0.893`
- IV-factor corr MAE: `0.083`
- conditional median MAE reduction vs rolled deck: `3.15%`
- history-activity vs generated-width Spearman: `0.922`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.160 | 1.073 | [1829, 2648] | [1475, 2584] |
| factor:usdcad | 0.040 | 1.172 | [1.213, 1.408] | [1.059, 1.505] |
| factor:usdjpy | 0.095 | 1.088 | [100.1, 121.1] | [80.56, 136.4] |
| factor:dxy | 0.041 | 1.039 | [91.33, 103.3] | [83.44, 107.4] |
| factor:copper | 0.119 | 0.943 | [2.004, 3.224] | [1.478, 3.542] |
| factor:wheat | 0.077 | 1.313 | [361, 539.2] | [343.8, 776.4] |
| factor:crude_oil | 0.111 | 1.130 | [26.19, 58.94] | [13.78, 73.48] |
| factor:us2y | 0.122 | 0.920 | [0.56, 1.83] | [-0.06831, 2.149] |
| factor:us10y | 0.094 | 1.036 | [1.37, 2.62] | [0.878, 3.437] |
| factor:aaa_oas | 0.203 | 0.934 | [0.53, 0.99] | [0.2402, 2.276] |
| factor:bbb_oas | 0.265 | 1.157 | [1.29, 3.03] | [1.246, 4.668] |
| factor:nikkei | 0.049 | 1.256 | [1.495e+04, 2.294e+04] | [6606, 2.409e+04] |
| factor:gold | 0.108 | 0.966 | [1116, 1365] | [768.8, 1623] |
