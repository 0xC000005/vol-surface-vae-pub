# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.108`
- factor delta KS pass <0.20: `12/13`
- factor q99 abs-delta ratio median: `1.043`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.877`
- factor-factor corr MAE: `0.126`
- IV-factor corr matrix corr: `nan`
- IV-factor corr MAE: `nan`
- conditional median MAE reduction vs rolled deck: `2.97%`
- history-activity vs generated-width Spearman: `0.943`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.127 | 1.122 | [1829, 2648] | [1420, 2625] |
| factor:usdcad | 0.053 | 1.120 | [1.213, 1.408] | [1.123, 1.528] |
| factor:usdjpy | 0.115 | 1.100 | [100.1, 121.1] | [94.18, 146.7] |
| factor:dxy | 0.081 | 1.025 | [91.33, 103.3] | [87.89, 112.8] |
| factor:copper | 0.103 | 0.905 | [2.004, 3.224] | [1.643, 3.663] |
| factor:wheat | 0.069 | 1.043 | [361, 539.2] | [303.7, 835.3] |
| factor:crude_oil | 0.094 | 1.040 | [26.19, 58.94] | [7.719, 71.22] |
| factor:us2y | 0.127 | 0.928 | [0.56, 1.83] | [0.07923, 2.088] |
| factor:us10y | 0.084 | 1.011 | [1.37, 2.62] | [0.986, 3.291] |
| factor:aaa_oas | 0.244 | 1.218 | [0.53, 0.99] | [0.03475, 2.183] |
| factor:bbb_oas | 0.194 | 1.154 | [1.29, 3.03] | [1.203, 3.863] |
| factor:nikkei | 0.039 | 1.252 | [1.495e+04, 2.294e+04] | [1.056e+04, 2.805e+04] |
| factor:gold | 0.072 | 0.926 | [1116, 1365] | [847.2, 1579] |
