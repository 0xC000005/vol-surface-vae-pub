# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.096`
- factor delta KS pass <0.20: `12/14`
- factor q99 abs-delta ratio median: `1.040`
- factor q99 abs-delta pass [0.5,2.0]: `14/14`
- factor-factor corr upper-triangle corr: `0.866`
- factor-factor corr MAE: `0.135`
- IV-factor corr matrix corr: `0.942`
- IV-factor corr MAE: `0.082`
- conditional median MAE reduction vs rolled deck: `4.87%`
- history-activity vs generated-width Spearman: `0.950`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.109 | 1.020 | [1829, 2648] | [1442, 2647] |
| factor:usdcad | 0.027 | 1.213 | [1.213, 1.408] | [1.012, 1.641] |
| factor:usdjpy | 0.058 | 1.044 | [100.1, 121.1] | [83.02, 137.4] |
| factor:dxy | 0.048 | 1.121 | [91.33, 103.3] | [80.38, 113.1] |
| factor:copper | 0.052 | 1.009 | [2.004, 3.224] | [1.488, 3.881] |
| factor:wheat | 0.059 | 1.010 | [361, 539.2] | [225.3, 828.7] |
| factor:crude_oil | 0.079 | 1.080 | [26.19, 58.94] | [7.915, 78.9] |
| factor:us2y | 0.161 | 0.964 | [0.56, 1.83] | [0.01322, 2.126] |
| factor:us10y | 0.094 | 1.085 | [1.37, 2.62] | [0.6733, 3.516] |
| factor:aaa_oas | 0.241 | 0.891 | [0.53, 0.99] | [0.1684, 1.343] |
| factor:bbb_oas | 0.279 | 1.189 | [1.29, 3.03] | [1.007, 3.941] |
| factor:nikkei | 0.058 | 1.210 | [1.495e+04, 2.294e+04] | [9104, 2.523e+04] |
| factor:gold | 0.033 | 1.036 | [1116, 1365] | [810.8, 1629] |
| factor:vix | 0.039 | 0.806 | [9.14, 28.14] | [1.978, 72.34] |
