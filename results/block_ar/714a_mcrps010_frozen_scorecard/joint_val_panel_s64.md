# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.122`
- factor delta KS pass <0.20: `11/13`
- factor q99 abs-delta ratio median: `1.083`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.814`
- factor-factor corr MAE: `0.146`
- IV-factor corr matrix corr: `0.909`
- IV-factor corr MAE: `0.077`
- conditional median MAE reduction vs rolled deck: `4.06%`
- history-activity vs generated-width Spearman: `0.941`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.177 | 1.102 | [1829, 2648] | [1333, 2589] |
| factor:usdcad | 0.065 | 1.242 | [1.213, 1.408] | [1.076, 1.616] |
| factor:usdjpy | 0.070 | 1.051 | [100.1, 121.1] | [88.55, 137.8] |
| factor:dxy | 0.045 | 1.157 | [91.33, 103.3] | [85.94, 113.2] |
| factor:copper | 0.086 | 0.972 | [2.004, 3.224] | [1.608, 3.64] |
| factor:wheat | 0.069 | 1.223 | [361, 539.2] | [304.4, 871.7] |
| factor:crude_oil | 0.107 | 1.140 | [26.19, 58.94] | [4.873, 67.56] |
| factor:us2y | 0.161 | 0.916 | [0.56, 1.83] | [0.05089, 2.13] |
| factor:us10y | 0.093 | 1.038 | [1.37, 2.62] | [0.9975, 3.459] |
| factor:aaa_oas | 0.278 | 0.859 | [0.53, 0.99] | [0.2943, 2.602] |
| factor:bbb_oas | 0.338 | 1.083 | [1.29, 3.03] | [1.232, 4.324] |
| factor:nikkei | 0.064 | 1.190 | [1.495e+04, 2.294e+04] | [6808, 2.408e+04] |
| factor:gold | 0.027 | 0.940 | [1116, 1365] | [916, 1689] |
