# 627a Joint-Panel Scenario Quality Audit

- finite rate: `1.0000`
- windows / samples / horizon: `441` / `64` / `30`
- factor delta KS mean: `0.122`
- factor delta KS pass <0.20: `11/13`
- factor q99 abs-delta ratio median: `1.051`
- factor q99 abs-delta pass [0.5,2.0]: `13/13`
- factor-factor corr upper-triangle corr: `0.899`
- factor-factor corr MAE: `0.102`
- IV-factor corr matrix corr: `nan`
- IV-factor corr MAE: `nan`
- conditional median MAE reduction vs rolled deck: `4.52%`
- history-activity vs generated-width Spearman: `0.946`

| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |
| --- | ---: | ---: | ---: | ---: |
| factor:spx | 0.138 | 1.150 | [1829, 2648] | [1493, 2691] |
| factor:usdcad | 0.081 | 1.139 | [1.213, 1.408] | [1.126, 1.662] |
| factor:usdjpy | 0.066 | 1.017 | [100.1, 121.1] | [85.57, 150.7] |
| factor:dxy | 0.068 | 1.137 | [91.33, 103.3] | [86.88, 115.9] |
| factor:copper | 0.097 | 1.011 | [2.004, 3.224] | [1.412, 3.622] |
| factor:wheat | 0.055 | 1.051 | [361, 539.2] | [282.8, 781.7] |
| factor:crude_oil | 0.093 | 1.122 | [26.19, 58.94] | [4.176, 72.14] |
| factor:us2y | 0.151 | 0.900 | [0.56, 1.83] | [0.1362, 2.113] |
| factor:us10y | 0.065 | 1.091 | [1.37, 2.62] | [0.8489, 3.579] |
| factor:aaa_oas | 0.296 | 0.871 | [0.53, 0.99] | [0.4173, 2.028] |
| factor:bbb_oas | 0.349 | 1.008 | [1.29, 3.03] | [1.217, 4.924] |
| factor:nikkei | 0.041 | 1.190 | [1.495e+04, 2.294e+04] | [9440, 2.574e+04] |
| factor:gold | 0.090 | 0.940 | [1116, 1365] | [850.3, 1678] |
