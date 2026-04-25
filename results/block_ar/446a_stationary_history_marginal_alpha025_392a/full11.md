# 446a Stationary History-Marginal Quantile Policy

- policy: `stationary history-marginal quantile map around frozen 392a`
- deployable: `True`
- uses validation future: `False`
- alpha: `0.250`
- regime bins: `True`
- calibration windows/samples: `441` / `48`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Key Metrics**
- cov90 overall: `0.870`
- h1 cov90: `0.840`
- h30 cov90: `0.880`
- conditionality MAE reduction: `4.27%`
- turb/calm ratio: `1.052`
- regime layer2: `0/8`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `11/25`
- median-bias fraction/magnitude: `20/25` / `25/25`
- corr ratio/rank ratio: `0.954` / `1.501`
- mean-reversion active pass: `0.833`
- max-jump KS: `0.362`
