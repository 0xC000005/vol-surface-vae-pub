# 438a Deployable Residual Bootstrap Risk System

- policy: `pre-validation residual-error bootstrap around frozen 392a median`
- deployable: `True`
- uses validation future: `False`
- regime bins: `True`
- calibration windows/samples: `441` / `48`
- residual shape scale: `0.100`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, time_series, regime_coverage, distributional_fidelity`

**Key Metrics**
- cov90 overall: `0.855`
- h1 cov90: `0.882`
- h30 cov90: `0.857`
- conditionality MAE reduction: `5.07%`
- turb/calm ratio: `1.076`
- regime layer2: `0/8`
- daily-change KS pass cells: `23/25`
- level KS pass cells: `4/25`
- corr ratio/rank ratio: `1.032` / `1.142`
- mean-reversion active pass: `0.792`
- max-jump KS: `0.424`
