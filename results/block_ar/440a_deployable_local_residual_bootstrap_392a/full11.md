# 440a Deployable Local Residual Bootstrap Risk System

- policy: `pre-validation history-local residual-error bootstrap around frozen 392a median`
- deployable: `True`
- uses validation future: `False`
- neighbor count: `64`
- feature recent window: `5`
- calibration windows/samples: `441` / `48`
- residual shape scale: `0.100`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity`

**Key Metrics**
- cov90 overall: `0.838`
- h1 cov90: `0.876`
- h30 cov90: `0.851`
- conditionality MAE reduction: `4.88%`
- turb/calm ratio: `1.047`
- regime layer2: `0/8`
- daily-change KS pass cells: `23/25`
- level KS pass cells: `10/25`
- corr ratio/rank ratio: `1.059` / `1.104`
- mean-reversion active pass: `0.792`
- max-jump KS: `0.282`
