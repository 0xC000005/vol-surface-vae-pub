# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/410a_recent_rollout_marginal_crps_w005_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.849`
- h30 cov90: `0.898`
- turb/calm ratio: `1.062`
- MR ratio h1: `0.970`
- MR ratio h30: `0.782`

**Additional v2 Suites**
- time-series ACF corr: `0.953`
- block boundary ratio: `0.964`
- cointegration gen/GT ratio: `0.566`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `10/25`
- corr ratio: `0.972`
- rank ratio: `1.465`
- max-jump KS: `0.375`
