# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/455a_recent_joint_mmd_w05_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.849`
- h30 cov90: `0.888`
- turb/calm ratio: `1.072`
- MR ratio h1: `0.923`
- MR ratio h30: `0.773`

**Additional v2 Suites**
- time-series ACF corr: `0.957`
- block boundary ratio: `1.004`
- cointegration gen/GT ratio: `0.656`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `7/25`
- corr ratio: `0.979`
- rank ratio: `1.462`
- max-jump KS: `0.375`
