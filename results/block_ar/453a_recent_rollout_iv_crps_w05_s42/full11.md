# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/453a_recent_rollout_iv_crps_w05_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.853`
- h30 cov90: `0.890`
- turb/calm ratio: `1.087`
- MR ratio h1: `0.975`
- MR ratio h30: `0.769`

**Additional v2 Suites**
- time-series ACF corr: `0.956`
- block boundary ratio: `0.995`
- cointegration gen/GT ratio: `0.665`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `7/25`
- corr ratio: `0.989`
- rank ratio: `1.439`
- max-jump KS: `0.376`
