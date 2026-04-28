# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.848`
- h30 cov90: `0.885`
- turb/calm ratio: `1.068`
- MR ratio h1: `0.980`
- MR ratio h30: `0.776`

**Additional v2 Suites**
- time-series ACF corr: `0.953`
- block boundary ratio: `0.938`
- cointegration gen/GT ratio: `0.597`
- regime coverage overall: `False`
- risk-state allocation: `False`
- risk-state width/future rho: `0.129`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `10/25`
- corr ratio: `0.963`
- rank ratio: `1.471`
- max-jump KS: `0.377`
