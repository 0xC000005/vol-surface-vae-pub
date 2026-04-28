# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.853`
- h30 cov90: `0.897`
- turb/calm ratio: `1.085`
- MR ratio h1: `0.979`
- MR ratio h30: `0.778`

**Additional v2 Suites**
- time-series ACF corr: `0.952`
- block boundary ratio: `0.977`
- cointegration gen/GT ratio: `0.597`
- regime coverage overall: `False`
- risk-state allocation: `False`
- risk-state width/future rho: `0.050`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `9/25`
- corr ratio: `0.971`
- rank ratio: `1.463`
- max-jump KS: `0.382`
