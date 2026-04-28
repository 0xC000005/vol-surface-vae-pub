# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.852`
- h30 cov90: `0.889`
- turb/calm ratio: `1.069`
- MR ratio h1: `0.950`
- MR ratio h30: `0.780`

**Additional v2 Suites**
- time-series ACF corr: `0.957`
- block boundary ratio: `0.950`
- cointegration gen/GT ratio: `0.606`
- regime coverage overall: `False`
- risk-state allocation: `True`
- risk-state observable response: `True`
- risk-state oracle future alignment: `False`
- risk-state width/history rho: `0.298`
- risk-state width/future rho: `0.036`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `10/25`
- corr ratio: `0.960`
- rank ratio: `1.480`
- max-jump KS: `0.380`
