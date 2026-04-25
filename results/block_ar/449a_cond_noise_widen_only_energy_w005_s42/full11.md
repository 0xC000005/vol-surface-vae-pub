# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/449a_cond_noise_widen_only_energy_w005_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.847`
- h30 cov90: `0.883`
- turb/calm ratio: `1.047`
- MR ratio h1: `1.002`
- MR ratio h30: `0.783`

**Additional v2 Suites**
- time-series ACF corr: `0.954`
- block boundary ratio: `0.949`
- cointegration gen/GT ratio: `0.760`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio: `0.968`
- rank ratio: `1.482`
- max-jump KS: `0.364`
