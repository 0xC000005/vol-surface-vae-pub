# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/467a_recent_condcritic_w001_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.863`
- h30 cov90: `0.903`
- turb/calm ratio: `1.044`
- MR ratio h1: `0.957`
- MR ratio h30: `0.809`

**Additional v2 Suites**
- time-series ACF corr: `0.958`
- block boundary ratio: `0.977`
- cointegration gen/GT ratio: `0.656`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `8/25`
- corr ratio: `0.984`
- rank ratio: `1.472`
- max-jump KS: `0.367`
