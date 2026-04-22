# 220h Full 11-Suite Multi-Horizon Validation

- model: `293d`
- checkpoint: `models/backfill/293d_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.863`
- h30 cov90: `0.943`
- turb/calm ratio: `0.984`
- MR ratio h1: `0.093`
- MR ratio h30: `0.546`

**Additional v2 Suites**
- time-series ACF corr: `0.957`
- block boundary ratio: `1.029`
- cointegration gen/GT ratio: `0.656`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `1/25`
- corr ratio: `1.301`
- rank ratio: `0.849`
- max-jump KS: `0.647`
