# 220h Full 11-Suite Multi-Horizon Validation

- model: `293e`
- checkpoint: `models/backfill/293e_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.831`
- h30 cov90: `0.946`
- turb/calm ratio: `1.029`
- MR ratio h1: `0.075`
- MR ratio h30: `0.451`

**Additional v2 Suites**
- time-series ACF corr: `0.958`
- block boundary ratio: `0.984`
- cointegration gen/GT ratio: `0.768`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `24/25`
- level KS pass cells: `3/25`
- corr ratio: `1.215`
- rank ratio: `0.914`
- max-jump KS: `0.625`
