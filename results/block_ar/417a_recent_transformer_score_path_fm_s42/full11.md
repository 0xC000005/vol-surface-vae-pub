# 220h Full 11-Suite Multi-Horizon Validation

- model: `339a`
- checkpoint: `models/backfill/417a_recent_transformer_score_path_fm_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion`

**Horizon Summary**
- h1 cov90: `0.884`
- h30 cov90: `0.829`
- turb/calm ratio: `0.972`
- MR ratio h1: `0.909`
- MR ratio h30: `0.760`

**Additional v2 Suites**
- time-series ACF corr: `0.946`
- block boundary ratio: `1.010`
- cointegration gen/GT ratio: `0.626`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `7/25`
- corr ratio: `0.829`
- rank ratio: `1.959`
- max-jump KS: `0.405`
