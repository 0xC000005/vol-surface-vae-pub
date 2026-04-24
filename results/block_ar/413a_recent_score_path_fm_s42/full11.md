# 220h Full 11-Suite Multi-Horizon Validation

- model: `339a`
- checkpoint: `models/backfill/413a_recent_score_path_fm_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, cross_cell_correlation, mean_reversion`

**Horizon Summary**
- h1 cov90: `0.804`
- h30 cov90: `0.862`
- turb/calm ratio: `1.045`
- MR ratio h1: `1.058`
- MR ratio h30: `0.794`

**Additional v2 Suites**
- time-series ACF corr: `0.932`
- block boundary ratio: `1.032`
- cointegration gen/GT ratio: `0.671`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `20/25`
- corr ratio: `0.486`
- rank ratio: `2.712`
- max-jump KS: `0.290`
