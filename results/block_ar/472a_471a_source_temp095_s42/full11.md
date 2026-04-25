# 220h Full 11-Suite Multi-Horizon Validation

- model: `471a`
- checkpoint: `models/backfill/471a_frozen_center_source_transport_score_flow_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion`

**Horizon Summary**
- h1 cov90: `0.869`
- h30 cov90: `0.905`
- turb/calm ratio: `1.097`
- MR ratio h1: `1.091`
- MR ratio h30: `0.799`

**Additional v2 Suites**
- time-series ACF corr: `0.954`
- block boundary ratio: `0.897`
- cointegration gen/GT ratio: `0.735`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `9/25`
- corr ratio: `0.945`
- rank ratio: `1.562`
- max-jump KS: `0.314`
