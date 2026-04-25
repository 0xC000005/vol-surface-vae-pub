# 220h Full 11-Suite Multi-Horizon Validation

- model: `471a`
- checkpoint: `models/backfill/471a_frozen_center_source_transport_score_flow_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, time_series, regime_coverage, distributional_fidelity, mean_reversion`

**Horizon Summary**
- h1 cov90: `0.888`
- h30 cov90: `0.914`
- turb/calm ratio: `1.103`
- MR ratio h1: `1.088`
- MR ratio h30: `0.796`

**Additional v2 Suites**
- time-series ACF corr: `0.959`
- block boundary ratio: `0.953`
- cointegration gen/GT ratio: `0.824`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `10/25`
- corr ratio: `0.943`
- rank ratio: `1.570`
- max-jump KS: `0.243`
