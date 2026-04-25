# 220h Full 11-Suite Multi-Horizon Validation

- model: `471a`
- checkpoint: `models/backfill/471a_frozen_center_source_transport_score_flow_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `5/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity, mean_reversion`

**Horizon Summary**
- h1 cov90: `0.864`
- h30 cov90: `0.891`
- turb/calm ratio: `1.075`
- MR ratio h1: `1.024`
- MR ratio h30: `0.798`

**Additional v2 Suites**
- time-series ACF corr: `0.962`
- block boundary ratio: `1.002`
- cointegration gen/GT ratio: `0.629`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `13/25`
- corr ratio: `0.962`
- rank ratio: `1.500`
- max-jump KS: `0.349`
