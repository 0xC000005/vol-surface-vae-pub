# 220h Full 11-Suite Multi-Horizon Validation

- model: `471a`
- checkpoint: `models/backfill/471a_frozen_center_source_transport_score_flow_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, regime_coverage, distributional_fidelity, mean_reversion`

**Horizon Summary**
- h1 cov90: `0.866`
- h30 cov90: `0.899`
- turb/calm ratio: `1.057`
- MR ratio h1: `1.044`
- MR ratio h30: `0.800`

**Additional v2 Suites**
- time-series ACF corr: `0.957`
- block boundary ratio: `0.942`
- cointegration gen/GT ratio: `0.690`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `11/25`
- corr ratio: `0.954`
- rank ratio: `1.518`
- max-jump KS: `0.310`
