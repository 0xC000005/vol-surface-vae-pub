# 220h Full 11-Suite Multi-Horizon Validation

- model: `289a`
- checkpoint: `models/backfill/289a_v0_s42_small/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `1/11`
- failed suites: `surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.686`
- MR ratio h30: `0.750`

**Additional v2 Suites**
- time-series ACF corr: `0.574`
- block boundary ratio: `0.992`
- cointegration gen/GT ratio: `0.476`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `7/25`
- level KS pass cells: `1/25`
- corr ratio: `0.056`
- rank ratio: `2.871`
- max-jump KS: `1.000`
