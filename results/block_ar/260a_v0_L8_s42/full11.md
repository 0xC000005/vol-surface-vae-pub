# 220h Full 11-Suite Multi-Horizon Validation

- model: `260a`
- checkpoint: `models/backfill/260a_v0_L8_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `surface, coverage, conditionality, time_series, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.999`
- h30 cov90: `1.000`
- turb/calm ratio: `0.997`
- MR ratio h1: `1.011`
- MR ratio h30: `1.615`

**Additional v2 Suites**
- time-series ACF corr: `0.950`
- block boundary ratio: `0.888`
- cointegration gen/GT ratio: `1.016`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- corr ratio: `0.039`
- rank ratio: `4.060`
- max-jump KS: `0.896`
