# 220h Full 11-Suite Multi-Horizon Validation

- model: `274a`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/274a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.635`
- h30 cov90: `0.624`
- turb/calm ratio: `0.979`
- MR ratio h1: `2.336`
- MR ratio h30: `1.187`

**Additional v2 Suites**
- time-series ACF corr: `0.948`
- block boundary ratio: `0.977`
- cointegration gen/GT ratio: `1.066`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `6/25`
- level KS pass cells: `1/25`
- corr ratio: `1.546`
- rank ratio: `0.450`
- max-jump KS: `0.998`
