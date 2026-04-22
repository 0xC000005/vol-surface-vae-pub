# 220h Full 11-Suite Multi-Horizon Validation

- model: `269a`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/269a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `1/11`
- failed suites: `surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.966`
- h30 cov90: `0.998`
- turb/calm ratio: `0.979`
- MR ratio h1: `0.050`
- MR ratio h30: `1.104`

**Additional v2 Suites**
- time-series ACF corr: `0.948`
- block boundary ratio: `0.987`
- cointegration gen/GT ratio: `0.735`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `2/25`
- level KS pass cells: `0/25`
- corr ratio: `0.298`
- rank ratio: `3.607`
- max-jump KS: `0.652`
