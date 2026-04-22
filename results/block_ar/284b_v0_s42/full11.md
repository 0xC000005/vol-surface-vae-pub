# 220h Full 11-Suite Multi-Horizon Validation

- model: `284b`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/283a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.729`
- h30 cov90: `0.730`
- turb/calm ratio: `0.959`
- MR ratio h1: `1.689`
- MR ratio h30: `1.085`

**Additional v2 Suites**
- time-series ACF corr: `0.951`
- block boundary ratio: `1.042`
- cointegration gen/GT ratio: `0.795`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `23/25`
- level KS pass cells: `3/25`
- corr ratio: `1.012`
- rank ratio: `1.130`
- max-jump KS: `0.446`
