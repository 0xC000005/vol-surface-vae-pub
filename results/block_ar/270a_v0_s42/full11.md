# 220h Full 11-Suite Multi-Horizon Validation

- model: `270a`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/270a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `surface, coverage, conditionality, time_series, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.959`
- h30 cov90: `0.970`
- turb/calm ratio: `1.009`
- MR ratio h1: `2.856`
- MR ratio h30: `1.154`

**Additional v2 Suites**
- time-series ACF corr: `0.899`
- block boundary ratio: `0.970`
- cointegration gen/GT ratio: `1.263`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `1/25`
- level KS pass cells: `0/25`
- corr ratio: `2.289`
- rank ratio: `0.178`
- max-jump KS: `0.623`
