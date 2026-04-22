# 220h Full 11-Suite Multi-Horizon Validation

- model: `268b`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/268b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `1/11`
- failed suites: `surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.902`
- h30 cov90: `0.896`
- turb/calm ratio: `1.017`
- MR ratio h1: `1.682`
- MR ratio h30: `0.751`

**Additional v2 Suites**
- time-series ACF corr: `0.932`
- block boundary ratio: `0.945`
- cointegration gen/GT ratio: `2.485`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `6/25`
- level KS pass cells: `1/25`
- corr ratio: `0.461`
- rank ratio: `2.883`
- max-jump KS: `0.398`
