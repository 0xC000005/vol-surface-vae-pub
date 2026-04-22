# 220h Full 11-Suite Multi-Horizon Validation

- model: `273a`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/273a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `surface, coverage, conditionality, time_series, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `1.000`
- h30 cov90: `0.995`
- turb/calm ratio: `0.996`
- MR ratio h1: `-1.673`
- MR ratio h30: `-5.915`

**Additional v2 Suites**
- time-series ACF corr: `0.940`
- block boundary ratio: `1.002`
- cointegration gen/GT ratio: `1.513`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- corr ratio: `1.599`
- rank ratio: `0.337`
- max-jump KS: `1.000`
