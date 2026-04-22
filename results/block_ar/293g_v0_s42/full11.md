# 220h Full 11-Suite Multi-Horizon Validation

- model: `293g`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/293g_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.931`
- h30 cov90: `0.954`
- turb/calm ratio: `0.697`
- MR ratio h1: `2.420`
- MR ratio h30: `0.987`

**Additional v2 Suites**
- time-series ACF corr: `0.947`
- block boundary ratio: `0.946`
- cointegration gen/GT ratio: `1.035`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `14/25`
- level KS pass cells: `3/25`
- corr ratio: `0.690`
- rank ratio: `1.933`
- max-jump KS: `0.578`
