# 220h Full 11-Suite Multi-Horizon Validation

- model: `460a`
- checkpoint: `models/backfill/460a_day_vector_density_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `surface, coverage, conditionality, time_series, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.963`
- h30 cov90: `0.992`
- turb/calm ratio: `0.974`
- MR ratio h1: `0.388`
- MR ratio h30: `0.833`

**Additional v2 Suites**
- time-series ACF corr: `0.947`
- block boundary ratio: `0.948`
- cointegration gen/GT ratio: `0.782`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `8/25`
- level KS pass cells: `0/25`
- corr ratio: `0.428`
- rank ratio: `3.156`
- max-jump KS: `0.588`
