# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/452a_scaled_340a_score_transition_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.852`
- h30 cov90: `0.729`
- turb/calm ratio: `0.723`
- MR ratio h1: `0.992`
- MR ratio h30: `0.729`

**Additional v2 Suites**
- time-series ACF corr: `0.958`
- block boundary ratio: `1.036`
- cointegration gen/GT ratio: `0.698`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `0/25`
- corr ratio: `1.012`
- rank ratio: `1.254`
- max-jump KS: `0.348`
