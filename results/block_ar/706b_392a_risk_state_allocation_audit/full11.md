# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.856`
- h30 cov90: `0.882`
- turb/calm ratio: `1.039`
- MR ratio h1: `1.027`
- MR ratio h30: `0.783`

**Additional v2 Suites**
- time-series ACF corr: `0.957`
- block boundary ratio: `0.991`
- cointegration gen/GT ratio: `0.640`
- regime coverage overall: `False`
- risk-state allocation: `False`
- risk-state width/future rho: `-0.010`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio: `0.955`
- rank ratio: `1.503`
- max-jump KS: `0.377`
