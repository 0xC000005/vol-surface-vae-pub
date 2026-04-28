# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.850`
- h30 cov90: `0.887`
- turb/calm ratio: `1.053`
- MR ratio h1: `1.024`
- MR ratio h30: `0.793`

**Additional v2 Suites**
- time-series ACF corr: `0.956`
- block boundary ratio: `0.977`
- cointegration gen/GT ratio: `0.613`
- regime coverage overall: `False`
- risk-state allocation: `True`
- risk-state observable response: `True`
- risk-state oracle future alignment: `False`
- risk-state width/history rho: `0.270`
- risk-state width/future rho: `0.031`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio: `0.945`
- rank ratio: `1.527`
- max-jump KS: `0.374`
