# 220h Full 11-Suite Multi-Horizon Validation

- model: `287e`
- checkpoint: `models/backfill/287e_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.928`
- MR ratio h30: `0.788`

**Additional v2 Suites**
- time-series ACF corr: `0.956`
- block boundary ratio: `0.975`
- cointegration gen/GT ratio: `1.316`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio: `0.955`
- rank ratio: `1.336`
- max-jump KS: `0.349`
