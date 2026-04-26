# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/558a_hard_state_replay_patch_energy_s558/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.852`
- h30 cov90: `0.889`
- turb/calm ratio: `1.112`
- MR ratio h1: `0.988`
- MR ratio h30: `0.786`

**Additional v2 Suites**
- time-series ACF corr: `0.956`
- block boundary ratio: `0.962`
- cointegration gen/GT ratio: `0.560`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `11/25`
- corr ratio: `0.983`
- rank ratio: `1.436`
- max-jump KS: `0.320`
