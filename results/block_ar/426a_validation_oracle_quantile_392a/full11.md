# 426a Validation-Oracle Quantile Diagnostic

- base model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- diagnostic: `validation oracle marginal quantile map; not deployable`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `conditionality, cointegration, regime_coverage, mean_reversion`

**Horizon Summary**
- h1 cov90: `0.825`
- h30 cov90: `0.822`
- turb/calm ratio: `0.968`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `25/25`
- median-bias fraction cells: `25/25`
- median-bias magnitude cells: `23/25`
- regime layer2: `1/8`
- corr ratio: `0.908`
- rank ratio: `1.600`
- max-jump KS: `0.179`
