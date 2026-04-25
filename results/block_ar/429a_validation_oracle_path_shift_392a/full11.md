# 429a Validation-Oracle Path-Shift Diagnostic

- base model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- diagnostic: `validation oracle path-center shift; not deployable`
- shift mode: `per_cell`
- shift range: `-0.2750 / -0.0019 / 0.3873`
- residual scale: `1.000`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, cointegration, regime_coverage, distributional_fidelity, mean_reversion`

**Fidelity / Structure**
- cov90 overall: `0.927`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `22/25`
- median-bias fraction cells: `25/25`
- median-bias magnitude cells: `24/25`
- regime layer2: `0/8`
- conditionality MAE reduction: `15.78%`
- cointegration worst-cell ratio: `0.246`
- mean-reversion active pass: `0.458`
- max-jump KS: `0.431`
