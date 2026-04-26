# 555a Local-Factor Conditioned Core Fine-Tune

- checkpoint: `models/backfill/555a_local_factor_core_e4_s555/best_model.pt`
- windows: `192`
- local factor columns: `ret, price, slopes, skews, levels`
- conditionality mode: `live`
- suite score: `8/11`
- failed suites: `coverage, regime_coverage, distributional_fidelity`

**Key Metrics**
- cov90 overall: `0.873`
- h30 cov90: `0.872`
- conditionality MAE reduction: `6.20%`
- turb/calm ratio: `1.101`
- cointegration ratio: `0.705`
- regime layer2: `0/8`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `9/25`
- mean-reversion active pass: `0.875`
- max-jump KS: `0.346`
