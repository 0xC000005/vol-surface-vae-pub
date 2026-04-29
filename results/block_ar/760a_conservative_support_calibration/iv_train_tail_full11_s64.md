# 759a State-Local Support Calibration

- source: `759a state-local support calibration`
- base checkpoint: `models/backfill/755a_iv_shortprefix_fm_k5_w02_e2_s7551/best_model.pt`
- calibration split: `train`
- eval split: `train_tail`
- suite score: `5/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, pathwise_jump_realism`

**Key Metrics**
- cov90 overall: `0.872`
- calibration error: `0.007`
- level KS pass cells: `23/25`
- median-bias cells: `25/25`
- cointegration ratio: `0.536`
- worst-cell cointegration ratio: `0.342`
- regime layer2: `1/8`
- mean reversion pass: `True`
- pathwise max-jump KS: `0.560`
