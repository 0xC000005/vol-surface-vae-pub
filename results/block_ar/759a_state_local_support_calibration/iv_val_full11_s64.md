# 759a State-Local Support Calibration

- source: `759a state-local support calibration`
- base checkpoint: `models/backfill/755a_iv_shortprefix_fm_k5_w02_e2_s7551/best_model.pt`
- calibration split: `train_tail`
- eval split: `val`
- suite score: `5/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity, pathwise_jump_realism`

**Key Metrics**
- cov90 overall: `0.889`
- calibration error: `0.017`
- level KS pass cells: `15/25`
- median-bias cells: `16/25`
- cointegration ratio: `0.660`
- worst-cell cointegration ratio: `0.227`
- regime layer2: `2/8`
- mean reversion pass: `True`
- pathwise max-jump KS: `0.501`
