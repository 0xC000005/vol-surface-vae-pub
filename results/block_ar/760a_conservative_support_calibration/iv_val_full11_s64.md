# 759a State-Local Support Calibration

- source: `759a state-local support calibration`
- base checkpoint: `models/backfill/755a_iv_shortprefix_fm_k5_w02_e2_s7551/best_model.pt`
- calibration split: `train_tail`
- eval split: `val`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Key Metrics**
- cov90 overall: `0.874`
- calibration error: `0.014`
- level KS pass cells: `15/25`
- median-bias cells: `15/25`
- cointegration ratio: `0.777`
- worst-cell cointegration ratio: `0.328`
- regime layer2: `1/8`
- mean reversion pass: `True`
- pathwise max-jump KS: `0.489`
