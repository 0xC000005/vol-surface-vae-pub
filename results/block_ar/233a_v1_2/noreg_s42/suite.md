# 220b Multi-Horizon Path Suite

- model: `233a_v1_2`
- checkpoint: `models/backfill/233a_v1_2_noreg_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.558`
- h30 cov90: `0.567`
- worst cell h30: `0.115`
- best cell h30: `0.880`

**Conditionality**
- turb/calm ratio: `1.032`
- MAE reduction vs shuffled: `4.5%`

**Mean Reversion**
- aggregate slope ratio: `0.554`
- active pass count: `1/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `11/25`
- level KS pass cells: `12/25`
- worst floor occupancy: `12.689%`

**Cross-Cell / Pathwise**
- corr ratio: `0.950`
- rank ratio: `1.347`
- max-jump KS: `0.806`
