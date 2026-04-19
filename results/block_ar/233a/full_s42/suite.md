# 220b Multi-Horizon Path Suite

- model: `233a_full`
- checkpoint: `models/backfill/233a_v1_full_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.693`
- h30 cov90: `0.721`
- worst cell h30: `0.385`
- best cell h30: `0.922`

**Conditionality**
- turb/calm ratio: `1.010`
- MAE reduction vs shuffled: `4.8%`

**Mean Reversion**
- aggregate slope ratio: `0.794`
- active pass count: `4/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `13/25`
- level KS pass cells: `8/25`
- worst floor occupancy: `8.361%`

**Cross-Cell / Pathwise**
- corr ratio: `0.876`
- rank ratio: `1.648`
- max-jump KS: `0.735`
