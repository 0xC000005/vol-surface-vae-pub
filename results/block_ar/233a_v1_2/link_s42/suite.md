# 220b Multi-Horizon Path Suite

- model: `233a_v1_2`
- checkpoint: `models/backfill/233a_v1_2_link_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.641`
- h30 cov90: `0.667`
- worst cell h30: `0.240`
- best cell h30: `0.922`

**Conditionality**
- turb/calm ratio: `1.011`
- MAE reduction vs shuffled: `4.2%`

**Mean Reversion**
- aggregate slope ratio: `0.781`
- active pass count: `8/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `23/25`
- level KS pass cells: `12/25`
- worst floor occupancy: `13.367%`

**Cross-Cell / Pathwise**
- corr ratio: `1.099`
- rank ratio: `1.091`
- max-jump KS: `0.737`
