# 220b Multi-Horizon Path Suite

- model: `233a_B`
- checkpoint: `models/backfill/233a_v1_B_25d_s1337/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.658`
- h30 cov90: `0.716`
- worst cell h30: `0.370`
- best cell h30: `0.995`

**Conditionality**
- turb/calm ratio: `1.044`
- MAE reduction vs shuffled: `0.3%`

**Mean Reversion**
- aggregate slope ratio: `0.823`
- active pass count: `1/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `16/25`
- level KS pass cells: `13/25`
- worst floor occupancy: `15.560%`

**Cross-Cell / Pathwise**
- corr ratio: `1.472`
- rank ratio: `0.675`
- max-jump KS: `0.740`
