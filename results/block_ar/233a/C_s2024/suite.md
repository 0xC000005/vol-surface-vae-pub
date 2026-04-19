# 220b Multi-Horizon Path Suite

- model: `233a_C`
- checkpoint: `models/backfill/233a_v1_C_25d_s2024/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.596`
- h30 cov90: `0.623`
- worst cell h30: `0.312`
- best cell h30: `0.932`

**Conditionality**
- turb/calm ratio: `0.994`
- MAE reduction vs shuffled: `3.4%`

**Mean Reversion**
- aggregate slope ratio: `0.750`
- active pass count: `4/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `16/25`
- level KS pass cells: `10/25`
- worst floor occupancy: `7.325%`

**Cross-Cell / Pathwise**
- corr ratio: `1.182`
- rank ratio: `0.975`
- max-jump KS: `0.848`
