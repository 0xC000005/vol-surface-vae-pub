# 220b Multi-Horizon Path Suite

- model: `233a_full`
- checkpoint: `models/backfill/233a_v1_full_25d_s2024/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.612`
- h30 cov90: `0.608`
- worst cell h30: `0.245`
- best cell h30: `0.875`

**Conditionality**
- turb/calm ratio: `0.986`
- MAE reduction vs shuffled: `3.9%`

**Mean Reversion**
- aggregate slope ratio: `0.925`
- active pass count: `4/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `12/25`
- level KS pass cells: `7/25`
- worst floor occupancy: `12.463%`

**Cross-Cell / Pathwise**
- corr ratio: `0.950`
- rank ratio: `1.761`
- max-jump KS: `0.838`
