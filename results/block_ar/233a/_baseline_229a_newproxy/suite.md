# 220b Multi-Horizon Path Suite

- model: `227a`
- checkpoint: `models/backfill/factor_ar_229a_wide_decoder/checkpoint_ep30.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/7`
- failed suites: `coverage, conditionality, distributional_fidelity, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.725`
- h30 cov90: `0.582`
- worst cell h30: `0.271`
- best cell h30: `0.917`

**Conditionality**
- turb/calm ratio: `1.025`
- MAE reduction vs shuffled: `3.3%`

**Mean Reversion**
- aggregate slope ratio: `1.318`
- active pass count: `21/24`
- full-horizon overall: `True`

**Distributional Fidelity**
- daily-change KS pass cells: `19/25`
- level KS pass cells: `6/25`
- worst floor occupancy: `4.313%`

**Cross-Cell / Pathwise**
- corr ratio: `1.279`
- rank ratio: `0.836`
- max-jump KS: `0.945`
