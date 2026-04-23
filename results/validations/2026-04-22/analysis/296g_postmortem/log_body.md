### Context
`296g-v0` tested the smallest remaining fast-shell activation hypothesis in the `296` hybrid family:
- frozen `277d` backbone
- `296f` budget-plus-redistribution multiresolution zero-mean shell
- one learned scalar gate on the fast shell knots before budget renormalization

### Result
`296g-v0` scored `4/11`.

Passes:
- `block_ar`
- `cointegration`
- `cross_cell_correlation`
- `mean_reversion`

Artifacts:
- `results/block_ar/296g_v0_s42/full11.json`
- `results/block_ar/296g_v0_s42/full11.md`
- `results/validations/2026-04-22/analysis/296g_postmortem/summary.md`
- `results/validations/2026-04-22/analysis/296g_postmortem/gate_diagnostic.json`

### Mechanism Read
The gate did not become a conditional activation mechanism.

Gate diagnostic:
- mean `0.778`
- std `0.004`
- Spearman(gate, history realized variance) `-0.005`
- calm mean `0.777`
- turbulent mean `0.778`

So `296g` mostly reproduced `296f` with a mild static fast-shell rescaling:
- h1 coverage stayed high but slightly regressed
- regime width did not recover
- surface validity still failed
- level KS stayed `0/25`
- pathwise jump KS barely moved

### Decision
The local `296e -> 296f -> 296g` fast-shell branch is near a cap.

Next step:
- post-experiment analysis over the broader `296` hybrid family
- decide whether the fixed `277d` backbone plus zero-mean shell interface can ever repair level-law fidelity
- do not add another shell knob before that analysis
