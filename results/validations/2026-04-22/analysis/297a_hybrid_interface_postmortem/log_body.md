### Context
`297a` tested whether the fixed `277d` center plus shell interface could repair level-law fidelity if given a strong empirical residual shell.

Setup:
- frozen `277d` center
- residual raw-change bank from training windows only
- mean-centered residual paths
- validation full 11-suite
- independent and paired-symmetric variants

### Result
Both variants scored `4/11`.

Artifacts:
- `experiments/backfill/block_ar/analyze_297a_fixed_center_shell_oracle.py`
- `results/block_ar/297a_v0_s42/full11.json`
- `results/block_ar/297a_paired_v0_s42/full11.json`
- `results/validations/2026-04-22/analysis/297a_hybrid_interface_postmortem/summary.md`

Key metrics:
- `297a`: level KS `0/25`, change KS `4/25`, cov90 `0.830`, jump KS `0.342`
- `297a_paired`: level KS `0/25`, change KS `4/25`, cov90 `0.832`, jump KS `0.336`

### Mechanism Read
The diagnostic is a clean negative.

Even an empirical training residual shell around fixed `277d` does not move level KS and badly damages local change-law fidelity. This means the learned `296` shells were not merely underpowered; the fixed-center shell interface itself is capped for the remaining objective.

### Decision
Close the fixed-`277d` center plus shell program.

Next step:
- paradigm-shift ideation for `298a`
- stochasticity must enter the structural center/support object itself, not only a zero-mean shell around one fixed path
