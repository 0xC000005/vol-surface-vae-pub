### Context
After `296g`, the loop analyzed the broader `296` hybrid family:
- `277d`: structural deterministic backbone
- `296c`: best global zero-mean shell
- `296e`: multiresolution fast shell
- `296f`: budgeted multiresolution shell
- `296g`: gated budgeted multiresolution shell

### Result
The comparison shows the current interface is locally capped.

Key invariant:
- every `296` shell preserves some structural suites
- every `296` shell keeps level KS at `0/25`
- fast-shell variants improve h1 coverage and jump scale, but overpay through global overdispersion, surface failures, and weak regime differentiation

Artifact:
- `results/validations/2026-04-22/analysis/296_family_postmortem/summary.md`

### Mechanism Read
The zero-mean shell around one fixed `277d` center path is too constrained.

It can add width, but it cannot repair the unconditional level law if the fixed center support is wrong. `296g` also showed that scalar fast-shell activation does not become conditional: the gate was nearly constant and uncorrelated with historical realized variance.

### Decision
Stop adding shell knobs inside the same fixed-center zero-mean interface.

Next step:
- `297a` ideation
- shift the interface from one fixed center path plus shell to a small learned structural center-support distribution plus secondary shell
