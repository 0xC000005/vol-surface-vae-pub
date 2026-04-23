### Context
The `296` family postmortem left one unresolved question:
- is the fixed `277d` center plus zero-mean shell interface fundamentally capped,
- or have we only failed to learn the right shell distribution?

### Result
`297a` ideation selected an oracle/interface diagnostic before another architecture reset.

Artifact:
- `results/validations/2026-04-22/analysis/297a_hybrid_interface_ideation/memo.md`

### Mechanism Read
Another learned shell gate or per-cell activation would be knob accumulation.

The more principled next step is to test the interface itself:
- keep `277d` center fixed
- use an empirical training residual bank as an oracle-like shell
- evaluate whether level KS and distributional fidelity can improve without changing center support

### Decision
Next step:
- implement `297a-v0` as an evaluation-only fixed-center shell oracle diagnostic

Interpretation:
- if level KS remains dead, the fixed-center shell interface is falsified
- if level KS improves, the issue is shell learning/allocation rather than interface structure
