## 297a hybrid interface ideation

### Context
The `296` family is locally capped:

- the frozen `277d` center preserves structural suites
- zero-mean shells add stochastic width
- multiresolution shells restore h1 coverage and improve jump scale
- but level KS remains `0/25` across `277d` and all `296` variants

This suggests the problem may be the interface itself: one fixed center path plus a zero-mean shell may not be able to repair unconditional level-law fidelity.

### Candidate next moves

1. **Add another learned shell mechanism**
- Examples: per-cell gate, regime classifier, jump-specific scale.
- Rejected: this is knob accumulation. `296g` already showed scalar activation does not become conditional, and level KS remains untouched.

2. **Immediate architecture shift to structural center-support distribution**
- Replace the single center path with a stochastic distribution over plausible structural center paths.
- Plausible, but premature without an interface feasibility diagnostic.

3. **Run a fixed-center shell oracle diagnostic**
- Keep the `277d` center fixed.
- Build oracle-style shell samples from empirical residuals around that center.
- Evaluate whether a very strong shell can repair level KS and distributional fidelity without changing the center-support object.

### Recommendation
Choose option 3 first.

Reason:
- It directly tests the unresolved mechanism.
- It prevents another blind paradigm reset.
- If the oracle shell still cannot fix level KS, the fixed-center interface is falsified.
- If the oracle shell can fix level KS, the implementation problem is shell learning/allocation, not the backbone interface.

### 297a-v0
Run an evaluation-only oracle diagnostic:

- construct frozen `277d` center paths for validation windows
- construct residual paths in normalized-change coordinates
- sample residual paths from the training residual bank with paired/centered variants
- add them around the `277d` center path
- evaluate the full 11-suite

Read:
- If level KS remains near `0/25`, the fixed-center zero-mean shell interface is fundamentally capped.
- If level KS improves materially, the next architecture should learn a better residual law rather than replacing the center interface.

Constraint:
- This is diagnostic-only, not a deployable model.
- Do not count an oracle diagnostic as a new frontier model.
- Do not use validation future residuals for the main diagnostic sample bank.
