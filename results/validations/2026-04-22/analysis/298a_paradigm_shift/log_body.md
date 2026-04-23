### Context
`297a` falsified the fixed-center shell interface:
- empirical residual shells around frozen `277d` kept level KS at `0/25`
- change KS collapsed to `4/25`
- paired residuals did not change the diagnosis

### Decision
Paradigm shift to `298a`: stochastic structural center support.

Artifact:
- `results/validations/2026-04-22/analysis/298a_paradigm_shift/memo.md`

### Mechanism Read
The scenario generator needs stochasticity in the structural center/support object itself, not only residual width around one fixed center path.

The new object is:
- sample a plausible structural center path
- then optionally add a small secondary shell later

### 298a-v0
Use the existing `277d` representation as a structural future-path manifold:
- freeze `277d` history and future encoders
- train a small conditional density over future embeddings
- sample future-embedding codes conditioned on history
- decode sampled codes through the training future library
- replay selected normalized-change paths from the query current state

Constraint:
- do not return to top-k weighting, daily residual correction, or per-cell shell knobs
