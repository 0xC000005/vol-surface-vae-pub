## 288a-v0 Postmortem

### Result
- Model: `288a-v0`
- Eval: `results/block_ar/288a_v0_s42/full11.json`
- Score: `4/11`
- Passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### Design
`288a` kept the full `287e` mixed query key fixed and changed only the support use:
- from single-nearest deterministic replay
- to deterministic soft top-k replay over normalized future changes

No retraining was used. The same `287e` checkpoint was reused.

### Mechanism Read
The top-k soft replay hypothesis is falsified as the next smallest fix.

What improved:
- `level KS pass cells`: `12/25 -> 20/25`
- floor/ceiling support improved materially:
  - floor rate now passes
  - ceiling rate now passes
- overall local MAE stayed strong

What broke:
- `change KS pass cells`: `25/25 -> 8/25`
- `max-jump KS`: `0.349 -> 0.766`
- `kurtosis ratio`: `1.463 -> 3.040`
- `active_pass_rate`: `8.3% -> 12.5%` only trivial recovery
- active-cell structure is still far from gate

Interpretation:
- soft replay smooths the support object in a way that helps level-side fidelity
- but it destroys the sharp deterministic change law that the local-history delta replay family was finally getting right

So the cap is not just “single nearest is too brittle.”
The support-use change interacts directly with the change-law realism we care about.

### Decision
Do not continue local smoothing variants inside `288`.

Next step:
- `post_experiment_analysis`
- compare:
  - `277d` deterministic Stage A frontier
  - `287e` best current local-history replay key
  - `288a` soft-replay support-use probe

The live program-level question is now:
- is the deterministic Stage A retrieval line capped,
- or is there still one clean non-smoothing support-use move left?
