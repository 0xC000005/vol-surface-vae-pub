## 289a-v0 Postmortem

### Result
- Model: `289a-v0`
- Eval: `results/block_ar/289a_v0_s42/full11.json`
- Score: `1/11`
- Passes:
  - `block_ar`

### Design
`289a` was the first true Stage A reset away from retrieval:
- deterministic causal Transformer world model
- direct observation-space joint next-change prediction
- raw normalized-change target
- autoregressive generated-state rollout
- no retrieval bank
- no low-rank head
- no bounded side paths

### Mechanism Read
The clean failure is **direct observation-space collapse**.

The model did not fail by exploding.
It failed by becoming too smooth and too decorrelated:
- `corr_ratio = 0.056`
- `rank_ratio = 2.871`
- `cointegration ratio = 0.476`
- `max-jump KS = 1.000`
- `path q90 ratio = 0.077`
- `path q99 ratio = 0.125`
- `kurtosis ratio = 0.068`

Interpretation:
- the model suppresses almost all extreme motion
- it produces a near-factorless / over-high-rank dynamic
- direct raw panel prediction is too hard in this form

So the problem is not the world-model idea itself.
It is the **direct observation-space parameterization**.

### What This Rules Out
Do not continue local tuning on:
- raw direct next-change target
- same direct observation-space world-model parameterization
- larger/smaller Transformer-only retries without changing the state representation

### Decision
Keep the Stage A world-model paradigm.

Change the state representation.

Next step: `289b-v0`
- deterministic latent world model with a learned observation bottleneck
- learn:
  - encoder from panel state to latent state
  - autoregressive latent transition model
  - decoder from latent state back to panel level / change

This keeps the parametric world-model idea while removing the hardest part of `289a`:
- direct joint observation-space dynamics learning from raw panel state
