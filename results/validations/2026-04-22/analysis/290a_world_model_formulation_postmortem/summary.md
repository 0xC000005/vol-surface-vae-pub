## 290a-v0 Postmortem

### Result
- `290a-v0` trained stably but did not produce a valid full `11`-suite result.
- The full evaluator crashed in suite 9 (`cross_cell_correlation`) with:
  - `LinAlgError: Eigenvalues did not converge`

### Context
`289e` established that the Stage A world-model backbone was structurally viable:
- shared deterministic structure near gate
- active mean-reversion support near gate
- remaining blocker shifted to oversmoothed local move-size law

`290a` kept the `289e` history-memory backbone and changed only the next-step output formulation:
- discretized normalized-change bins
- teacher-forced cross-entropy target
- continuous rollout during training via expected-bin decode
- hard argmax decode at inference

### Training summary
- best epoch: `5`
- best val total: `2.940`

### What improved before evaluator failure
The partial full-suite output before the crash already showed that the formulation shift was directionally real:
- ACF corr: `0.832`
- kurtosis ratio: `0.402`
- level KS pass cells: `13/25`
- level median bias pass cells: `21/25`

So the discrete target improved deterministic level-side fidelity and temporal smoothness proxies.

### Failure mechanism
The evaluator crash was not caused by NaNs in the generated surfaces themselves.

Direct probes showed:
- hard rollout outputs were finite
- hard rollout std was nontrivial: `0.0969`
- hard decode used `43` distinct values
- no cell was globally zero-variance in a small probe

But the cross-cell suite produced invalid correlation matrices under hard decode:
- `numpy` emitted divide-by-zero warnings during correlation computation
- the suite then failed at eigendecomposition

The clean diagnosis is:
- training rollout used **soft expected-bin decode**
- inference used **hard argmax decode**
- that decode mismatch produced a brittle, piecewise-constant sample law with degenerate correlation slices

### What 290a means
`290a` does **not** falsify the discrete-output formulation itself.

It falsifies this narrower deployment choice:
- discrete-output world model
- trained with soft rollout
- evaluated with hard argmax decode

### Decision
Keep the `289e` backbone and the `290a` discrete target alive, but test the smallest formulation-consistent fix:

Next step: `290b-v0`
- same trained checkpoint class
- same discrete target
- same world-model backbone
- change inference/sample decode to expected-bin-center (`soft`) to match training rollout

### Kill criteria for 290b
`290b` is only alive if it:
- completes the full `11`-suite without support/correlation failure
- preserves or improves the `290a` local-law gains
- keeps cross-cell structure near the `289e` regime instead of collapsing again
