# 258a Follow-Up Analysis
## Core Finding
The fast stochastic branch collapsed almost immediately and stayed effectively dead for the full path. The surviving model is dominated by a slow stochastic branch with monotone long-horizon spread accumulation.
## Evidence
- collapse epoch (`fast_scale_mean < 1e-3`): `2`
- best val fast scale mean: `0.00000779`
- best val slow scale mean: `0.01040985`
- best val common RMS: `0.00698608`
- best val idio RMS: `0.00070313`

### Per-Horizon Means
- `slow_scale_mean` every 5 steps: 0.0389 0.0001 0.0002 0.0008 0.0037 0.0327
- `fast_scale_mean` every 5 steps: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
- `fast_gate_mean` every 5 steps: 0.0183 0.0000 0.0001 0.0005 0.0015 0.0072
- `sample_std_mean` every 5 steps: 0.0009 0.0053 0.0093 0.0114 0.0144 0.0179
- `common_rms` every 5 steps: 0.0017 0.0017 0.0017 0.0125 0.0124 0.0037
- `idio_rms` every 5 steps: 0.0003 0.0003 0.0003 0.0013 0.0010 0.0003

## Mechanistic Conclusion
`258a-v0` is not failing because the whole stochastic family is dead. It is failing because training routes almost all useful variance into the slow branch and collapses the fast branch before it can learn local shocks. That gives good long-horizon coverage and cross-cell rank, but wipes out jump realism and mean-reversion strength.

## Most Principled Next Step
Stay in the `258` family and design `258b` as an explicit anti-collapse fast-branch follow-up: add a fast-scale floor/penalty, or a fast-branch-specific reconstruction target on short-horizon changes / pathwise jumps, while preserving the slow branch that is already giving useful long-horizon coverage.
