# 274b Ideation

## Why 274a Is Still Alive
- `274a` fixed the central `273a` pathology: generated paths no longer leave support immediately, and surface validity now passes cleanly.
- The posterior/decoder pair also remains usable, so the explicit observation model should be kept.
- The remaining failure concentrates in the one-step prior: samples are too narrow, too smooth, and too common-mode.

## Next Family
`274b-v0`: probabilistic latent-token state-space model with short-memory latent prior

## Minimal Change
- Keep:
  - posterior `q(z_t | x_t)`
  - decoder `p(x_t | z_t)`
  - tokenized latent state
  - autoregressive ELBO training
- Change only:
  - prior from `p(z_t | z_{t-1})`
  - to `p(z_t | z_{t-k:t-1})` with a short latent history buffer

## Why This Is the Smallest Principled Extension
- `274a` already established that a probabilistic latent manifold is necessary.
- The new evidence says the prior is too memoryless, not that the whole family is wrong.
- Giving the prior a short latent memory is a direct response to:
  - under-dispersion
  - weak regime width allocation
  - over-common-mode rank collapse
  - over-strong mean reversion

## Constraints
- no hard low-rank head
- no bounded side paths
- no EC baseline
- no KL warmup, free bits, or auxiliary collapse patches unless the new family actually collapses

## Kill Criteria
- if KL collapses after adding memory, close the family quickly
- if rank ratio and coverage do not improve together, do not add more memory machinery
- if `274b` preserves surface validity but still leaves `prior_step1_std << gt_step1_std`, the prior family is too narrow and a broader transition class is needed

## Next Step
Implement `274b-v0` with the same observation model and a short latent-memory prior, then rerun the full train/eval/postmortem loop.
