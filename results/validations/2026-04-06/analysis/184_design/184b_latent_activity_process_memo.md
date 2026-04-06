# 184b: Latent Activity-Process Transport

## Motivation

`184a` kept the generalized latent-activity direction viable, but the activity gate
was still trained directly against a soft teacher activity target.

That left one structural gap:

- activity was still a per-window guided control rather than a true latent process

## Principle

Keep:

- explicit mean branch
- explicit covariance branch
- geometry-aware pathwise residual transport

Replace:

- `184a`'s guided activity gate

With:

- a learned latent residual activity **process**
  - prior over activity from history/context/state
  - posterior over activity from target residual structure during training
  - KL alignment between posterior and prior
  - prior-only activity at inference

## Why this is more generalized

This keeps the abstraction generic:

- the current IV surface is only the first geometry instance
- later grouped-factor systems can reuse the same latent activity-process interface
- only the geometry/group metadata changes

## Expected behavior

If `184b` works, it should:

- preserve the broad `183c/184a` strengths
- improve `S3` and `S7`
- do so without another teacher-guided static allocation patch

## Result

`184b_v0` is a stable generalized branch, but not a frontier break.

Best checkpoint:

- [184b_best_v2_s3mrj_full_30d/summary.json](/home/max/Documents/vol-surface-vae-pub/results/block_ar/184b_best_v2_s3mrj_full_30d/summary.json)

Final checkpoint:

- [184b_final_v2_s3mrj_full_30d/summary.json](/home/max/Documents/vol-surface-vae-pub/results/block_ar/184b_final_v2_s3mrj_full_30d/summary.json)

Interpretation:

- the latent process trains stably
- it preserves broad realism
- but this v0 still does not solve the remaining `S3/S7` hard-slice problem
