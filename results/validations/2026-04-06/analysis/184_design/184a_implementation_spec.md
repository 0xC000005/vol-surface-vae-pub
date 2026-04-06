# 184a_v0 Implementation Spec

## Goal

Implement the first **single-surface IV instance** of a broader
graph/group-aware latent residual activity model.

Immediate objective:

- improve the remaining `S3/S7` concentration failure from `183c`

without turning the solution into an IV-only patch.

## Anchor

Warm start from:

- [183c_best_v2_s3mrj_full_30d/summary.json](/home/max/Documents/vol-surface-vae-pub/results/block_ar/183c_best_v2_s3mrj_full_30d/summary.json)

## Core design

Keep:

- explicit mean branch
- explicit covariance branch
- pathwise residual-law transport in whitened geometry-aware space

Add:

- latent residual activity state
- reusable group-aware geometry interface

## Geometry interface

Use:

- [activity_geometry.py](/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/activity_geometry.py)

Current v0 instance:

- one IV surface group only

Future extension:

- multiple groups via `group_ids`
- same latent activity mechanism can then operate on grouped factor systems

## 184a model

### Quiet branch

Use the existing `183c`-style state metric controls:

- smooth background allocation
- state-conditioned local and band metrics

### Event branch

Add an explicit latent activity branch with:

- activity gate `p(event | context, state)`
- activity budget
- group router
- local positive/negative event allocator

For v0:

- one group => group router is structurally trivial but already part of the interface

### Final transport

`metric_local = quiet_metric + event_gate * event_budget * event_metric`

with:

- quiet branch preserved
- event branch centered
- local event allocation sparse over `horizon x cell`

## Teacher target for activity

Do not use manual labels.

Use a soft teacher activity target from the teacher residual allocation:

- concentration score from top-k mass of `|target_local_log|`
- amplitude score from target local max / mean
- combined soft target in `[0,1]`

This is a learned residual-activity target, not a hand-made turbulence label.

## Training

### Stage 1

Freeze:

- encoder
- decoder
- covariance branch
- flow
- prior
- path transport

Train only:

- activity gate
- activity budget
- group router
- event allocator

### Stage 2

Unfreeze lightly:

- path transport
- path context adapter

Keep backbone frozen.

## Losses

Base:

- flow matching loss
- local control loss
- band control loss

Add:

- activity gate loss to teacher activity target
- event support loss on sparse local allocation
- mild event-rate regularizer toward teacher event frequency

Do **not** use a generic “don’t use events” penalty.

## Checkpoint selection

Use existing frontier key.

Interpretation priority:

1. preserve `S2`, `S8`, `S10`, `S11`
2. improve `S3`
3. improve `S7`

## Expected success condition

`184a_v0` is successful if it:

- preserves the broad `183c` pass profile
- improves local concentration enough to move `S3` and/or `S7`
- does not regress on `S4` or `S8`

## Files

- new: [activity_geometry.py](/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/activity_geometry.py)
- new: [train_184a_latent_activity_transport.py](/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/train_184a_latent_activity_transport.py)
- update loader: [test_block_ar_requirements_v2.py](/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/test_block_ar_requirements_v2.py)
