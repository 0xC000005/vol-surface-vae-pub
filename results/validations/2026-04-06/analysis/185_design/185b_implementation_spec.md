# 185b Implementation Spec

## Branch

`185b = graph/group event-residual decomposition`

## Motivation

The focused `185a` mechanism review showed:

- the activity gate stays alive
- but the event-path local amplitude is effectively near zero on the hard turbulent late-horizon slices
- prior and posterior look nearly identical
- turning the event path off improves `worstLate` and `S8`, so the current event branch perturbs the quiet backbone more than it helps

So `185b` should not add another gate variant. It should make the event branch learn an explicit residual event path target.

## Keep

- explicit mean branch
- explicit covariance branch
- `183c` quiet pathwise residual-law backbone
- graph/group-aware latent activity process from `185a`

## Change

- build a sparse teacher event decomposition from the target residual path
- train the event branch against that target in basis space with a direct event flow-matching loss
- warm-start from `185a` and keep the learned activity/event stack instead of reinitializing it

## Teacher Event Decomposition

1. Compute `target_basis`
2. Compute `target_local_log`
3. Compute `target_event_local = target_local_log - quiet_metric_local`
4. Build sparse top-k positive / negative support from `target_event_local`
5. Convert that support into a block-weighted teacher mask
6. Define:

`target_event_white = target_white * teacher_mask`

`target_event_basis = to_basis(target_event_white)`

This makes the event branch own an explicit sparse residual event path instead of only a weak modulation field.

## New Loss

Add:

`event_residual_fm_loss = MSE(event_scale * event_transport_velocity, target_event_velocity)`

where `target_event_velocity` is the flow-matching target for `target_event_basis`.

Keep the existing:

- total path FM loss
- quiet local/band losses
- event local/band losses
- activity support / KL / BCE terms

## Training

- stage 1: train event/activity stack only
- stage 2: light joint tuning with path-context adapter
- warm-start from `185a_best`

## Success Criterion

`185b` is only interesting if it improves at least one of:

- `S3` turb/calm width separation
- `S7` layer-2 regime-by-cell coverage
- tightened `S4` kurtosis / spectrum

without giving back:

- `S2`
- `S8`
- `S10`
- `S11`
