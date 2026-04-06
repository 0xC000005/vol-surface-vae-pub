# 185a Implementation Spec

## Branch

`185a_v0 = graph/group-aware latent event-path residual law`

Single-surface IV instance first, with graph/group abstraction kept in the code path.

## Backbone

Keep from the `183c` anchor:

- explicit mean-reverting mean branch
- structured covariance branch
- whitened pathwise residual-law backbone
- state-conditioned quiet transport

Warm start from:

- [183c best checkpoint](/home/max/Documents/vol-surface-vae-pub/models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt)

## New `185a_v0` components

- latent blockwise activity prior/posterior over the future path
- sparse positive/negative node support allocator
- graph-propagated event local field in whitened node space
- separate event-path transport network
- additive residual decomposition:
  - quiet background path
  - event-path contribution

## Concrete decoder

At each transport step:

1. build quiet controls from the `183c` state-metric backbone
2. infer latent activity state from context and optional posterior teacher signal
3. decode sparse event support over `time x node`
4. decode positive event amplitudes
5. propagate the event seed over the geometry graph
6. decode event band shifts
7. generate:
   - quiet transport velocity
   - event transport velocity
8. add them:
   - `v_total = v_quiet + event_scale * v_event`

## Training plan

Stage 1:

- freeze mean/covariance backbone
- freeze quiet path transport
- train only:
  - latent activity process
  - sparse support allocator
  - event amplitude head
  - event band head
  - event transport
  - event scale

Stage 2:

- keep quiet path transport frozen
- unfreeze only the path-context adapter for light joint adjustment

## Losses

- conditional flow matching on the full path
- quiet local/band reconstruction losses
- event local/band residual-gap losses
- event support loss against top-k teacher targets
- latent activity KL / budget consistency / group KL
- posterior/prior BCE to event targets
- small event amplitude regularizer

## Success target

Improve the remaining stricter frontier:

- `S3`
- `S4`
- `S7`

while preserving:

- `S2`
- `S8`
- `S10`
- `S11`
