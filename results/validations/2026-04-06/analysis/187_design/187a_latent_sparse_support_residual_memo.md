# 187a Latent Sparse-Support Residual Memo

## Phase Reset

The smooth-transport family is closed.

Current strict anchor:

- `results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json`

Reason for reset:

- repeated model and objective changes preserved broad realism
- but failed to solve selective concentration on rare hard slices
- the recurring failure is now interpreted as a smoothness bias in the current model family

## New Direction

Move to a **latent sparse-support residual model**.

Keep:

- explicit mean dynamics
- explicit covariance dynamics

Change:

- replace smooth residual transport as the primary concentration mechanism
- model residual activity with explicit sparse support

Core idea:

- most nodes are quiet most of the time
- a small latent support activates during event-like conditions
- residual amplitude is generated only on active support
- support lives over `time x node` now, and later `time x node x group`

## Why This Is Different

The old family tried to sharpen one smooth residual law.

The new family should let the model represent:

- quiet background residual structure
- sparse selective event support
- event amplitude on active support

That makes concentrated residual allocation part of the generative structure itself.

## Generality

This is not IV-specific.

Geometry should be abstract:

- grid for the single-surface IV instance
- graph/group structure for multi-factor systems later

The sparse-support idea is generic across financial factor systems because intermittent residual activity is generic.

## Immediate Implication

Do not run more smooth transport variants.

If work continues from here, the next step should be:

1. full design/spec for the sparse-support residual class
2. single-surface IV implementation as the first instance
3. later extension to graph/group geometry
