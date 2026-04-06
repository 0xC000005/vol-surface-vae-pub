# 187a Sparse-Support Residual Implementation Spec

## Objective

Build the first non-smooth residual model class after the `183c -> 186a` plateau.

Keep:

- explicit mean-reverting mean branch
- explicit structured covariance branch
- support-aware outputs

Replace:

- smooth residual transport as the primary concentration mechanism

With:

- a **quiet background residual path**
- a **latent sparse support process** over `time x node`
- an **event amplitude law** defined only on active support

The target is to address the unresolved strict-benchmark frontier:

- `S3` local conditional width concentration
- `S7` regime-by-cell hard-slice allocation
- tightened `S4` quiet / shoulder / extreme concentration

without giving back:

- `S2`
- `S8`
- `S10`
- `S11`

## Why This Class, Not Another Transport Variant

The closed `183c -> 186a` family showed:

- the model often knows roughly where stress should go
- but smooth residual laws keep smearing stress into neighbors
- objective reweighting alone widens too broadly

So the next model must make **selective concentration part of the generative structure**, not a smooth correction on top of one diffuse law.

This is the narrowest model-class change that still preserves:

- the validated mean/covariance backbone
- end-to-end likelihood training
- reuse beyond IV surfaces

## Model Class

Name:

- `187a = latent sparse-support residual model`

Single-surface IV v0 instance:

- `grid` geometry
- support over `time x node`

Generalized form later:

- support over `time x node x group`
- same latent mechanism on graph/group geometry

## Core Latent Variables

### 1. Quiet background residual path

`q_t`:

- low-amplitude residual path
- captures ordinary quiet-day residual behavior
- should dominate most windows

### 2. Sparse support mask

`s_tn in {0,1}`:

- latent activity indicator for time `t` and node `n`
- determines where event residual can be nonzero

Interpretation:

- most entries off
- a small structured subset on during event-like windows

### 3. Event amplitude path

`e_tn`:

- event residual amplitude
- only contributes where support is active

Combined residual:

`r_tn = q_tn + s_tn * e_tn`

This is the key structural change:

- quiet path handles ordinary residual law
- sparse event path handles concentrated stress

## Geometry Abstraction

### v0

Use current single-surface grid geometry:

- 30 horizons x 25 cells flattened as `time x node`
- local neighborhood defined on the 5x5 surface grid at each horizon

### later

Replace grid adjacency with graph/group adjacency:

- groups like `SPX vol`, `commodity vol`, `rates`, `FX`
- node-level and group-level support priors

So the sparse-support mechanism is reusable:

- geometry changes
- model class does not

## Generative Structure

Given history `H`:

1. mean branch produces deterministic conditional mean path `mu(H)`
2. covariance branch produces conditional covariance structure `Sigma(H)`
3. whitened residual generator produces:
   - quiet path `q`
   - support logits `logit_s`
   - event amplitudes `e`
4. final whitened residual path:
   - `r = q + s * e`
5. map back through covariance and support transform to output space

This preserves the proven separation:

- mean branch owns drift / mean reversion
- covariance branch owns second order
- sparse residual branch owns concentrated higher-order path behavior

## Support Process Design

### v0 requirement

The support process must not collapse to:

- always off
- always on
- diffuse weak activation everywhere

So v0 should use:

- a **latent posterior / prior pair**
- support-rate regularization
- support-size regularization
- neighborhood-aware total-variation regularization

### recommended parameterization

Support logits come from:

- history-conditioned context
- optional quiet-path state summary
- horizon embedding
- node embedding

Then:

- sample relaxed Bernoulli / BinaryConcrete during training
- hard threshold only for diagnostic analysis

Why relaxed support first:

- keeps the model end-to-end differentiable
- avoids discrete optimization complexity in v0

## Event Amplitude Law

Event amplitudes must carry real magnitude, unlike `185a`.

So:

- event amplitude network outputs signed amplitudes in whitened space
- scale must not be multiplicatively tied to a near-zero gate
- amplitude supervision should be active whenever teacher event decomposition is nonzero

Design rule:

- support decides **where**
- amplitude decides **how much**

Do not let the same scalar gate control both.

## Teacher Decomposition for v0

Use a simple teacher residual split for training guidance only:

1. compute teacher whitened residual path from conditional mean/covariance
2. define quiet baseline via robust local smooth component
3. define event teacher residual as the sparse leftover above a threshold or local contrast criterion
4. train:
   - support logits toward teacher support
   - event amplitudes toward teacher event residual on active support

Important:

- teacher split is a training scaffold, not the final probabilistic claim
- at inference, support and event amplitudes are fully model-generated

## Training Objective

Total objective should contain:

### 1. Background residual loss

For quiet path:

- standard flow / residual matching on low-amplitude residual structure

### 2. Support supervision loss

For sparse support:

- BCE or focal-style loss to teacher support mask
- positive support weighting for rare active points

### 3. Event amplitude loss

On active support only:

- weighted L1 / Huber / MSE toward teacher event residual

### 4. Support regularization

- target active-rate prior
- sparsity penalty
- local TV penalty over `time x neighborhood`

### 5. Quiet / shoulder / extreme spectrum loss

Retain the stricter `S4` concern directly:

- quiet mass
- shoulder mass
- extreme mass

But apply it to the **combined residual law** after decomposition.

### 6. Mean/cov protection

Keep:

- backbone frozen in stage 1
- light joint fine-tune only in stage 2

This protects:

- `S10`
- `S2/S8`
- broad structure

## Training Stages

### Stage 1: sparse residual head only

Freeze:

- encoder
- decoder
- flow
- prior
- quiet backbone transport

Train:

- support head
- event amplitude head
- any small support-context adapter

Goal:

- verify support does not collapse
- verify event amplitude is nontrivial on teacher-event points

### Stage 2: light joint fine-tune

Unfreeze:

- sparse residual head
- optionally path-context adapter
- optionally a small residual combiner

Keep core backbone mostly fixed.

Goal:

- integrate event path without losing `S2/S8/S10/S11`

## Checkpoint Selection

Use strict benchmark priorities, not loss only.

Order:

1. `S3`
2. tightened `S4`
3. `S7`
4. `S2`
5. `S8`
6. `S10`
7. `S11`
8. val loss as tie-breaker

Rationale:

- this phase exists to solve concentration
- broad realism is already available in the anchor

## File Plan

### new

- `experiments/backfill/block_ar/train_187a_sparse_support_residual.py`
- `results/validations/2026-04-06/analysis/187_design/187a_sparse_support_residual_implementation_spec.md`

### likely reuse

- `experiments/backfill/block_ar/train_183c_state_metric_transport.py`
- `experiments/backfill/block_ar/train_186a_hard_slice_tail_objective.py`
- `experiments/backfill/block_ar/activity_geometry.py`
- `experiments/backfill/block_ar/basis_geometry.py`

### harness

- `experiments/backfill/block_ar/test_block_ar_requirements_v2.py`

## v0 Acceptance Criteria

The first acceptable sign of life is not full frontier break. It is:

- support does not collapse
- event amplitude is materially nonzero on hard slices
- event-off ablation now hurts the model

Then benchmark-wise, v0 should ideally:

- improve `S3`
- improve `S7`
- improve tightened `S4`
- without breaking `S2/S8/S10/S11`

If support stays alive but amplitudes still collapse, then the next issue is amplitude parameterization.

If support becomes diffuse / always on, then the issue is support prior / regularization.

If support is selective and amplitudes are real but `S3/S7` still fail, then the limitation is deeper than the current sparse-support parameterization.

## Decision

This is the next coding phase if research continues.

Do not return to another smooth transport variant before testing this class.
