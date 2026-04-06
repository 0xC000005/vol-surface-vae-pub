# 184c Implementation Spec: State-Dependent Sparse Precision Transport

## Why 184c

After tightening `S4` in the benchmark, the current anchor
[183c_best_v2_s3mrjspec_full_30d/summary.json](/home/max/Documents/vol-surface-vae-pub/results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json)
now fails:

- `S3`: conditional width is still too weak on hard turbulent slices
- `S4`: tail concentration is still wrong
- `S7`: regime-by-cell coverage still misses the hard late-horizon cells

Under the stricter `S4`, the failure is explicit:

- kurtosis ratio: `0.507`
- quiet mass ratio: `0.831`
- shoulder mass ratio: `1.144`
- extreme mass ratio: `1.322`

So the remaining issue is not just “activity detection.”
It is that residual mass is still allocated too diffusely:

- not enough quiet mass on most cells / horizons
- too much shoulder mass spread broadly
- not enough concentrated width on the true hard slices

## Research Takeaways

The targeted literature supports keeping the current path and changing the transport
operator, not the entire model class.

### 1. Keep explicit mean/covariance structure

- CW-Gen: conditional whitening improves generation when conditional mean/covariance
  information is already informative.
  - https://arxiv.org/abs/2509.20928
- TSFlow: data-dependent priors simplify the transport problem for time series.
  - https://arxiv.org/abs/2410.03024

Implication:

- keep the current explicit mean branch
- keep the current structured covariance branch
- keep residual generation in whitened space

### 2. Keep joint pathwise conditional generation

- ProFITi: conditional flows are useful because they model joint future laws rather
  than fixed parametric marginals.
  - https://arxiv.org/abs/2402.06293
- Conditioned normalizing flows for multivariate forecasting support the same
  direction at larger multivariate scale.
  - https://arxiv.org/abs/2002.06103

Implication:

- do not go back to independent per-cell or per-horizon fixes
- keep one joint residual path law

### 3. Make the geometry abstraction real

- “Connecting the Dots” shows learned graph structure is a principled way to model
  dependencies in multivariate time series.
  - https://arxiv.org/abs/2005.11650
- BernNet shows graph spectral filters can be learned flexibly without fixing one
  oversimplified filter shape.
  - https://arxiv.org/abs/2106.10994

Implication:

- IV now uses a grid graph
- later multi-factor systems can reuse the same operator on a grouped graph

### 4. Sparse support is the right inductive bias

- Sparsemax provides exact-zero selective support instead of dense soft assignments.
  - https://arxiv.org/abs/1602.02068
- Entmax generalizes that idea and allows tunable sparsity with differentiable support.
  - https://arxiv.org/abs/1905.05702

Implication:

- if the model needs concentrated residual allocation, the allocator should be able to
  set many locations to exact zero
- dense soft modulation is the wrong bias for the current failure mode

### 5. Observable/contextual activity structure is still principled

- BCT-SSM supports the idea that time-series mixture/state structure can be imposed
  through observable/contextual partitions without data-specific hand rules.
  - https://arxiv.org/abs/2106.03023

Implication:

- the current latent activity-process direction remains valid
- but activity alone is not enough; it must drive a sharper operator

## Core Principle

`184c = 184b backbone + state-dependent sparse transport kernel / precision field`

Keep:

- explicit mean-reverting mean branch
- structured covariance branch
- whitened pathwise residual law
- geometry abstraction from [activity_geometry.py](/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/activity_geometry.py)
- latent activity process from `184b`

Change:

- replace diffuse metric modulation with a **sparse local operator**
- the operator decides how residual energy moves across `time x node`
- it must be capable of exact-zero support outside a small active neighborhood

This is the narrowest general fix that matches the diagnosis.

## What “Sparse Precision Transport” Means

At each transport step, conditioned on `(z_t, t, context, activity state)`:

1. produce a node activity budget
2. produce a sparse local kernel over each node’s neighborhood
3. turn that into a positive local precision / diffusion operator
4. apply transport in that state-dependent geometry

Conceptually:

- quiet nodes should stay close to identity transport
- active nodes should receive concentrated radial energy
- neighbors can share some energy, but only through a sparse local kernel

This is more general than an IV-specific patch:

- on the IV surface: neighborhood = grid adjacency
- on grouped factors later: neighborhood = graph adjacency

## Minimal `v0` Design

### Geometry

Extend [activity_geometry.py](/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/activity_geometry.py):

- expose adjacency for the current `5x5` grid
- allow later injection of group / graph adjacency
- expose neighborhood index lists for sparse local kernels

### Sparse kernel head

New head:

- input: node state features from current `184b` path context and latent activity state
- output: neighborhood logits per node
- transformation: `entmax15` or `sparsemax` over each local neighborhood

Why:

- exact-zero support
- differentiable
- better aligned with “quiet most of the time, concentrated when active”

### Node budget head

Separate positive head:

- input: same node state features
- output: positive event strength per node
- normalized by block/time budget

This splits:

- where support lives
- how much local strength it receives

### Precision / diffusion operator

Build a state-dependent local operator:

- `K_t(node -> neighbor)` from sparse kernel
- `a_t(node)` from positive budget
- operator is identity plus a sparse graph-Laplacian-style correction

High-level form:

`P_t = I + lambda * L_t`

where `L_t` is a sparse positive semidefinite operator induced by the kernel and budgets.

Use `P_t` to precondition the transport velocity or residual basis coefficients.

The point is not the exact algebraic form.
The point is:

- local energy reallocation happens through the operator itself
- not through a tiny additive control field afterward

### Quiet background

Keep a quiet branch:

- if activity is low, transport remains close to the current `183c/184b` behavior
- the sparse operator only matters when activity is high

This protects:

- `S2`
- `S8`
- `S10`
- `S11`

## Losses

Keep current backbone losses and add only generic residual-geometry terms.

### Keep

- flow matching loss
- conditional width / frontier subset selection
- current activity-process KL structure

### Add

1. Sparse support regularizer

- encourage small support size when activity is low or moderate
- do not force sparsity on every window equally

2. Quiet / shoulder / extreme spectrum loss

- differentiable penalty on pooled exceedance-spectrum mismatch
- use GT-derived thresholds
- targets:
  - quiet mass up
  - shoulder mass down
  - extreme mass not broadly inflated

This is generic distribution-shape control, not an IV-specific hack.

3. Neighborhood concentration loss

- encourage target cells on hard slices to dominate neighbor spillover
- use teacher target only as a training aid in the single-surface `v0`
- keep it local and generic:
  - target node should exceed neighborhood average when target allocation is positive

## Training Plan

Warm start from:

- [184b best checkpoint](/home/max/Documents/vol-surface-vae-pub/models/backfill/latent_activity_process_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_184b/best_model.pt)

### Stage 1

- freeze mean branch
- freeze covariance branch
- freeze most of path transport
- train:
  - sparse kernel head
  - node budget head
  - activity-process coupling layers

Goal:

- learn selectivity without destroying the broad anchor behavior

### Stage 2

- unfreeze residual transport
- joint fine-tune with small LR

Goal:

- let transport adapt to the new geometry without relearning the backbone

## Selection Rule

Primary ranking:

1. `S3`
2. `S7`
3. tightened `S4` (`kurtosis + spectrum`)
4. keep `S2/S8/S10/S11`

So this branch only counts as success if it improves concentration
without giving back broad realism.

## Expected Win Condition

`184c_v0` is successful if it:

- keeps `S2` pass
- keeps `S8` pass
- keeps `S10` pass
- keeps `S11` pass
- materially improves at least one of:
  - `S3` turb/calm width ratio
  - `S7` Layer 2 regime×cell pass count
  - tightened `S4` quiet/shoulder/extreme spectrum

## Why This Is General

This is not “fix the IV surface.”

It is a reusable operator class:

- explicit mean for drift
- explicit covariance for second order
- residual transport in whitened space
- state-dependent sparse geometry operator over nodes

The only thing that changes across domains is the geometry:

- single IV surface now: grid graph
- multi-factor later: grouped graph

So the mechanism is generic by construction.

## File Plan

New / changed files for `184c_v0`:

- `/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/activity_geometry.py`
- `/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/train_184c_sparse_precision_transport.py`
- `/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/test_block_ar_requirements_v2.py`

## Decision

The next principled build is:

**`184c_v0`: latent activity-process transport with a state-dependent sparse precision operator.**

That is the narrowest operator-level change supported by:

- the stricter benchmark
- the mechanistic reviews
- and the external literature on conditional whitening, joint flows, graph structure, and sparse support.
