# 185a Memo: Graph/Group-Aware Latent Event-Path Residual Law

## Status

This memo starts the **new model-class phase** after the `184x` operator/gate family plateau.

It is different from the `184x` line in one decisive way:

- `184x` kept one residual transport law and tried to sharpen it with gates, operator swaps, or quiet/event mixtures.
- `185a` treats **residual event structure itself as a first-class latent path** over `time x node x group`.

So this is not another operator variant.

## Why the `184x` family is done

The current stricter anchor is still:

- [183c_best_v2_s3mrjspec_full_30d/summary.json](/home/max/Documents/vol-surface-vae-pub/results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json)

The cleanest final attempt in the family was:

- [184d_best_v2_s3mrjspec_full_30d/summary.json](/home/max/Documents/vol-surface-vae-pub/results/block_ar/184d_best_v2_s3mrjspec_full_30d/summary.json)

What remained true across `183c -> 184d`:

- `S3` still fails because hard turbulent late-horizon cells do not get enough concentrated width.
- `S7` still fails because regime-by-cell local allocation remains too diffuse.
- tightened `S4` still fails by business standard because quiet mass is too low and shoulder mass is too high.

Interpretation:

- the model already has broadly correct mean/covariance structure
- it already has useful state signal
- but the current residual module still does **not** represent sparse intermittent event structure as a core latent object

That is why more gates and operator heads are no longer the right path.

## Online Research Takeaways

I did the missing online ideation pass for the new model class. The main sources and takeaways are:

### 1. Keep explicit mean/covariance conditioning

- CW-Gen: conditional whitening is a principled way to separate mean/covariance structure from higher-order generation.
  - https://arxiv.org/abs/2509.20928
- TSFlow: data-dependent priors make path generation easier when temporal structure is already partially known.
  - https://arxiv.org/abs/2410.03024

Implication:

- keep the current explicit mean-reverting mean branch
- keep the structured covariance branch
- keep generation in whitened residual space

### 2. Keep one joint future-path generator

- ProFITi: conditional flows over the joint future law are better aligned with multivariate forecasting than fixed-shape marginals.
  - https://arxiv.org/abs/2402.06293
- CANF: joint multi-step conditional flows are designed for correlated multi-step forecast laws.
  - https://arxiv.org/abs/2201.02753
- MOSES: when the factor set grows, marginalization consistency matters across subsets, not just full-joint fit.
  - https://arxiv.org/abs/2406.07246

Implication:

- do not split the problem into independent cells or horizons
- keep one joint residual path law

### 3. Make event structure a first-class latent sequence

- Add and Thin: whole-sequence diffusion for temporal point processes shows that event sequences should be modeled as sequence objects, not just one-step corrections.
  - https://arxiv.org/abs/2311.01139
- Conditional Generative Modeling for High-dimensional Marked Temporal Point Processes: high-dimensional event marks can be generated directly without forcing a low-capacity parametric intensity form.
  - https://arxiv.org/abs/2305.12569
- Neural Jump SDEs: continuous evolution plus discrete jumps is a natural hybrid latent dynamics model.
  - https://arxiv.org/abs/1905.10403
- Generative modelling with jump-diffusions: jump-aware generative processes are a principled way to capture heavy-tailed behavior and rare events.
  - https://arxiv.org/abs/2503.06558

Implication:

- the next model should not bolt jumps onto a residual flow
- it should learn a latent **event-path process** jointly with the residual path

### 4. Geometry should be abstract, not IV-specific

- Connecting the Dots: graph structure is the right abstraction for multivariate dependencies.
  - https://arxiv.org/abs/2005.11650
- BernNet: local structure should be able to vary by region/node rather than using one homogeneous filter everywhere.
  - https://arxiv.org/abs/2106.10994
- Marked Neural Spatio-Temporal Point Process with Dynamic GNN: dynamic graph event processes are a natural model class for graph-structured events.
  - https://arxiv.org/abs/2206.03469
- Graph Regularized Point Process: event propagation can be regularized and interpreted on a graph of interacting nodes.
  - https://arxiv.org/abs/2211.11758

Implication:

- IV surface now = grid graph instance
- future multi-factor setup = grouped graph with surface groups, cross-group edges, and local nodes

## Core Design

`185a = graph/group-aware latent event-path residual law`

Keep:

- explicit mean branch
- explicit covariance branch
- whitened residual target
- geometry abstraction

Replace:

- residual transport with optional activity/operator modulation

With:

- a **joint latent event-path process**
- a **smooth residual path**
- a **graph/group-aware event propagation decoder**

In other words:

`future residual path = smooth background path + event-path contribution`

where the event-path contribution is a learned latent sparse process over the future path, not a post-hoc control field.

## Latent Variables

For each future block or step:

- `z_t^smooth`: continuous smooth residual state
- `z_t^event`: latent event activity state
- `m_t`: sparse event support over nodes or node-groups
- `a_t`: event amplitudes / marks in whitened geometry-aware space

Conceptually:

- `z_t^smooth` handles ordinary residual evolution
- `z_t^event` decides whether event structure is active and what type it is
- `m_t` decides where event mass lives
- `a_t` decides how large and in what direction it moves

This is cleaner than `184x` because concentration is now part of the generative object itself.

## Geometry Abstraction

The design is intentionally reusable beyond IV surfaces.

### Single-surface IV instance

- nodes = `5 x 5` surface cells
- graph = grid adjacency + optional spectral basis
- groups = optional coarse surface regions / horizon blocks

### Future multi-factor extension

- nodes = factors or factor-cells
- groups = `SPX surface`, `commodity surface`, `rates`, `FX`, etc.
- graph = within-group adjacency + cross-group edges

So the same machinery can represent:

- local events within one surface
- group-specific events
- cross-group co-events

## Why this is different from `184x`

`184x` tried to answer:

- "given one residual law, how do we sharpen or activate it?"

`185a` instead answers:

- "what is the latent residual event path that generates the concentrated part of the future law?"

That is a model-class difference, not a parameter tweak.

## First `v0` Recommendation

The first implementation should stay conservative:

### Backbone

- warm-start from the current best generalized anchor
- keep mean/covariance modules fixed initially

### Event-path prior

- blockwise latent event activity process
- sparse node/group support per block
- graph-aware support propagation

### Event-path inference

- posterior over event support and amplitudes from whitened residual targets during training
- prior from history/context only at generation time

### Residual decoder

- smooth background path via conditional flow matching in whitened space
- event decoder adds sparse graph-propagated event contributions

### Training

1. fit posterior event inference and event decoder against backbone residual errors
2. freeze mean/covariance at first
3. then lightly joint-tune

This preserves the existing wins while testing the new class cleanly.

## Why this remains principled

This is still an elegant separation:

- mean branch owns drift / mean reversion
- covariance branch owns second order
- event-path residual law owns intermittent concentrated higher-order structure

It is not IV-specific because:

- no hand-coded "turbulent cell" rules
- no fixed quiet/event labels
- no per-cell hacks
- geometry is abstracted

The inductive bias is generic:

- financial residual paths are mostly quiet
- occasionally they activate concentrated event structure
- that structure propagates over related nodes and groups

## Success Criteria

`185a_v0` is successful if it improves the remaining stricter frontier together:

- `S3`
- `S4`
- `S7`

while preserving:

- `S2`
- `S8`
- `S10`
- `S11`

In business terms:

- more quiet mass
- less shoulder-heavy residual behavior
- sharper concentration on true hard slices
- without breaking broad realism

## Failure Interpretation

If `185a_v0` also fails, the failure will be much more informative than the `184x` failures.

Why:

- `185a` is the first version that makes concentrated residual events a first-class latent path
- if it still cannot solve `S3/S4/S7`, the limit is likely deeper than operator selection

At that point, the next rethink would be about the residual-path factorization itself, not about adding more heads.

## Recommendation

The next principled step is:

- write a concrete implementation spec for `185a_v0`
- implement the **single-surface IV instance first**
- keep the graph/group abstraction in the code path from day one

That is the clean new model-class phase.
