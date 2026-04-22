## 293a One-Stage Joint Conditional-Law Reset

### Context
The active `289 -> 292` line is being retired as the main search tree.

Why:
- the deterministic retrieval hierarchy capped at `5/11`
- the deterministic world-model line (`289e -> 291a`) devolved into output-head surgery
- the `292a` AR reset was not materially new enough relative to the earlier `212 -> 221/222/223/226/227/233` autoregressive line
- the repo already contains strong evidence that flat "strong H=1 law + rollout" is a different problem from directly learning the 30-day joint path law

So the reset has to be specific:
- not "AR again"
- not "one-shot again"
- not "183c with new words"

### Decision
Next step: `293a-v0`

### Core object
Model the native target directly:

\[
p(\Delta X_{1:30} \mid H)
\]

where:
- `H` = recent history
- `\Delta X_{1:30}` = full 30-day future normalized-change panel

This is a fixed-horizon conditional **joint law** model.

The model should learn:
- conditional marginals for each future `(horizon, cell)` point
- conditional dependence across future times and cells

It should **not** be trained as:
- deterministic path first, uncertainty later
- next-day density rolled out recursively
- transport around a hand-designed base law

### Minimal 293a-v0 workflow
1. Encode history.
2. Build tokens for future `(horizon, cell)` points.
3. Predict a conditional marginal distribution for each future point.
4. Model the dependence structure across all future points jointly.
5. Train by maximizing the conditional joint log-likelihood.

This is distribution-first, not path-reconstruction-first.

### Minimal 293a-v0 architecture

#### History encoder
- input: recent normalized levels, normalized changes, and simple calendar / horizon context
- architecture: causal transformer encoder or equivalent sequence encoder
- output: a compact history context representation

#### Future query tokens
- one token per future `(horizon, cell)` pair
- token features:
  - horizon embedding
  - cell embedding
  - history-context cross-attention
  - current normalized state anchor for that cell

#### Marginal model
- one conditional monotone spline flow per future token
- outputs an invertible 1D conditional marginal density for that token
- this avoids:
  - fixed Gaussian / Student-t assumptions
  - per-cell discretization artifacts

#### Dependence model
- attentional copula / joint dependency module over the future tokens
- consumes token embeddings and transformed uniforms from the marginal layer
- outputs the copula density contribution for the full 30-day future panel

#### Training objective
- exact or tractable conditional log-likelihood:
  - sum of token marginal log-densities
  - plus copula log-density

No:
- AR rollout loss
- retrieval bank
- hard-coded low-rank decoder
- transport residual-law backbone
- deterministic Stage A / stochastic Stage B decomposition

### Why this is materially different

#### Different from `170e`
`170e` was already a one-shot joint-likelihood model, but it hard-coded:
- Student-t marginals
- separable Kronecker time/cell covariance
- scalar tail parameterization

`293a` differs by:
- learned nonparametric conditional marginals
- learned dependence model instead of fixed separable covariance
- no explicit Student-t family assumption

So `293a` is not "170e again." It removes the strongest explicit law assumptions.

#### Different from `183c`
`183c` is a state-dependent transport / residual-law generator built on:
- explicit mean branch
- explicit covariance branch
- whitened geometry-aware residual transport

`293a` differs by:
- modeling the conditional density directly
- no transport backbone
- no mean/covariance/pathwise residual decomposition
- no IV-grid-specific geometry assumptions

So `293a` is not "183c again." It is density-first rather than transport-first.

#### Different from `212`
`212ai` and its AR descendants learn:
- a strong next-step conditional law
- then rely on recursive rollout

`293a` differs by:
- treating the 30-day future as the native object
- learning the full fixed-horizon conditional law directly
- removing autoregressive compounding from the core formulation

So `293a` is not "212 again." It does not ask a one-day law to become a 30-day law via recursion.

### Why this reset is principled
This is the narrowest reset that is genuinely different from the main prior families:
- explicit joint-likelihood one-shot, but less hand-parametric than `170e`
- fixed-horizon joint generator, but density-first rather than transport-first like `183c`
- no AR rollout, avoiding the repeated `212`-style compounding failure

It also matches the conceptual concern that:
- realized next-day changes may be largely irreducible
- but the conditional **joint future law** may still be learnable

### Scope
`293a-v0` is intentionally fixed to 30 days.

If it works:
- extend later by chunking or hierarchical window composition

If it fails:
- we will know whether the failure is in the explicit joint-law idea itself, not in retrieval, AR compounding, or transport assumptions

### Kill criteria
`293a` is only alive if it materially improves over the current reset lines on at least one of:
- change KS
- level KS
- pathwise jump realism
- cointegration
- conditionality / regime coverage

while preserving:
- surface validity
- non-degenerate cross-cell structure

If it collapses into:
- overly smooth marginals
- dead dependence model
- or factorized-per-cell behavior

then the explicit joint-law reset is falsified quickly.

### Immediate next step
Implement `293a-v0` as:
- history encoder
- future token grid
- conditional marginal spline flows
- attentional copula dependency head

Train and evaluate one clean baseline before any ablations.
