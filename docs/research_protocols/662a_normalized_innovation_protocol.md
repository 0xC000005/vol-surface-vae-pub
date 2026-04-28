# 662a State-Aware Normalized-Innovation Protocol

## Core Object

The methodology is a conditional law over state-aware normalized innovations:

`p(z_{t+1:t+H} | encoded state path, recent innovations, local scale/regime features)`

The generated object is not raw level and not pure return. The model generates
future movement in a normalized coordinate, then reconstructs raw scenarios by
unnormalizing and integrating from the last observed market state.

## First Normalization Choice

Use support-aware encoded levels first:

- Positive level-like variables: log level.
- Difference-level variables: identity/difference coordinate.
- Bounded or surface-like variables: keep the existing IV encoded level used by
  the state-space utilities unless a later diagnostic proves it is wrong.

Then compute daily encoded increments:

`dx_t = encoded_level_t - encoded_level_{t-1}`

The first 662a implementation uses scale normalization:

`z_t = dx_t / sigma_history`

where `sigma_history` is computed from the observed history window only. The
default estimator is a fixed EWMA/root-mean-square scale over recent encoded
increments with a robust floor. Do not tune the EWMA decay per experiment; use a
fixed half-life tied to the history length unless a diagnostic specifically
identifies the scale estimator as the failure.

Do not subtract a strong rolling mean in the first implementation. Conditional
drift and mean reversion should be learned from the state features. Trend and
level information are input features, not hidden inside deterministic centering.

## State Conditioning

The encoder must receive enough information to be level-aware:

- current encoded level;
- recent encoded level path;
- recent normalized innovation path;
- local scale used for normalization;
- recent trend/slope and volatility summaries when available through the same
  generic computation for all channels.

This forces the model to learn relationships such as high-IV mean reversion,
credit-spread regime behavior, equity return scale, and cross-factor stress
co-movement from state rather than from a factor-specific branch.

## Backend Policy

Flow matching is the first backend because it is already integrated and does
not impose a Gaussian or Student-t tail assumption. It is not the methodology.
The methodology is the state-aware normalized-innovation conditional law.

Do not switch backend because a score is disappointing. Backend changes are only
allowed after a diagnosis shows the backend is the bottleneck.

Allowed backend conclusions:

- Data framing failure: normalized innovations are unrealistic or not stable.
- Reconstruction failure: innovations are realistic but decoded levels fail.
- Conditionality failure: shuffled histories produce similar distributions.
- Diversity failure: same-history samples collapse.
- Dependency failure: marginal innovations are plausible but cross-factor
  dependence is weak.
- Distribution-shift failure: train-tail works but validation fails.
- Backend failure: diagnostics show the object/framing works but the sampler or
  training objective cannot allocate conditional probability mass.

Only the last category justifies moving from flow matching to diffusion, copula,
or another backend.

## Required Audits Before Backend Switch

Every 662a-family iteration must report:

- train-tail IV full-suite score;
- validation IV full-suite score;
- train-tail joint-panel audit;
- validation joint-panel audit;
- condition-shuffle audit;
- same-history sample diversity audit;
- innovation-space realism audit;
- reconstruction sanity audit.

If an iteration fails, the next step must name one failure category from the
Backend Policy and attempt the smallest fix for that category before changing
backend or architecture.

## Switch Rules

Do not switch AR versus one-shot, flow versus diffusion, or transformer versus
another backbone until the diagnostics above identify the current component as
the bottleneck.

The first committed 662a backend is:

- autoregressive daily normalized-innovation generation;
- shared Transformer-style state encoder;
- conditional flow matching transition;
- one shared stochastic source and one checkpoint for all channels.

This is a baseline implementation choice, not a claim that Transformer or flow
matching is required.
