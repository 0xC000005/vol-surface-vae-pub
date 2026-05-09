# World Model Autoresearch Plan

## Objective

Develop a self-supervised latent time-series world model for forecasting and
scenario generation, using the repository's existing IV-surface, joint-panel,
and state-normalized-innovation data before broadening to external/public
benchmarks.

The target architecture is:

```text
past multivariate window
-> JEPA context encoder
-> latent predictive world state
-> multi-horizon latent predictor
-> conditional flow decoder
-> calibrated multi-day scenario paths
```

The workflow has two distinct research obligations:

1. **Part 1: latent world model.** Prove the learned representation is
   future-predictive, non-collapsed, and non-redundant.
2. **Part 2: conditional flow decoder.** Prove the learned latent state can
   condition realistic, calibrated, diverse future scenario distributions.

Part 2 success does not prove Part 1 success, and Part 1 success does not prove
Part 2 success. Reports must keep the two tables separate.

## Long-Term Research Target

The long-run claim is not "we added a Barlow/VICReg loss to a time-series
model." The target claim is:

> A redundancy-controlled joint-embedding predictive objective can learn a
> compact latent world state for multivariate time series, and a conditional
> flow decoder can use that state to generate calibrated, diverse scenario
> paths.

This should eventually support both:

- **forecasting tasks**, through frozen probes and simple heads on the learned
  latent state;
- **generation tasks**, through conditional flow matching in path space.

The first domain is the existing repo data. Broader public benchmarks should
come later, only after the local data object and evaluation contract are clean.

## Data Sources To Leverage First

Use local data and artifacts before online searches or new datasets:

- `data/vol_surface_with_ret.npz`: IV-surface data used by many backfill and
  flow experiments.
- `data/multi_factor_data.npz`: broader factor panel for generalization checks.
- `data/regime_labels.npz`, `data/gt_cumulative_variance.npz`, and existing
  scale/uncertainty caches when useful for probes.
- `results/baselines_current_panel/manifest.json` and related baseline results
  for current-panel comparison.
- Existing SNI and flow experiments in `experiments/backfill/block_ar/`.
- Existing model code under `diffusion/block_ar/`, especially deterministic
  world-model and flow-matching variants.
- Existing evaluation harnesses, including IV full-suite, scorecard, Energy
  Score, Variogram Score, CRPS, coverage, correlation, and PCA/effective-rank
  diagnostics where available.

The first data milestone is a small reusable world-model dataset builder under
`experiments/world/` that can produce:

- past context windows;
- future target windows for one or more horizons;
- normalized/state-mapped or SNI-compatible coordinates;
- deterministic train/validation/test split identifiers;
- metadata needed for retrieval, frozen probes, and decoder evaluation.

## HEAD Loop

Each iteration is one complete HEAD cycle:

- **Hypothesis:** one falsifiable claim.
- **Execute:** one focused analysis, implementation, or experiment.
- **Analyze:** compare against saved metrics, baselines, and failure gates.
- **Decide:** update state, append the research log, and pick the next step.

Allowed iteration types:

- `post_experiment_analysis`
- `research_ideation`
- `paradigm_shift`
- `experiment`

Decision law:

1. If a fresh result is not understood enough to choose the next move, run
   `post_experiment_analysis`.
2. If the current family appears capped or underdetermined, run
   `research_ideation`.
3. If evidence says the active decomposition is wrong, run `paradigm_shift`.
4. If a clear falsifier is already identified, run `experiment`.

Do not run an experiment merely because more training looks active.

## Part 1: Latent World Model Gates

Part 1 evaluates representation quality before scenario generation.

Primary objective:

```text
past window -> context encoder -> z_context
z_context + horizon tokens -> predictor -> z_future_pred
actual future window -> target/EMA encoder -> z_future_target
```

Core loss:

```text
L_part1 =
    L_future_latent_prediction
  + lambda_var L_variance
  + lambda_cov L_covariance
```

The prediction term is the main task. Variance and covariance terms are
representation-health regularizers:

- variance prevents constant collapse;
- covariance/off-diagonal correlation prevents duplicated latent dimensions;
- neither replaces future-latent prediction.

Required Part 1 metrics:

- future-latent prediction MSE/cosine error by horizon;
- retrieval top-k or MRR: predicted future latent retrieves the true future
  among distractors;
- embedding variance per dimension;
- effective rank / participation ratio;
- singular-value spectrum;
- off-diagonal covariance/correlation norm;
- frozen probes for future volatility, jump/tail indicator, drawdown,
  cross-cell/cross-factor correlation, or downstream forecast/classification.

Forbidden Part 1 shortcuts:

- do not align arbitrary time windows as positives;
- do not call reconstruction quality a proof of world-model quality;
- do not use scenario-generation metrics to hide representation collapse.

## Part 2: Conditional Flow Decoder Gates

Part 2 evaluates whether the latent state can condition a scenario law.

Default decoder family: conditional flow matching.

```text
epsilon ~ base_noise
x_1 = future path
x_tau = (1 - tau) * epsilon + tau * x_1
v_theta(x_tau, tau | z_context, z_future_pred, tokens, past) -> x_1 - epsilon
```

At inference, sample base noise and integrate the learned vector field to obtain
multi-day scenario paths.

Fixed Gaussian or Student-t mixture heads are not the intended method for this
workflow. They may be used only as weak sanity baselines, because fixed
parametric decoders are too restrictive for the conditional, high-dimensional
scenario distribution targeted here.

Required Part 2 metrics:

- CRPS or pinball/quantile scores;
- Energy Score;
- Variogram Score;
- empirical coverage by horizon and by cell/channel;
- sample variance ratio;
- pairwise scenario distance;
- correlation matrix error;
- PCA/eigenvalue spectrum error and effective rank;
- tail metrics such as extreme quantile coverage, max-jump KS, drawdown
  distance, or stress-window coverage;
- conditionality by regime/context bucket.

Part 2 reports must state whether the decoder used:

- frozen Part 1 encoder/predictor;
- semi-frozen Part 1 components;
- jointly trained Part 1 and Part 2;
- teacher-forced or generated prefix/context during rollout.

## Integration Gate

A combined model is a candidate only if:

1. Part 1 passes non-collapse, redundancy, prediction, and retrieval/probe
   gates.
2. Part 2 passes distributional scenario gates.
3. A direct generator without JEPA pretraining is included as a baseline when
   making claims about the world-model representation.
4. Ablations identify the role of `L_prediction`, `L_variance`, `L_covariance`,
   and the flow decoder.

If the flow decoder works but Part 1 probes fail, the result is a useful
generator branch, not a validated latent world model. If Part 1 works but the
flow decoder fails, the result is a representation branch, not a scenario
generator.

## Failure Classification

Every failed iteration must assign one primary class:

- `data_object`: windows/coordinates/splits are not stable or learnable;
- `latent_prediction`: future-latent alignment fails;
- `collapse`: embeddings are constant, low-rank, or duplicated;
- `probe_failure`: latent looks healthy but does not support forecasting/state
  probes;
- `decoder_train_fit`: flow decoder cannot learn training distribution;
- `decoder_calibration`: realistic samples but wrong width/location;
- `decoder_diversity`: same-context samples collapse;
- `dependency`: marginals pass but cross-cell/cross-factor structure fails;
- `conditionality`: generated law ignores context/latent state;
- `rollout`: teacher-forced quality does not survive generated rollout;
- `test_mismatch`: metric is unstable or misaligned with the research goal.

Only switch backend or temporal factorization when the failure class points to
that component and simpler audits have been run.

## First Iteration

The first HEAD iteration should be a data-and-evaluation inventory, not model
training.

Hypothesis:

> The repository already contains enough IV/joint/SNI data and evaluation code
> to define a smoke-scale world-model dataset and Part 1/Part 2 metric contract
> without introducing new data or online dependencies.

Execute:

1. Inventory candidate local data files and existing split conventions.
2. Identify the minimum smoke dataset shape for Part 1:
   `past_window`, `future_window`, `horizon`, `metadata`.
3. Identify which existing metrics can be reused directly and which need new
   wrappers under `experiments/world/evaluation/`.
4. Write a concise report under `experiments/world/reports/`.

Analyze:

- whether the first smoke can use `data/vol_surface_with_ret.npz`;
- whether `multi_factor_data.npz` should wait until after IV smoke passes;
- whether existing full-suite/scorecard metrics can be reused as-is;
- what the first minimal Part 1 falsifier should be.

Decide:

- build the dataset/metric harness next, or run post-analysis if the data
  object is unclear.

## State And Logging

Local resumability files:

- `autoresearch-session/world_model_goal.json`
- `autoresearch-session/world_model_state.json`
- `autoresearch-session/world_model_driver_prompt.md`
- `autoresearch-session/check_goal_world_model.py`

`autoresearch-session/` is ignored by Git by default. Treat these as local
session state unless explicitly force-added.

Every completed iteration must:

1. update `autoresearch-session/world_model_state.json`;
2. append a concise entry to the true tail of `RESEARCH_LOG.md` using
   `research-log-tail-append`;
3. stage only files owned by the current iteration;
4. make one focused commit unless the user explicitly says not to or the
   iteration was only local state setup.

Research-log entries should include:

- context;
- hypothesis;
- execution;
- result;
- mechanism read;
- decision / next step;
- artifact paths and verification commands when applicable.

## Stop Conditions

The workflow stops only when one of these is true:

- `goal_reached` is true in local state;
- the target stage in `world_model_goal.json` is reached;
- `autoresearch-session/WORLD_MODEL_STOP` exists;
- the requested iteration budget is exhausted;
- runtime/tool limits make further work unreasonable.

Research/model blockers are not stop conditions by default. Convert blockers
into analysis, ideation, or paradigm-shift iterations.
