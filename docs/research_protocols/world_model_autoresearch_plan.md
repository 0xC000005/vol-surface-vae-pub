# World Model Autoresearch Plan

## Objective

Develop a self-supervised latent time-series world model whose Part 1
pretraining learns robust **market-state representations** from masked
multiview observations, using the repository's existing IV-surface,
joint-panel, and state-normalized-innovation data before broadening to
external/public benchmarks.

The revised Part 1 pretraining architecture is:

```text
same market window / same relative time position
-> structured partial view A + observed-mask channel
-> structured partial view B or clean target view + observed-mask channel
-> encoder / EMA encoder
-> aligned latent market-state embeddings
-> redundancy and collapse control
```

The downstream scenario architecture remains:

```text
learned market-state embedding
-> downstream forecast/range probes
-> conditional flow decoder
-> calibrated multi-day scenario paths
```

The workflow has two distinct research obligations:

1. **Part 1: latent market-state representation.** Prove the learned
   representation is stable under valid masked views, non-collapsed,
   non-redundant, and useful to downstream forecast/range/state probes.
2. **Part 2: conditional flow decoder.** Prove the learned latent state can
   condition realistic, calibrated, diverse future scenario distributions.

Part 2 success does not prove Part 1 success, and Part 1 success does not prove
Part 2 success. Reports must keep the two tables separate.

Future prediction, range estimation, retrieval, and scenario generation are
downstream probes or consumers of Part 1. They are not the Part 1 pretraining
objective.

## Long-Term Research Target

The long-run claim is not "we added a Barlow/VICReg loss to a time-series
model." The target claim is:

> A masked multiview joint-embedding objective can learn a compact,
> redundancy-controlled market-state representation for multivariate time
> series, and a conditional flow decoder can use that state to generate
> calibrated, diverse scenario paths.

This should eventually support both:

- **forecasting and range tasks**, through frozen probes and simple heads on
  the learned latent state after pretraining;
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
- structured masked views with observed/missing mask channels;
- same-window/same-relative-index positive-pair metadata;
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

## Literature Gate For Objective Changes

The workflow is local-first for data, metrics, artifacts, and prior results, but
it is not literature-blind. Before implementing or promoting a nonstandard JEPA
objective, decoder objective, or collapse-prevention mechanism, run a short
primary-source literature check and classify the proposal.

Use three labels:

- `canonical_jepa`: the change directly follows established JEPA practice,
  including masking or target-block design, EMA/stop-gradient target encoders,
  context-to-target latent prediction inside a masked view, horizon/position
  tokens, and simple latent L1/L2 alignment losses.
- `supported_adjacent`: the change is not canonical JEPA, but is supported by
  adjacent self-supervised learning, contrastive learning, relational learning,
  variance/covariance regularization, distribution matching, or probabilistic
  latent-variable literature.
- `speculative_local_heuristic`: the change is motivated mainly by local
  diagnostics, metrics, or failure modes in this repository.

Promotion rules:

1. Prefer `canonical_jepa` changes when the failure class can plausibly be fixed
   through target construction, target encoder design, masking/horizon design,
   prediction loss, or variance/covariance controls.
2. A `supported_adjacent` change must name the supporting paper family in the
   report and explain why the assumptions carry over to multivariate time
   series.
3. A `speculative_local_heuristic` may be tested only as a diagnostic or
   deliberately bounded experiment. It must not become the main workflow
   direction unless it beats the current reference candidate on the appropriate
   Part 1 or Part 2 gate.
4. Every report involving a nonstandard objective must include a
   `Literature Status` section with the classification, sources, and whether the
   objective is being treated as canonical, adjacent, or speculative.

For the revised JEPA Part 1 branch, the preferred order of attack is:

1. define structured masked views and observed/missing mask channels;
2. define the positive-pair rule: same window, same relative index, same
   underlying panel state;
3. use EMA or stop-gradient target encoders where appropriate;
4. add Barlow Twins, variance/covariance, or related redundancy control only
   after true masked/multiview positives are defined;
5. treat future prediction, range estimation, and scenario generation as
   downstream probes, not as pretraining losses;
6. treat neighborhood/contrastive/relational losses as secondary diagnostics
   unless primary sources and local metrics both support promotion.

## Part 1: Latent Market-State Gates

Part 1 evaluates representation quality before scenario generation.

Primary objective:

```text
same market window and relative index
-> structured masked view A + observed-mask channel
-> online encoder -> z_a
same market window and relative index
-> structured masked view B or clean target view + observed-mask channel
-> target/EMA encoder -> stopgrad(z_b)
```

Core loss:

```text
L_part1 =
    L_multiview_alignment(z_a, stopgrad(z_b))
  + lambda_redundancy L_redundancy_or_barlow
  + optional lambda_var L_variance
  + optional lambda_cov L_covariance
```

The alignment term is the main pretraining task. Redundancy, variance, and
covariance terms are representation-health controls:

- variance prevents constant collapse;
- covariance/off-diagonal correlation prevents duplicated latent dimensions;
- Barlow-style cross-correlation terms are appropriate only for true
  same-state masked/multiview positives;
- none of these terms should be treated as a future-prediction objective.

Required Part 1 metrics:

- view-alignment MSE/cosine/cross-correlation for same window and same relative
  index;
- retrieval top-k or MRR: masked-view embedding retrieves the matching clean or
  differently masked same-state embedding among distractors;
- embedding variance per dimension;
- effective rank / participation ratio;
- singular-value spectrum;
- off-diagonal covariance/correlation norm;
- mask-artifact diagnostics: the model must not win by detecting sentinel
  corruption;
- frozen downstream probes for future volatility range, jump/tail indicator,
  drawdown, cross-cell/cross-factor correlation, regime/state labels, or
  forecast/classification tasks.

Forbidden Part 1 shortcuts:

- do not align arbitrary time windows as positives;
- do not align the same absolute date across different relative indices unless
  the representation is explicitly non-contextual;
- do not use visible Gaussian/uniform sentinel values as missing content without
  an observed/missing mask channel;
- do not shuffle factor identity or time order unless the model and task are
  explicitly permutation-invariant;
- do not treat future prediction as the pretraining objective;
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

1. Part 1 passes non-collapse, redundancy, masked-view alignment, retrieval,
   mask-artifact, and downstream probe gates.
2. Part 2 passes distributional scenario gates.
3. A direct generator without JEPA pretraining is included as a baseline when
   making claims about the world-model representation.
4. Ablations identify the role of masked view construction, alignment loss,
   redundancy/collapse control, downstream probes, and the flow decoder.

If the flow decoder works but Part 1 probes fail, the result is a useful
generator branch, not a validated latent world model. If Part 1 works but the
flow decoder fails, the result is a representation branch, not a scenario
generator.

## Failure Classification

Every failed iteration must assign one primary class:

- `data_object`: windows/coordinates/splits are not stable or learnable;
- `view_alignment`: same-state masked views fail to align;
- `mask_artifact`: the model exploits missingness/sentinel artifacts rather
  than market-state structure;
- `latent_prediction`: downstream future-latent probe fails after pretraining;
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
   `window`, `masked_view_a`, `mask_a`, `masked_view_b`, `mask_b`,
   `relative_index`, `absolute_index`, `metadata`.
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

## Gated Modeling Tracks

Modeling guardrails are constraints on what the next iteration may do; they are
not stop conditions by themselves.

If Part 1 is frozen or explicitly marked "no more knobs", do not add new Part 1
losses, target sweeps, retrieval/neighborhood objectives, split/horizon changes,
or future-prediction pretraining objectives unless a new Part 1 failure is
documented and the literature gate is satisfied.

Barlow/VICReg-style terms are not forbidden in the masked-multiview branch, but
they require true same-state positive pairs, observed/missing mask channels, and
mask-artifact diagnostics. Do not add them as a patch to a poorly specified
prediction target.

If decoder work has not been explicitly requested, do not start conditional
flow training, decoder baselines, decoder conditioning changes, or Part 2
experiments.

When Part 1 and Part 2 are both gated, continue only with bounded process work
until a real stop condition appears or the user redirects the loop. Allowed
work is limited to:

- provenance, manifest, digest, and artifact-identity checks;
- report, metric, and research-log reconciliation;
- handoff criteria, restart checklists, and open-risk ledgers;
- workflow guardrails that prevent objective creep;
- explicit decision reports that preserve caveats for future work.

These iterations still need a hypothesis, falsifier, state update,
research-log tail append, verification, and one focused commit.

## Stop Conditions

The workflow stops only when one of these is true:

- `goal_reached` is true in local state;
- the target stage in `world_model_goal.json` is reached;
- `autoresearch-session/WORLD_MODEL_STOP` exists;
- the requested iteration budget is exhausted;
- runtime/tool limits make further work unreasonable.

Research/model blockers are not stop conditions by default. Convert blockers
into analysis, ideation, or paradigm-shift iterations.
