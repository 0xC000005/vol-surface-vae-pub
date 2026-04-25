# 524a Factor-Conditioned Surface-Level Learned Law

## Motivation
522a showed that a joint IV+factor daily-change generator is not the right object for the official 11-suite. It can learn plausible local changes while failing future IV-level occupancy, long-horizon coverage, conditionality, cointegration, mean reversion, and pathwise realism.

523a showed the active `8/11` frontier wins because it keeps the learned law close to the future IV-level path being evaluated. The next clean route should therefore preserve the 340c/392a target coordinate and only add broader financial state as observed conditioning.

## Hypothesis
A factor-conditioned surface-level law can improve regime/coverage/distributional behavior without losing the 392a/510a structural passes if:

- the generated object remains future IV levels in empirical normal-score coordinates;
- factors are used only as observed history conditioning, never as generated daily-change targets;
- the core objective remains vanilla flow matching in the future-level coordinate;
- the first acceptance gate is recovery of the 392a/510a structural passes, not immediate `11/11`.

## Minimal Architecture
Start from `EmpiricalNormalScoreCausalMemoryTransitionFlowMatching`.

Add one optional side-channel:

- `factor_history`: `(batch, history_len, factor_dim)` containing observed factor returns and/or standardized levels available at forecast origin.
- `factor_encoder`: a small GRU or transformer encoder over the factor-history sequence.
- `factor_context_proj`: maps the final factor context to `memory_dim`.
- add the projected factor context to every IV memory token before the causal memory transformer or before the velocity memory projection.

Do not add:

- generated factor paths;
- low-rank decoders;
- separate deterministic center/residual paths;
- bounded residual rules;
- policy calibration;
- regime hand-label branches.

This keeps the only new inductive bias to: "observable market-state history may condition the future IV-level law."

## Data Framing
Use the aligned 38-d data infrastructure only to provide factor histories. The forecast target stays the official IV future:

- history IV surfaces: official `build_multistep_windows` history;
- future IV surfaces: official `build_multistep_windows` future;
- factor history: aligned factor returns/levels over the same observed history dates;
- no future factors, no validation futures in training, no oracle residuals.

Standardize factor features on the train split only. Prefer a generic feature vector over financial handcrafting: all available factor returns plus standardized levels, with missing returns filled by the existing 38-d loader policy.

## Objective
Keep the same empirical normal-score flow-matching objective as 340c/392a:

- target: future IV scores relative to current prefix score;
- source: vanilla Gaussian path source;
- memory: causal IV prefix memory plus learned factor-history context;
- loss: FM velocity MSE only.

No extra CRPS, MMD, coverage, regime, KS, or calibration loss in the first prototype.

## Acceptance Gate
524a/525a should be treated as a mechanics falsifier, not a sweep.

Minimum gate:

- full 11-suite score must be at least `6/11` after a short run;
- it must pass `surface`, `time_series`, `block_ar`, and `cross_cell_correlation`;
- it must not destroy `mean_reversion` and `pathwise_jump_realism` versus 392a/510a by more than the 509a/510a tolerance;
- if it cannot approach `8/11`, close the factor-conditioned 340c branch rather than adding knobs.

Success gate:

- recover the `8/11` structural frontier while improving at least one of the three hard failures: coverage, regime coverage, or distributional fidelity.

## Decision
Proceed to an executable prototype only if it can be implemented as a small extension of the 340c empirical-normal-score surface-level model. The clean path is not "larger 38-d daily-change diffusion"; it is "surface-level learned law with optional observed factor conditioning."
