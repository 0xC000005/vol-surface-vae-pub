# World Model HEAD137: Context-To-Target JEPA Design

Date: 2026-05-10

## Iteration Type

`paradigm_shift`

## Objective Family

`context_to_target_jepa` for same-window masked current/history state blocks.

## Literature Status

`canonical_jepa`.

This is the JEPA family used by I-JEPA and V-JEPA: predict latent
representations of masked target regions from visible context, then evaluate the
resulting frozen representations. It is not future prediction and not a value
reconstruction decoder.

## Why Change Family

The current `masked_multiview_invariance` branch is healthy but capped:

- representation health is non-collapsed and seed-stable;
- corruption robustness passes;
- downstream path-shape/risk-width utility exists;
- regime probes contain some balanced signal;
- exact current-state geometry is still compressed away;
- representation-surface changes do not fix exact-state retention.

This is the precise failure mode where pure invariance can be too aggressive:
if two views must agree despite missing different factors and surface regions,
the easiest stable representation may discard exact local state details.

## Proposed Diagnostic

Train a minimal same-window context-to-target JEPA smoke test:

```text
clean same market window
-> choose typed target blocks inside the history/current window
-> context view masks those target blocks and keeps mask channels
-> context encoder produces context/time embeddings
-> target encoder sees the clean target blocks and target metadata
-> predictor maps context + target metadata to target latent
-> loss aligns predicted target latent to stop-gradient/EMA target latent
-> Barlow/VICReg-style health checks remain on evaluated encoder surfaces
```

Target blocks are not future horizons. They are masked current/history pieces:

- IV surface rectangles, wings, maturity bands, or moneyness bands;
- side-channel groups;
- factor families;
- contiguous days in the history window.

The target metadata must include geometry id, factor id/family, geometry
coordinates, relative time index, and synthetic/observed mask indicators.

## Minimal Scope

The first diagnostic should be intentionally small:

- reuse the existing `build_masked_multiview_batch` data contract;
- keep train/validation windows at the current scaled smoke level;
- use the same base encoder size where possible;
- add only the target-block sampler, target encoder path, and predictor needed
  for context-to-target latent prediction;
- do not add future targets, scenario decoder losses, or value reconstruction;
- report both context encoder health and target-latent prediction quality.

## Falsifiers

Reject or demote this route if:

- effective rank or per-dimension variance collapses;
- target-latent prediction improves while frozen exact-state probes do not;
- exact-state probes improve only by harming path-shape/risk-width utility;
- mask-family leakage becomes large;
- the branch requires many target/mask knobs before showing a clean signal.

## Acceptance For A Smoke Candidate

A successful first context-to-target diagnostic should show:

- representation health comparable to or better than HEAD127/HEAD131;
- corruption robustness without mask-family shortcuts;
- current-IV MSE materially closer to raw surface features than HEAD127;
- no loss of factor-return and path-shape/risk-width signal;
- regime diagnostics reported with both accuracy and macro recall;
- no claim of Part B readiness until the formal Part 1 gate is rerun.

## Decision

Proceed to a minimal context-to-target JEPA diagnostic only after coding the data
surface carefully. The goal is missing-state representation learning inside the
same market window, not future forecasting and not generative reconstruction.

This is a principled family change. It should be implemented as a separate
diagnostic branch and compared against the frozen scaled Barlow candidate, not
silently mixed into the existing two-view invariance objective.
