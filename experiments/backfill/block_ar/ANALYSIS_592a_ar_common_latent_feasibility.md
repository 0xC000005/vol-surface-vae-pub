# 592a AR Common-Latent Feasibility Audit

## Context

591a selected an AR common-latent transition flow as the next paradigm after
closing one-shot unified MLP path flows. Before coding, I inspected the existing
AR frontier model and the relevant prior experiments.

## Existing Mechanism

`diffusion/block_ar/empirical_normal_score_causal_memory_transition_flow_matching.py`
already has fixed source-correlation support:

- `path_source_corr`;
- `path_source_ar`;
- `_source_noise_like`;
- sampling-time path-common source noise.

This means a simple "add common source noise" implementation would duplicate
prior work rather than create a new learned-latent model.

## Prior Falsifiers

The log already contains the relevant experiments:

- 334a tested fixed path-common source noise inside the 330c AR flow and scored
  `3/11`;
- 423a tested persistent source-noise FM from the 392a frontier with
  `path_source_corr=0.35` and scored `6/11`;
- 424 closed fixed persistent source-noise as a primary route and explicitly
  rejected a rho sweep.

Mechanism from those runs:

- fixed common noise preserves some shared geometry;
- it acts mostly as a width/amplitude actuator;
- it cannot learn when uncertainty should be calm, turbulent, mean-reverting, or
  level-consistent;
- it breaks frontier structural passes before repairing level/regime failures.

## What Would Be New

A valid 592a cannot be fixed common source noise.

It must be a learned scenario-level latent state with three properties:

1. The latent must be inferred or sampled as a scenario variable, not just a fixed
   Gaussian source correlation.
2. The latent must enter the transition network persistently across rollout steps.
3. The training objective must force the latent to affect final path/regime
   allocation, otherwise teacher-forced FM can ignore it.

The minimal viable version is therefore not a one-line config change. It requires
either:

- a learned latent-conditioned rollout objective around the 510a patch-energy
  trainer; or
- a small conditional latent prior plus path decoder inside the AR transition
  memory, trained with rollout samples.

## Risk

This can easily become another latent hierarchy or posterior/prior scaffold if
implemented carelessly. That would violate the clean-pathology guard and repeat
the 270-328 latent-state failures.

The only acceptable implementation is a narrow, scenario-level stochastic input
to the existing AR transition model, with no separate deterministic center path,
no IV-specific clamps, and no evaluator-facing filter.

## Decision

Do not implement 592a as `path_source_corr` or a rho/temperature sweep.

The next executable experiment should be narrower:

- `593a = 510a frontier diagnostic with learned-common-latent design sketch`, or
- directly implement a small wrapper that injects a scenario latent into the
  existing memory state during rollout and trains only through patch/final-series
  sample loss.

Acceptance remains strict:

- if the implementation cannot be made as a small wrapper around 510a, do not
  code it;
- if it cannot recover at least the 8/11 frontier, close neural architecture
  search and return to the separated risk-policy product around 510a/564a.
