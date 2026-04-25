# 519a In-Session Closure Checkpoint

## Status

The configured in-session hard cap is reached at iteration `519`.

The objective is not achieved:

- target: deployable `11/11`;
- current deployable learned frontier: `8/11`;
- goal reached: no.

## Current Frontier

### 392a

- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- score: `8/11`
- failed suites: coverage, regime coverage, distributional fidelity
- role: safer structural anchor

### 510a

- checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- score: `8/11`
- failed suites: coverage, regime coverage, distributional fidelity
- role: frontier tie with better aggregate coverage/path geometry but thinner
  conditionality/cointegration margins

### 435a

- artifact: `results/block_ar/435a_suite_feasibility_upper_bound/full11.json`
- score: `11/11`
- role: oracle feasibility proof only
- deployable: no, uses validation futures

## Closed Branches

The following were tested or audited and should not be resumed as local tweaks:

- local proper-score / objective changes around `392a`;
- joint sliced-Wasserstein and MMPD-style patch energy;
- persistent and AR(1) source-prior changes;
- checkpoint interpolation and learned-law ensembling;
- marginal quantile maps, rank-copula conditional marginals, and center policies;
- source/residual transport and density-ratio wrappers;
- MixLinear/Minkowski-style compact direct cores;
- quick observable-state conditioning repair based on current `ret`, `price`,
  `slopes`, `skews`, and `levels` audit.

## Mechanism

The learned generator can model local shape, serial behavior, correlation, and
conditional response well enough to pass eight suites. It fails when asked to
match unconditional future absolute level/regime occupancy while also preserving
conditionality, cointegration, and jump realism.

Oracle centering proves this is mechanically satisfiable, but not learnable from
the current local IV-only setup under the tested objectives.

## Resume Paths

### Path A: New Learned-Law Program

Resume from `experiments/backfill/block_ar/PLAN_518a_next_learned_law_program.md`.

Use a separate branch or worktree. The first milestone is not `11/11`; it is
matching the `8/11` frontier while showing positive signal on level/regime proxy
targets.

### Path B: Deployable Risk System

Resume from the policy-calibration framing:

- report base learned generator metrics separately;
- add only auditable policy calibration layers;
- never present calibrated-system improvements as learned conditional-law
  improvements.

This can be useful for risk managers, but it is a different claim from a pure
learned generator.

## Stop Reason

Stop because the configured hard cap was reached, not because the scientific
goal was reached. Continuing the same local line would be brute-force knob
search and would violate the clean-pathology guard.

