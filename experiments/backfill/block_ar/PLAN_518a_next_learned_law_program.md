# 518a Next Learned-Law Program Handoff

## Objective

Build a deployable learned conditional scenario generator that can plausibly
move beyond the current `8/11` frontier without relying on validation futures,
oracle centering, or post-hoc evaluator-specific corrections.

The target remains the same official full 11-suite.

## Why This Needs A New Scoped Program

The local `392a` / `510a` neighborhood is exhausted. It already has enough local
geometry to pass most structural suites, but it cannot match future absolute
level/regime occupancy. Local corrections either:

- keep the center fixed and leave level KS/regime coverage unchanged;
- move the center and damage conditionality, cointegration, or mean reversion;
- widen/smooth the residual law and break jump realism or overcoverage gates.

Recent foundation-model work points to data/pretraining scale and simple
probabilistic sequence objectives, not another small repair head.

## Proposed Program

### Data Scope

Use a broader panel than IV history alone:

- IV surface cells;
- available market observables from the current data (`ret`, `price`, `slopes`,
  `skews`, `levels`);
- if available later, the broader multi-factor panel already referenced in the
  repo (`multi_factor_data.npz`).

The model should condition on the full observable history but only generate the
required IV surface scenarios for the 11-suite unless a separate multi-factor
evaluation is explicitly enabled.

### Model Class

Use a single-stage probabilistic sequence model:

- decoder-only or encoder-decoder transformer/state-space backbone;
- narrow latent/token bottleneck allowed only as compression, not as hand-coded
  low rank;
- joint future path generation, not per-cell independent heads;
- vanilla probabilistic objective such as tokenized cross-entropy, quantile
  likelihood, diffusion/flow matching, or CRPS/energy score;
- no bounded idio path, no explicit EC baseline, no hard low-rank readout, no
  validation-future calibration in the core.

### Pretraining

The key difference from the failed local experiments should be data scale:

- pretrain on all available rolling histories, not only the final recent block;
- include masking / next-token / multi-token forecasting tasks over the broader
  factor panel;
- fine-tune to 30-day IV scenario generation only after the model learns the
  general sequence law.

### Evaluation Contract

Report three numbers separately:

1. Base learned generator full 11-suite.
2. Optional policy-calibrated risk system full 11-suite.
3. Oracle feasibility reference (`435a`) clearly marked non-deployable.

The base learned generator must be judged independently. A policy layer can be
useful for risk management, but it cannot be used as evidence that the learned
conditional law improved.

## Acceptance Criteria

A new program is worth continuing only if an early checkpoint satisfies all of:

- reaches at least `8/11` without policy calibration;
- does not lose surface validity, conditionality, block-AR smoothness,
  correlation structure, or mean reversion;
- improves at least one of the three frontier failures versus both `392a` and
  `510a` without making another frontier failure materially worse;
- shows positive pre-validation signal on future level/regime proxies, unlike
  `516a`'s negative-R2 audit.

If an early checkpoint cannot match `8/11`, stop before scaling.

## Non-Goals

- Do not tune another scalar width/temperature/source parameter around `392a`.
- Do not add evaluator-specific post-processing to claim a base-model win.
- Do not treat a calibrated risk-system score as learned-law evidence.
- Do not restart old `205-265` low-rank/bounded-idio families unless they are
  explicitly reframed as ablations.

## Immediate Next Action

Outside this exhausted local loop, create a separate branch or worktree for the
new learned-law program. The first milestone should be a small pretraining
prototype with a clear early-stop gate: match `392a`'s structural passes before
any optimization toward coverage/regime/distributional failures.

