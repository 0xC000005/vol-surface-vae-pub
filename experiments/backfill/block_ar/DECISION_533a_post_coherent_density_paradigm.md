# 533a Post-Coherent-Density Paradigm Decision

## Context
532a falsified the simplest marginalization-consistent density route. It was methodologically clean but scored only `3/11`, with weak conditionality, poor level KS, broken cross-cell rank structure, and over-strong mean reversion.

This result should be read together with the prior closed branches:

- `347a-351a` transition exact-likelihood models peaked at `5/11`.
- `459/460` future-token and day-vector density reset stayed below frontier.
- `489/490` H60 context recovered to only `7/11`.
- `491/492` rank-copula conditional marginals collapsed to `3/11`.
- `525-530` broader factor conditioning stayed at `7/11`.
- `532a` one-shot coherent Gaussian density scored `3/11`.
- `392a`/`510a` remain the best learned deployable frontier at `8/11`.
- `435a` remains the `11/11` oracle feasibility result, but it uses validation futures and is not deployable.

## Decision
Do not continue with another local repair around:

- global Gaussian copulas;
- separable marginal maps;
- H60 context;
- factor side-channels;
- 392a-local CRPS/MMD/energy/critic losses;
- transition-likelihood tail/base variants;
- posthoc interval or level calibration.

Those are now research knobs, not first-principles progress.

## Remaining Principled Route
The only route still methodologically clean is a larger learned-law program, not another local patch:

```text
learn p(future financial panel | history financial panel)
```

with:

- broader financial panel tokens, not IV-only local windows;
- pretraining or multi-task training on all available factor/IV channels;
- a single probabilistic sequence objective;
- no validation-future oracle, no evaluator-specific calibration, no hand-written regime branch;
- IV-surface scenario generation as one conditional query of the learned law.

This is aligned with the recent time-series foundation-model trend and with the Bitter Lesson: move signal and capacity into a general sequence model rather than keep adding hand-designed patches around the current IV-only frontier.

## Immediate Next Step
The next in-session HEAD iteration should not try to beat `392a` directly with another small model. It should create a data/model readiness audit for this larger learned-law program:

- enumerate exactly which aligned factor/IV channels are available locally;
- define tokenization and train/validation framing without lookahead;
- choose the smallest first prototype that is still a genuine multi-factor probabilistic sequence model;
- define how its IV conditional scenario samples will be evaluated by the same 11-suite.

If the local repository lacks enough aligned financial panel data for this program, that is a real blocker and should be documented as such rather than hidden by more architecture tuning.
