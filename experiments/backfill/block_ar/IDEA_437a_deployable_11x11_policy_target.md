# 437a: Deployable 11/11 Policy Target

## Context

`435a` reached `11/11`, but only by using validation futures. It is a suite
feasibility upper bound, not a deployable generator.

The new target is stricter:

- reach `11/11`;
- no validation future information at inference;
- no per-window oracle center;
- no per-window oracle miss placement;
- calibration is allowed only if fit on past/calibration outcomes and then frozen.

## Deployability Definition

A candidate is deployable if, for each validation history, it may use only:

- the current history window;
- model parameters/checkpoints trained before validation;
- calibration tables or residual banks fit on pre-validation windows;
- random noise.

It may not use:

- the validation future path;
- validation-future errors;
- validation-future coverage outcomes;
- any per-validation-window correction fitted after seeing the target.

## Next Hypothesis

Convert the `435a` oracle idea into a deployable calibration policy by replacing the
oracle future center with a frozen forecast-error residual bank:

1. Use frozen `392a` as the learned conditional core.
2. On pre-validation calibration windows, compute `392a` sample medians and realized
   future errors.
3. Store a residual-error library, optionally binned by a generic history volatility
   proxy.
4. At validation inference, generate scenarios as:

   `validation 392a median + sampled calibration residual error + small 392a residual shape`

This is a split-conformal / residual-bootstrap risk policy. It is deployable because the
residual bank is fit only on past outcomes.

## Falsifier

Run one frozen-policy evaluator with no validation-future fitting.

Success:

- reaches `11/11` as a deployable calibrated risk system.

Failure:

- improves coverage/regime/distribution but loses structure;
- or remains below `392a`;
- or exposes that validation forecast errors are not stable enough to transfer from the
  pre-validation calibration block.

If this fails, do not tune many calibration knobs. Analyze whether a deployable policy
route is viable at all.

