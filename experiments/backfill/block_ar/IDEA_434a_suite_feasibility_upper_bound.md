# 434a: Suite Feasibility Upper-Bound Diagnostic

## Context

The post-340 audit found no clean untested learned architecture route past the `392a`
`8/11` frontier. Before more model search, we need to know whether the current full
11-suite is satisfiable by any controlled scenario sample law.

This is not a publishable generator and must not be reported as a learned conditional
law. It is a feasibility upper bound for the suite/product target.

## Hypothesis

If even a validation-informed controlled sample law cannot reach or approach `11/11`,
then the suite is either internally too strict for this data slice or is certifying a
policy-calibrated risk system rather than a learned conditional generator.

If such a controlled sample law can reach `11/11`, then the suite is at least logically
satisfiable and the remaining question is which parts are learnable versus policy
calibrated.

## Diagnostic Object

Construct a non-learned scenario system with explicit provenance:

1. Use the validation ground truth only to build an oracle diagnostic.
2. Preserve realistic path residuals by borrowing residual path shapes from `392a` or
   historical futures.
3. Control coverage miss rates so the `[70%, 95%]` coverage caps are respected rather
   than trivially overcovering.
4. Evaluate the unchanged full 11-suite.

The diagnostic should report:

- base learned `392a` metrics;
- oracle-system metrics;
- exact oracle ingredients used;
- which suites remain impossible even with oracle control.

## Guardrails

This must be labeled as:

- `oracle_uses_validation_future=true`;
- `not_deployable`;
- `not_learned_conditional_law`;
- `suite_feasibility_upper_bound`.

It should not update `current_best_model`, because it is not a valid learned generator.

## Decision Criterion

If the upper bound reaches `11/11`:

- the suite is not contradictory;
- but a paper must separate learned-law performance from policy/oracle calibration.

If the upper bound remains below `11/11`:

- stop model architecture search;
- audit the suite gates or product definition before further autoresearch.

