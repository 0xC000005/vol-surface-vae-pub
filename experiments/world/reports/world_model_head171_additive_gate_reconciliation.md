# World Model HEAD171: Additive Gate Reconciliation

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`part1_quality_gate_reconciliation`; no model change.

## Hypothesis

The additive-signal evidence should change the interpretation of the learned
embedding, but not promote Part 1, unless raw-plus-learned clears the
exact-state and persistence guardrails.

## Falsifier

This reconciliation would fail if HEAD167-170 showed the learned embedding
passes the additive gate, unblocks Part B, or reduces the IV exact-state miss to
a single probe artifact.

## Reconciled Additive Gate

| layer | current status | evidence |
| --- | --- | --- |
| abstract/path-shape utility | PASS | HEAD167: path-shape/risk-width learned wins `2/2`, raw-plus-learned improves `2/2` |
| balanced regime signal | PARTIAL | HEAD167: macro-recall delta `0.168540`, accuracy delta `-0.039062` |
| persistence guardrail | PARTIAL | HEAD167: learned wins `0/2`, raw-plus-learned improves `1/2` |
| IV exact-state guardrail | FAIL | HEAD168: raw-plus-learned IV MSE is `1.048173x` raw-only |
| IV topology | FAIL | HEAD169: raw-plus-learned worsens `14/25` IV cells and improves `11/25` |
| probe scale artifact check | FAIL to clear blocker | HEAD170: standardized raw-plus/raw IV ratio remains failed at `1.213267` |

## Interpretation

The scaled Barlow embedding is not useless. It adds signal for non-surface
geometry, path-shape/risk-width probes, and balanced regime diagnostics. The
current failure is narrower and more useful to know: the embedding still
interferes with exact IV state and persistence under simple frozen probes.

This means the correct status is:

```text
additive abstract signal: present
exact-state/persistence guardrails: not passed
Part 1 promotion: blocked
Part B decoder: blocked
```

## Decision

Do not demote the scaled Barlow embedding as empty, but do not promote it as a
joint Part 1 representation. Future work should either explain the raw-plus IV
guardrail failure without adding a reconstruction loss, or broaden
additive-signal coverage beyond the current IV-future/regime diagnostics.

Part 1 remains `DO_NOT_PROMOTE`; Part B remains blocked.
