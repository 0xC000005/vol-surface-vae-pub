# World Model HEAD167: Additive-Signal Gate Audit

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_additive_signal_gate`; frozen probes only.

## Hypothesis

If the learned Part 1 embedding is useful as abstract market-state
information, it should add value to explicit raw state on path-shape,
risk-width, or balanced state probes without being treated as a
replacement for exact raw conditioning.

## Falsifier

The additive framing would fail if raw-plus-learned features add no
value on the abstract target families, or if exact-state guardrails
are treated as solved by a learned-only embedding that still loses to
raw current state.

## Feature Surfaces

- `raw_only`: explicit current-state floor.
- `learned_only`: frozen scaled Barlow embedding.
- `raw_plus_learned`: explicit raw state plus frozen embedding.

## Gate Layers

| layer | status | key evidence |
| --- | --- | --- |
| exact-state guardrail | FAIL | learned/raw IV MSE ratio `2.443584` |
| path-shape/risk-width | PASS | learned wins `2/2` and raw+learned improves `2/2` |
| persistence guardrail | PARTIAL | learned wins `0/2` and raw+learned improves `1/2` |
| regime balanced signal | PARTIAL | macro-recall delta `0.168540`, accuracy delta `-0.039062` |

## Decision

- Additive signal present: `True`.
- Gate passed: `False`.
- Promotion decision: `DO_NOT_PROMOTE`.
- Part B blocked: `True`.

Existing frozen probes show additive path-shape/risk-width and balanced-regime signal, but exact-state and persistence guardrails still block Part 1 promotion.
