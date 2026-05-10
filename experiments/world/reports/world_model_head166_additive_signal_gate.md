# World Model HEAD166: Additive-Signal Quality Gate

Date: 2026-05-10

## Iteration Type

`research_ideation`

## Objective Family

`part1_additive_signal_gate_design`; no model change, no decoder work.

## Hypothesis

If raw current state remains the explicit exact-conditioning surface, then the
learned Part 1 embedding should be judged by whether it adds abstract
market-state information beyond raw features, not by whether it replaces raw
identity information.

## Falsifier

This gate would be the wrong next step if raw-plus-learned features cannot
improve raw-only probes on any abstract/path-shape/risk-width/state target
family, or if the added embedding only helps by leaking mask artifacts,
target-window information, or probe-specific tuning.

## Evidence Basis

- HEAD132: scaled Barlow loses exact IV-state retention to raw last-surface
  features (`0.013756` versus `0.005630` IV MSE).
- HEAD133: scaled Barlow loses regime accuracy to the majority baseline but has
  better macro and rare-class recall than raw-last features.
- HEAD134: scaled Barlow wins `2/2` path-shape/risk-width targets and
  raw-last plus Barlow improves raw-last on `4/5` targets, while persistence
  and exact-state targets remain weak.
- HEAD160: Part 1 remains `DO_NOT_PROMOTE` after context-to-target demotions.
- HEAD165: raw exact state should remain explicit; learned embeddings should
  prove additive abstract state.

## Gate Definition

Evaluate the frozen representation with three feature surfaces on the same
splits and target definitions:

| surface | purpose |
| --- | --- |
| raw-only | exact-state and persistence floor |
| learned-only | standalone abstract representation quality |
| raw-plus-learned | additive market-state value over explicit raw state |

The gate is a downstream probe gate, not a pretraining loss. It must not update
the encoder, introduce future targets into Part 1 pretraining, or start Part B.

## Required Probe Families

| family | expected role | pass condition |
| --- | --- | --- |
| exact-state / persistence | guardrail | raw-plus-learned should not materially degrade raw-only |
| path-shape / risk-width | primary additive target | raw-plus-learned should beat raw-only, and learned-only should be nontrivial |
| regime / market-state labels | diagnostic | report balanced metrics, not only accuracy against majority |
| non-surface / factor geometry | coverage | include when data are available so the embedding is not IV-only |

Use simple frozen probes, shared splits, and paired validation comparisons. Do
not introduce a new Part 1 knob to make one probe pass.

## Decision

HEAD166 changes the acceptance framing, not the model status. The current
candidate is still not Part-B-ready. The next empirical step, if run, should be
an additive-signal audit over `raw-only`, `learned-only`, and
`raw-plus-learned` surfaces using existing frozen probe families.

Part 1 remains `DO_NOT_PROMOTE`; Part B remains blocked.
