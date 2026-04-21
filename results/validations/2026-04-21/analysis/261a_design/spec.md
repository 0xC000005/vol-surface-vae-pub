# 261a-v0 Spec

## Goal
Add a stochastic residual scenario layer on top of frozen `260e` without reopening the deterministic search.

## Design
- freeze `260e` as the deterministic center-path core
- compute the center path in the same `asinh` local-scale change coordinate
- define residual targets as:
  - `target_residual_coord = target_change_coord - center_change_coord`
- train a shallow residual flow-matching model on those residuals
- at inference:
  - sample residual paths
  - convert them back to raw changes around the frozen center path
  - subtract the sample-mean raw residual per window so the scenario layer is exactly mean-preserving

## Architecture
- frozen base model: `260e`
- residual model:
  - shallow temporal conv backbone
  - direct residual-path output in transformed change space
  - conditioned on history context from the frozen base and the frozen center path
- no new anchor branch
- no motif/token machinery
- no extra deterministic architecture changes

## Training Loss
- vanilla flow-matching loss on residual targets
- plus one regularizer:
  - zero-mean raw-residual penalty from a small multi-sample draw

## Kill Criteria
Reject `261a-v0` if either occurs:
- it materially moves / destroys the `260e` deterministic structure
- it fails to improve at least one of:
  - coverage
  - conditionality
  - regime coverage
  - pathwise jump realism

## Elegance Constraint
Do not add new research knobs to `260`.
If `261a` works, it should remain a separate residual layer in both code and theory.
