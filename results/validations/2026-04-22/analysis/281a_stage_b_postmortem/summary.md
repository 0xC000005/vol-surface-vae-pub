# Stage B Comparison Postmortem

## Compared Models
- `277d`: deterministic Stage A core only
- `278a`: raw retrieved future scenarios
- `279a`: fully centered residual scenarios
- `279b`: partially centered residual scenarios
- `280a`: scalar residual scaling
- `280b`: horizon-scale residual scaling

## Main Comparison
- `277d` proves the deterministic Stage A backbone is real:
  - `5/11`
  - perfect center preservation by construction
  - zero useful spread
- `278a` proves the raw retrieved residual bank already contains useful Stage B shape:
  - best coverage (`74.5%`)
  - strongest coverage floor (`14` bad windows)
  - strongest change-KS fidelity (`23/25`)
  - but center preservation is too weak, so mean reversion fails
- `279a` proves center-preserving residualization is directionally right:
  - `5/11`
  - mean_reversion restored
  - jump realism gets closest to the gate
  - but full centering removes too much of the useful Stage B shape
- `279b` proves centering strength alone is not enough:
  - recovers some spread and change-KS
  - but still loses too much mean_reversion and jump realism
- `280a` proves learned residual scaling is a live mechanism class:
  - global coverage and jump-scale metrics improve
  - but one scalar is too blunt and flattens regime allocation
- `280b` proves horizon-structured scaling is also real:
  - full-horizon mean_reversion returns
  - but coverage and jump realism deteriorate again

## Mechanism Read
The Stage B residual bank is not the problem.
The raw retrieved bank already contains the useful spread, coverage, and fidelity signal.

The live problem is how to use that bank **without moving the fixed Stage A center path**.

The recent sequence also shows something narrower:
- pure centering is too destructive
- pure scale control is too limited

So the next Stage B mechanism should not be another scale variant.
It should change the **selection shape** over the residual bank while preserving the
fixed center path.

## Decision
Next step: `281a-v0`

- keep the fixed `277d` Stage A center path
- keep the same top-k residual bank
- keep center preservation by centering residuals under the actual selection
  distribution
- replace scale-only control with a learned query-conditioned **temperature** over the
  fixed retrieval scores

This is the smallest non-scale Stage B mechanism that:
- preserves the Stage A center path in expectation
- changes which residual shapes are sampled
- reuses the existing retrieval geometry instead of learning a new scorer
