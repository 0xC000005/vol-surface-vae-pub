# World Model HEAD165: Exact-State Conditioning Boundary

Date: 2026-05-10

## Iteration Type

`research_ideation`

## Objective Family

Exact-state blocker analysis; no model change.

## Question

Should the learned Part 1 embedding be required to replace raw current-state
features for exact IV state, or should exact state remain an explicit
conditioning channel while the learned representation carries invariant/abstract
market-state information?

## Evidence

- HEAD132/149 show raw current IV features are the exact-state floor.
- HEAD127/130 scaled Barlow has healthy representation rank and retrieval but
  loses exact IV state to raw features.
- HEAD134 shows scaled Barlow can help path-shape/risk-width probes and can add
  to raw-last features, even while failing standalone exact-state baselines.
- HEAD154-157 show context-to-target routes do not fix exact-state retention;
  the target latent surfaces become low-rank or geometry-dominated.
- HEAD164 says raw-value reconstruction is a separate MAE-style diagnostic, not
  a JEPA patch.

## Interpretation

Raw current-state features are not just a weak baseline for exact IV state; they
are nearly the identity information needed to answer exact-state probes. A
compact invariant embedding can be useful without being the best standalone
container for every current IV value.

This means there are two different questions:

1. Can a learned representation replace raw current state for exact value
   retention? Current answer: no.
2. Can a learned representation add useful abstract/path-shape information when
   raw state is retained separately? Current answer: partially yes.

Conflating those questions makes the workflow oscillate between representation
learning and reconstruction. The cleaner boundary is:

```text
raw current state / observed-mask channels -> exact conditioning surface
learned JEPA embedding -> abstract/invariant market-state summary
downstream probes/decoder -> evaluate raw-only, embedding-only, and raw+embedding
```

## Decision

Do not try to make a single compact JEPA embedding replace the raw exact-state
conditioning channel by adding reconstruction-like patches. If future Part B is
authorized, it should evaluate raw-only, learned-only, and raw+learned
conditioning separately. For Part 1, the learned embedding still needs stronger
evidence that it adds robust abstract state beyond raw features before
promotion.

Part 1 remains `DO_NOT_PROMOTE`; Part B remains blocked until explicitly
authorized or a later gate changes status.
