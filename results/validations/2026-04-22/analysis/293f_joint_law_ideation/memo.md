## 293f ideation

### Context
The `293d` vs `293e` comparison showed a clean support-object tradeoff:

- `293d` monolithic support:
  - harder to predict
  - stronger structural anchoring

- `293e` compositional support:
  - easier to learn
  - weaker structural anchoring

So the branch does not yet need a family shift. It needs one clean hybrid support representation.

### Decision
Next step: `293f-v0`

Keep:
- the `293` daily joint-token local-law core
- explicit support as the path-shape mechanism

Change:
- use a **hybrid coarse support object**

### Hybrid representation
Use:
1. one global coarse path anchor code over the full knot panel
2. one small continuous knotwise refinement path conditioned on that anchor

So the support scaffold becomes:

`coarse_anchor_panel + knot_refinement_panel`

before interpolation over the 30-day window.

### Why this is the clean hybrid
This preserves the strongest piece of `293d`:
- one explicit global coarse anchor that encodes structural path shape

And it adds the lightest possible version of what `293e` helped with:
- knotwise flexibility
- easier local adaptation of levels and jumps

Crucially, it does **not** add:
- a second support codebook
- another latent branch
- a retrieval bank
- an AR rollout reset

So it stays legible:
- one anchor
- one refinement

### Proposed mechanism
Training targets:
- global coarse path code:
  - same full `(5 x 25)` knot-panel codebook as `293d`
- knotwise refinement:
  - residual between the true knot panel and the decoded anchor knot panel
  - predict that residual as a bounded continuous refinement

Model:
1. Encode history.
2. Predict the global coarse anchor code.
3. Decode that anchor into a coarse knot panel.
4. Predict a small knotwise residual path conditioned on history and the chosen anchor.
5. Interpolate the refined knot panel across the full 30-day window.
6. Condition the daily joint-token decoder on that hybrid scaffold.

### Why this is preferable to another discrete hybrid
A second discrete support codebook would add too much branch machinery.

The point of `293f` is to test whether:
- explicit global structural anchoring
- plus light continuous refinement

is enough.

That is the narrowest clean test.

### Kill criteria
`293f` is alive only if it keeps or improves the structural gains of `293d`:
- change KS
- cointegration robustness
- MR structure

while also improving at least one softer path-shape suite:
- level KS
- regime differentiation
- pathwise jump realism

If it just averages the two failure modes and lands in the middle, then the support-object branch is probably near a local cap.

### Most principled next step
Implement `293f-v0` with:
- the `293d` monolithic coarse anchor code
- a small bounded knotwise refinement head
- unchanged daily joint-token local-law decoder
