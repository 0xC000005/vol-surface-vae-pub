## 291b Stage A Discrete Joint-Support World Model

### Context
`291a` showed that a learned joint next-change support object is directionally better than factorized per-cell bins, but a **continuous** support code is too entangled to predict directly from the world-model hidden state.

The next clean move is therefore not:
- more continuous-code tuning
- bigger support MLPs
- adding side corrections

It is to keep the same world-model backbone and change only the support representation again.

### Decision
Next step: `291b-v0`

### Core idea
Learn a **discrete codebook** over one-step normalized-change panels.

Then:
- encode each target next-step panel to a discrete support code
- decode that code back to the full joint 25-cell change panel
- train the world-model hidden state to predict the next code logits or code index

This preserves:
- learned support from data
- joint panel structure
- deterministic Stage A rollout

while making the prediction target simpler and more stable than a free continuous support code.

### Minimal 291b-v0 design
Backbone:
- same `289e` history-memory world-model backbone

Support object:
- learned encoder for one-step normalized-change panels
- vector-quantized or nearest-codebook bottleneck
- learned decoder from codebook embedding to full 25-cell normalized change

Prediction head:
- world-model hidden state predicts next code logits
- rollout uses the argmax codebook entry deterministically

Training:
- codebook reconstruction loss
- code-prediction cross-entropy
- decoded next-change loss
- rollout level loss

### Why this is the smallest justified shift
The evidence now says:
- `289e` backbone: viable
- factorized per-cell output: capped
- continuous joint support: too entangled

So the smallest next move is:
- keep the backbone
- keep joint support
- make the support target discrete and learnable

### Kill criteria
`291b` is only alive if it recovers at least some of the `289e` shared-structure behavior while improving local-law realism over `290b`:
- corr ratio above `0.50`
- MR ratio above `0.70`
- change KS above `0/25`
- jump q90/q99 above `0.071 / 0.111`
