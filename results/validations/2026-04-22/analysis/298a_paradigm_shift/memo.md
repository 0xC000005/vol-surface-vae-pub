## 298a paradigm shift: stochastic structural center support

### Why a shift is required
The fixed-center hybrid program is now falsified enough to retire as the active line.

Evidence:

- `296c` showed a zero-mean shell can add broad calibration while preserving `277d` structure.
- `296e/f/g` showed fast-shell mechanisms can improve h1 coverage and jump scale.
- But every `296` variant kept level KS at `0/25`.
- `297a` showed even empirical training residual shells around fixed `277d` still keep level KS at `0/25` and collapse change KS to `4/25`.

So the issue is not just learned-shell weakness.

The issue is the interface:
- one fixed structural center path
- plus residual width around it

That interface cannot repair the unconditional level-law support.

### New hypothesis
The scenario generator needs stochasticity in the **structural center/support object** itself.

Instead of:

```text
one fixed center path + zero-mean shell
```

move to:

```text
sample a plausible structural center path + optional small shell
```

The center path should be sampled from a learned conditional distribution over realistic future path shapes, not chosen as one deterministic nearest neighbor and not averaged from top-k futures.

### Why not return to old retrieval weighting
The old `278-286` family mostly reweighted or transported retrieved futures. It failed because:

- top-k weighting/soft replay smoothed away sharp change-law behavior
- raw future support improved levels but broke MR
- anchored-delta support preserved structure but failed level-law fidelity
- local support interpolation did not reconcile both sides

`298a` should not repeat that.

### 298a-v0 proposal
Use the existing `277d` learned representation as a structural path manifold, but change the stochastic object:

1. Freeze the trained `277d` history and future encoders.
2. Train a small conditional density over future embeddings:
   - input: history embedding
   - target: future embedding
   - output: sampled future-embedding codes
3. Decode each sampled future embedding by nearest-neighbor / medoid lookup in the training future-embedding library.
4. Replay the selected future's normalized-change path from the query current state.
5. Evaluate the resulting structural center-support samples directly.

This is not a residual shell.
This is not top-k reweighting.
It is a stochastic model over structural future-path codes.

### Expected read
If `298a-v0` improves level KS while preserving `277d` structure, then the new support paradigm is alive.

If it fails with the same raw/anchored tradeoff, then the learned-representation retrieval manifold itself is probably capped, and the next paradigm must leave empirical support decoding entirely.

### Constraints
- no daily residual correction
- no per-cell gates
- no hard finance-specific level transport
- no train+validation support leakage
- do not count oracle diagnostics as frontier models
