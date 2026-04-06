# 184d Memo: Latent Activity Operator Mixture

## Why 184d

The focused `183c_best` vs `184c_best` review shows that `184c` did **not** fail because the generalized geometry-aware direction is wrong. It failed because the new sparse-precision operator was effectively turned off during training:

- `184b_best` activity gate mean: about `0.150`
- `184c_best` activity gate mean: about `3.2e-05`
- `184c_best` precision-delta mean: about `1.47e-07`
- node top-1 share stayed diffuse at about `0.11`
- kernel top-1 stayed flat at about `0.33`

So `184c_best` mostly preserved the old `183c` behavior:

- `S2` still passes
- `S8` still passes
- `S10` still passes
- `S11` still passes
- but `S3`, `S4`, and `S7` remain effectively unchanged

This means the next fix should target **operator engagement**, not another broad architecture rewrite.

## Core Design

`184d = latent residual activity mixture of operators`

Keep:

- explicit mean branch
- explicit covariance branch
- geometry-aware pathwise residual-law backbone
- grid geometry now, graph/group geometry later

Replace:

- free multiplicative sparse-precision gate

With:

- a **latent activity-state mixture** over residual operators
  - quiet operator: the current `183c`-style state-metric transport
  - event operator: the sparse precision / sharpened transport operator

## Why This Is Principled

This is not a hand-labeled quiet/event classifier.

It is a latent residual activity state:

- inferred from the conditional path context
- trained with a posterior during learning
- sampled from a prior at generation time

That keeps the model general:

- IV surface now: grid instance
- future multi-factor setup: graph/group instance

The principle is generic:

- most paths are governed by a quiet residual law
- a minority require a concentrated event residual law

What `184c` taught is that a **free scalar gate** can collapse to zero while the backbone still explains most of the data. A mixture of operators is the narrowest generalized fix because it forces the model to learn operator selection, not just operator shrinkage.

## Training Plan

1. Warm-start from `183c_best`.
2. Add posterior-guided operator assignments during training.
3. Stage 1:
   - freeze the backbone
   - train only the latent activity process and event operator
4. Stage 2:
   - light joint fine-tuning
5. Selection:
   - prioritize stricter `S4`
   - then `S3/S7`
   - preserve `S2/S8/S10/S11`

## Success Criterion

`184d_v0` is successful if it:

- materially raises quiet mass and lowers shoulder mass under tightened `S4`
- improves `S3` turb/calm differentiation
- improves `S7` regime-by-cell allocation
- preserves `S2`, `S8`, `S10`, `S11`

## Failure Interpretation

If `184d` also collapses, the conclusion is stronger:

- the generalized latent-activity path is still viable
- but the current operator family is not enough
- the next step would need a stronger event-law operator, not another gating tweak
