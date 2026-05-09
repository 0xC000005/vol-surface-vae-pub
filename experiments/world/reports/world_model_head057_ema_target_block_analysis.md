# World Model HEAD057: EMA Target-Block Analysis

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

After HEAD056 improved over earlier EMA JEPA runs but still failed against the
fixed-PCA reference, what is the next non-ad-hoc Part 1 JEPA move?

## Evidence

Old EMA runs:

- HEAD006 absolute prefix/frame EMA JEPA: MRR around `0.033-0.034`, predicted
  rank around `1.7-2.2`.
- HEAD025 delta-frame EMA JEPA: MRR `0.042356`, predicted rank `1.262683`.

HEAD056 fused relative context with delta-frame EMA target:

- learned-target MRR/top5: `0.054649`/`0.057031`;
- predicted/target/context rank: `2.883115`/`3.806809`/`4.441587`;
- frozen-context fixed-PCA probe MRR/top5: `0.069033`/`0.086719`;
- frozen-context raw-delta probe MSE/MRR/top5:
  `0.019864`/`0.067651`/`0.074219`.

Current fixed-PCA reference:

- fixed-PCA ridge MRR/top5: `0.103763`/`0.136719`;
- raw-delta ridge MSE/MRR/top5: `0.015701`/`0.096147`/`0.128125`.

## Mechanism Read

HEAD056 shows the context-side repair mattered: fused relative context improves
rank and learned-target retrieval versus older EMA runs.

It also shows the isolated horizon-delta-frame target is too weak. The EMA
target geometry remains less discriminative than the fixed-PCA target contract,
and frozen-context probes are still below the active reference.

The next canonical target-construction change should follow the I-JEPA/A-JEPA
target-block principle more closely:

```text
relative past block -> fused context encoder
future relative/delta prefix block -> EMA target encoder
context + horizon token -> predictor
loss = MSE(z_pred, stopgrad(z_target))
```

The key difference from HEAD056 is target block size. The target should be a
future delta prefix ending at each horizon, not an isolated horizon frame. This
tests whether a richer future block gives the EMA target encoder a more stable
and discriminative latent geometry without adding another loss.

## Falsifier For Next Experiment

The prefix-block EMA run fails if:

- learned-target retrieval/top-k remains below HEAD056 or far below the
  fixed-PCA reference;
- predicted or target effective rank collapses below the HEAD056 selected
  values;
- frozen-context fixed-PCA/raw-delta probes remain materially below the
  reference.

## Decision

Run one target-construction experiment: fused relative context plus EMA target
encoder over future delta-prefix target blocks.

Do not add Barlow Twins, VICReg, retrieval/neighborhood objectives, target
sweeps, decoder work, or any new scalar loss weight.

## Artifacts

- `experiments/world/reports/world_model_head057_ema_target_block_analysis.md`
