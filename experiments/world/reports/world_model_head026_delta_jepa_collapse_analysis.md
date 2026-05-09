# World Model HEAD026: Delta JEPA Collapse Analysis

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Question

Why did the canonical delta-target EMA horizon-JEPA from HEAD025 improve over
the old learned-target JEPA runs but still fail the Part 1 gate?

## Evidence Reviewed

HEAD025 history:

```text
epoch 1  MRR 0.023945  top5 0.017969  MSE 0.232170  pred rank 3.187364
epoch 2  MRR 0.024205  top5 0.021094  MSE 0.097684  pred rank 3.481381
epoch 3  MRR 0.024242  top5 0.021875  MSE 0.103887  pred rank 4.356162
epoch 4  MRR 0.024882  top5 0.021875  MSE 0.117920  pred rank 5.410567
epoch 5  MRR 0.025631  top5 0.022656  MSE 0.131088  pred rank 4.140211
epoch 6  MRR 0.026882  top5 0.022656  MSE 0.143746  pred rank 1.785323
epoch 7  MRR 0.028670  top5 0.021875  MSE 0.164003  pred rank 1.405544
epoch 8  MRR 0.027964  top5 0.025000  MSE 0.196536  pred rank 1.317041
epoch 9  MRR 0.029736  top5 0.024219  MSE 0.227996  pred rank 1.295986
epoch 10 MRR 0.042356  top5 0.043750  MSE 0.263569  pred rank 1.262683
epoch 11 MRR 0.034972  top5 0.028125  MSE 0.287803  pred rank 1.239630
epoch 12 MRR 0.030655  top5 0.021094  MSE 0.310662  pred rank 1.234572
```

HEAD025 selected-epoch per-horizon health:

```text
h1  MRR 0.042209  top5 0.050781  pred rank 1.233237  target rank 2.128783
h5  MRR 0.037001  top5 0.023438  pred rank 1.228867  target rank 2.067704
h10 MRR 0.049518  top5 0.066406  pred rank 1.229386  target rank 1.961267
h20 MRR 0.043025  top5 0.042969  pred rank 1.242761  target rank 2.161259
h30 MRR 0.040028  top5 0.035156  pred rank 1.232791  target rank 1.971792
```

HEAD006 learned-target comparators:

```text
prefix EMA: MRR 0.034284  top5 0.035156  pred rank 2.207024  target rank 2.494334
frame EMA:  MRR 0.033023  top5 0.032031  pred rank 1.715300  target rank 1.650369
```

Raw horizon-frame persistence reference:

```text
MRR 0.052426  top5 0.086719  top10 0.135156
```

## Mechanism Read

The delta coordinate helps, but only partially:

- HEAD025 improves learned-target MRR over HEAD006 prefix/frame EMA JEPA.
- The selected target latent remains low-rank across every horizon
  (`~1.96` to `~2.16` effective rank).
- The predicted latent is even more collapsed at the selected epoch
  (`~1.23` effective rank across horizons).
- The best retrieval epoch is not a healthy-representation epoch. The highest
  rank epoch is epoch 4 (`5.41`), but its MRR is only `0.024882`; the best MRR
  is epoch 10 (`0.042356`), where rank is `1.26`.

This means target coordinate was not the whole issue. EMA learned-target JEPA is
still finding a low-dimensional ordering signal rather than a compact but
usable multivariate world state.

## Failure Class

Primary: `latent_prediction`.

Secondary: `collapse`.

More specifically: target-space collapse precedes predictor collapse. The
target encoder is not providing a rich enough future latent for the predictor to
learn against.

## Decision

Do not add another retrieval loss, neighborhood loss, or scalar regularization
knob.

The next non-ad-hoc move should be a target-space contract, not a loss patch:

1. define a fixed target embedding baseline from horizon deltas, such as PCA or
   whitened low-rank SVD fitted on training deltas;
2. evaluate whether fixed target embeddings preserve rank and support
   retrieval/probes better than the EMA target encoder;
3. only after a fixed target contract works, revisit learned target encoders as
   an approximation to that contract.

This keeps the next step diagnostic and bounded: first prove the target space is
worth predicting before asking EMA JEPA to learn it.

## Next Step

Run `research_ideation` for a fixed target embedding contract. The literature
classification should be explicit: this is not canonical I-JEPA, but a
diagnostic target-space stabilization step motivated by the HEAD007 supervised
delta lower bound and HEAD025 target-encoder collapse.
