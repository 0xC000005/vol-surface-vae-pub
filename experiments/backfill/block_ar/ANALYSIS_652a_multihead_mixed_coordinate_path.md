# 652a Shared-Source Multi-Head Mixed-Coordinate Path Flow

## Purpose

652a tests the user's decoder-expressiveness hypothesis without changing the
scientific object. The model remains one joint conditional path law with one
history encoder, one sampled stochastic source, and one future denoising
backbone. The only specialization is the observation readout:

- IV channels use an IV value adapter and IV output head.
- Anchor-factor channels use a generic factor value adapter and factor output
  head.
- IV-only and joint38 are both supported by the same implementation.

This is not the 574a-style composition deck. The IV surface and anchor factors
are sampled in one model call from shared randomness.

## Data Preconditions

651a verified the data framing before this run:

- The source panel is a same-day wide market panel with `25` IV cells and `13`
  anchor-factor state variables.
- The modeled state target is `38` channels, not `51`; the extra `13` source
  columns are reference return/diff diagnostics.
- Future states reconstruct exactly from modeled one-day encoded increments for
  30/60/90/152-day horizons.
- No validation/test leakage was detected.

## Commands

Joint38:

```bash
python experiments/backfill/block_ar/train_647a_mixed_coordinate_path_flow.py \
  --state_scope joint38 \
  --head_mode multihead \
  --head_hidden 128 \
  --epochs 8 \
  --max_train_windows 2048 \
  --batch_size 32 \
  --memory_dim 128 \
  --memory_layers 3 \
  --memory_heads 4 \
  --memory_ff 256 \
  --token_dim 128 \
  --token_layers 3 \
  --token_heads 4 \
  --token_ff 256 \
  --flow_steps 16 \
  --prefix_feature_mode scale \
  --sample_count 4 \
  --sample_steps 8 \
  --chunk_size 2 \
  --seed 652 \
  --device cuda \
  --output_dir models/backfill/652a_joint38_multihead_mixed_path_e8_w2048_s652
```

IV-only used the same command with `--state_scope iv_only` and output directory
`models/backfill/652a_ivonly_multihead_mixed_path_e8_w2048_s652`.

## Results

| model | IV suite | failed suites |
| --- | ---: | --- |
| `652a_joint38` | `4/11` | coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| `652a_ivonly` | `5/11` | coverage, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |

IV-only is the clearer IV result:

- Surface validity passes.
- Horizon-level coverage passes at h1/h7/h14/h30, but per-cell coverage still
  has simultaneous undercoverage and overcoverage.
- Conditionality passes: MAE reduction is `6.6%`; per-cell worst MAE reduction is
  `-7.4%`; worst per-cell width ratio is `1.182`.
- ACF passes with correlation `0.941`.
- Daily-change KS passes with `19/25` cells.
- Cross-cell correlation passes with correlation ratio `0.804` and rank ratio
  `1.703`.
- Aggregate h1 mean reversion is nearly exact with gen/GT ratio `0.989`, but
  the full-horizon profile still fails.
- Tail and path realism remain weak: kurtosis ratio is `0.333`, q99 tail-scale
  passes only `12/25` cells, and pathwise max-jump KS is `0.591`.

Joint38 is weaker on the IV 11-suite but strong on native joint-panel checks:

- Factor daily-change KS mean is `0.0849`; `13/13` factors pass KS < `0.20`.
- Factor q99 tail ratio median is `1.126`; `13/13` factors pass `[0.5, 2.0]`.
- Factor-factor upper-triangle correlation similarity is `0.842`.
- IV-factor correlation similarity is `0.846`.

## Interpretation

The decoder-expressiveness concern was valid. A homogeneous 38-channel readout
was too crude for mixed IV/factor observations, and typed heads improve the
joint story without breaking the one-law requirement.

However, 652a does not solve the main deployability bottleneck. The remaining
error is not "IV and factors are glued together"; it is IV path-law calibration:
the model allocates movement unevenly across cells and horizons, producing too
many moderate moves in some cells, too few realistic extreme-shape events in the
aggregate, unstable level occupancy, and weak long-horizon mean reversion.

## Decision

Keep the shared-source multi-head architecture as the clean native joint
backbone. Do not return to composed stress decks for the joint product.

The next minimal step should target the loss/data coordinate, not add another
architectural branch. The most principled next candidate is a same-backbone
variant with better positive-level factor preprocessing and/or a path
distribution objective that directly scores realized future movement and tail
allocation while preserving the shared stochastic source.
