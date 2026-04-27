# 588a Official Full 11 Bridge for Unified Increment Flows

## Hypothesis

587a looked promising under the custom unified-flow audit, but it had not been
scored by the official IV 11-suite. A bridge is required before treating it as a
frontier candidate.

## Implementation

Added `experiments/backfill/block_ar/evaluate_588a_unified_flow_full11_bridge.py`.

The bridge:

- loads any unified increment-flow checkpoint;
- builds the same official validation windows as the full 11-suite;
- verifies IV history/future alignment between the unified panel and official
  `build_rollout_windows` tensors;
- samples the unified model;
- reconstructs IV future surfaces from generated increments;
- uses `FixedDeployableSampler` so conditionality tests call the same frozen
  samples;
- writes official `full11.json` and `full11.md`.

Added `test_code/test_588a_unified_flow_full11_bridge.py` for alignment
diagnostics.

## Run

```bash
pytest test_code/test_588a_unified_flow_full11_bridge.py \
  test_code/test_577a_unified_increment_flow.py -q
```

Result: `9 passed`.

Official score:

```bash
python experiments/backfill/block_ar/evaluate_588a_unified_flow_full11_bridge.py \
  --checkpoint models/backfill/587a_conditional_affine_cumulative_flow_s587/best_model.pt \
  --max_windows 441 \
  --samples 48 \
  --sample_steps 16 \
  --batch_size 32 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 588 \
  --device cuda \
  --output_json results/autoresearch/588a_587a_unified_flow_full11/full11.json \
  --output_md results/autoresearch/588a_587a_unified_flow_full11/full11.md
```

## Result

587a official score:

- `1/11`;
- only block-AR boundary/growing-uncertainty suite passed;
- failed: surface, coverage, conditionality, time-series, cointegration, regime
  coverage, distributional fidelity, cross-cell correlation, mean reversion,
  pathwise jump realism.

Key metrics:

- explosion rate: `14.9%` versus gate `<5%`;
- overall 90% coverage: `55.7%`;
- h1 / h7 / h14 / h30 90% coverage: `46.5% / 48.9% / 52.4% / 67.5%`;
- conditionality MAE reduction: `5.2%`, but per-cell conditionality fails;
- kurtosis ratio: `10.09`;
- per-cell tail-scale passes: `6/25`;
- cointegration gen/GT ratio: `0.284`;
- regime layer2: `0/8`;
- daily-change KS passes: `7/25`;
- level KS passes: `0/25`;
- cross-cell corr ratio/rank ratio: `0.197 / 3.228`;
- h1 mean-reversion ratio: `0.307`;
- pathwise max-jump KS: `0.788`.

## Mechanism Read

The custom audit hid official-suite failures because it summarized broad path
scale and endpoint movement, but the official suite checks local structure.

587a is not merely undercovered:

- the full-path conditional-affine prior still has too many independent stochastic
  degrees of freedom, so cross-cell correlation collapses and effective rank is
  too high;
- cumulative loss makes most daily moves too smooth, while rare paths still
  explode enough to fail surface validity;
- h1 mean reversion is too weak, even though h7/h14/h30 mean reversion profiles
  are closer;
- distributional mass is misplaced across cells and levels.

This is below the historical `392a`/`510a` frontier and cannot be treated as a
candidate deployable learned law.

## Decision

Close full-dimensional conditional-affine source as a frontier route.

The next clean experiment should keep the cumulative path loss but reduce the
stochastic source degrees of freedom through a learned latent bottleneck. This is
not a hard low-rank readout; it is the allowed narrow encoder-decoder stochastic
bottleneck and directly targets the cross-cell/rank failure. If that cannot
recover cross-cell structure while preserving 587a's scale control, abandon the
unified MLP path-flow branch.
