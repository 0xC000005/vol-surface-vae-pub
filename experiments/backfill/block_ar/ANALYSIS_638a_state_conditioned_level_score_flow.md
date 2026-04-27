# 638a State-Conditioned Level-Score Flow Result

## Hypothesis

637a selected a coordinate experiment after 634a/636a showed that rollout-loss fine-tuning from 631a damages IV calibration. The hypothesis was that raw encoded-increment integration is the source of IV level drift and explosions. 638a therefore kept 629a/631a's level-plus-increment conditioning but changed the generated object to next empirical level-score changes.

## Execution

Implemented:

- `GenericStateConditionedLevelScoreFlowMatching`;
- `train_638a_state_conditioned_level_score_flow.py`;
- `evaluate_638a_state_conditioned_level_score_flow.py`;
- joint-panel audit support for `--model_type 638a`;
- focused tests in `test_638a_state_conditioned_level_score_flow.py`.

Verification:

```bash
pytest test_code/test_638a_state_conditioned_level_score_flow.py test_code/test_629a_state_conditioned_increment_flow.py -q
```

Training command:

```bash
python experiments/backfill/block_ar/train_638a_state_conditioned_level_score_flow.py \
  --state_scope joint38 --epochs 8 --max_train_windows 2048 --batch_size 32 \
  --memory_dim 128 --memory_layers 3 --memory_heads 4 --memory_ff 256 \
  --token_dim 128 --token_layers 3 --token_heads 4 --token_ff 256 \
  --flow_steps 16 --prefix_feature_mode scale \
  --sample_count 4 --sample_steps 8 --chunk_size 2 \
  --seed 638 --device cuda \
  --output_dir models/backfill/638a_joint38_statecond_levelscore_scale_e8_w2048_s638
```

Best epoch was 5 with validation loss `0.4391`.

## IV Result

638a scored `4/11`, tying 631a's count but with a very different profile.

Passed:

- `surface`;
- `time_series`;
- `block_ar`;
- `cross_cell_correlation`.

Key metrics:

- surface explosion `0.0%`;
- overall 90% coverage `69.7%`;
- conditionality MAE reduction `4.3%`;
- turbulent/calm width ratio `0.970`;
- daily-change KS `23/25`;
- level KS `6/25`;
- median-bias pass `9/25`;
- kurtosis ratio `0.940`;
- q99 tail-scale pass `22/25`;
- cross-cell corr/rank ratios `0.998 / 1.316`;
- mean-reversion ratio `0.702`, but active cells only `7/12`;
- pathwise q90/q99 ratios `0.733 / 0.896`, but max-jump KS `0.555`.

Artifacts:

- `models/backfill/638a_joint38_statecond_levelscore_scale_e8_w2048_s638/best_model.pt`
- `results/autoresearch/638a_joint38_statecond_levelscore_scale_e8_w2048_s638/full11.md`

## Joint-Panel Result

The joint-panel audit was materially worse than 629a/631a:

- factor daily-change KS mean `0.156`, pass `9/13`;
- factor q99 abs-change median ratio `2.135`, pass `6/13`;
- factor-factor correlation shape `0.514`, generated mean absolute correlation `0.052` versus GT `0.225`;
- IV-factor correlation shape `0.663`, generated mean absolute correlation `0.073` versus GT `0.149`.

Artifact:

- `results/autoresearch/638a_joint38_statecond_levelscore_scale_e8_w2048_s638/joint_panel.md`

## Mechanism Read

638a confirms the coordinate diagnosis for IV: generating next level scores directly eliminates surface explosions, fixes global kurtosis/tail scale, and greatly improves pathwise jump-size ratios. It also preserves IV cross-cell geometry.

But it fails as a native 25+13 joint law. Anchor-factor changes become too wide, and factor-factor / IV-factor absolute co-movement collapses. This is the mirror image of 631a: encoded-increment generation gives much better joint factor behavior but accumulates IV level drift; level-score generation stabilizes IV levels but is a poor generated coordinate for anchor-factor changes.

## Decision

Do not promote 638a as the active deployable joint model. Keep 631a as the incumbent native joint learned base and keep 638a as a decisive coordinate milestone.

The next principled question is whether a single architecture can use a spec-driven generated coordinate without becoming two separate models: IV-like mean-reverting level variables need support-valid level-score generation, while random-walk-like anchor levels need increment generation. If pursued, this must be framed as data-coordinate selection, not an IV/factor model branch.
