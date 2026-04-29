# 764a Objective Reformulation Ideation

## Context

763a showed that 762a did not fail because proper level scoring is conceptually invalid. It failed because the new level-marginal CRPS term was added as a large extra objective component: `0.0985`, about `7.2x` the existing weighted channel-level-energy contribution. The generated validation spread narrowed by `4-6%`, and train-tail dropped from `8/11` to `6/11`.

## Options

| option | decision | reason |
|---|---|---|
| Budget-matched level-score replacement | recommended | Changes one objective-coordinate axis, keeps a proper score, and avoids uncontrolled objective scale. |
| Self-normalized multi-score objective | defer | More principled long-term, but introduces new optimizer dynamics before a simpler falsifier is run. |
| Conditional source scale / new readout | reject for next | Simple variants were already falsified in 678a, 689a, 692b, and 702b/c. |

## Recommended 765a

Run a budget-matched replacement, not another additive scalar stack:

- Source checkpoint: `models/backfill/674a_iv_channel_level_alltrain_w005_e3_s6731/best_model.pt`.
- Preserve 755a short-prefix generated FM: `free_running_fm_weight=0.2`, `free_running_fm_prefix_steps=5`.
- Remove raw channel-level energy: `channel_level_energy_weight=0.0`.
- Add standardized level-marginal CRPS at matched objective budget: `level_marginal_crps_weight=0.007`.
- Use `level_marginal_crps_coordinate=scaled_delta`.

The weight is derived from the incumbent level-score budget rather than swept: `0.05 * 0.275 / 1.970 = 0.00698`.

## Falsifier

Reject the replacement if train-tail remains below 755a by more than one suite or if validation coverage/level-KS remains worse than 755a. If it fails, the next step should be a broader objective/readout reformulation, not another nearby scalar weight.
