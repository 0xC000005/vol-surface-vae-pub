# 754a Generated-Prefix FM Attribution

## Scorecards

| run | pass | cov90 | calerr | cond MAE% | risk | level KS | bias | coint | worst coint | regime L2 | MR | MR ratio | path KS | kurt |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 734a_val_incumbent | 6/11 | 0.836 | 0.046 | 9.24 | True | 17/25 | 16/25 | 0.723 | 0.209 | 0/8 | True | 1.050 | 0.442 | 1.003 |
| 746a_train_tail_incumbent | 6/11 | 0.861 | 0.014 | -1.63 | False | 20/25 | 25/25 | 0.492 | 0.276 | 2/8 | True | 1.093 | 0.529 | 1.783 |
| 752a_val_freeprefix_w020 | 5/11 | 0.806 | 0.075 | 9.08 | True | 11/25 | 13/25 | 0.739 | 0.200 | 2/8 | False | 0.853 | 0.360 | 1.029 |
| 752a_train_tail_freeprefix_w020 | 5/11 | 0.855 | 0.031 | -3.90 | False | 18/25 | 23/25 | 0.448 | 0.162 | 1/8 | False | 0.890 | 0.490 | 1.802 |
| 753a_val_freeprefix_w005 | 5/11 | 0.785 | 0.097 | 9.18 | True | 9/25 | 13/25 | 0.734 | 0.224 | 1/8 | False | 1.019 | 0.373 | 1.062 |
| 753a_train_tail_freeprefix_w005 | 5/11 | 0.837 | 0.050 | -4.70 | False | 18/25 | 23/25 | 0.458 | 0.243 | 2/8 | False | 1.044 | 0.477 | 1.868 |

## Deltas Versus Baselines

### 752a_val_minus_734a_val
- `cov90`: `-0.030676`
- `calibration_error`: `0.028824`
- `level_ks_pass_cells`: `-6.0`
- `median_bias_pass_cells`: `-3.0`
- `regime_layer2_pass_count`: `2.0`
- `regime_layer3_catastrophic_rate`: `0.021315`
- `cointegration_ratio`: `0.015873`
- `cointegration_worst_cell_ratio`: `-0.008955`
- `history_width_spearman`: `-0.033729`
- `future_width_spearman`: `0.020291`
- `mean_reversion_ratio`: `-0.19757`
- `mean_reversion_full_horizon_active_rate`: `-0.201666`
- `pathwise_max_jump_ks`: `-0.081859`
- `kurtosis_ratio`: `0.026817`

### 753a_val_minus_734a_val
- `cov90`: `-0.050927`
- `calibration_error`: `0.05059`
- `level_ks_pass_cells`: `-8.0`
- `median_bias_pass_cells`: `-3.0`
- `regime_layer2_pass_count`: `1.0`
- `regime_layer3_catastrophic_rate`: `0.026304`
- `cointegration_ratio`: `0.011205`
- `cointegration_worst_cell_ratio`: `0.014926`
- `history_width_spearman`: `-0.029596`
- `future_width_spearman`: `0.018088`
- `mean_reversion_ratio`: `-0.031156`
- `mean_reversion_full_horizon_active_rate`: `-0.15`
- `pathwise_max_jump_ks`: `-0.069161`
- `kurtosis_ratio`: `0.059386`

### 752a_train_tail_minus_746a_train_tail
- `cov90`: `-0.006389`
- `calibration_error`: `0.017105`
- `level_ks_pass_cells`: `-2.0`
- `median_bias_pass_cells`: `-2.0`
- `regime_layer2_pass_count`: `-1.0`
- `regime_layer3_catastrophic_rate`: `0.001995`
- `cointegration_ratio`: `-0.044412`
- `cointegration_worst_cell_ratio`: `-0.114154`
- `history_width_spearman`: `0.014662`
- `future_width_spearman`: `0.033105`
- `mean_reversion_ratio`: `-0.203073`
- `mean_reversion_full_horizon_active_rate`: `-0.107777`
- `pathwise_max_jump_ks`: `-0.039002`
- `kurtosis_ratio`: `0.0189`

### 753a_train_tail_minus_746a_train_tail
- `cov90`: `-0.023786`
- `calibration_error`: `0.036181`
- `level_ks_pass_cells`: `-2.0`
- `median_bias_pass_cells`: `-2.0`
- `regime_layer2_pass_count`: `0.0`
- `regime_layer3_catastrophic_rate`: `0.004535`
- `cointegration_ratio`: `-0.034384`
- `cointegration_worst_cell_ratio`: `-0.033073`
- `history_width_spearman`: `0.001059`
- `future_width_spearman`: `0.025516`
- `mean_reversion_ratio`: `-0.049933`
- `mean_reversion_full_horizon_active_rate`: `-0.07`
- `pathwise_max_jump_ks`: `-0.05238`
- `kurtosis_ratio`: `0.084819`

## Attribution

- primary read: `static_generated_prefix_fm_rejected_as_repair`
- Both generated-prefix weights score 5/11 on validation and 5/11 on train-tail, so the issue is not only validation distribution shift.
- The intended regime/path axes move partly in the right direction, but coverage, level-KS, median allocation, and full-horizon active mean reversion regress versus the incumbent.
- Lowering the weight to 0.05 recovers aggregate mean-reversion ratio but not the active-cell full-horizon gate; it also worsens validation cov90 and level-KS versus 0.2 and versus 734a.

## Mechanism Candidates

- `auxiliary_loss_conflict`: plausible. The generated-prefix FM target pulls the velocity field toward true future innovations under off-manifold generated prefixes, while level/channel energy separately scores integrated paths. The two losses may give incompatible gradients once the generated prefix is already biased.
- `too_much_generated_prefix_too_early`: plausible. A static auxiliary term uses generated-prefix states throughout fine-tuning, rather than gradually increasing generated-prefix exposure. This can train on low-quality prefixes before the model has adapted.
- `pure_weight_tuning`: rejected. Weights 0.2 and 0.05 both fail with similar gate pattern. More scalar search would be a research knob, not a first-principles fix.

## Decision

Do not continue generated-prefix FM scalar tuning. If exposure-bias work continues, make the next test structural and curriculum-based: generated-prefix exposure should be scheduled from teacher-forced to free-running prefixes, or limited to short prefixes, while keeping the same normalized-innovation AR flow core.
