# 743a IV Hard-Cell Train/Validation Shift Audit

## Summary
- hard_cells_with_level_ks_gt_020: `4`
- hard_cells_total: `4`
- hard_cells_with_val_inside_train_tail_90_ge_080: `0`
- median_hard_level_ks: `0.450113`
- median_hard_val_in_train_tail_90: `0.554422`

## Hard Cells

| horizon | cell | base cov | level KS | cum KS | inc KS | val in train 90 | mean shift std |
|---:|---|---:|---:|---:|---:|---:|---:|
| 14 | (2,3) | 0.673 | 0.401 | 0.159 | 0.084 | 0.603 | -1.023 |
| 30 | (0,2) | 0.603 | 0.454 | 0.209 | 0.103 | 0.542 | -1.006 |
| 30 | (2,3) | 0.524 | 0.447 | 0.238 | 0.087 | 0.567 | -1.163 |
| 30 | (3,3) | 0.578 | 0.456 | 0.252 | 0.144 | 0.499 | -1.217 |

## Grid Shift

- level KS train-tail vs val: `{'median': 0.27891156462585037, 'p90': 0.4151927437641724, 'max': 0.5419501133786848, 'n_gt_020': 570, 'n_total': 750}`
- cumulative-change KS train-tail vs val: `{'median': 0.11564625850340138, 'p90': 0.20861678004535156, 'max': 0.2789115646258503, 'n_gt_020': 94, 'n_total': 750}`
- one-day-increment KS train-tail vs val: `{'median': 0.08843537414965985, 'p90': 0.13174603174603186, 'max': 0.1541950113378684, 'n_gt_020': 0, 'n_total': 750}`

## Mechanism Read

The hard cells show meaningful train-tail/validation distribution shift if their level KS exceeds 0.20, but validation targets remain mostly inside train-tail empirical intervals if the 90% containment is high. That pattern separates data drift from pure impossibility.

## Decision

Treat the defect primarily as a split-shift/oracle limitation before adding model complexity. A model-side fix is unlikely to be clean unless it uses a generic robustness/local-geometry mechanism.
