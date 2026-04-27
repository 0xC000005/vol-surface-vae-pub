# 648a Mixed-Coordinate Path Energy Fine-Tune

## Hypothesis

647a showed that the clean native joint path model preserves support,
cross-cell structure, and anchor-factor realism, but plain flow-matching MSE is
misaligned with conditional path-law evaluation. 648a keeps the 647
architecture fixed and fine-tunes with a minimal proper scoring objective:
full-path energy score in the same generated mixed-coordinate space, anchored
by the original flow-matching loss.

## Implementation

- Base checkpoint:
  `models/backfill/647a_joint38_mixed_path_flow_e8_w2048_s647/best_model.pt`
- Fine-tune target: generated mixed-coordinate future path.
- Loss: `fm_anchor_weight * FM + energy_weight * full_path_energy`.
- No architecture changes, no scalar sampling calibration, no factor-specific
  branches, no retrieval, and no post-hoc stress deck.

## Results

Artifacts:

- model: `models/backfill/648a_joint38_mixed_path_energy_ft_e3_w2048_s648/best_model.pt`
- IV suite: `results/autoresearch/648a_joint38_mixed_path_energy_ft_e3_w2048_s648/full11.json`
- joint audit: `results/autoresearch/648a_joint38_mixed_path_energy_ft_e3_w2048_s648/joint_panel.json`

Focused tests passed:

```bash
pytest test_code/test_648a_mixed_coordinate_path_energy_finetune.py test_code/test_647a_mixed_coordinate_path_flow.py -q
```

IV 11-suite:

- score: 4/11, unchanged from 647a
- coverage worsened: overall 90% CI coverage 66.3% -> 54.0%
- conditionality worsened: MAE reduction -0.7% -> -12.7%
- level KS worsened: passing cells 10/25 -> 1/25
- median bias worsened: fraction gate 14/25 -> 3/25
- pathwise max-jump KS improved slightly: 0.618 -> 0.575, but still failed the
  relaxed 0.50 gate

Joint-panel audit:

- factor delta KS mean worsened: 0.087 -> 0.115
- factor KS pass dropped: 13/13 -> 12/13
- factor q99 tail pass dropped: 11/13 -> 10/13
- factor-factor correlation stayed high: 0.829 -> 0.836
- IV-factor correlation stayed high: 0.845 -> 0.838

## Mechanism Read

The minimal full-path energy score pushes movement/tail amplitude but does not
fix conditional probability allocation. It introduces an upward IV level bias,
damages coverage, and slightly damages factor marginal realism. This confirms
that simply adding a sample-level proper score to the path model is not enough;
the objective is still blind to the specific directional/level calibration
failure that dominates the IV suite.

## Decision

Close 648a as a negative result. The next move should not be a stronger scalar
energy weight or temperature sweep. The failure suggests that the model needs a
clean way to learn conditional location and conditional dispersion together,
rather than pushing spread through an undirected full-path distance.
