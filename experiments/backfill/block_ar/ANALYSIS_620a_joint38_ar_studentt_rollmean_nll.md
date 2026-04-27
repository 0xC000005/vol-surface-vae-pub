# 620a Joint38 Student-t AR Likelihood With Recursive Mean-Rollout Loss

## Hypothesis

The 619 analysis identified a train/sample mismatch in the likelihood path: one-step teacher-forced Student-t AR training learns broad conditional increments, but free-running samples do not preserve multi-step level occupancy or mean reversion. 620a tested the narrowest direct fix: keep the same unified 38-channel Student-t transition law and add a modest smooth-L1 loss on the model's recursively rolled predicted mean path.

This was intended to improve multi-step level placement without adding separate IV/factor treatment, retrieval, hand-built stress decks, or post-hoc evaluator calibration.

## Execution

Model:

- `GenericGaussianTransitionLaw`
- `state_scope=joint38`
- `distribution_family=student_t`
- `student_t_df=5.0`
- `rollout_mean_loss_weight=0.5`
- `rollout_mean_loss_steps=30`
- `rollout_mean_loss_beta=0.5`

Training selected epoch 2 as best validation NLL:

- epoch 1: train `-16.5749`, val `-6.6764`, rollout loss `0.2403`
- epoch 2: train `-30.0798`, val `-8.3243`, rollout loss `0.2000`
- epoch 8: train `-49.3462`, val `15.4570`, rollout loss `0.1679`

The rollout loss kept decreasing while validation NLL degraded sharply after epoch 2, indicating conflict between free-running mean placement and the probabilistic transition objective.

## Result

Official broad-frame IV 11-suite:

- score: `3/11`
- passed: `block_ar`, `cointegration`, `cross_cell_correlation`
- failed: `surface`, `coverage`, `conditionality`, `time_series`, `regime_coverage`, `distributional_fidelity`, `mean_reversion`, `pathwise_jump_realism`

Key metrics:

- cov90 overall: `0.920`
- conditionality MAE reduction: `3.22%`
- turbulent/calm width ratio: `0.927`
- regime layer2: `0/8`
- daily-change KS pass cells: `12/25`
- level KS pass cells: `0/25`
- median-bias cells: `13/25`
- mean-reversion active pass: `0.083`
- max-jump KS: `0.339`

## Mechanism Read

620a falsifies this particular multi-step mean-placement implementation. The added loss does not repair level occupancy or mean reversion and instead damages conditionality and surface realism. The likely mechanism is that a deterministic rolled mean path is the wrong training target for a stochastic risk scenario law: it encourages the model to chase the realized future center while the test and use case require calibrated multi-sample conditional path distributions.

This does not fully invalidate the 619 train/sample mismatch diagnosis. It does show that simply adding a free-running mean loss is not a clean or sufficient fix. Continuing to add auxiliary losses inside this same one-step likelihood branch would accumulate research knobs without a clear first-principles gain.

## Decision

Close the recursive mean-rollout-loss branch. The next principled step is analysis/ideation, not another local loss tweak. The open question is whether the deployable general model should remain an AR transition law or move to a direct sequence-level conditional path law that trains the same object it samples: full future paths for arbitrary channel panels, with one shared model for IV-only and 25+13 joint scenarios.

