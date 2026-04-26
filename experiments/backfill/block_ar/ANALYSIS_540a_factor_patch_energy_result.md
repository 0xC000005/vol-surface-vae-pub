# 540a Factor-Conditioned Patch-Energy Fine-Tune Result

## Setup
540a tested the targeted synthesis selected in 539a:

```text
525a factor-conditioned empirical-score AR transition flow
+ 509a overlapping patch-energy proper score
```

Source checkpoint:

- `models/backfill/525a_factor_conditioned_surface_fm_e4_s525/best_model.pt`

Training:

- recent pre-validation adaptation windows: `441`;
- holdout split: `353/88`;
- objective: teacher-forced FM anchor + patch energy on differentiable free rollouts;
- patch length: `5`;
- patch energy weight: `0.05`;
- best holdout epoch: `3`.

Artifacts:

- Trainer: `experiments/backfill/block_ar/train_540a_factor_patch_energy_finetune.py`
- Checkpoint: `models/backfill/540a_factor_patch_energy_s540/best_model.pt`
- Full suite: `results/autoresearch/540a_factor_patch_energy_s540/full11.json`
- Summary: `results/autoresearch/540a_factor_patch_energy_s540/summary.md`

## Result
Score: `5/11`.

Passed:

- surface validity;
- block-AR checks;
- cross-cell correlation;
- mean reversion;
- pathwise jump realism.

Failed:

- coverage;
- conditionality;
- time-series properties;
- IV-EWMA cointegration;
- regime coverage;
- distributional fidelity.

Key metrics:

- overall 90% coverage: `0.879`;
- h30 90% coverage: `0.894`;
- conditionality MAE reduction: `4.2%`;
- daily-change KS cells: `25/25`;
- level KS cells: `9/25`;
- median-bias cells: `20/25`;
- cointegration ratio: `0.695`, but worst-cell ratio `0.246`;
- cross-cell correlation ratio: `0.970`;
- mean-reversion aggregate ratio: `0.988`, active cells `20/24`, active-cell corr `0.888`;
- pathwise max-jump KS: `0.348`.

## Mechanism Read
The synthesis did not collapse. It preserved strong structural geometry and recovered clean daily-change fidelity. It also nearly passed several thin gates:

- conditionality missed at `4.2%` versus `>5%`;
- cointegration worst-cell ratio missed at `0.246` versus `0.25`;
- time-series kurtosis ratio missed at `0.799` versus `0.80`;
- coverage had good aggregate calibration but failed per-cell over-95 checks at later horizons.

But it did not meet the acceptance gate:

- score is `5/11`, below the `8/11` frontier;
- level KS is `9/25`, below the `392a`/`510a` `10/25` and the factor branch's `12/25`;
- regime layer2 remains `0/8`;
- conditionality is still below gate.

The clean read is that combining factor conditioning with patch energy does not break the frontier. It moves the geometry toward a balanced structural sampler but does not solve future level occupancy or regime cell coverage.

## Decision
Close the factor-plus-patch synthesis. Do not sweep patch length, patch weight, checkpoint epoch, or factor adaptation length.

The local `392a` repair neighborhood has now been re-tested with the strongest constructive ingredients and remains below the existing `392a`/`510a` deployable frontier. The next step should not be another local fine-tune; it should be either a larger data/pretraining program or a frank frontier/deployability closure.

