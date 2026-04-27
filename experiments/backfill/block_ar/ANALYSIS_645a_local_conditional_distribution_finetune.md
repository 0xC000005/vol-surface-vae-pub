# 645a Local Conditional Distribution-Alignment Finetune

## Hypothesis

644a proposed that each history has only one realized future, so one-step FM and one-realization path losses do not estimate enough conditional future-path mass. 645a tested a local conditional distribution-alignment finetune on top of 641a:

- keep the 641a mixed-coordinate native joint model;
- differentiably roll generated level-score futures;
- use nearby histories in score-space as a local empirical conditional future target;
- align generated local future distributions with a sliced-Wasserstein-style loss;
- keep the original FM loss as an anchor;
- no inference-time retrieval;
- no separate IV/factor heads or losses.

## Execution

Implemented:

- `experiments/backfill/block_ar/train_645a_local_conditional_distribution_finetune.py`
- `test_code/test_645a_local_conditional_distribution_finetune.py`

Verification:

```bash
pytest test_code/test_645a_local_conditional_distribution_finetune.py test_code/test_641a_state_conditioned_mixed_coordinate_flow.py -q
```

Result: `6 passed`.

Training:

- source checkpoint: `models/backfill/641a_joint38_mixedcoord_scale_e8_w2048_s641/best_model.pt`
- train windows: `1024`
- epochs: `2`
- batch size: `8`
- train samples: `4`
- rollout flow steps: `4`
- local weight: `0.05`
- FM anchor: `1.0`
- output: `models/backfill/645a_641a_local_distalign_e2_b8_s645/best_model.pt`

Training selected epoch 2:

- epoch 1: train total `0.6487`, validation total `0.7639`
- epoch 2: train total `0.6471`, validation total `0.7598`

## Result

IV full suite:

| metric | 641a | 645a |
| --- | ---: | ---: |
| score | 4/11 | 4/11 |
| cov90 | 0.647 | 0.581 |
| conditional MAE reduction | 4.51% | -1.08% |
| turbulent/calm width | 0.997 | 1.012 |
| daily-change KS pass | 24/25 | 23/25 |
| level KS pass | 4/25 | 1/25 |
| median-bias pass | 8/25 | 2/25 |
| kurtosis ratio | 0.946 | 0.661 |
| q99 tail pass | 23/25 | 21/25 |
| cross-cell corr ratio | 0.949 | 0.967 |
| mean-reversion ratio | 0.919 | 0.932 |
| pathwise max-jump KS | 0.610 | 0.425 |

Native joint audit:

| metric | 641a | 645a |
| --- | ---: | ---: |
| factor KS mean | 0.098 | 0.123 |
| factor KS pass <0.20 | 12/13 | 10/13 |
| factor q99 pass [0.5,2.0] | 13/13 | 12/13 |
| factor-factor corr shape | 0.866 | 0.873 |
| IV-factor corr shape | 0.852 | 0.863 |
| generated factor abs corr | 0.137 | 0.151 |
| generated IV-factor abs corr | 0.102 | 0.123 |

## Mechanism Read

The local distributional objective moved pathwise max-jump realism and correlation strength in the right direction, but it worsened the central IV risk qualities:

- coverage fell sharply;
- conditionality became worse than unconditional;
- level KS and median placement collapsed;
- global kurtosis failed;
- factor marginal quality weakened.

The likely mechanism is that local neighbor futures provide a useful distributional shape signal but a poor conditional location signal for this validation frame. The objective pulls paths toward a local historical-center distribution, which improves some path/correlation geometry while making the generated IV deck less conditionally calibrated.

## Decision

Close 645a at this weight and do not weight-sweep by default. The causal result is clear enough: local distribution alignment as implemented is not the missing deployability layer.

The next step should be a paradigm-level decision, not another auxiliary loss. The remaining viable choices are:

- return to 641a as the clean native joint baseline and treat it as a risk stress generator with explicit limitations;
- or shift to a direct sequence-level probabilistic model that trains the full future path law natively instead of one-step FM plus finetune losses.
