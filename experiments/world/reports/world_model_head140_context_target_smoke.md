# World Model HEAD140: Context-Target JEPA Smoke

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`context_to_target_jepa`.

## Hypothesis

A minimal same-window context-to-target JEPA can train on masked current/history
target blocks without using future targets or value reconstruction.

## Execution

Ran:

```bash
python experiments/world/part1_jepa_latent/context_target_jepa_smoke.py --device cpu --epochs 8 --max_train_windows 384 --max_val_windows 128 --seed 2140 --output_json results/world/context_target_jepa_smoke_head140.json --checkpoint models/world/checkpoints/part1_jepa_latent/context_target_jepa_smoke_head140.pt
```

## Result

Training loss decreased from `0.291315` to `0.027290`.

Validation metrics:

| metric | value |
| --- | ---: |
| loss | 0.036270 |
| alignment | 0.025615 |
| barlow | 0.213105 |
| clean-last effective rank | 9.414709 |
| clean-last variance min | 0.002395 |
| clean-last offdiag | 0.303647 |
| clean-flat effective rank | 6.495410 |
| clean-flat variance min | 0.003939 |
| clean-flat offdiag | 0.381745 |

## Interpretation

The scaffold is runnable and learns the target-latent smoke loss, but the
resulting clean representation health is weaker than the scaled Barlow
candidate. This is not evidence that context-to-target JEPA improves Part 1.

## Decision

`DO_NOT_PROMOTE`.

The next diagnostic should compare the frozen HEAD140 context encoder against
scaled Barlow on exact-state probes. If exact-state improves but rank/redundancy
remain weak, the model branch needs representation-health correction before any
Part 1 gate rerun.
