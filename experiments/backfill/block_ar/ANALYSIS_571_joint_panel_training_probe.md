# 571 Joint Panel Training Probe

## Context

The requested extension has two parts:

- stack implied-volatility cells with additional financial factors;
- test both the regular 30-day horizon and a longer horizon to see whether joint training exposes data-curation or preprocessing blockers.

This is a controlled training-feasibility probe, not a claim that the joint model is better than the current IV-only risk-deck frontier.

## Method

Reused the existing `537a` daily panel transition law because it is already a genuine 51-channel autoregressive panel model:

- `25` IV surface cells;
- `13` broad factor levels;
- `13` factor returns/differences;
- normal-score marginalization plus a full daily Cholesky Gaussian transition in transformed panel space.

Two short CUDA training probes were run with the same model family and recent-window cap:

- `h30`: native 30-day future horizon;
- `h90`: longer 90-day future horizon.

The intent was to test whether the data path, tensor shapes, checkpointing, and sampling remain stable when IV and factor data are jointly modeled.

## Verification

Pre-run checks:

- `python -m py_compile experiments/backfill/block_ar/train_537a_panel_daily_cholesky_transition.py diffusion/block_ar/panel_daily_cholesky_transition_model.py experiments/backfill/block_ar/_panel_law_535_utils.py`
- `pytest test_code/test_537a_panel_daily_transition_density.py test_code/test_569a_factor_panel_readiness.py -q`
- Result: `7 passed`.

Training commands:

```bash
python experiments/backfill/block_ar/train_537a_panel_daily_cholesky_transition.py \
  --future_len 30 \
  --epochs 2 \
  --max_train_windows 512 \
  --n_quantiles 101 \
  --history_hidden 64 \
  --context_hidden 96 \
  --batch_size 32 \
  --lr 7e-4 \
  --device cuda \
  --seed 57130 \
  --output_dir models/backfill/571a_joint_panel_h30_probe_s57130
```

```bash
python experiments/backfill/block_ar/train_537a_panel_daily_cholesky_transition.py \
  --future_len 90 \
  --epochs 2 \
  --max_train_windows 512 \
  --n_quantiles 101 \
  --history_hidden 64 \
  --context_hidden 96 \
  --batch_size 24 \
  --lr 7e-4 \
  --device cuda \
  --seed 57190 \
  --output_dir models/backfill/571b_joint_panel_h90_probe_s57190
```

## Results

| run | future horizon | train shape | val shape | params | best epoch | best validation NLL |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `571a` | `30` | `(512, 30, 51)` history, `(512, 30, 51)` future | `(441, 30, 51)` history, `(441, 30, 51)` future | `193935` | `2` | `1.1407808321` |
| `571b` | `90` | `(512, 30, 51)` history, `(512, 90, 51)` future | `(441, 30, 51)` history, `(441, 90, 51)` future | `199695` | `2` | `1.0971591096` |

Checkpoint sampling smoke:

| run | sample shape | finite rate | IV min | IV max | factor min | factor max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `571a h30` | `(1, 4, 30, 51)` | `1.0` | `0.010893` | `0.898255` | `-0.110000` | `20809.419922` |
| `571b h90` | `(1, 4, 90, 51)` | `1.0` | `0.010748` | `0.898255` | `-0.110000` | `20809.419922` |

Artifacts:

- `models/backfill/571a_joint_panel_h30_probe_s57130/train_summary.json`
- `models/backfill/571b_joint_panel_h90_probe_s57190/train_summary.json`
- `results/autoresearch/571_joint_panel_training_probe/sample_smoke.json`

## Preprocessing Read

The joint data path is mechanically viable for both 30-day and 90-day training. The model trains, checkpoints, and samples finite 51-channel paths without shape or support failures.

The remaining issue is not "can it train"; it is whether the factor preprocessing policy is research-grade enough for full-scale use:

- existing `537a` utilities align factor parquet data to IV dates and use `ffill().fillna(0.0)`;
- `569a` showed the broad factor panel has nontrivial missingness before fill;
- the short recent-window probes likely avoid most early-start missingness, so they do not certify full-period data curation;
- factor scales are heterogeneous in raw space, so reporting and downstream risk interpretation need explicit units and transformations even if the model trains in transformed coordinates.

For a full joint model, the preprocessing policy should be made explicit before claiming generalization:

- use a documented per-factor missing policy, such as start-date truncation or backfill-then-forward-fill;
- avoid silent structural zero fills for factor levels unless zero has a real economic meaning;
- record factor source, transform, and unit in the training manifest;
- define evaluation metrics for factors and cross-factor co-movement, not just IV subpanel tests.

## Decision

Joint IV-plus-factor training is feasible at the regular 30-day horizon and at a longer 90-day horizon. No extra data curation is required just to make the joint version run.

Extra curation is still required before treating it as a defensible risk-manager model. The next principled step is to turn the preprocessing policy into a first-class manifest and then run an official 30-day IV-subpanel evaluation from the `571a` checkpoint, plus a separate long-horizon evaluation definition for `60/90/152/252`-day use.
