# Independent Verification: Support-Max Quality-Guard Default Candidate

## Verdict

`AGREE`, with precise scope:

> `924e` is a legitimate **default-candidate** support policy under the current
> three-seed paired-CRN stability gate. It is not yet a settled production
> default without broader split/time-block confirmation.

## Checked

The verifier inspected:

- `experiments/backfill/block_ar/nl_portfolio_response_quality_guard_policy.py`
- `test_code/test_924a_nl_portfolio_response_quality_guard_policy.py`
- `experiments/backfill/block_ar/nl_portfolio_response_overlay_stability_gate.py`
- `test_code/test_924c_nl_portfolio_response_overlay_stability_gate.py`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_quality_guard_924e_t025_supportmaxq25/quality_guard_policy_bridge_report.json`
- CRN `925`, `926`, and `927` equal/candidate comparison artifacts.
- `nl_equal_top5_test_s8_crn*/scenario_eval/` equal floor reports.

The verifier also recomputed the three comparisons from eval JSON/NPZ inputs
instead of trusting the saved comparison JSON alone, and ran the focused 924a
and 924c tests.

## Confirmed

- The support-weight-max threshold is train-derived in the final implementation.
  The `924e` artifact records `min_support_weight_max_quantile: 0.25` and
  threshold `0.23393332011672088`.
- Labels and priors are train-only: train source is
  `nl_rollout_response_full_906e_train_mixture_labels` plus
  `nl_portfolio_response_train_mixture_labels_922f_crn_full`; held-out candidate
  source is separate.
- Sign convention is correct: lower-is-better rollout scores become
  higher-is-better utilities through `labels.append(-float(value))`.
- Equal/candidate comparisons are apples-to-apples for the tested claim:
  `samples=8`, `top_k=5`, `66` held-out windows, same seeds, same CRN base
  seeds, and intended equal-vs-field-weight support sampling.
- All three paired CRN comparisons are clean wins:
  - CRN `925`: CRPS `-0.0007958`, energy `-0.0001859`, coverage `+0.0006475`,
    reliable portfolio path `-0.0045456`.
  - CRN `926`: CRPS `-0.0014033`, energy `-0.0014312`, coverage `+0.0041699`,
    reliable portfolio path `-0.0059369`.
  - CRN `927`: CRPS `-0.0003380`, energy `-0.0000694`, coverage `+0.0024605`,
    reliable portfolio path `-0.0034396`.
- The stability gate returns `overlay_default_candidate`, with `3/3` clean seeds.

## Warnings

- The support-concentration idea came from held-out postmortem analysis. That is
  not leakage in the final implementation, but the exploratory origin must be
  reported honestly.
- Three paired CRN seeds are enough for the repo's current default-candidate
  gate, not enough to declare a final production default.
- The policy activates on `49/66` held-out windows and falls back on `17/66`, so
  part of the win comes from a conservative confidence gate.
- Tests cover arithmetic, fallback behavior, and gate classification, but do
  not fully prevent accidental train/test artifact path mixups beyond recorded
  source-path provenance.

## Recommended Status

Proceed with `924e` as the current default candidate and next production-facing
research line. Before replacing equal support mixture as the unconditional
default, run broader split/time-block confirmation and keep the exploratory
origin explicit in paper/demo claims.
