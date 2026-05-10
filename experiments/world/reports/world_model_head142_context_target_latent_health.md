# World Model HEAD142: Context-Target Latent Health

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`context_to_target_jepa_diagnostic`; no model or objective change.

## Hypothesis

HEAD140 may be weak because the trained loss acts on a target/predicted
latent surface that is lower-rank or more mask-family driven than the clean
context embedding used by downstream probes.

## Falsifier

The shortcut diagnosis is false if target and predicted latent rows have
healthy rank and mask-family probes do not beat majority by a meaningful
margin.

## Validation Mask Coverage

| family | windows | hidden rate | target row rate | last-row rate | tokens / target row |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 128 | 0.063802 | 0.767969 | 0.750000 | 4.818583 |
| factor_family | 20 | 0.015977 | 0.143333 | 0.100000 | 6.465116 |
| surface_maturity | 22 | 0.086207 | 1.000000 | 1.000000 | 5.000000 |
| surface_moneyness | 26 | 0.086207 | 1.000000 | 1.000000 | 5.000000 |
| surface_rectangle | 25 | 0.037241 | 1.000000 | 1.000000 | 2.160000 |
| time_block | 15 | 0.162222 | 0.162222 | 0.066667 | 58.000000 |
| vol_side_channel | 20 | 0.017241 | 1.000000 | 1.000000 | 1.000000 |

## Latent Health

| surface | effective rank | variance min | offdiag abs mean |
| --- | ---: | ---: | ---: |
| clean_context_last | 9.414709 | 0.002395 | 0.303647 |
| masked_context_last | 10.713954 | 0.009986 | 0.290194 |
| selected_context | 10.174329 | 0.015407 | 0.298217 |
| selected_predicted | 5.908201 | 0.001104 | 0.368069 |
| selected_target | 13.964280 | 0.008746 | 0.230051 |
| selected_target_values | 21.357869 | 0.000679 | 0.172881 |

## Alignment

| pair | MSE | cosine mean | top1 | top10 |
| --- | ---: | ---: | ---: | ---: |
| predicted_to_target | 0.025583 | 0.979068 | 0.005859 | 0.042969 |
| context_to_target | 0.497315 | 0.691316 | n/a | n/a |

## Mask-Family Probe

| feature | accuracy | majority | lift | macro recall |
| --- | ---: | ---: | ---: | ---: |
| selected_context_to_target_family | 0.253645 | 0.264496 | -0.010851 | 0.444013 |
| selected_predicted_to_target_family | 0.263140 | 0.264496 | -0.001356 | 0.455183 |
| selected_target_to_target_family | 0.732452 | 0.264496 | 0.467955 | 0.606307 |

## Decision

- Clean context last rank: `9.414709`.
- Target latent rank: `13.964280`.
- Predicted latent rank: `5.908201`.
- Target-family lift from target latent: `0.467955`.
- Target-family lift from predicted latent: `-0.001356`.
- Low-rank warning: `True`.
- Target latent mask-family warning: `True`.
- Predicted latent low-rank warning: `True`.
- High-cosine/low-retrieval warning: `True`.
- Promotion decision: `DO_NOT_PROMOTE`.

The branch is weak because the supervised target latent is heavily mask-family identifiable while the predictor collapses to a low-rank surface with high cosine but poor row retrieval. Do not add model knobs before fixing this target/predictor diagnostic.
