# World Model HEAD131: Scale Seed Stability

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`masked_multiview_invariance` scale/stability diagnostic.

## Hypothesis

If the HEAD127 scale improvement is real, same-config seeds should keep
retrieval, rank, variance, and redundancy in the same healthy range.

## Falsifier

A new seed collapses rank or variance, loses most same-state retrieval,
or shows materially worse redundancy than the HEAD127 seed.

## Seed Results

| seed | label | top1 | top10 | raw top10 | top10-raw | mrr | rank | var min | offdiag | final loss |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 680 | HEAD127 | 0.353776 | 0.849740 | 0.343229 | 0.506510 | 0.497244 | 22.323718 | 0.021119 | 0.162719 | 0.007212 |
| 681 | HEAD131_seed681 | 0.364844 | 0.877083 | 0.350521 | 0.526563 | 0.520146 | 22.176785 | 0.015874 | 0.163740 | 0.007328 |
| 682 | HEAD131_seed682 | 0.338151 | 0.839453 | 0.343359 | 0.496094 | 0.485981 | 22.353826 | 0.021701 | 0.159666 | 0.006819 |

## Summary Ranges

| metric | min | mean | max | std |
| --- | ---: | ---: | ---: | ---: |
| top1 | 0.338151 | 0.352257 | 0.364844 | 0.010950 |
| top10 | 0.839453 | 0.855425 | 0.877083 | 0.015880 |
| mrr | 0.485981 | 0.501124 | 0.520146 | 0.014215 |
| top10_minus_raw | 0.496094 | 0.509722 | 0.526563 | 0.012644 |
| effective_rank | 22.176785 | 22.284777 | 22.353826 | 0.077344 |
| variance_min | 0.015874 | 0.019565 | 0.021701 | 0.002621 |
| offdiag_abs_mean | 0.159666 | 0.162042 | 0.163740 | 0.001731 |
| final_loss | 0.006819 | 0.007119 | 0.007328 | 0.000218 |

## Decision

- Representation stability smoke passed: `True`.
- Promotion decision: `DO_NOT_PROMOTE`.

The scaled flat Barlow representation-health metrics are stable over three smoke-scale seeds. This upgrades scale/stability evidence for representation health, but it does not clear Part 1 because baseline superiority, regime probes, exact-state retention, and full-data stability remain unresolved.
