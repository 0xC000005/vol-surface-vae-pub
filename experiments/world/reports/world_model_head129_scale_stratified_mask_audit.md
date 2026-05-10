# World Model HEAD129: Scale Stratified Mask Audit

## Objective Family

`masked_multiview_invariance` geometry/mask-family diagnostic.

## Hypothesis

If HEAD127 scaled checkpoint is robust under structured masking, retrieval/rank should not
collapse for one mask family while aggregate metrics look healthy.

## Grouped By View A Mask Family

| stratum | windows | top1 | top5 | top10 | eff rank A/B | offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factor_family | 31 | 0.668817 | 0.935484 | 0.946237 | 21.707070 / 21.697401 | 0.168058 |
| surface_maturity | 36 | 0.661111 | 0.950000 | 0.961111 | 21.500024 / 21.806755 | 0.170515 |
| surface_moneyness | 51 | 0.632680 | 0.945752 | 0.958170 | 21.358034 / 21.864775 | 0.168727 |
| surface_rectangle | 52 | 0.674359 | 0.938462 | 0.964744 | 22.209735 / 22.295904 | 0.163487 |
| time_block | 42 | 0.520635 | 0.808730 | 0.831746 | 21.089090 / 21.900987 | 0.153323 |
| vol_side_channel | 44 | 0.699242 | 0.968182 | 0.973485 | 21.467010 / 21.984848 | 0.167287 |

## Grouped By View B Mask Family

| stratum | windows | top1 | top5 | top10 | eff rank A/B | offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factor_family | 38 | 0.662281 | 0.903509 | 0.940351 | 21.914534 / 21.698616 | 0.160715 |
| surface_maturity | 46 | 0.702174 | 0.987681 | 0.988406 | 21.983817 / 21.614000 | 0.169753 |
| surface_moneyness | 44 | 0.715909 | 0.969697 | 0.970455 | 22.017691 / 21.475758 | 0.166423 |
| surface_rectangle | 52 | 0.654487 | 0.976923 | 0.980769 | 22.023664 / 21.964733 | 0.168168 |
| time_block | 46 | 0.640580 | 0.791304 | 0.804348 | 21.485262 / 20.667570 | 0.159182 |
| vol_side_channel | 30 | 0.761111 | 0.964444 | 0.964444 | 21.758789 / 21.614601 | 0.168638 |

## Decision

`no_large_stratified_failure`.

This is a validation diagnostic. It should guide masking/probe audits,
not introduce a new representation objective by itself.
