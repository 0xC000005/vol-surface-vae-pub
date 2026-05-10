# World Model HEAD084: Stratified Mask-Family Audit

## Objective Family

`masked_multiview_invariance` geometry/mask-family diagnostic.

## Hypothesis

If HEAD070 is robust under structured masking, retrieval/rank should not
collapse for one mask family while aggregate metrics look healthy.

## Grouped By View A Mask Family

| stratum | windows | top1 | top5 | top10 | eff rank A/B | offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factor_family | 17 | 0.649020 | 0.911765 | 0.925490 | 13.740261 / 13.949887 | 0.215548 |
| surface_maturity | 17 | 0.731373 | 0.949020 | 0.956863 | 13.429407 / 13.969673 | 0.227577 |
| surface_moneyness | 26 | 0.575641 | 0.958974 | 0.970513 | 13.781823 / 14.462535 | 0.223083 |
| surface_rectangle | 27 | 0.674074 | 0.922222 | 0.956790 | 13.903622 / 13.951172 | 0.224370 |
| time_block | 16 | 0.512500 | 0.816667 | 0.827083 | 13.773838 / 14.511665 | 0.201137 |
| vol_side_channel | 25 | 0.694667 | 0.965333 | 0.977333 | 13.930140 / 14.518521 | 0.223129 |

## Grouped By View B Mask Family

| stratum | windows | top1 | top5 | top10 | eff rank A/B | offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factor_family | 16 | 0.627083 | 0.935417 | 0.956250 | 14.062426 / 13.900599 | 0.218308 |
| surface_maturity | 19 | 0.645614 | 0.992982 | 0.992982 | 13.873443 / 13.647177 | 0.231172 |
| surface_moneyness | 18 | 0.801852 | 0.979630 | 0.981481 | 14.217367 / 13.722375 | 0.225101 |
| surface_rectangle | 33 | 0.511111 | 0.974747 | 0.981818 | 14.084602 / 14.008144 | 0.224831 |
| time_block | 28 | 0.638095 | 0.809524 | 0.826190 | 14.428731 / 13.957516 | 0.199849 |
| vol_side_channel | 14 | 0.769048 | 0.957143 | 0.957143 | 14.186917 / 13.543651 | 0.223871 |

## Decision

`no_large_stratified_failure`.

This is a validation diagnostic. It should guide masking/probe audits,
not introduce a new representation objective by itself.
