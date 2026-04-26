# 545a Dual-Coordinate Joint Distribution Alignment Idea

## Context

544a tested the clean local-data idea from shift-aware normalization:

- remove causal local level/scale from each history window,
- model future paths in local normalized coordinates,
- invert samples back to IV levels.

The result was informative but below frontier: `5/11`. It passed daily-change realism,
cross-cell structure, cointegration, and path-jump realism, but failed level KS, mean
reversion, regime layer2 coverage, conditionality, and time-series kurtosis/skew.

The pathology is now specific: a purely local normalized frame is excellent for local
motion but too weak for long-horizon level allocation and reversion strength.

## Literature Signal

Recent time-series work supports a targeted change in objective/framing, not another
module stack:

- DistDF (`https://openreview.net/forum?id=VrdLwUmzBy`, ICLR 2026 poster) argues that
  direct forecast objectives can be biased when label sequences are autocorrelated and
  proposes joint-distribution Wasserstein alignment between forecast and label sequences.
- TimeBridge (`https://proceedings.mlr.press/v267/liu25cb.html`, ICML 2025) argues that
  short-term non-stationarity should be mitigated, but long-term non-stationarity and
  cointegration should be preserved rather than fully normalized away.
- EvoMSN (`https://openreview.net/forum?id=eQDdfqacoR`, ICLR 2025 withdrawn submission)
  reinforces the point that distribution shifts are multi-scale and that normalization
  and denormalization statistics matter dynamically.

The useful translation for this repo is:

1. keep the local normalized coordinate for daily motion realism,
2. also train generated raw IV-level paths against the realized raw IV-level path distribution,
3. use a path-level proper/distributional objective rather than a new handcrafted decoder.

## Proposed 546a Falsifier

Train the 544a model with a dual-coordinate loss:

- anchor loss: the existing local-score flow-matching transition loss,
- path loss: differentiable rollout samples inverted to raw IV levels,
- alignment: energy/Wasserstein-style path score on raw IV levels.

This is not a center/residual architecture and not a post-hoc calibration wrapper. The
generator remains single-stage. The added requirement is objective-level: generated raw
future paths must align with realized raw future paths, while the local frame still handles
short-horizon shift-normalized motion.

## Acceptance Gate

Run one decisive experiment only:

- must recover at least the `8/11` learned frontier before further work,
- must improve level KS and mean-reversion strength relative to 544a,
- must not lose daily-change KS, cross-cell correlation, or pathwise jump realism.

Do not sweep local-scale floors, temperatures, feature modes, path loss weights, or sample
counts. If the first clean dual-coordinate alignment falls below frontier, close the local
normalization branch.

## Why This Is Still Clean

This is the minimal response to the 544a failure mechanism:

- 544a over-optimized the normalized local-motion law,
- the suite still asks for the raw future IV-level law,
- therefore the next objective must include raw future path distribution alignment.

The bias is methodological, not domain-specific: align the generated joint future path law
in the coordinate where the deployment risk manager consumes scenarios.
