# 595a Recent Time-Series Research Ideation

## Context

594a closed the learned common-latent wrapper branch. The same-frame comparison showed that 510a and 593a share the broad-frame bottleneck: the model can produce locally realistic daily moves and cross-cell structure, but the full future-path law is not calibrated on level location, per-window coverage floors, regime-cell coverage, aggregate kurtosis, and active-cell mean-reversion geometry.

595a uses recent time-series deep-learning work to choose the next clean direction without adding another architectural patch.

## Sources Checked

- DistDF, ICLR 2026: `https://openreview.net/forum?id=VrdLwUmzBy`
- vLinear / WFMLoss, arXiv 2026: `https://arxiv.org/abs/2601.13768`
- Sundial / TimeFlow, arXiv 2025: `https://arxiv.org/abs/2502.00816`
- TSFlow, ICLR 2025: `https://openreview.net/forum?id=uxVBbSlKQ4`
- TimePFN, AAAI 2025 / arXiv: `https://arxiv.org/abs/2502.16294`
- MixLinear, ICLR 2026: `https://openreview.net/forum?id=QUj0KuCumD`

## Read

The relevant movement is not "make the architecture fancier." The useful thread is that time-series objectives should align the future sequence distribution, not only one-step transitions or pointwise errors.

DistDF is the closest match to our bottleneck. It frames forecasting as conditional distribution alignment and argues that standard direct losses can be biased under label autocorrelation. Its practical implication for this repo is to train against a joint future-sequence discrepancy rather than only a local transition or per-step objective.

vLinear/WFMLoss points in the same direction from flow matching: final-series-oriented objectives can outperform velocity-oriented objectives, and horizon/path weighting can be plug-and-play. That maps directly to our history: velocity/path-flow architectural shifts did not solve the suite, while 510a's broad-frame failures are about the final path distribution.

Sundial and TSFlow support continuous generative time-series modeling with flow matching, but they do not argue for our next move being another wrapper. They support keeping a clean generative core while changing how the future distribution is learned. TSFlow's data-dependent GP prior is interesting, but adding an explicit GP prior now would be another inductive knob before exhausting the objective-level fix.

TimePFN supports the idea that synthetic priors can help small-data multivariate time series, but that is a larger pretraining paradigm and should not be the immediate next move. It is a fallback if direct joint-distribution finetuning cannot move broad-frame calibration.

MixLinear is useful as a warning: low-resource time-series progress often comes from simple temporal/frequency structure, not huge model complexity. It does not directly address probabilistic scenario law calibration, so it is not the next experiment.

## Decision

The next clean experiment should keep the 510a/392a AR flow architecture fixed and change the finetuning objective to a final-path joint-distribution objective.

Proposed 596 direction:

- Start from 510a final checkpoint.
- Finetune the native AR flow model, not a wrapper.
- Keep architecture unchanged.
- Replace/augment patch energy with a broader final-path objective:
  - full-path energy score over `[horizon, cells]`;
  - sliced/projection Wasserstein terms over full paths;
  - explicit horizon weights that emphasize h7/h14/h30 because broad-frame failures grow with horizon;
  - a small FM anchor only to prevent transition-law drift.
- Evaluate on the same 441-window broad frame first.

This is first-principles aligned because it targets the learned conditional law itself, not a post-hoc stress deck. It is also Bitter Lesson aligned relative to the current evidence: the inductive bias is moved into the objective that tells the model what distribution to learn, not into a hand-engineered scenario path component.

## Guardrails

Do not add regime labels, per-cell hand corrections, low-rank decoders, bounded idiosyncratic paths, or separate IV/anchor-factor logic. If 596 needs many weights to work, close it. The objective should have one readable causal statement: "train the generator to match the joint future-path distribution on the broad validation-adjacent history frame."
