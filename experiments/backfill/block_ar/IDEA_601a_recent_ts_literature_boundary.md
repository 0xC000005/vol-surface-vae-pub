# 601a Recent Time-Series Literature Boundary

## Question

After 599a and 600a, the current bottleneck is no longer a missing obvious module. Current
anchor factors did not explain the hard IV failures, and a local-geometry IV-history proxy
hurt the out-of-sample failure probes. 601a reviews recent time-series DNN directions and
decides what should be tried next without accumulating architecture knobs.

## External Literature Signals

- Local Geometry Attention, ICLR 2026:
  `https://openreview.net/forum?id=NCQPCxN7ds`. The paper motivates local Gaussian
  process style attention for robustness under corrupted time series. 600a locally tested
  the relevant cheap falsifier and found negative lift, so LGA should not be escalated
  into this generator now.
- DistDF, ICLR 2026:
  `https://openreview.net/forum?id=VrdLwUmzBy`. It motivates joint-distribution
  Wasserstein alignment instead of MSE-style direct forecasting under autocorrelated
  labels. This route has already been locally tested by 504a and 596a; the latter kept the
  510a architecture and used final-path energy plus sliced-Wasserstein style alignment,
  but it narrowed the deck and stayed at 5/11 broad-frame.
- vLinear/WFMLoss:
  `https://arxiv.org/abs/2601.13768`. It supports final-series-oriented flow objectives
  and simple efficient multivariate structure. This overlaps with the 596a final-path
  objective lesson: objective-level alignment alone did not add missing conditional
  variance here.
- FACT, ICLR 2026:
  `https://openreview.net/forum?id=j3gNYqrHtl`. It supports efficient fine-grained
  cross-variable convolution from time/frequency structure. This is relevant for a future
  panel backbone, but it is not the immediate fix for IV-only sparse regime undercoverage.
- TimeRecipe, ICLR 2026:
  `https://openreview.net/forum?id=CsoR8ztROC`. It supports module-level evidence rather
  than architecture folklore. This aligns with the current discipline: cheap falsifiers
  before adding learned modules.
- The Forecast After the Forecast, ICLR 2026:
  `https://openreview.net/forum?id=syfWdclGE1`. It directly supports a frozen-backbone,
  bounded post-processing direction via small input/output adapters, quantile calibration,
  and conformal correction. This is the only recent direction that matches the empirical
  situation without pretending that the base learned law has solved the sparse tail problem.

## Local Evidence Boundary

- Native law repair has been tried repeatedly: AR, one-shot, hierarchical, joint panel,
  local-shift score flows, common latent wrappers, joint Wasserstein/energy objectives,
  scalar temperature, current factor histories, and local geometry features.
- The best short-frame learned IV laws remain 392a/510a at 8/11, but broad-frame 510a is
  5/11. The 574a joint stress deck remains more useful as a risk-manager product than as a
  calibrated learned conditional law.
- The hard failures are not local daily-change realism. They are sparse conditional
  interval allocation, regime/tail width, level occupancy, and pathwise extreme placement.
- 597a shows global widening helps aggregate undercoverage but damages law quality; 600a
  shows local geometry does not identify the missing hard windows.

## Decision

Do not start another native architecture branch purely because a new paper has a module.
The clean next step is a bounded post-forecast calibration audit on a frozen generator:

- keep base generator samples unchanged and report them as the learned conditional law;
- add a disclosed, bounded adapter as a risk-policy/calibration layer;
- train or fit it only on historical calibration splits;
- evaluate whether it improves risk-manager usability without hiding base-law failures;
- report base-model metrics separately from final calibrated-system metrics.

This is not a Bitter-Lesson base-model win. It is a deployability boundary: if the frozen
law cannot learn sparse conditional tail allocation from available signal, a calibrated
risk overlay is defensible for risk management as long as it is disclosed and bounded.

## Next Experiment

Run a 602a frozen-postforecast adapter audit rather than a new learned core. The minimal
experiment should start from 510a broad-frame samples and test a small, auditable output
adapter against the current risk-manager acceptance criteria. If this cannot improve
coverage/pathwise realism without destroying conditionality, then the honest conclusion is
that more deployability requires new data/state variables, not another neural module.
