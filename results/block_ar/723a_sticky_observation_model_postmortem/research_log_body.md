### Context

723a is a research-ideation/postmortem step after the sticky-coordinate branch accumulated negative evidence: sticky-zero thresholding, empirical atom gates, and hybrid sticky-score coordinates all failed in different ways.

### Findings

The failure is best reframed as an observation-model problem:

- OAS exact-zero daily changes likely mix true zero economic movement, stale quotes, tick rounding, and no-update events.
- A single continuous innovation target forces the flow to learn both no-update atoms and nonzero update sizes as one object.
- Threshold readouts and atom gates repair the no-change mass only superficially and weaken correlation amplitude.
- Score coordinates repair rank shape too aggressively and create extreme spread tails.

Artifact:

- `results/block_ar/723a_sticky_observation_model_postmortem/analysis.md`

### Decision

The next decisive experiment should keep the AR flow core but add a generic sticky observation adapter:

- select sticky channels by empirical train no-change rate;
- mask exact-zero sticky-channel targets out of the continuous flow loss so the flow learns nonzero update sizes;
- apply an empirical update/no-update readout after sampling as a diagnostic;
- leave IV and all non-sticky factors unchanged.

If this diagnostic is positive, replace the empirical atom gate with a learned gate. If it fails, the normalized-innovation family likely needs a broader observation-model paradigm shift rather than more coordinate tweaks.
