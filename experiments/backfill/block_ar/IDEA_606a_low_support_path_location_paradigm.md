# 606a Low-Support Path-Location Paradigm

## Context

605a clarified the active bottleneck. The hard windows are near the edge of training
history support, especially in IV/joint IV+factor feature space, but the realized hard-cell
h30 moves are not extreme relative to training futures. The model is not failing because
the future amplitude is impossible. It is failing because, under a low-support history
state, it places future path mass in the wrong location/direction.

This closes the interval-width family:

- 602a state/horizon symmetric widening improved aggregate coverage but failed sparse
  regime/cell inclusion and damaged authenticity.
- 603a per-cell/horizon symmetric widening improved a few inclusion numbers but remained
  far below risk-readiness.
- 604a asymmetric lower/upper tail widening failed to learn the missing stress direction.

## Principle

For low-support conditioning states, the system needs a path-location allocation mechanism,
not a wider envelope. A valid next paradigm must satisfy:

- it changes where future paths go, not only how wide the interval is;
- it is activated by support uncertainty, not by validation-future knowledge;
- it preserves path authenticity by drawing from learned or empirical path dynamics;
- it keeps base learned-law metrics separate from policy/fallback metrics;
- it does not add per-evaluator knobs to the base model.

## Candidate Direction

The cleanest falsifier is a support-aware path-location fallback:

- estimate history support from training nearest-neighbor distance in IV/factor feature
  space;
- keep frozen 510a samples for in-support histories;
- for low-support histories, inject a bounded fraction of path samples built from
  historical future increments or a learned empirical increment source;
- anchor injected paths at the current surface so they remain valid scenarios for the
  current condition;
- evaluate as a disclosed risk-policy system, not as the base learned conditional law.

This is deliberately different from 602a-604a. It is not interval scaling around the same
wrong median. It can move sample paths into different future locations using historically
observed path shapes.

## Risk

This is not a pure Bitter-Lesson learned-law improvement. It is closer to a support-aware
fallback or stress overlay. That is acceptable only if:

- base 510a learned-law metrics remain reported separately;
- the fallback trigger is determined only from history support;
- the injected paths come from training/calibration history, not validation futures;
- the method is framed as risk deployment under low support, not as a calibrated general
  conditional density.

## Next Experiment

Run 607a as a support-aware path-location fallback audit:

- use 510a as the base generator;
- compute history support scores from train/validation history features;
- replace or augment only high-support-distance validation windows;
- generate injected paths by adding sampled training future increments to the current last
  surface and clipping to valid IV bounds;
- compare full suite and risk-readiness against broad 510a, 602a, 603a, and 604a.

Acceptance is not 11/11. The falsifier is whether path-location fallback improves
lower-only regime/cell inclusion materially without destroying scenario authenticity.
