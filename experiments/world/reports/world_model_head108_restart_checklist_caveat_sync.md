# World Model HEAD108: Restart Checklist Caveat Sync

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Package consistency for `masked_multiview_invariance`.

## Hypothesis

If the manifest and package summary now carry newer Part 1 caveats, the restart
checklist should force future runs to read those caveats before any experiment
or downstream probe.

## Falsifier

The iteration fails if the restart checklist still omits the mask-policy
coverage, sample-scale, downstream raw-baseline, or IV-surface-only target-scope
caveats.

## Execution

- Added the HEAD097, HEAD100, HEAD102, and HEAD105 caveat reports to the
  checklist read-first list.
- Marked the HEAD070 checkpoint as smoke-scale directly in the fixed reference
  section.
- Added explicit restart caveats for validated mask families, full-data
  convergence, mixed downstream utility, and IV-surface-only future targets.

## Result

The checklist now matches the current manifest/package boundary and should
prevent a resumed run from overclaiming the HEAD070 reference candidate.

## Decision

Continue autoresearch in manual-stop mode.
