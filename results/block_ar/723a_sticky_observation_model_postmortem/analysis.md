# 723a Sticky Observation-Model Postmortem

## Why The Current Sticky Branch Is Exhausted

The last four iterations separated the failure mechanism:

- 718a showed AAA/BBB OAS have high exact no-change mass and calmer validation tails than training, so the failure is not a simple OOD/tail-width issue.
- 719a showed a deterministic sticky-zero readout is directionally useful but insufficient: it improves factor KS to `12/13` but leaves BBB above gate and weakens correlation amplitude.
- 720a showed stronger thresholds are not the answer: q10/q50 already overproduce zero moves, and q90 makes BBB and correlation worse.
- 721a showed atom probability alone is not enough: empirical global/history atom gates improve AAA but leave BBB high and reduce factor-correlation amplitude.
- 722a showed score-coordinate on sticky channels is too aggressive: it blows out OAS tails and contaminates broader factor marginal realism.

This falsifies the local family of threshold, atom-only, and score-coordinate repairs.

## Cleaner Data-Object Interpretation

For OAS-like channels, an observed zero daily change is ambiguous:

- It may mean the economic spread truly did not move.
- It may mean the quote was stale or rounded to a tick.
- It may mean an update event did not occur that day.

The current continuous normalized-innovation flow treats all three cases as one continuous target. That creates two failures at once:

- The model is penalized into placing too much mass near exactly zero.
- The same continuous path also has to explain the nonzero update sizes, which creates unstable tails when we try score or threshold repairs.

The first-principles object should be:

`observed move = update/no-update observation process + continuous nonzero economic move`

This is not an OAS-specific hack. It is a generic observation model for discrete, stale, ticked, or asynchronously updated financial series. IV cells remain continuous dense observations, so they receive the identity observation adapter.

## Most Principled Next Experiment

Keep the AR flow core and normalized innovation coordinate. Add a generic sticky observation adapter only for channels selected by the train no-change-rate rule.

Minimal diagnostic version:

- Sticky selector: same empirical train no-change rule as 719a/720a (`zero_rate >= 0.25`).
- Continuous path target: for sticky channels, downweight or mask exact-zero target increments in the flow-matching loss so the continuous path learns the conditional nonzero-update distribution.
- Observation readout: after sampling, apply an empirical update/no-update gate to restore no-change observations.
- Non-sticky channels: unchanged normalized-innovation loss and unchanged readout.

This tests the key hypothesis:

`The core flow can model nonzero spread updates if stale/no-update observations are removed from the continuous target.`

## Guardrails

- This must remain one frozen framework across IV-only, anchor-only, and joint38.
- The sticky selector must be data-derived and formulaic, not a list of OAS names.
- The atom gate should be diagnostic first; if positive, replace it with a learned gate later.
- Do not use score-coordinate for sticky channels unless the nonzero-masked experiment proves the remaining error is rank-shape rather than tail blowout.

## Decision

Next HEAD iteration should implement the minimal sticky-observation diagnostic: nonzero-masked flow loss for sticky channels plus the existing empirical atom readout. If this fails, the active normalized-innovation family likely needs a broader observation-model paradigm shift rather than more coordinate tweaks.
