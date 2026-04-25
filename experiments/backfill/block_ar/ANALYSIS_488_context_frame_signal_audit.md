# 488 Context-Frame Signal Audit

## Context

After 487a, the 392a objective neighborhood looked capped: density-ratio resampling and marginal CRPS both moved calibration but did not solve conditional level allocation. The next clean question was whether the remaining failures are partly caused by the 30-day conditioning frame rather than the generative core.

The audit keeps validation forecast starts aligned across history lengths. For a fixed validation offset, `idx + history_len` is identical, so the future target is unchanged and only the available past context changes.

Artifact:

- `results/block_ar/488_context_frame_audit/context_signal_audit.json`
- `results/block_ar/488_context_frame_audit/context_signal_audit.md`

## Result

Chronological ridge probes on generic history summaries gave:

| H | level R2 | persistence R2 | MAE / persistence | h30 R2 | path move R2 | future vov R2 | hist-vov/future-vov corr | future abs turb/calm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.3445 | -0.1115 | 0.7314 | 0.3534 | 0.4883 | 0.5766 | -0.1964 | 1.0119 |
| 60 | 0.4905 | -0.1146 | 0.7767 | 0.3402 | 0.5095 | 0.5954 | -0.4735 | 0.8529 |
| 90 | 0.4845 | -0.1241 | 0.7754 | 0.3337 | 0.5083 | 0.5892 | -0.6776 | 0.9286 |
| 120 | 0.4539 | -0.1313 | 0.7650 | 0.2981 | 0.4298 | 0.6168 | -0.4983 | 1.0730 |
| 180 | 0.4593 | -0.1446 | 0.7625 | 0.2873 | 0.4284 | 0.5766 | -0.4470 | 0.9902 |
| 252 | 0.4750 | -0.1606 | 0.7373 | 0.3087 | 0.3780 | 0.3189 | -0.2046 | 1.1826 |

## Mechanism Read

Longer context is not a regime-width silver bullet: the simple history vol-of-vol split does not consistently imply larger future movement in the turbulent bucket, matching earlier oracle audits around the width gate.

But H60 adds real generic level/path signal over H30 without adding architectural machinery. H90+ does not add a clean incremental benefit and starts to look like capacity/data-framing drift.

## Decision

Run a single H60 empirical-normal-score causal-memory AR experiment next. This is methodologically clean: same vanilla one-step transition FM core and same empirical score coordinate system, only a larger conditioning window. If H60 cannot improve the 392a frontier, stop treating history length as the primary bottleneck.
