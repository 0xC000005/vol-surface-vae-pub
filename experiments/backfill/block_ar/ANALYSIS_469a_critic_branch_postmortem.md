# Autoresearch 469a: Conditional Critic Branch Postmortem

## Context

After calibration-table variants capped below the `392a` frontier, the next
learned-law idea was to replace hand-written marginal calibration pressure with
a learned conditional two-sample objective. The generator architecture stayed in
the `392a` family; only the training objective changed.

## Evidence

| Run | Objective | Score | Main result |
| --- | --- | ---: | --- |
| `392a` | FM + rollout energy | `8/11` | best deployable frontier |
| `467a` | global conditional path critic, adv `0.01` | `6/11` | coverage improves, but conditionality, cointegration, level/bias degrade |
| `468a` | same critic, adv `0.002` | `7/11` | less structural damage, but conditionality falls to `3.12%` and level/regime remain failed |

## Mechanism Read

The global path critic is not precise enough. It sees the whole `(history,
future)` path and can learn broad differences between realized and generated
paths, but the generator update is not constrained to preserve the conditional
center or the local structural features that `392a` already gets right.

Observed pattern:

1. Average coverage/calibration improves.
2. Conditionality weakens.
3. Level KS and regime layer-2 do not cross gates.
4. Structural suites can be preserved at low weight, but the score remains below
   `392a`.

This is effectively a learned marginal calibration pressure, not a clean
conditional-law improvement.

## Decision

Stop global critic variants. Do not add critic depth, heads, or weight sweeps
without a more local objective design.

The next viable direction must preserve what `392a` already gets right by
construction:

- conditional center / median behavior,
- daily-change KS,
- cross-cell correlation,
- mean reversion,
- pathwise jump shape.

The remaining failures are localized in level distribution and regime-cell
coverage. A future learned objective must therefore be local and residual-based
rather than global path-level discrimination.

## Candidate Next Paradigm

Train a residual-law model conditional on the frozen `392a` center:

```text
Y = center_392(H) + R(H, noise)
```

where `center_392(H)` is frozen or heavily anchored, and only the residual law is
learned. This is different from prior posthoc calibration because the residual
generator is trained as a model, not an empirical table. It is also different
from old center/residual hand engineering because the center comes from the best
learned generator and the residual model is generic.

First falsifier:

- freeze or detach a `392a` center estimate per history;
- train a small conditional residual flow/FM in empirical score or IV residual
  coordinates;
- objective: transition/FM or proper score on residual paths;
- sample: center plus learned residual path;
- no regime labels, no validation oracle, no calibration table.

If this cannot beat `392a`, then the suite is likely asking for information not
learnable from the current conditional signal without policy calibration.
