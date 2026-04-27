# 619 likelihood-pathology analysis and next step

## Context

614a-618a shifted from flow-matching MSE to conditional likelihood training over the same generic joint38 state panel. This was a principled objective shift: the model still uses one shared memory, one shared transition law, and the same IV-only/joint38 state-scope mechanism.

The likelihood path changed the failure mode but did not produce a deployable model.

## Evidence

Best likelihood-family result so far:

- 617a Student-t temp `0.90`;
- score `5/11`;
- passed: surface, conditionality, block-AR, cointegration, cross-cell correlation;
- failed: coverage, time-series properties, regime coverage, distributional fidelity, mean reversion, pathwise jump realism.

Key 617a temp `0.90` metrics:

- cov90 `91.6%`;
- conditional MAE reduction `9.2%`;
- persistent severe undercoverage `3.7%`;
- daily KS `17/25`;
- level KS `0/25`;
- median-bias cells `21/25`;
- kurtosis ratio `0.474`;
- mean-reversion ratio `0.370`;
- pathwise max-jump KS `0.479` pass;
- per-cell q99 jump-scale `6/25`.

618a tested whether a one-step residual mean-placement loss fixed the level/mean-reversion gap. It did not:

- score dropped to `4/11`;
- surface failed;
- daily KS fell to `14/25`;
- median-bias cells fell to `15/25`;
- cointegration worst-cell weakened to `0.269`;
- level KS stayed `0/25`;
- mean-reversion ratio stayed weak at `0.404`;
- per-cell q99 jump-scale stayed `6/25`.

## Stable Failure Pattern

Across Gaussian NLL, Student-t NLL, temperature diagnostics, and one-step mean loss:

- Aggregate coverage can be made high.
- Persistent severe undercoverage can pass.
- Conditionality can pass.
- Cointegration and cross-cell structure can pass.
- Daily-change shape can often pass.
- Level occupancy remains bad.
- Mean reversion remains weak.
- Regime layer2 remains `0/8`.
- Per-cell q99 tail scale remains highly imbalanced.

This is not the old narrow-interval problem. The likelihood models know how to spread mass, but they do not place multi-step future levels correctly.

## Mechanism Read

The core pathology is teacher-forced one-step AR likelihood.

During training, every transition is conditioned on the true prior future prefix:

```text
history, true future up to h-1 -> likelihood of true h increment
```

During sampling, every transition is conditioned on the model's sampled prefix:

```text
history, sampled future up to h-1 -> sampled h increment
```

This train/sample mismatch lets a one-step likelihood look acceptable while its free-running multi-step level process drifts into the wrong unconditional level occupancy. Broad Student-t innovations then recover coverage and undercoverage, but they do not fix the generated path law.

The failed 618a result supports this interpretation. A one-step mean penalty still acts under teacher forcing. It does not train the model to keep its own recursive future path in the right level regions.

## What Not To Do Next

Do not keep adding local width or temperature knobs:

- scalar temperature already failed for flow and only partially helped Student-t;
- learned source scale collapsed;
- scale-prefix features were a falsifier;
- one-step mean loss regressed important structure.

Do not split IV and anchor factors into separate paths. The native joint panel route remains methodologically cleaner.

## Next Clean Move

Add a differentiable multi-step mean-rollout placement loss to the likelihood model.

Principle:

- keep exact one-step Student-t NLL as the probabilistic core;
- during training, free-run the model's predicted transition mean for selected horizons;
- compare the resulting mean path levels to realized future score levels;
- backpropagate through the recursive mean rollout;
- keep one shared model and one shared panel.

This directly targets the train/sample mismatch without introducing post-hoc calibration or separate factor-specific treatment.

The loss should be generic:

```text
total = StudentT_NLL
      + lambda_rollout * smooth_l1(predicted_mean_score_path, true_future_score_path)
```

Expected effect:

- stronger multi-step level placement;
- stronger h1/h7/h14/h30 mean-reversion profile;
- better level KS;
- less need for broad stochastic coverage;
- potentially better per-cell tail balance because the stochastic component is not compensating for center drift.

Risk:

- too large a rollout weight may collapse stochastic diversity or hurt coverage;
- start with a modest weight and compare against 617a temp `0.90`, not against the weaker 618a.

## Decision

Proceed with a multi-step mean-rollout placement experiment next. This is more principled than another scalar calibration because it attacks the identified train/sample mismatch while keeping the architecture and methodology clean.
