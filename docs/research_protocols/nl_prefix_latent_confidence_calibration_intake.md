# NL Prefix-Latent Scenario Confidence Calibration Intake

## Objective

Make the demo's 30-day directional summary table calibrated, interpretable, and
useful to risk managers without changing the underlying nearest-similar top3/90
support-grounded scenario generator.

The current table is directionally correct but easy to misread because it shows
three different notions of confidence in the same screen:

- grounding confidence: how explicit the story's current/recent market claim is;
- terminal direction confidence: how consistently the generated day-30
  distribution points above or below the accepted start;
- baseline-change strength: how much the narrative-conditioned distribution
  differs from the start-only baseline.

These must remain separate. A high-confidence grounding claim such as
``VIX down'' says the conditioning prefix should display lower volatility. It
does not imply the generated 30-day future distribution must have high
confidence that VIX is lower at day 30.

## Reproduction

The motivating live demo case is the start-22 fragile-risk-on narrative:

- report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_story_gradio_demo/prefix_latent_live_smoke/condition_only_run/prefix_latent_story_smoke_report.json`
- baseline report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_story_gradio_demo/prefix_latent_live_smoke/condition_only_run/start_only_baseline/prefix_latent_story_smoke_report.json`
- selected support:
  `joint39_train_3952`, `joint39_train_3668`, `joint39_train_3760`
- selected support direction checks:
  `0/6` mismatches for all three supports

The initial demo confidence rule was:

```text
High   if the 10%-90% terminal band is entirely above or below zero.
Medium if the band crosses zero but |mean terminal move| >= 25% of band width.
Low    otherwise, or when direction is flat/mixed.
```

This was intentionally conservative, but the label was too opaque for the demo.
In the reproduced run, most factors had wide terminal bands crossing zero. For
example:

- SPX narrative mean is positive, but the 10%-90% band is `[-44.8, 160.4]`,
  so terminal sign certainty is low.
- DXY narrative mean is positive, but the band is `[-1.89, 4.56]`.
- VIX narrative mean is negative and the mean is larger relative to the band,
  so it reaches medium confidence.
- IV surface narrative mean is negative with a tighter relative band, so it
  reaches medium confidence.

Therefore the old low labels were mostly not a support-selection failure. They
were a readout calibration issue: the table asked a strict terminal-sign
question, while the product user needed separate path-count and mean-magnitude
evidence for both baseline and narrative.

## Candidate Improvement

Replace the single heuristic with a calibrated readout:

1. **30-day typical view.**
   Use the empirical terminal sign split when available. A strong majority
   becomes `Up` or `Down`; a close split becomes mixed (`-`). This prevents a
   skewed mean from showing `Up` when most generated paths do not end up.
2. **Path count.**
   Use generated terminal samples to estimate sign probability and effect size:
   `P(delta > 0)`, `P(delta < 0)`, mean terminal move, 10%-90% band, and
   `abs(mean) / band_width`.
3. **Mean move.**
   Show the raw mean terminal move plus a standardized size, for example
   `+38 pts / +0.4σ`. This is kept separate from the view because expected
   value and majority path direction can disagree.
4. **Baseline-impact strength.**
   Separately compare narrative-conditioned samples to start-only baseline
   samples using mean difference, robust standardized difference, and
   probability of narrative terminal level being above the baseline terminal
   level. This supports labels such as `less down than baseline` without
   pretending the terminal outcome itself is high confidence.

The UI should keep arrows only in the view columns and the baseline-change
column. Path-count and mean-move columns use plain quantitative text.

## Prototype Readout

The prototype now persists empirical terminal sign shares in live top3/90
reports:

```text
terminal_probability_up
terminal_probability_down
terminal_sample_count
```

The demo table now displays:

```text
Market | Baseline View | Baseline Path Count | Baseline Mean Move |
Narrative View | Narrative Path Count | Narrative Mean Move |
30d Change vs Baseline
```

This keeps the baseline/narrative comparison explicit while making the
path-count/mean-magnitude disagreement visible. A case with a mixed path count
and a positive mean is shown with a mixed view, a dominant path-count percentage,
and a positive mean move such as `+38 pts / +0.4σ`, rather than a vague
high/medium/low confidence label.

## Experiments

1. Build a reproducible confidence diagnostic script for saved reports/arrays.
   It should output mean, p10, p90, sign probability, current confidence label,
   proposed confidence label, and baseline-impact strength for each market.
2. Run the diagnostic on the live start-22 fragile-risk-on case and the six
   professional narrative deck.
3. Scale to held-out historical backtests where realized 30-day futures are
   known. Evaluate whether high/medium/low buckets are calibrated against
   realized sign, CRPS/energy, and interval coverage.
4. Compare old and proposed confidence labels. Promotion requires clearer
   product interpretation without overstating certainty.

## Promotion Gate

Promote a new demo confidence rule only if:

- the rule is calibrated on held-out historical backtests;
- high confidence has materially higher realized sign hit rate than medium,
  and medium is higher than low;
- the rule does not convert broad, zero-crossing distributions into false
  certainty;
- the UI remains compact and does not add extra clutter;
- scenario generation, support selection, and top3/90 defaults remain unchanged.

## Non-Goal

This work does not change the scenario generator, support selector, grounding
schema, or top3/90 posterior ensemble. It only calibrates the risk-manager-facing
summary of generated 30-day distributions.
