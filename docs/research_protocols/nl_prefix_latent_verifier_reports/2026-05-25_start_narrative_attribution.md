# Start-versus-Narrative Attribution Verification

Date: 2026-05-25

## Verification Result

Verdict: `AGREE_WITH_LIMITS`

The narrow claim is supported: the new crossed diagnostic correctly consumes
the saved three-start rollout bundles, verifies that each start is held fixed
within its six narrative cases, and reports that raw-level scenario variation is
mostly explained by accepted starting level while start-normalized variation
retains a material narrative and interaction component.

This is not a causal attribution study and does not prove production readiness.
It is a model-behavior diagnostic over three accepted starts, six narratives,
64 generated paths per cell, and the current saved rollout bundles.

## What I Checked

- `experiments/backfill/block_ar/nl_start_narrative_attribution.py`
- `test_code/test_944a_nl_start_narrative_attribution.py`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_narrative_attribution_944a/start_narrative_attribution.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_narrative_attribution_944a/start_narrative_attribution.md`
- `paper/narrative_grounded_scenarios/generated_tables/table_start_narrative_attribution.tex`
- `paper/narrative_grounded_scenarios/main.tex`
- `docs/research_protocols/nl_prefix_latent_current_truth.md`
- `RESEARCH_LOG.md` tail entry dated 2026-05-25.

## Confirmed Correct

- The attribution cube is complete: starts `18`, `22`, and `40` crossed with
  six public professional narratives.
- For the current narrative-conditioned policy, each start has identical raw
  starting state across all six narratives (`max_start_diff = 0.0`).
- Each current-policy cell has generated state shape `(64, 30, 39)`.
- The decomposition shares sum to one for every reported policy/feature-space
  row.
- Reported headline values match the JSON artifact:
  - raw-level current policy: start `81.3%`, narrative `7.8%`, interaction
    `10.9%`;
  - start-normalized current policy: start `57.6%`, narrative `22.3%`,
    interaction `20.1%`;
  - start-only null: start `100.0%`, narrative `0.0%`, interaction `0.0%`.
- Pairwise KS metrics support the interpretation:
  - fixed-start narrative effect under current policy: factor KS `0.176`,
    portfolio KS `0.149`;
  - same-narrative start effect under current policy: factor KS `0.566`,
    portfolio KS `0.486`;
  - fixed-start narrative effect under the start-only null: factor KS `0.000`,
    portfolio KS `0.000`.
- The generated paper table and figure compile into the paper. The rebuilt PDF
  has no missing references, missing citations, missing assets, or overfull /
  underfull warnings.

## Issues Found

- `NOTE`: The analysis uses saved rollout bundles. It does not regenerate
  scenarios or perform a new OpenAI/Codex labeling pass.
- `NOTE`: The three-start grid is a useful attribution diagnostic, not a full
  global sensitivity study over all possible market starting levels.
- `NOTE`: The ANOVA-style decomposition is over standardized distribution
  summary features, not over full path distributions directly. Pairwise KS
  metrics are included to keep the distribution-level interpretation grounded.

## Independent Assessment

The diagnostic is worth keeping in the paper. It answers a real product
question more precisely than the previous fixed-start-only evidence:

- raw-level charts are expected to be dominated by the accepted start;
- after anchoring paths by the accepted start, narrative and interaction still
  explain a material share of scenario-summary variation;
- the start-only null confirms that the measured narrative effect is not coming
  from the starting level alone.

The correct wording is: "the narrative effect is demonstrated and material
after start-normalization, but the accepted starting level remains the dominant
driver of raw-level scenario geometry." Avoid stronger wording such as "the
narrative dominates the scenario distribution."

## Commands

- `uv run python experiments/backfill/block_ar/nl_start_narrative_attribution.py`
- `uv run pytest test_code/test_944a_nl_start_narrative_attribution.py -q`
- `uv run python -m py_compile experiments/backfill/block_ar/nl_start_narrative_attribution.py`
- `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
- paper source checks for missing refs, missing cites, unused bibliography, and
  missing figure assets.
