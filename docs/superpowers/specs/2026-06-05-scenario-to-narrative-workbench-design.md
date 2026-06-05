# Scenario-to-Narrative Analyst Workbench Design

Date: 2026-06-05

## Purpose

Build a reverse demo for the natural-language scenario workflow:

```text
scenario or factor table
-> normalized numerical sidecar
-> visualization for human spot-checking
-> 14 positive narratives and 14 paired hard-negative narratives
-> strict JSON, Markdown review, and validation report
```

The demo should feel like a wrapped chatbot only in the sense that it has a simple
interactive surface. Its output must be structured, validated, and exportable.

## Product Shape

Use the selected **Analyst Workbench** layout, not a step wizard. The first screen
should show one dense review surface:

- left rail for source selection and upload;
- top status cards for normalization and validation state;
- numerical visualization before narrative generation;
- factor move table;
- paired narrative packet preview;
- audit and export controls.

The numerical visualization is not decorative. It is the review anchor that lets
users decide whether the generated narratives make sense against the scenario
evidence.

## Supported Input Modes

### Historical Joint39 Case

The user selects a historical `joint39_*` case. The app loads the existing
numeric historical prefix, support metadata, and factor evidence from the
repository artifacts. This is the highest-provenance mode and should be the
default demo path.

### Generated Deck Artifact

The user selects or uploads an existing generated-deck report and arrays artifact.
The app summarizes the generated paths using the same numeric deck-summary logic
already used by the reverse-caption audit. When available, use the calibrated
report terminal summary; otherwise derive terminal deltas from generated states
and the requested start vector.

### Uploaded Factor Table

The MVP requires numeric `start` and `end` levels for each supplied factor. This
matches the actual pipeline's numeric-state contract. Direction, magnitude, and
confidence may be accepted as optional annotations, but they cannot replace
numeric evidence.

Accepted CSV shape:

```csv
factor,start,end,confidence
SPX,1294.0,1311.0,medium
DXY,90.3,87.2,high
CRUDE_OIL,62.1,70.5,medium
GOLD,548.0,622.5,medium
BBB_OAS,1.42,1.42,medium
```

For strict Joint39-compatible uploads, require a full `values_by_name` object or
`state_vector` in raw-state coordinates. For lightweight factor-table uploads,
allow partial factor coverage, but label the sidecar as `factor_table_partial`
and do not claim it is a full Joint39 pipeline state.

## Internal Data Contract

Normalize every input into `ScenarioSidecarV1`.

Required fields:

- `scenario_id`
- `scenario_type`: `historical_joint39`, `generated_deck`, `factor_table_full`,
  or `factor_table_partial`
- `horizon_days`
- `factor_rows`
- `source_artifacts`
- `normalization_warnings`

Each `factor_row` should include:

- `factor`
- `start`
- `end`
- `delta`
- `direction`
- `magnitude`
- `confidence`
- `evidence`

Generated decks may also include:

- `sample_count`
- `p10_delta`
- `p50_delta`
- `p90_delta`
- `summary_source`

## Narrative Output Contract

Return `ScenarioNarrativePacketV1`.

Required fields:

- `scenario_sidecar`
- `positive_narratives`
- `hard_negative_narratives`
- `paired_review`
- `validation`
- `artifact_paths`

The packet must contain exactly fourteen positive narratives and fourteen paired
hard-negative narratives in the existing view set:

- `sparse_user_query`
- `weekly_risk_monitor`
- `mechanism_first`
- `technical_factor_evidence`
- `factor_list_baseline`
- `institutional_risk_committee_note`
- `risk_manager_memo`
- `full_professional`
- `sparse_variant_tape_read`
- `sparse_variant_portfolio_concern`
- `sparse_variant_macro_channel`
- `sparse_variant_credit_ambiguity`
- `sparse_variant_rates_commodities`
- `sparse_variant_desk_note`

All searchable narrative text must be authored by Codex/GPT or trusted human
source text. Local code may compute facts, summaries, visualizations, candidate
negative windows, validation checks, and audit metadata, but must not synthesize
training/demo narrative prose.

## Hard Negatives

For historical Joint39 and generated-deck modes, prefer real historical near-miss
negative candidates:

1. Convert the sidecar into a sign/magnitude vector over available factors.
2. Search the historical support bank for near but directionally incompatible
   candidates.
3. Assign one candidate per narrative view using focus-channel logic.
4. Ask Codex/GPT to author same-style hard-negative narratives from those
   assigned candidates.

For partial factor-table uploads, negative generation is allowed only if enough
factor evidence exists to identify meaningful contradiction channels. If not,
the app should return a structured warning and either skip hard-negative
generation or generate clearly labeled demo-only synthetic counterfactual
sidecars. Synthetic negatives must not enter training data.

## Validation

The workbench must validate:

- schema completeness;
- numeric start/end availability for uploaded tables;
- finite numeric values;
- factor-name mapping;
- full versus partial factor coverage;
- positive and negative text uniqueness;
- exactly fourteen expected views;
- negative candidate linkage where real historical negatives are used;
- no same-window negative link;
- no local prose generation flag;
- no internal training language;
- no direct negation shortcuts;
- no future/terminal scenario leakage, VaR/ES, target P&L, or post-horizon facts.

Validation failures should block export unless the user explicitly requests a
diagnostic-only artifact.

## UI Sections

### Input Rail

- Source tabs: `Historical Joint39`, `Generated Deck`, `Factor Table`.
- Historical selector or artifact upload.
- Factor-table upload with displayed accepted schema.
- Buttons: `Normalize and Visualize`, `Generate Narrative Packet`, `Export`.

### Numerical Review

- Factor path charts for historical cases and generated decks.
- Terminal distribution charts for generated decks.
- Factor move table with start, end, delta, direction, magnitude, confidence.
- Warnings table for missing factors, partial coverage, or unsupported formats.

### Narrative Packet

- Review-pair table showing positive and hard-negative text side by side.
- Tabs for all fourteen views, sparse-only views, validation, raw JSON, and
  Markdown.
- Export links for packet JSON, Markdown review, and validation report.

### Audit

- Source artifact paths.
- Prompt/schema version.
- Specialist-document hashes where applicable.
- Codex/GPT model and usage metadata.
- Local-prose flag.
- Validation summary.

## Implementation Plan Outline

1. Add `ScenarioSidecarV1` and `ScenarioNarrativePacketV1` schemas.
2. Add normalizers for historical Joint39 cases, generated deck artifacts, and
   uploaded factor tables.
3. Refactor the 14-view generator to consume a normalized sidecar instead of
   only a historical episode card.
4. Reuse existing deck-summary, hard-negative assignment, and validation logic.
5. Build the Gradio workbench screen using the selected analyst layout.
6. Run a small TestFlight across historical, generated-deck, full-table, partial
   table, and malformed-input cases.
7. Compile a human-reviewable packet before using the demo with external users.

## Non-Goals

- Do not train or promote a new text-to-latent model in this demo task.
- Do not use rule-based or deterministic local narrative generation.
- Do not accept direction-only uploaded tables for MVP narrative generation.
- Do not claim the generated deck caption is the original conditioning story.
  It describes the scenario evidence being captioned.
