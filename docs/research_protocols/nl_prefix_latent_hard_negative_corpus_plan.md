# NL Hard-Negative Corpus Audit and Regeneration Plan

## Objective

Build the hard-negative corpus required by the training method described in the
paper and presentation. The current direct Codex/GPT-authored 982g corpus is
valid as a positive multi-view narrative bank, but its stored hard negatives are
short rejection labels and the current bridge training reports do not consume
them as explicit paired hard-negative narratives.

This work is a gate before any new bridge-retraining claim. Until this gate
passes, the paper-facing method must not claim explicit hard-negative narrative
training as an implemented result.

## Required Standard

For each eligible historical 30-day prefix:

- preserve the existing direct Codex/GPT-authored positive narratives;
- generate hard negatives only through Codex's agentic authoring framework,
  GPT/Codex-authored calls, or trusted human/source text;
- provide at least one hard-negative narrative for each positive training view;
- link hard negatives to real incompatible historical windows when the loss
  needs a negative SNI memory target;
- require the negative to contradict at least one high-confidence grounding
  channel or regime mechanism;
- keep all text prefix-only, with no realized future leakage;
- ban deterministic/template/rule-based prose for both positives and
  negatives, including any local EpisodeCardV3-style narrative generator;
- keep frozen SNI rollout and top3/90 assembly unchanged.

## Why Links Matter

For text-space retrieval, a hard-negative text can be used directly. For the
text-to-memory bridge, a hard-negative text alone is not enough: the loss also
needs the incompatible historical window's SNI memory. The regenerated corpus
therefore has two layers:

1. hard-negative narrative views, written in the same style family as positives;
2. linked negative support windows that supply real incompatible SNI memories.

## Work Plan

1. Audit the 982g corpus and current bridge reports.
   - Count positive views, hard-negative texts, view coverage, and authoring.
   - Check whether hard negatives are linked to actual historical windows.
   - Check whether current bridge reports used explicit stored negatives.
2. Define the hard-negative schema.
   - Window id, positive view name, negative view name, negative text.
   - Linked negative window id, contradiction channels, contradiction rationale.
   - Source authoring metadata and leakage validation.
3. Select candidate negative windows.
   - Use structured current/recent market facts only.
   - Prefer windows with contradictory high-confidence directions or regime
     mechanisms.
   - Enforce train/heldout split safety and temporal non-overlap.
4. Regenerate hard-negative narratives.
   - Use Codex/GPT only.
   - Generate matched-view negatives rather than short labels.
   - Preserve the same risk-manager professional standard as the positives.
5. Validate the corpus.
   - Coverage: every training window has the required negative views.
   - Link integrity: every negative view points to a valid incompatible window.
   - Contradiction: at least one required channel or mechanism differs.
   - Leakage: no future, terminal, or realized-horizon language.
   - Authoring: no deterministic/template generated prose.
6. Retrain and compare only after validation passes.
   - Projected-memory bridge with explicit text and memory hard negatives.
   - Text-space contrastive retriever with explicit matched negative texts.
   - Keep top3/90 support assembly fixed for all comparisons.

## Deliverables

- `hard_negative_corpus_audit.json`
- `hard_negative_corpus_audit.md`
- `hard_negative_bank.jsonl`
- `hard_negative_validation_report.json`
- bridge/retriever retraining reports that explicitly list the negative rows
  used in training
- research-log entry and current-truth update before any public claim changes

## Non-Promotion Rule

This is not a paper/demo default change. It is a data-method consistency gate.
The current top3/90 paper/demo workflow remains protected until a verifier
reviews both the corpus validation and downstream scenario results.

## Current Status

As of 2026-06-03, the audit and manifest phases are complete and the
Codex/GPT-only generation path has a validated partial pass:

- audit result: `fail_needs_regeneration`;
- full target manifest: `32080` rows = `4010` windows x `8` positive views;
- generated hard-negative rows: `12433`;
- complete target windows: `1554`, plus one partial target window with one
  valid pre-existing sparse-user view;
- independent validation: `12433` valid rows, `0` errors, `65` warnings;
- warning interpretation: the sixty-five warnings are mechanical/factor-list
  high-overlap warnings where factor names overlap but directions are
  contradictory; the two new warnings are the `technical_factor_evidence` and
  `factor_list_baseline` rows for `joint39_train_1462`; no new warnings were
  introduced in the latest tranche;
- same-window negative links: `0`;
- distinct linked negative windows: `1978`;
- latest tranche note: a serialized `--batch-size 8 --max-new 256` tranche
  accepted all `256` requested rows, continued the normal scale-tranche path,
  extended contiguous full eight-view coverage through `joint39_train_1553`,
  and left the first missing manifest row at
  `joint39_train_1554__weekly_risk_monitor`;
- validator hardening: same-window negative links and exact positive-text
  copies are explicit validation errors;
- local generated prose: `False`;
- remaining rows before the full corpus passes: `19647`.

This is enough to prove the authoring/validation route, but not enough to claim
the full explicit hard-negative corpus is complete or to retrain promoted
retrievers.

## 2026-06-10 owner decision: daily 32,080-row bank de-scoped

The project owner decided the full daily bank is not required. The accepted
matched hard-negative corpus is the completed stride-5 14+14 lane (`988b`
bank: 802/802 targets pass; `990a` manifest: 11,228 matched pair rows, 0
validation errors), which `990e` training fully consumed. `985c` authoring is
paused permanently at 12,433/32,080 rows; the partial bank stays on disk as an
artifact. Completion criteria in this plan that reference "all 32,080 rows"
are superseded by the stride-5 criterion. The retraining/promotion criteria
are unchanged and still unmet (see `nl_prefix_latent_current_truth.md`,
Active HEAD Objective section).
