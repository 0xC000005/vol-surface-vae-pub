# NL 14+14 Retrieval Training Plan

Generated: 2026-06-07

## Purpose

This plan defines the next self-supervised training work after the stride-5
14-positive / 14-hard-negative narrative bank was validated. It covers the two
retrieval families now under comparison:

1. narrative-to-narrative text-space retrieval with frozen-SNI preference
   reranking;
2. narrative-to-projected-memory retrieval with explicit paired hard-negative
   memory training.

The plan does not change the paper/demo default, the frozen SNI encoder/decoder,
or the top-3/90 component-preserving support assembly.

## Current Training Data

Use the validated stride-5 14+14 manifest, not the stale 8-view partial
hard-negative bank.

- Manifest report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_self_supervised_training_manifest_990a/training_manifest_report.json`
- Training examples:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_self_supervised_training_manifest_990a/training_examples.jsonl`
- Training pairs:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_self_supervised_training_manifest_990a/training_pairs.jsonl`
- Validation status: `pass`
- Target episodes: `802`
- Positive examples: `11228`
- Hard-negative examples: `11228`
- Total examples: `22456`
- Pair rows: `11228`
- Memory target rows: `4010`
- Validation errors: `0`
- Text digest:
  `c1586f2f5a4c1787ff4b874b21e3e742f6b011964f13d59ee9413dd52ea002ac`

Labeling rule:

- A positive text for target window `i` maps to `memory_targets[i]`.
- A hard-negative text maps to the linked incompatible window memory it
  describes, `memory_targets[j]`.
- A pair row preserves the source relation: target `i`, positive text for `i`,
  hard-negative text for incompatible window `j`.

## Method 1: Narrative-To-Narrative Text-Space Retrieval

Goal: preserve rich narrative information before applying numerical guardrails.
This branch stays in text-embedding space for retrieval and uses frozen SNI only
as a teacher/evaluator downstream.

Training input:

- all `22456` manifest texts embedded with `text-embedding-3-large`;
- same `label_window_id` across views forms positives;
- `training_pairs.jsonl` supplies explicit matched hard negatives;
- view names are retained for sparse-query and long-form robustness checks.

Training objective:

- supervised contrastive loss over examples sharing the same
  `label_window_index`;
- explicit pairwise margin that pushes a source positive away from its matched
  hard-negative text;
- optional view-balanced sampling so short sparse variants are not drowned by
  long professional views.

Retrieval/evaluation:

- retrieve candidate historical episode texts from adapted text embeddings;
- collapse text hits to support windows with diversity and temporal-overlap
  controls;
- apply grounded direction checks, accepted-start fit, and the unchanged
  top-3/90 component-preserving rollout;
- use frozen-SNI historical replay as Stage-2 teacher/evaluator to train or
  evaluate support preferences.

Required reports:

- text-space training diagnostics: same-window recall, hard-negative margin,
  sparse-query recall, view-family robustness;
- support audit: selected windows, direction pass rate, temporal diversity,
  overlap with raw OpenAI retrieval and projected-memory retrieval;
- frozen-SNI scenario report: CRPS, energy score, coverage, fixed-start
  conditionality, start-only lift.

## Method 2: Narrative-To-Projected-Memory Retrieval

Goal: implement the paper/presentation bridge method against the new 14+14
corpus. This is the method where text is projected into the frozen SNI
128-dimensional memory space.

Training input:

- same `training_examples.jsonl` and `training_pairs.jsonl`;
- `support_bank_arrays.npz::memory_targets` supplies the frozen SNI memory;
- every example has a `label_window_index` pointing to the memory it describes.

Correct objective:

- alignment: map each text embedding to its own described window memory;
- same-window multi-view alignment: different positive styles for the same
  window should project near the same memory;
- explicit paired hard-negative memory margin:

```text
cos(B(e(pos_i)), m_i) > cos(B(e(pos_i)), m_j) + gamma
```

where `j` is the linked incompatible negative window;

- optional reciprocal margin:

```text
cos(B(e(neg_j)), m_j) > cos(B(e(neg_j)), m_i) + gamma
```

so authored negative text is treated as a real description of window `j`, not
as a fake class attached to source `i`;

- optional in-batch contrastive loss over unique memory labels.

Implementation requirement:

The current bridge code is not enough by itself. It must be updated or wrapped
so the report proves it consumed `training_pairs.jsonl` and computed explicit
paired negative-memory margins. A run that only maps each row to its label
memory with generic in-batch contrastive cannot be described as the full
explicit hard-negative bridge method.

Retrieval/evaluation:

- rank support windows by cosine between predicted memory and historical SNI
  memory;
- combine with accepted-start fit and grounded direction checks;
- keep top-3/90 rollout unchanged;
- compare against raw projected-memory bridge, text-space method, start-only,
  and current paper/demo baseline.

Required reports:

- bridge alignment: target cosine, true-memory rank, hard-negative source
  margin, reciprocal negative margin, in-batch contrastive loss;
- support audit: support windows, direction pass rate, start fit, temporal
  diversity;
- frozen-SNI scenario report: CRPS, energy score, coverage, fixed-start
  conditionality, start-only lift.

## Baselines

Every promoted comparison must include:

- raw `text-embedding-3-large` narrative-to-narrative retrieval;
- existing projected-memory plus grounding;
- start-only terminal-state support retrieval;
- current nearest-similar top-3/90 paper/demo candidate;
- historical replay/direct text-memory ablations where already available.

## Execution Order

1. Keep the validated 990a manifest as the shared data source.
2. Build a manifest-aware embedding cache for all `22456` texts.
3. Implement the text-space contrastive trainer over the manifest.
4. Implement the projected-memory bridge trainer with explicit paired
   hard-negative memory margins.
5. Run small smoke training for both methods on a bounded target subset.
6. Run full stride-5 training for both methods only after smoke reports prove
   the pair rows are consumed.
7. Run support-level audits before any frozen-SNI scenario rollout.
8. Run matched frozen-SNI top-3/90 scenario evaluations for both methods.
9. Append results to the research log and update current-truth surfaces only
   after verifier-style artifact checks.

## Non-Promotion Rules

- Do not claim the full 4010 overlapping-window corpus was trained; this plan
  starts from the validated stride-5 training slice.
- Do not claim explicit hard-negative bridge training unless the run report
  lists consumed pair rows and paired memory-margin metrics.
- Do not change paper/demo defaults until downstream scenario metrics and
  independent verification support promotion.
- Do not generate local/template narrative prose at any stage.
- Do not condition on realized future paths or use generated future scenario
  descriptions as training text.
