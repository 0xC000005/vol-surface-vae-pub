# Paper Bug Sweep After Figure 7 Fix

Date: 2026-05-22

## Verification Result

Verdict: `PASS`

The specific Figure 7 bug is fixed, and the paper compiles without missing
assets, missing references, missing citations, or stale fixed-start labels in
the compiled sources. This pass does not certify every scientific claim in the
paper; it is a paper-surface and artifact-consistency audit after the realized
future overlay issue. A follow-up cosmetic pass also removed the remaining
LaTeX underfull-table warning and unused bibliography entries.

## What I Checked

- Rebuilt `paper/narrative_grounded_scenarios/main.pdf` with `pdflatex`.
- Checked `main.log` for missing references, missing citations, missing files,
  errors, and rerun warnings.
- Parsed expanded TeX sources, including generated table inputs, for:
  - missing `\includegraphics` assets;
  - missing `\input` files;
  - missing labels referenced by `\Cref`, `\cref`, or `\ref`;
  - missing `\bibitem` entries for all `\cite` keys.
- Searched compiled paper sources for stale public-facing strings:
  `start18`, `fixed_start_18`, `384 generated`, `384 samples`,
  `response-preview`, `current_start_checked_gap30`, `diverse_topk`, and
  `joint39_`.
- Compared the fixed-start rollout table against the 943a JSON artifacts.
- Extracted PDF text around Figure 7 to verify that the caption now describes
  only the accepted start marker, not a realized-future overlay.

## Confirmed

- Figure 7 no longer overlays a realized historical future. The casebook figure
  shows generated raw-level fans plus a black accepted-start marker.
- Realized future language remains only where it is appropriate: historical
  backtest and caption-leakage discussion.
- The current PDF has 33 pages and was rebuilt after the Figure 7 fix and
  cosmetic cleanup pass.
- No missing figures, tables, references, or citation keys were found.
- No unused bibliography entries remain.
- No overfull or underfull box warnings remain in `main.log`.
- The fixed-start table now matches the cleaned 943a paper-facing artifact:
  64 generated paths per narrative; incumbent plus start-only null.
- The stale text saying `384 generated paths per narrative` was removed from
  the main text.
- The stale ablation sentence about removing start-distance ranking pressure
  was removed from the current fixed-start section because that ablation is not
  part of the cleaned table.

## Remaining Notes

- Some ignored JSON/Markdown artifact summaries under `paper/.../figures/`
  still contain internal `joint39_*` ids because they are provenance artifacts,
  not compiled public paper text.

## Commands

- `uv run python -m py_compile experiments/backfill/block_ar/plot_narrative_casebook_backtest.py`
- `uv run python experiments/backfill/block_ar/plot_narrative_casebook_backtest.py --device cuda`
- `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
- TeX/source audit scripts for missing assets, references, citations, stale
  strings, and fixed-start table consistency.
- `rg -n "Warning|undefined|Undefined|Citation|Reference|Rerun|Missing|Error|Overfull|Underfull" paper/narrative_grounded_scenarios/main.log`
- `rg -n '384 generated|384 samples|start-distance|barely changes|start 18|fixed_start_18|start18|933a|932a|942d|response-preview diagnostic|Response-preview|current_start_checked_gap30|diverse_topk|joint39_' paper/narrative_grounded_scenarios/main.tex paper/narrative_grounded_scenarios/generated_tables/*.tex`
