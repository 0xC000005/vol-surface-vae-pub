### Context

721a tested whether the next repair should be a learned sticky/hurdle gate. The diagnostic separated no-change probability from the continuous generated path without retraining: for channels selected by empirical zero mass, it applied either a global train-derived atom probability or a simple history-stickiness binned atom probability.

### Findings

- The empirical atom gate improved AAA/BBB zero mass relative to identity but did not pass factor KS.
- Anchor identity: `11/13`, AAA KS `0.287`, BBB KS `0.293`.
- Anchor global atom: `11/13`, AAA KS `0.226`, BBB KS `0.276`.
- Anchor history-bin atom: `11/13`, AAA KS `0.225`, BBB KS `0.275`.
- Joint identity: `11/13`, AAA KS `0.239`, BBB KS `0.260`.
- Joint global atom: `11/13`, AAA KS `0.211`, BBB KS `0.274`.
- Joint history-bin atom: `11/13`, AAA KS `0.210`, BBB KS `0.272`.
- Joint factor-correlation amplitude worsened: identity ratio `0.355`; atom variants around `0.330` to `0.331`.

Artifacts:

- `experiments/backfill/block_ar/analyze_721a_empirical_atom_gate.py`
- `results/block_ar/721a_empirical_atom_gate/analysis.json`
- `results/block_ar/721a_empirical_atom_gate/analysis.md`

### Mechanism Read

Matching no-change probability alone is not enough. The empirical atom gate helps AAA somewhat but leaves BBB worse or unchanged and weakens correlation amplitude. Therefore the residual blocker is the conditional nonzero movement distribution, not merely the atom probability.

### Decision

Do not implement a learned atom gate by itself. The next coordinate experiment should repair sticky-channel nonzero movement shape while leaving non-sticky channels in the incumbent normalized coordinate. The clean candidate is a hybrid empirical-score flow coordinate applied only to sticky channels selected by the train no-change rule; this reuses the existing score machinery without applying score-coordinate globally to IV.
