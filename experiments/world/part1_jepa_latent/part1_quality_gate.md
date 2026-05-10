# World Model Part 1 Quality Gate

Date: 2026-05-10

This gate must be checked before using the Part 1 representation as a Part 2
decoder condition. It is stricter than the current HEAD070 smoke package.

## Literature Grounding

Primary JEPA and JEPA-adjacent work evaluates learned embeddings by downstream
utility and robustness, not by latent loss alone:

- I-JEPA evaluates ImageNet linear/low-shot transfer, object counting, depth
  prediction, masking ablations, and compute/data scaling:
  <https://arxiv.org/abs/2301.08243>.
- V-JEPA evaluates frozen video/image tasks, feature-vs-pixel prediction,
  masking strategies, data scale, and probing method:
  <https://arxiv.org/abs/2404.08471>.
- V-JEPA 2 separates understanding, prediction, and planning, then evaluates
  motion understanding, action anticipation, video QA, physical reasoning, and
  robot planning:
  <https://arxiv.org/abs/2506.09985>.
- LeCun's position paper frames JEPA world models as representation-space
  systems for missing-state estimation, future-state prediction, and planning,
  with collapse as a core joint-embedding risk:
  <https://openreview.net/pdf?id=BZ5a1r-kVsf>.
- Time-series JEPA work evaluates embeddings on classification, forecasting,
  predictive control, anomaly/early-warning, and embedding informativeness:
  <https://arxiv.org/abs/2509.25449>,
  <https://arxiv.org/abs/2405.10093>,
  <https://arxiv.org/abs/2406.04853>,
  <https://arxiv.org/abs/2602.04643>.
- Time-series non-contrastive masked-embedding work uses classification and
  regression linear evaluation/fine-tuning to test generalization:
  <https://ojs.aaai.org/index.php/AAAI/article/view/33828>.

## Gate Layers

### 1. Package Integrity

- `reference_package_check.py` passes.
- Checkpoint/result/data artifact identities match recorded digests.
- Guardrail docs still carry caveat terms.

### 2. Representation Health

- Same-state masked-view alignment is good but not sufficient.
- Same-state retrieval top-k/MRR is materially above raw masked baselines.
- Effective rank, per-dimension variance, singular spectrum, and off-diagonal
  redundancy reject collapse or low-dimensional shortcuts.
- Barlow/VICReg-style terms are evaluated on the exact representation surface
  consumed by downstream probes.

### 3. Corruption Robustness

- Performance is stable across held-out mask seeds and each default mask family.
- Mask-family prediction remains at or below simple majority baselines.
- Richer mask families are evaluated separately before being claimed:
  sparse, wing, ATM-strip, whole-surface day dropout, and cross-family stress.
- Synthetic missingness does not become the easiest feature to detect.

### 4. Baseline Superiority

The frozen representation must be compared against:

- raw last-surface features;
- raw full-history flattened features;
- simple PCA/fixed target features;
- simple temporal baselines such as persistence or rolling-window statistics.

No broad utility claim is allowed unless the Barlow representation beats the
appropriate raw/simple baselines for the target family being claimed.

### 5. Market-State Linear Probes

Use frozen embeddings only. These are representation-quality probes, not
pretraining losses.

- IV-surface shape probes: level, slope/skew, curvature, term-structure
  summary, wing/ATM summaries, realized surface displacement.
- Factor-panel state probes: factor family level/return summaries, stress
  indicators, cross-family spread or correlation summaries.
- Regime/state probes: train-derived regime labels, volatility buckets,
  drawdown/stress states, high/low dispersion states.
- Similarity/retrieval probes: nearest neighbors should match market-state
  similarity better than raw masked inputs and simple baselines.

### 6. Temporal Utility Probes

These are downstream tests only.

- IV-surface future range/path-shape/terminal/mean-delta probes.
- Factor-panel future target probes before any claim about joint factor futures.
- Anomaly or early-warning style probes where labels can be defined without
  leakage.
- Split and horizon sensitivity: results must survive held-out time splits and
  at least one reasonable alternative horizon/window setting.

### 7. Scale And Stability

- Repeat the reference candidate beyond the current smoke scale before any
  full-data claim.
- Report seed sensitivity for the representation-health and probe layers.
- Preserve the no-new-knobs rule: failures first become diagnostics, not
  immediate objective changes.

## Promotion Rule

Part 1 can be promoted from "smoke-scale reference candidate" to "Part 1
quality gate passed" only when:

- layers 1-4 pass;
- layer 5 shows meaningful market-state information beyond raw/simple
  baselines;
- layer 6 supports every downstream family we intend to claim;
- layer 7 shows the result is not a one-seed or tiny-sample artifact.

Until then, Part 2 work remains gated. HEAD070 may be used for diagnostics, but
not as a certified joint market-state world-model representation.
