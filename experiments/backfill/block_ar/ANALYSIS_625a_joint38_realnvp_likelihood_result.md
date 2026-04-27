# 625a Joint38 Conditional RealNVP Transition Likelihood Result

## Hypothesis

624a selected a conditional normalizing-flow transition likelihood as the clean next model-side falsifier. The goal was to combine:

- exact density training from the Gaussian/Student-t likelihood branch;
- flexible learned non-elliptical increments from the flow branch;
- explicit AR/state feedback from the strongest historical frontier.

The model uses one shared state panel and one transition law for `iv_only` or `joint38`. It does not introduce separate IV/factor heads, retrieval, post-hoc calibration, deterministic mean-rollout loss, or evaluator-specific rules.

## Change

Implemented:

- `diffusion/block_ar/generic_realnvp_transition_law.py`
- `experiments/backfill/block_ar/train_625a_unified_ar_realnvp_likelihood.py`
- `experiments/backfill/block_ar/evaluate_625a_unified_ar_realnvp_likelihood.py`
- `test_code/test_625a_generic_realnvp_transition.py`

Also fixed `sample_smoke(...)` backward compatibility in `train_609a_unified_ar_transition_flow.py` so raw-coordinate callers do not need to pass encoded-state specs.

Verification:

```bash
python -m py_compile \
  diffusion/block_ar/generic_realnvp_transition_law.py \
  experiments/backfill/block_ar/train_625a_unified_ar_realnvp_likelihood.py \
  experiments/backfill/block_ar/evaluate_625a_unified_ar_realnvp_likelihood.py \
  experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py
```

```bash
pytest test_code/test_625a_generic_realnvp_transition.py test_code/test_622a_encoded_state_coordinate.py -q
```

Result: `6 passed`.

## Run

Training:

```bash
python experiments/backfill/block_ar/train_625a_unified_ar_realnvp_likelihood.py \
  --state_scope joint38 \
  --epochs 8 \
  --max_train_windows 2048 \
  --batch_size 32 \
  --memory_dim 128 \
  --memory_layers 3 \
  --memory_heads 4 \
  --memory_ff 256 \
  --coupling_layers 6 \
  --coupling_hidden 256 \
  --coupling_depth 2 \
  --scale_clip 1.5 \
  --sample_count 4 \
  --sample_steps 8 \
  --chunk_size 2 \
  --seed 625 \
  --device cuda \
  --output_dir models/backfill/625a_joint38_ar_realnvp_nll_e8_w2048_s625
```

Training selected epoch 5:

- best validation NLL: `-7.123185`;
- final validation NLL: `-4.245587`;
- smoke finite rate: `1.0`.

Official bridge:

```bash
python experiments/backfill/block_ar/evaluate_625a_unified_ar_realnvp_likelihood.py \
  --checkpoint models/backfill/625a_joint38_ar_realnvp_nll_e8_w2048_s625/best_model.pt \
  --state_scope joint38 \
  --max_windows 441 \
  --samples 48 \
  --n_steps 30 \
  --batch_size 32 \
  --chunk_size 8 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 6251 \
  --device cuda \
  --output_json results/autoresearch/625a_joint38_ar_realnvp_nll_e8_w2048/full11.json \
  --output_md results/autoresearch/625a_joint38_ar_realnvp_nll_e8_w2048/full11.md
```

## Result

Score: `3/11`.

Passed:

- surface;
- block-AR;
- cointegration.

Failed:

- coverage;
- conditionality;
- time-series;
- regime coverage;
- distributional fidelity;
- cross-cell correlation;
- mean reversion;
- pathwise jump realism.

Key metrics:

- cov90 overall: `81.6%`;
- h1/h7/h14/h30 cov90: `77.8% / 81.4% / 82.1% / 81.7%`;
- conditionality MAE reduction: `4.2%`;
- turbulent/calm width ratio: `1.099`;
- ACF correlation: `0.980`;
- kurtosis ratio: `0.487`;
- daily-change KS cells: `22/25`;
- level KS cells: `4/25`;
- median-bias cells: `11/25`;
- persistent severe undercoverage: `7.2%`;
- cointegration gen/GT ratio: `0.747`, worst-cell `0.307`;
- cross-cell corr/rank: `0.426 / 2.658`;
- mean-reversion ratio: `0.684`, active pass `4/12`;
- pathwise max-jump KS: `0.534`;
- per-cell q99 jump-scale cells: `15/25`.

## Mechanism Read

625a confirms that flexible one-step likelihood is useful but insufficient.

What improved:

- average coverage is much better than 610a and 622a;
- daily-change KS is strong at `22/25`;
- cointegration recovers;
- median bias magnitude is mostly small.

What failed:

- per-cell coverage allocation still fails at every checked horizon;
- turbulent/calm width is still below the risk-policy target;
- level KS remains far below gate;
- cross-cell correlation ratio drops below gate;
- h1 mean reversion is too weak;
- pathwise max-jump KS and per-cell extreme-jump balance fail.

The causal read is that exact flexible transition likelihood learns local marginal increment density, but the free-running joint path still lacks the coherent common-shock/dependence geometry needed for realistic multi-cell scenarios. This is the same high-level tradeoff as prior flexible one-shot/path-flow attempts, but now observed inside a clean AR likelihood model.

## Decision

Do not promote 625a. Keep the RealNVP transition implementation as a useful generic likelihood baseline, but do not immediately sweep coupling depth, scale clips, or temperature. Those would be local knobs after a clear branch-level miss.

The next step should be a post-experiment synthesis of the 610-625 native joint results. The central question is now whether any clean learned-law route remains locally plausible, or whether the risk-manager-deployable route must be framed as a base learned joint law plus disclosed scenario-set/risk-policy calibration.

