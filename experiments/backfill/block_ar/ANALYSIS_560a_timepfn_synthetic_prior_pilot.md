# 560a TimePFN-Style Synthetic Prior Pilot

## Context

The current local frontier is `510a`: an `8/11` IV-only risk prototype that is risk-manager presentable only with caveats. Its persistent blocker is lower stress/regime inclusion, not high-side overcoverage or marginal level-KS alone.

560a tested the smallest faithful TimePFN-style move that is possible without external data: synthetic-prior pretraining before real SPX adaptation, while reusing the existing empirical-normal-score causal-memory AR flow backbone.

## Implementation

Added a deterministic synthetic IV-surface prior generator:

- latent level, slope, skew, and curvature factors;
- calm/stress regime labels;
- mean-reverting factor dynamics;
- stress jumps and volatility-of-volatility bursts;
- bounded, smoothed 5x5 IV surfaces.

Added `train_560a_synthetic_prior_pretrain.py`, which trains the existing `EmpiricalNormalScoreCausalMemoryTransitionFlowMatching` model on synthetic `(history, future)` windows and saves the standard 340c-family checkpoint format.

## Verification

Focused tests:

- `pytest test_code/test_560a_synthetic_iv_prior.py test_code/test_560a_synthetic_pretrain_smoke.py -q`
- Result: `4 passed, 1 warning`.

Smoke synthetic pretraining:

- Command: `python experiments/backfill/block_ar/train_560a_synthetic_prior_pretrain.py --output_dir models/backfill/560a_timepfn_synthetic_prior_smoke --device cuda --n_windows 512 --epochs 2 --batch_size 32 --history_len 30 --future_len 30 --n_quantiles 201 --model_hidden 128 --model_layers 2 --memory_dim 64 --memory_layers 2 --memory_heads 4 --memory_ff 128 --token_dim 64 --token_layers 2 --token_heads 4 --token_ff 128 --hidden_dim 64 --flow_steps 8 --seed 560`
- Result: train loss `1.0656 -> 0.9027`, validation loss `0.9385 -> 0.8483`.

Short real-data adaptation:

- Command: `python experiments/backfill/block_ar/train_377a_340c_recent_fm_adaptation.py --checkpoint models/backfill/560a_timepfn_synthetic_prior_smoke/best_model.pt --output_dir models/backfill/560a_timepfn_synthetic_prior_smoke_adapt --device cuda --adaptation_windows 128 --epochs 1 --batch_size 16 --lr 5e-5 --quantile_source recent --seed 560`
- Result: adaptation loss `1.2105`; checkpoint successfully entered the existing 340c-family real-data path.

Small 11-suite smoke evaluation:

- Artifact: `results/autoresearch/560a_timepfn_synthetic_prior_smoke/full11_smoke.json`
- Result: `1/11`.
- Passed: `block_ar`.
- Failed: surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism.

## Mechanism Read

The `1/11` smoke result is not a fair falsification of the TimePFN-style idea because the checkpoint was intentionally small, trained on only `512` synthetic windows for `2` epochs, and adapted on only `128` real windows for `1` epoch.

The useful result is that the scaffold works:

1. synthetic prior windows are deterministic, bounded, and regime-diverse;
2. the existing empirical-score AR flow can train on synthetic data;
3. the resulting checkpoint is compatible with existing real-data adaptation;
4. the checkpoint is compatible with the existing 11-suite evaluation path.

The smoke failure is still informative. A weak synthetic prior and undertrained small model produce poor cross-cell dependence and mean-reversion after adaptation. This means the next test must scale the same idea closer to the 340c/392a capacity and adaptation budget before making a research decision.

## Decision

Keep the TimePFN-style synthetic-prior branch alive. Do not compare the tiny smoke checkpoint to `510a` as a candidate model.

The next HEAD iteration should run a scaled 561a experiment:

- use the existing 340c/392a-sized backbone;
- pretrain on thousands of synthetic windows for enough epochs to reduce synthetic validation loss;
- adapt on the full `441` recent real windows;
- evaluate with the official `192` validation windows and `48` samples if runtime permits;
- judge only whether stress/regime inclusion improves without losing conditionality, dependence, mean reversion, and path realism.
