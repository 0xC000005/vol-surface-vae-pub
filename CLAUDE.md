# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Standing Directives (ALWAYS ACTIVE)

1. **HEDA Cycle**: Hypothesize → Experiment → Document → Analyze.
   After each experiment, IMMEDIATELY document results in RESEARCH_LOG.md before moving on.
2. **Don't stop** until ALL 8 test suites PASS on the raw model (no conformal), or you have
   exhausted all reasonable hypotheses. Check **MEMORY.md** for accumulated findings.
3. **Bitter Lesson**: Everything LEARNED from data. No per-cell/per-tenor constants, no
   data-derived lookup tables, no domain-specific heuristics. Must generalize to any
   conditional scenario generation problem (IV, rates, FX, etc.).
4. **Autoresearch auto-resume**: If `autoresearch-session/current_state.json` exists,
   an autoresearch loop is active. Invoke the `autoresearch` skill and follow its
   Context Recovery Protocol to resume. Check for running training processes first.

## Project Overview

**Branch: `diffusion-poc-v1`** — Single-pass afCRPS ensemble for multi-horizon IV surface forecasting.

Two-phase pipeline:
1. **Pretrained encoder** (DDPM with MSE): `models/backfill/block_ar_vol_scaled_30ep/best_model.pt`
2. **afCRPS fine-tuning**: Freeze encoder, train decoder with CRPS + auxiliary losses

Current best: **5/8 test suites PASS** (confirmed ceiling after 43+ experiments). Best models:
- **97a+qmap** (best CI: 93.6%, production recommended)
- **99m_v2** (best factor structure: eff_rank 2.66, kurtosis 0.845)
- **99l_v3** (best correlation: corr 0.60, coint 0.93)
- **99j_v3** (best per-cell KS: 24/25)

## Research Log

`RESEARCH_LOG.md` is 25,000+ lines. **Never read the full file.** Use the `research-log` skill
(MCP semantic search or targeted Read with offset/limit).

## Development Environment

- Python >=3.13 via `uv`: `uv sync`
- **Run all scripts from repo root** with `PYTHONPATH=.`
- GPU: RTX 3070 Ti (8 GB) — can fit 2 concurrent training jobs

## Architecture (afCRPS Single-Pass)

```
History (30×5×5) → GRUEncoder → condition (128-dim)
                                     ↓
Noise z~N(0,I) → NoiseMLP → noise_embed
                                     ↓
         AR Frame Loop (30 steps):
           prev_frame + condition + noise → FrameDecoder → delta
           iv_{t+1} = iv_t + vol_scale × cell_spread × delta + skip(z)
                                     ↓
                          K ensemble members → (B, K, 30, 5, 5)
```

**Key classes** (all in `diffusion/block_ar/single_pass_ar.py`):
- `SinglePassBlockAR`: Main model, `SinglePassConfig` (dataclass, ~100 hyperparams)
- `FrameDecoder`: Per-frame MLP (hidden=128, zero-init output)
- `GRUEncoder` (in `gru_encoder.py`): GRU(25→64) + attention pool → bottleneck(128)

**Noise process**: AR(1) with `rho=0.8`: `z_{t+1} = 0.8·z_t + √0.36·ε`

## Common Commands

```bash
# Train afCRPS (best recipe: 99m_v2 settings)
PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
    --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
    --no_ema --epochs 60 --batch_size 8 --noise_dim 32 --n_members 8 \
    --lr_decoder 1e-3 --lambda_vs 0.1 --lambda_es 1.0 --lambda_is 0.5 \
    --ar_frame --ar_cell_spread --ar_noise_skip --ar_skip_bypass_spread \
    --ar_reflect --ar_floor_clamp 0.01 --ar_bias_lambda 0.01 \
    --lambda_cell_var 1.0 --freeze_after_epoch 10 \
    --disable_early_stop \
    --output_dir models/backfill/afcrps_XXX --device cuda

# Full validation (9 test suites, ~5 min) — ALWAYS use v2
PYTHONPATH=. python experiments/backfill/block_ar/test_block_ar_requirements_v2.py \
    --model_path models/backfill/afcrps_XXX/best_model.pt \
    --no_ema --max_batches 20 --n_samples 50 \
    --output_dir results/block_ar/XXX_30d --device cuda

# Long-horizon test (252-day)
PYTHONPATH=. python experiments/backfill/block_ar/test_long_horizon.py \
    --model_path models/backfill/afcrps_XXX/best_model.pt \
    --no_ema --max_batches 10 --n_samples 50 --device cuda
```

## Validation Test Suites (9)

`test_block_ar_requirements_v2.py` outputs `summary.json` with pass/fail for each:

1. **Surface Validity**: Explosion rate, calendar/butterfly arbitrage
2. **CI Coverage**: Per-horizon + per-cell 90% CI (worst_cell_pass is the hard gate)
3. **Conditionality**: Turb/calm width ratio (>1.15), per-cell MAE reduction
4. **Time Series**: ACF correlation, kurtosis ratio (0.5-2.0)
5. **Block-AR Boundary**: Smoothness, growing uncertainty (monotonic with horizon)
6. **Cointegration**: Cell-cell cointegration pass rate
7. **Regime Coverage**: Per-regime per-cell CI (3-layer: horizon → regime → cell)
8. **Distributional**: KS on daily changes, KS on IV levels, median bias
9. **Cross-Cell Correlation**: Correlation ratio and effective rank ratio (v2 only)

## Loading Models

```python
import torch
from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig

checkpoint = torch.load("models/backfill/afcrps_XXX/best_model.pt", weights_only=False)
model = SinglePassBlockAR(SinglePassConfig(**checkpoint["config"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate samples
samples = model.sample(history, n_samples=50)  # (B, 50, 30, 5, 5)
```

## Data Format

**Input:** `data/vol_surface_with_ret.npz`
- `surface`: (N, 5, 5) — 5×5 IV grids (moneyness × tenor)
- `ret`: (N,) — Daily SPX returns

Training windows: history (B, 30, 5, 5) + future (B, 30, 5, 5), stride-1 sliding window.

## Key Gotchas

- **Always use `--no_ema`** — EMA destroys conditionality on small models
- **`sample_batched()`** folds n_samples into batch dim — 9.4x speedup over loop
- **Freeze-after-epoch**: MLP freeze at epoch 10 preserves cross-cell correlation (GT: 0.38)
  Without freeze, CRPS pulls correlation to 0.88 (rank-1 attractor)
- **rho=0.8 is essential**: Lower values break growing uncertainty and cointegration
- **Returns are useless**: `Corr(ret_t, IV_{t+1})` = 0.001. The -0.81 leverage effect is
  concurrent (same-day), not predictive. Encoder correctly ignores them.
- **GT data floor**: Calendar arb 7.0%, Butterfly arb 20.1% — these are NOT model failures

## Path Conventions

- Pretrained encoder: `models/backfill/block_ar_vol_scaled_30ep/best_model.pt`
- afCRPS checkpoints: `models/backfill/afcrps_*/best_model.pt`
- Test results: `results/block_ar/*/summary.json`
- Data: `data/vol_surface_with_ret.npz`

## Research Tools

- **QMD MCP** (configured in `.mcp.json`): Hybrid search (BM25 + vector + LLM reranking)
  over all project markdown files. Collection "research" indexes the entire repo.
  Tools: `mcp__qmd__query`, `mcp__qmd__get`, `mcp__qmd__multi_get`, `mcp__qmd__status`.
  After appending to RESEARCH_LOG.md, re-index with: `qmd update --collection research && qmd embed`
- **arxiv MCP** (configured in `.mcp.json`): Search, download, and read arXiv papers.
  Tools: `mcp__arxiv__search_papers`, `mcp__arxiv__download_paper`, `mcp__arxiv__read_paper`.
- **PaperQA2** (CLI at `/home/max/miniconda3/bin/pqa`): Deep Q&A over scientific papers
  with citations. Superhuman on literature search benchmarks. Use via Bash:
  `pqa ask "your question"` — it searches for papers, builds a local index, and answers
  with full citations. For local PDFs: `pqa ask --settings '{"paper_directory": "/path"}' "question"`

## Legacy Modules (Reference Only)

- `diffusion/simple_denoiser.py`, `diffusion/ddpm_scheduler.py` — DDPM POC (superseded by afCRPS)
- `vae/` — VAE baseline (33% CI coverage, for comparison only)
- `experiments/backfill/diffusion_poc/` — DDPM POC experiments (superseded)

## Disqualified Approaches (for raw model research)

Per Bitter Lesson: no conformal calibration, no per-cell data-derived constants, no
domain-specific heuristics. Everything must be learned end-to-end.

**Exception**: Quantile mapping (qmap) is acceptable for production deployment (97a+qmap
is the recommended production model). The Bitter Lesson constraint applies to research
toward 6+/8 — post-hoc fixes don't count toward passing test suites.
