# Context60 Experiment

This directory contains an experimental variant of the backfill model using an extended context window of 60 days (vs 20 days in production model).

## Research Question

Does increasing the context length from 20 to 60 days improve volatility surface forecasting performance?

## Configuration

- **Context length**: 60 days (vs 20 in baseline)
- **Training data**: Same as context20 (2004-2019)
- **Horizons**: [1, 7, 14, 30] days
- **Architecture**: CVAEMemRand with quantile regression decoder

Two configuration variants tested:
- **Full config** (`config/backfill_context60_config.py`): Full feature set
- **Short config** (`config/backfill_context60_config_short.py`): Reduced configuration

## Training Scripts

**Initial training:**
```bash
python experiments/backfill/context60/train_backfill_context60.py
```

**Phase 2 resumption:**
```bash
python experiments/backfill/context60/train_backfill_context60_resume.py
```

**Phase 3 resumption:**
```bash
python experiments/backfill/context60/train_backfill_context60_resume_phase3.py
```

## Outputs

**Model checkpoints:**
- `models/backfill/context60_experiment/checkpoints/backfill_context60_best.pt`
- `models/backfill/context60_experiment/checkpoints/backfill_context60_short_best.pt`
- Phase checkpoints: `*_phase[1-4]_ep*.pt`

**Training logs:**
- `models/backfill/context60_experiment/training_logs/context60_training_log*.txt`

## Status

**Experimental** - This is a research variant to test the hypothesis that longer context improves forecasting. Results should be compared against the context20 baseline to assess trade-offs between:
- Improved long-range temporal modeling
- Increased computational cost
- Risk of overfitting to recent patterns

## Next Steps

To evaluate context60 performance:
1. Generate predictions using the same evaluation scripts as context20
2. Compare RMSE, CI calibration, and computational cost
3. Analyze whether the 3× increase in context length justifies any improvements
