# Codebase Reorganization Summary

**Date:** November 21, 2025

## Changes Made

### Directory Structure

**Before:** 63 Python scripts in root, 16 subdirectories in tables/

**After:**
- 10 core scripts in root
- 51+ scripts organized in `experiments/` subdirectories
- 9 logical subdirectories in `results/`
- Clear separation of concerns

### New Organization

```
experiments/           # Experiment-specific scripts
├── backfill/
│   ├── context20/    # Main production model (18 scripts)
│   ├── context60/    # Context ablation (3 scripts)
│   └── horizon5/     # Horizon validation (4 scripts)
├── econometric_baseline/  # 9 econometric scripts
├── oracle_vs_prior/       # 4 evaluation scripts
├── cointegration/         # 2 analysis scripts
├── vol_smile/             # 3 analysis scripts
└── diagnostics/           # 1 diagnostic script

models/                # Reorganized model storage
├── backfill/
│   ├── context20_production/
│   ├── context60_experiment/
│   └── archived/
└── 1d_backfilling/

results/               # Reorganized from tables/
├── presentations/     # Markdown reports (4 files)
├── backfill_16yr/    # Context20 results
├── econometric_baseline/  # Baseline results
├── cointegration/
├── distribution_analysis/
├── vol_smile/
└── archived_temp/

archived_experiments/  # Old test outputs
├── test_horizon5/
├── test_phase3_output/
├── test_scheduled_sampling_output/
└── validation_scripts/
```

## Path Changes

### Models
- `models_backfill/backfill_16yr.pt` → `models/backfill/context20_production/backfill_16yr.pt`
- `models_backfill/backfill_context60*.pt` → `models/backfill/context60_experiment/checkpoints/`
- `models_backfill/*.npz, *.csv` → `results/backfill_16yr/predictions/` or `/evaluations/`

### Results
- `tables/` → `results/`
- `tables/MILESTONE_PRESENTATION.md` → `results/presentations/MILESTONE_PRESENTATION.md`
- `tables/backfill_plots/` → `results/backfill_16yr/visualizations/plotly_dashboards/`
- `tables/econometric_backfill/` → `results/econometric_baseline/predictions/`
- `tables/econometric_vs_vae_comparison/` → `results/econometric_baseline/comparisons/vs_vae_2008_2010/`
- `tables/cointegration_*` → `results/cointegration/*/`
- `tables/marginal_distribution_plots/` → `results/distribution_analysis/marginal_distributions/`

### Scripts
All `*_16yr.py` scripts moved to `experiments/backfill/context20/`
All `econometric_*.py` scripts moved to `experiments/econometric_baseline/`

## Running Scripts After Reorganization

### Option 1: Run from repository root (RECOMMENDED)

All scripts should be run from the repository root with updated paths:

```bash
# Before
python test_insample_reconstruction_16yr.py

# After
python experiments/backfill/context20/test_insample_reconstruction_16yr.py
```

### Option 2: Scripts with sys.path fixes

Many moved scripts may need their import paths updated. Two approaches:

**A. Add repository root to sys.path** (at top of script):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now absolute imports work
from vae.utils import train
import numpy as np
data = np.load("data/vol_surface_with_ret.npz")
```

**B. Update all paths to be relative from root:**
```python
# Update data loading
data = np.load("../../data/vol_surface_with_ret.npz")

# Update output paths
output_dir = Path("../../results/backfill_16yr/predictions")
```

## Documentation

New README files created:
- `experiments/README.md` - Overview of all experiments
- `experiments/backfill/context20/README.md` - Context20 model documentation
- `experiments/backfill/context60/README.md` - Context60 experiment documentation
- `results/README.md` - Results directory structure and usage

Updated:
- `CLAUDE.md` - Updated all paths and added repository structure section

## Next Steps

1. **Update script imports:** Scripts in `experiments/` subdirectories may need path fixes (see Option 2 above)
2. **Verify workflows:** Test key scripts to ensure they still work with new paths
3. **Update any external documentation** that references old paths
4. **Consider adding .gitignore updates** for new directory structure

## Benefits

- **Clearer organization:** Experiments grouped by research question
- **Easier navigation:** README files guide users to relevant scripts
- **Better maintainability:** Clear separation of core vs experimental code
- **Reduced clutter:** Test outputs and old experiments archived
- **Scalable structure:** Easy to add new experiments without cluttering root

## Rollback (if needed)

All files were moved, not deleted. To rollback:
```bash
# Backup current state first
git stash

# Reset to before reorganization
git checkout <commit_before_reorganization>
```
