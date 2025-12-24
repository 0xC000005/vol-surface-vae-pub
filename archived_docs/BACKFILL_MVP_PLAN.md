# Minimal MVP Plan: Autoregressive Backfilling with Current Architecture

**Date**: 2025-11-12
**Objective**: Add 30-day autoregressive backfilling capability to existing CVAEMemRand with minimal code changes

## Core Insight

Your current `CVAEMemRand` already has everything needed:
- LSTM encoder produces latent sequences
- Decoder can generate from latents
- Quantile regression works well

**Minimal changes needed**: Just change HOW we generate (autoregressive) and HOW we train (sequence loss).

---

## Phase 1: Autoregressive Generation (Week 1)

### 1.1 Multi-Step Rollout Method

**Objective**: Generate 30 days by feeding predictions back as context

**Single new method** in `vae/cvae_with_mem_randomized.py`:

```python
def generate_autoregressive_sequence(self, initial_context, horizon=30,
                                     use_ex_feats=False):
    """
    Generate multi-step sequence autoregressively.

    Args:
        initial_context: dict with "surface" (B, C, 5, 5) and optional "ex_feats"
        horizon: number of days to generate (default 30)

    Returns:
        Generated surfaces (B, horizon, 3, 5, 5) - 3 quantiles
    """
    generated = []
    context = initial_context.copy()

    for step in range(horizon):
        # Use existing get_surface_given_conditions (no changes!)
        pred = self.get_surface_given_conditions(context)  # (B, 1, 3, 5, 5)
        generated.append(pred)

        # Update context: drop oldest, add prediction (use p50 as point estimate)
        new_surface = pred[:, 0, 1, :, :]  # (B, 5, 5) - median
        context = self._update_context(context, new_surface, use_ex_feats)

    return torch.cat(generated, dim=1)  # (B, horizon, 3, 5, 5)

def _update_context(self, context, new_surface, use_ex_feats):
    """
    Update context window: drop oldest day, append new prediction.

    Args:
        context: dict with "surface" (B, C, 5, 5) and optional "ex_feats"
        new_surface: (B, 5, 5) - new day to append
        use_ex_feats: whether to handle ex_feats

    Returns:
        Updated context dict
    """
    # Shift surface context
    old_surfaces = context["surface"]  # (B, C, 5, 5)
    new_surfaces = torch.cat([old_surfaces[:, 1:, :, :],
                              new_surface.unsqueeze(1)], dim=1)

    updated = {"surface": new_surfaces}

    # If using extra features, need to generate them too
    if use_ex_feats and "ex_feats" in context:
        # For MVP: just repeat last ex_feat value (simple approximation)
        # TODO: Could decode ex_feats from model if using ex_loss variant
        old_ex_feats = context["ex_feats"]  # (B, C, 3)
        new_ex_feats = torch.cat([old_ex_feats[:, 1:, :],
                                  old_ex_feats[:, -1:, :]], dim=1)
        updated["ex_feats"] = new_ex_feats

    return updated
```

**Files to modify**: Only `vae/cvae_with_mem_randomized.py` (add 2 methods, ~60 lines)

---

## Phase 2: Sequence Training (Week 2)

### 2.1 Multi-Horizon Loss

**Objective**: Train on entire sequences, not just next day

**Modify existing** `train_step()` in `vae/cvae_with_mem_randomized.py`:

```python
def train_step_multihorizon(self, x, optimizer: torch.optim.Optimizer,
                           horizons=[1, 7, 14, 30]):
    """
    Train on multiple horizons simultaneously.
    Current train_step only uses horizon=1.

    Args:
        x: dict with "surface" (B, T_max, 5, 5) where T_max >= max(horizons)
        optimizer: PyTorch optimizer
        horizons: list of horizons to train on

    Returns:
        dict with loss components
    """
    # Weights decay with horizon
    weights = {1: 1.0, 7: 0.8, 14: 0.6, 30: 0.4}

    total_loss = 0
    total_re = 0
    total_kl = 0

    for h in horizons:
        # Extract sequence: context of length C, predict h steps ahead
        # If x has shape (B, T_max, 5, 5), extract subsequence ending at position C+h
        batch_h = self._extract_horizon_batch(x, h)

        # Use existing forward() - no changes!
        optimizer.zero_grad()

        if "ex_feats" in batch_h:
            surface_recon, ex_feats_recon, z_mean, z_log_var, z = self.forward(batch_h)
        else:
            surface_recon, z_mean, z_log_var, z = self.forward(batch_h)

        # Get ground truth at horizon h
        B = batch_h["surface"].shape[0]
        T = batch_h["surface"].shape[1]
        C = T - 1
        surface_real = batch_h["surface"][:, C:, :, :].to(self.device)

        # Existing loss calculation
        re_surface = self.quantile_loss_fn(surface_recon, surface_real)

        if "ex_feats" in batch_h:
            ex_feats_real = batch_h["ex_feats"][:, C:, :].to(self.device)
            if self.config["ex_loss_on_ret_only"]:
                ex_feats_recon = ex_feats_recon[:, :, :1]
                ex_feats_real = ex_feats_real[:, :, :1]
            re_ex_feats = self.ex_feats_loss_fn(ex_feats_recon, ex_feats_real)
            reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
        else:
            reconstruction_error = re_surface

        # KL divergence
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        # Weighted sum
        horizon_loss = reconstruction_error + self.kl_weight * kl_loss
        total_loss += weights[h] * horizon_loss
        total_re += weights[h] * reconstruction_error.item()
        total_kl += weights[h] * kl_loss.item()

    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "reconstruction_loss": total_re,
        "kl_loss": total_kl,
    }

def _extract_horizon_batch(self, x, horizon):
    """
    Extract batch for specific horizon.

    Args:
        x: dict with "surface" (B, T_max, 5, 5)
        horizon: number of steps ahead to predict (1, 7, 14, or 30)

    Returns:
        dict with "surface" (B, C+1, 5, 5) where C is context length
    """
    C = self.config.get("context_len", 5)
    T = C + horizon  # Total sequence length needed

    # Extract subsequence from random starting point
    B, T_max = x["surface"].shape[0], x["surface"].shape[1]

    if T_max < T:
        raise ValueError(f"Sequence too short: need {T}, have {T_max}")

    # Random starting index
    start_idx = torch.randint(0, T_max - T + 1, (1,)).item()

    batch = {
        "surface": x["surface"][:, start_idx:start_idx+T, :, :]
    }

    if "ex_feats" in x:
        batch["ex_feats"] = x["ex_feats"][:, start_idx:start_idx+T, :]

    return batch
```

**Files to modify**: `vae/cvae_with_mem_randomized.py` (add methods, ~100 lines)

### 2.2 Simple Scheduled Sampling

**Objective**: Mix teacher forcing with autoregressive during training

**Add to training loop** in `vae/utils.py`:

```python
def train_with_scheduled_sampling(model, train_loader, valid_loader,
                                  epochs=400, lr=1e-5, model_dir="models",
                                  file_name="backfill_model.pt"):
    """
    Simple two-phase training:
    - Phase 1 (epochs 1-200): Teacher forcing (existing single-step)
    - Phase 2 (epochs 201-400): Multi-horizon training

    Args:
        model: CVAEMemRand instance
        train_loader, valid_loader: DataLoaders
        epochs: Total training epochs
        lr: Learning rate
        model_dir, file_name: Checkpoint saving
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        if epoch < 200:
            # Phase 1: Standard single-step teacher forcing
            print(f"Epoch {epoch+1}/{epochs} - Teacher Forcing Mode")

            for batch in train_loader:
                # Use existing train_step
                losses = model.train_step(batch, optimizer)
        else:
            # Phase 2: Multi-horizon training
            print(f"Epoch {epoch+1}/{epochs} - Multi-Horizon Mode")

            for batch in train_loader:
                # Use new multihorizon train_step
                losses = model.train_step_multihorizon(batch, optimizer,
                                                      horizons=[1, 7, 14, 30])

        # Validation (use single-step for simplicity)
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                val_losses = model.test_step(batch)
                valid_loss += val_losses["loss"]

        valid_loss /= len(valid_loader)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_weights(model_dir, file_name)
            print(f"  → Saved best model (valid loss: {valid_loss:.6f})")

        print(f"  Train loss: {losses['loss']:.6f}, Valid loss: {valid_loss:.6f}")
```

**Files to modify**: `vae/utils.py` (add function, ~40 lines)

---

## Phase 3: Limited Data Setup (Week 3)

### 3.1 Config File for Data Splits

**New file**: `config/backfill_config.py`

```python
"""
Configuration for backfilling experiments with limited training data.
"""

class BackfillConfig:
    """Configuration for limited data training and backfilling."""

    # Training data parameters (CONFIGURABLE)
    train_period_years = 3  # Options: 1, 2, or 3 years
    train_end_idx = 5000    # Use recent data (before test set)

    # Auto-compute start index (assuming ~250 trading days per year)
    train_start_idx = train_end_idx - (250 * train_period_years)

    # Backfill period (historical period to generate)
    backfill_start_idx = 2000   # Start of 2008 crisis period
    backfill_end_idx = 2765     # End of 2010 period
    backfill_horizon = 30        # Days to generate per window

    # Model hyperparameters (reuse existing architecture)
    context_len = 5  # Initial value, can increase to 30 later
    latent_dim = 5
    mem_hidden = 100
    kl_weight = 1e-5

    # Training parameters
    epochs = 400
    batch_size = 64
    learning_rate = 1e-5

    # Horizons for multi-horizon loss
    training_horizons = [1, 7, 14, 30]
    horizon_weights = {1: 1.0, 7: 0.8, 14: 0.6, 30: 0.4}

    # Evaluation parameters
    eval_horizons = [1, 7, 14, 30]

    @classmethod
    def get_train_indices(cls):
        """Get training data indices."""
        return cls.train_start_idx, cls.train_end_idx

    @classmethod
    def get_backfill_indices(cls):
        """Get backfill period indices."""
        return cls.backfill_start_idx, cls.backfill_end_idx

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 60)
        print("BACKFILL CONFIGURATION")
        print("=" * 60)
        print(f"Training period: {cls.train_period_years} years")
        print(f"Training indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print(f"Backfill indices: [{cls.backfill_start_idx}, {cls.backfill_end_idx}]")
        print(f"Backfill horizon: {cls.backfill_horizon} days")
        print(f"Context length: {cls.context_len} days")
        print(f"Training horizons: {cls.training_horizons}")
        print("=" * 60)
```

### 3.2 Training Script

**New file**: `train_backfill_model.py`

```python
"""
Train CVAEMemRand for backfilling task using limited data.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import train_with_scheduled_sampling, set_seeds
from config.backfill_config import BackfillConfig
import os

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

# Print configuration
BackfillConfig.summary()

# Load data
print("\nLoading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[..., np.newaxis],
                         skew_data[..., np.newaxis],
                         slope_data[..., np.newaxis]], axis=-1)

# Extract limited training data
train_start, train_end = BackfillConfig.get_train_indices()
vol_train = vol_surf_data[train_start:train_end]
ex_train = ex_data[train_start:train_end]

print(f"Training data shape: {vol_train.shape}")
print(f"Training period: {BackfillConfig.train_period_years} years "
      f"({train_end - train_start} days)")

# Split into train/validation (80/20)
split_idx = int(0.8 * len(vol_train))
vol_train_split = vol_train[:split_idx]
ex_train_split = ex_train[:split_idx]
vol_valid_split = vol_train[split_idx:]
ex_valid_split = ex_train[split_idx:]

# Create datasets (need longer sequences for multi-horizon training)
max_horizon = max(BackfillConfig.training_horizons)
min_seq_len = BackfillConfig.context_len + max_horizon + 1
max_seq_len = min_seq_len + 10  # Some variability

print(f"\nCreating datasets with sequence length: {min_seq_len}-{max_seq_len}")

train_dataset = VolSurfaceDataSetRand(
    vol_train_split, ex_train_split,
    min_seq_len=min_seq_len,
    max_seq_len=max_seq_len
)

valid_dataset = VolSurfaceDataSetRand(
    vol_valid_split, ex_valid_split,
    min_seq_len=min_seq_len,
    max_seq_len=max_seq_len
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_sampler=CustomBatchSampler(train_dataset, BackfillConfig.batch_size, min_seq_len)
)

valid_loader = DataLoader(
    valid_dataset,
    batch_sampler=CustomBatchSampler(valid_dataset, 16, min_seq_len)
)

print(f"Train batches: {len(train_loader)}")
print(f"Valid batches: {len(valid_loader)}")

# Model configuration (reuse existing structure)
model_config = {
    "feat_dim": (5, 5),
    "latent_dim": BackfillConfig.latent_dim,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "kl_weight": BackfillConfig.kl_weight,
    "re_feat_weight": 1.0,  # Can train ex_loss variant
    "surface_hidden": [5, 5, 5],
    "ex_feats_dim": 3,
    "ex_feats_hidden": None,
    "mem_type": "lstm",
    "mem_hidden": BackfillConfig.mem_hidden,
    "mem_layers": 2,
    "mem_dropout": 0.3,
    "ctx_surface_hidden": [5, 5, 5],
    "ctx_ex_feats_hidden": None,
    "interaction_layers": None,
    "compress_context": True,
    "use_dense_surface": False,
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
    "context_len": BackfillConfig.context_len,
}

print("\n" + "=" * 60)
print("MODEL CONFIGURATION")
print("=" * 60)
for key, value in model_config.items():
    print(f"  {key}: {value}")
print("=" * 60)

# Create model
print("\nInitializing model...")
model = CVAEMemRand(model_config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
output_dir = "models_backfill"
os.makedirs(output_dir, exist_ok=True)
model_name = f"backfill_{BackfillConfig.train_period_years}yr.pt"

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

train_with_scheduled_sampling(
    model, train_loader, valid_loader,
    epochs=BackfillConfig.epochs,
    lr=BackfillConfig.learning_rate,
    model_dir=output_dir,
    file_name=model_name
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Model saved to: {output_dir}/{model_name}")
```

### 3.3 Context Length Ablation Study (Optional but Recommended)

**Objective**: If context_len=5 performs poorly, systematically try longer contexts.

**Rationale**:
- Context=5 is conservative (1 trading week) - may be too limited for crisis modeling
- 2008 financial crisis had volatility regimes lasting weeks/months
- Longer context can capture more market history but increases computation

**When to run this**:
- After Phase 4+5 evaluation if results are poor:
  - 30-day RMSE > 0.055
  - CI violations > 25%
  - Doesn't beat GARCH baseline
- Before Phase 4 if you want comprehensive comparison upfront

**Context lengths to try**:
1. **context_len=5** (baseline, already trained)
2. **context_len=10** (~2 weeks)
3. **context_len=20** (~1 month)
4. **context_len=30** (~6 weeks)

**How to run**:

For each context length:

1. Update `config/backfill_config.py`:
```python
context_len = 20  # Change this value
```

2. Retrain model:
```bash
python train_backfill_model.py
# Saves to: models_backfill/backfill_3yr.pt (overwrites!)
# Or rename model_name to preserve previous versions
```

3. Generate + evaluate:
```bash
python generate_backfill_sequences.py
python evaluate_backfill.py
```

4. Compare results in table

**Expected trade-offs**:

| Context Length | Pros | Cons | When to Use |
|----------------|------|------|-------------|
| **5 days** | Fast training, less memory | Limited historical info | Quick MVP validation |
| **10 days** | Balanced | Moderate improvement | Good default |
| **20 days** | ~1 month history | 4× slower training | Recommended for production |
| **30 days** | Maximum context | 6× slower, fewer samples | Best for crisis periods |

**Computational impact**:
- Training time: Linear with context length (context=20 is ~4× slower than context=5)
- Memory usage: Increases with context length
- Dataset size: Decreases (need longer sequences, fewer valid samples)

**Script for batch comparison** (optional):

Create `ablation_context_lengths.sh`:
```bash
#!/bin/bash
# Run ablation study across all context lengths

for ctx in 5 10 20 30; do
    echo "====================================================="
    echo "Training with context_len=$ctx"
    echo "====================================================="

    # Update config
    sed -i "s/context_len = [0-9]\+/context_len = $ctx/" config/backfill_config.py

    # Train
    python train_backfill_model.py

    # Rename model to preserve
    mv models_backfill/backfill_3yr.pt models_backfill/backfill_3yr_ctx${ctx}.pt

    # Generate
    python generate_backfill_sequences.py

    # Rename predictions
    mv models_backfill/backfill_predictions_3yr.npz \
       models_backfill/backfill_predictions_3yr_ctx${ctx}.npz

    # Evaluate
    python evaluate_backfill.py > results_ctx${ctx}.txt
done

echo "====================================================="
echo "Ablation study complete! Check results_ctx*.txt"
echo "====================================================="
```

**Expected results**:
- **context=5**: Fast baseline (may underperform on crisis)
- **context=10**: Small improvement (~5-10% RMSE reduction)
- **context=20**: Best balance (likely sweet spot for crisis modeling)
- **context=30**: Marginal gains over 20 (diminishing returns)

**Recommendation**: Start with context=5 for MVP, then try context=20 if results warrant it.

---

## Phase 4: Generation Script (Week 4)

### 4.1 Backfill Generation

**New file**: `generate_backfill_sequences.py`

```python
"""
Generate 30-day autoregressive backfill sequences.
"""

import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
from config.backfill_config import BackfillConfig
from tqdm import tqdm

# Set seeds and dtype
set_seeds(42)
torch.set_default_dtype(torch.float64)

# Configuration
BackfillConfig.summary()

# Load model
model_path = f"models_backfill/backfill_{BackfillConfig.train_period_years}yr.pt"
print(f"\nLoading model: {model_path}")
model_data = torch.load(model_path, weights_only=False)
model = CVAEMemRand(model_data["model_config"])
model.load_weights(dict_to_load=model_data)
model.eval()

# Load data
print("\nLoading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[..., np.newaxis],
                         skew_data[..., np.newaxis],
                         slope_data[..., np.newaxis]], axis=-1)

# Backfill period
backfill_start, backfill_end = BackfillConfig.get_backfill_indices()
horizon = BackfillConfig.backfill_horizon
context_len = BackfillConfig.context_len

print(f"\nGenerating backfill sequences:")
print(f"  Period: indices [{backfill_start}, {backfill_end}]")
print(f"  Horizon: {horizon} days")
print(f"  Context: {context_len} days")

# Storage
all_predictions = []
all_ground_truth = []
all_dates = []

# Generate for each window
num_windows = (backfill_end - backfill_start - horizon) // horizon
print(f"  Number of windows: {num_windows}")

use_ex_feats = model.config["ex_feats_dim"] > 0

for i in tqdm(range(num_windows)):
    # Starting point for this window
    start_day = backfill_start + i * horizon

    # Initial context (before window)
    context_surfaces = vol_surf_data[start_day - context_len:start_day]
    context_ex_feats = ex_data[start_day - context_len:start_day] if use_ex_feats else None

    # Prepare context dict
    context = {
        "surface": torch.from_numpy(context_surfaces).unsqueeze(0).float()
    }
    if use_ex_feats:
        context["ex_feats"] = torch.from_numpy(context_ex_feats).unsqueeze(0).float()

    # Generate 30 days autoregressively
    with torch.no_grad():
        generated = model.generate_autoregressive_sequence(
            context, horizon=horizon, use_ex_feats=use_ex_feats
        )  # (1, horizon, 3, 5, 5)

    # Ground truth
    ground_truth = vol_surf_data[start_day:start_day + horizon]  # (horizon, 5, 5)

    # Store
    all_predictions.append(generated.cpu().numpy())
    all_ground_truth.append(ground_truth)
    all_dates.append(start_day)

# Convert to arrays
predictions = np.concatenate(all_predictions, axis=0)  # (num_windows, horizon, 3, 5, 5)
ground_truth = np.array(all_ground_truth)  # (num_windows, horizon, 5, 5)
dates = np.array(all_dates)

print(f"\n✓ Generation complete!")
print(f"  Predictions shape: {predictions.shape}")
print(f"  Ground truth shape: {ground_truth.shape}")

# Save
output_file = f"models_backfill/backfill_predictions_{BackfillConfig.train_period_years}yr.npz"
np.savez(
    output_file,
    predictions=predictions,
    ground_truth=ground_truth,
    dates=dates,
    config=BackfillConfig.__dict__
)

print(f"\nSaved to: {output_file}")
```

---

## Phase 5: Simple Evaluation (Week 5)

### 5.1 Basic Metrics Script

**New file**: `evaluate_backfill.py`

```python
"""
Evaluate backfilling performance with simple metrics.
"""

import numpy as np
from sklearn.metrics import r2_score
from config.backfill_config import BackfillConfig

# Load predictions
predictions_file = f"models_backfill/backfill_predictions_{BackfillConfig.train_period_years}yr.npz"
print(f"Loading: {predictions_file}")
data = np.load(predictions_file)

predictions = data["predictions"]  # (num_windows, horizon, 3, 5, 5)
ground_truth = data["ground_truth"]  # (num_windows, horizon, 5, 5)

print(f"Predictions shape: {predictions.shape}")
print(f"Ground truth shape: {ground_truth.shape}")

# Extract quantiles
p05 = predictions[:, :, 0, :, :]  # (num_windows, horizon, 5, 5)
p50 = predictions[:, :, 1, :, :]  # Median as point forecast
p95 = predictions[:, :, 2, :, :]

def compute_metrics_for_horizon(h):
    """Compute metrics for specific horizon."""
    # Extract horizon h (0-indexed)
    pred_h = p50[:, h, :, :].flatten()
    true_h = ground_truth[:, h, :, :].flatten()
    p05_h = p05[:, h, :, :].flatten()
    p95_h = p95[:, h, :, :].flatten()

    # Point forecast metrics
    rmse = np.sqrt(np.mean((pred_h - true_h) ** 2))
    mae = np.mean(np.abs(pred_h - true_h))
    r2 = r2_score(true_h, pred_h)

    # CI calibration
    ci_violations = np.mean((true_h < p05_h) | (true_h > p95_h)) * 100

    # Direction accuracy
    pred_change = np.diff(p50[:, :h+1, :, :], axis=1)[:, -1, :, :].flatten()
    true_change = np.diff(ground_truth[:, :h+1, :, :], axis=1)[:, -1, :, :].flatten()
    direction_acc = np.mean(np.sign(pred_change) == np.sign(true_change)) * 100

    return {
        "horizon": h + 1,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ci_violations": ci_violations,
        "direction_acc": direction_acc
    }

# Compute for each evaluation horizon
print("\n" + "=" * 80)
print("BACKFILL EVALUATION RESULTS")
print("=" * 80)
print(f"\nHorizon  RMSE      MAE       R²       CI Viol  Direction")
print("-" * 80)

results = []
for h_idx in [0, 6, 13, 29]:  # Days 1, 7, 14, 30
    metrics = compute_metrics_for_horizon(h_idx)
    results.append(metrics)
    print(f"{metrics['horizon']:2d}-day   "
          f"{metrics['rmse']:.6f}  "
          f"{metrics['mae']:.6f}  "
          f"{metrics['r2']:.4f}  "
          f"{metrics['ci_violations']:5.1f}%   "
          f"{metrics['direction_acc']:5.1f}%")

print("-" * 80)

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print("=" * 80)
print(f"Model: CVAEMemRand (quantile regression)")
print(f"Training data: {BackfillConfig.train_period_years} years")
print(f"Training mode: Multi-horizon loss with scheduled sampling")
print(f"\nKey Findings:")
print(f"  - 1-day RMSE: {results[0]['rmse']:.4f}")
print(f"  - 30-day RMSE: {results[3]['rmse']:.4f}")
print(f"  - RMSE growth: {(results[3]['rmse'] / results[0]['rmse'] - 1) * 100:.1f}%")
print(f"  - CI violations (1d): {results[0]['ci_violations']:.1f}% (target: 10%)")
print(f"  - CI violations (30d): {results[3]['ci_violations']:.1f}% (target: 10%)")
print(f"  - Direction accuracy (30d): {results[3]['direction_acc']:.1f}% (random: 50%)")

# Save results
import pandas as pd
df = pd.DataFrame(results)
output_csv = f"models_backfill/evaluation_results_{BackfillConfig.train_period_years}yr.csv"
df.to_csv(output_csv, index=False)
print(f"\nResults saved to: {output_csv}")
```

### 5.2 Cointegration Test (Simple)

**Add to** `evaluate_backfill.py`:

```python
print("\n" + "=" * 80)
print("COINTEGRATION ANALYSIS")
print("=" * 80)

from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Reshape surfaces to (time × grid_points)
# Use all windows concatenated
pred_flat = p50.reshape(-1, 25)  # (num_windows * horizon, 25)
true_flat = ground_truth.reshape(-1, 25)

# Test on historical
print("\nTesting historical surfaces...")
result_hist = coint_johansen(true_flat, det_order=0, k_ar_diff=1)
rank_hist = np.sum(result_hist.lr1 > result_hist.cvt[:, 1])  # 95% critical values

# Test on generated
print("Testing generated surfaces...")
result_gen = coint_johansen(pred_flat, det_order=0, k_ar_diff=1)
rank_gen = np.sum(result_gen.lr1 > result_gen.cvt[:, 1])

print(f"\nCointegration Results:")
print(f"  Historical rank: {rank_hist}")
print(f"  Generated rank:  {rank_gen}")
print(f"  Rank preserved:  {rank_hist == rank_gen}")

if rank_hist == rank_gen:
    print("\n✓ Generated surfaces preserve cointegration structure!")
else:
    print(f"\n✗ Cointegration rank changed by {abs(rank_gen - rank_hist)}")
```

### 5.3 Visualization

**New file**: `visualize_backfill.py`

```python
"""
Visualize backfilling results (like teacher forcing plot).
"""

import numpy as np
import matplotlib.pyplot as plt
from config.backfill_config import BackfillConfig

# Load predictions
predictions_file = f"models_backfill/backfill_predictions_{BackfillConfig.train_period_years}yr.npz"
data = np.load(predictions_file)

predictions = data["predictions"]  # (num_windows, horizon, 3, 5, 5)
ground_truth = data["ground_truth"]  # (num_windows, horizon, 5, 5)

# Select 3 grid points (same as teacher forcing plot)
grid_points = [
    (2, 2, "6M ATM"),
    (3, 2, "1Y ATM"),
    (4, 2, "2Y ATM"),
]

# Use first window for visualization
window_idx = 0
p05 = predictions[window_idx, :, 0, :, :]
p50 = predictions[window_idx, :, 1, :, :]
p95 = predictions[window_idx, :, 2, :, :]
truth = ground_truth[window_idx, :, :, :]

# Create plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
days = np.arange(BackfillConfig.backfill_horizon)

for idx, (i, j, label) in enumerate(grid_points):
    ax = axes[idx]

    # Extract time series
    gt = truth[:, i, j]
    pred = p50[:, i, j]
    lower = p05[:, i, j]
    upper = p95[:, i, j]

    # Plot
    ax.fill_between(days, lower, upper, alpha=0.3, color='blue', label='90% CI')
    ax.plot(days, gt, 'k-', linewidth=2, label='Ground Truth')
    ax.plot(days, pred, 'b-', linewidth=1.5, label='Prediction (p50)')

    # Mark violations
    violations = (gt < lower) | (gt > upper)
    if np.any(violations):
        ax.scatter(days[violations], gt[violations], color='red', s=50,
                  zorder=5, label='CI Violation')

    # Calculate metrics
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    r2 = 1 - np.sum((gt - pred) ** 2) / np.sum((gt - np.mean(gt)) ** 2)
    ci_viol = np.mean(violations) * 100

    ax.set_ylabel(f'{label}\nImplied Vol')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}  R²: {r2:.3f}  CI Viol: {ci_viol:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)

axes[-1].set_xlabel('Days into Backfill Sequence')
plt.suptitle(f'Autoregressive Backfilling Performance\n'
             f'Training: {BackfillConfig.train_period_years} years, '
             f'Horizon: {BackfillConfig.backfill_horizon} days',
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_file = f'models_backfill/backfill_visualization_{BackfillConfig.train_period_years}yr.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved visualization to: {output_file}")
plt.close()
```

---

## Phase 6: Simple Baselines (Week 6)

### 6.1 Historical Mean Baseline

**New file**: `baselines/historical_mean.py`

```python
"""
Historical mean baseline for comparison.
"""

import numpy as np
from config.backfill_config import BackfillConfig

# Load data
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]

# Training data
train_start, train_end = BackfillConfig.get_train_indices()
train_surfaces = vol_surf_data[train_start:train_end]

# Compute mean surface
mean_surface = np.mean(train_surfaces, axis=0)  # (5, 5)

# Generate predictions (just repeat mean)
backfill_start, backfill_end = BackfillConfig.get_backfill_indices()
horizon = BackfillConfig.backfill_horizon

num_windows = (backfill_end - backfill_start - horizon) // horizon
predictions = np.tile(mean_surface, (num_windows, horizon, 1, 1))

# Ground truth
ground_truth = []
for i in range(num_windows):
    start_day = backfill_start + i * horizon
    ground_truth.append(vol_surf_data[start_day:start_day + horizon])
ground_truth = np.array(ground_truth)

# Compute RMSE
rmse_1d = np.sqrt(np.mean((predictions[:, 0, :, :] - ground_truth[:, 0, :, :]) ** 2))
rmse_30d = np.sqrt(np.mean((predictions[:, -1, :, :] - ground_truth[:, -1, :, :]) ** 2))

print(f"Historical Mean Baseline:")
print(f"  1-day RMSE:  {rmse_1d:.6f}")
print(f"  30-day RMSE: {rmse_30d:.6f}")

# Save
np.savez("baselines/historical_mean_predictions.npz",
         predictions=predictions,
         ground_truth=ground_truth,
         rmse_1d=rmse_1d,
         rmse_30d=rmse_30d)
```

### 6.2 GARCH Baseline

**New file**: `baselines/garch_forecast.py`

```python
"""
GARCH(1,1) baseline for each grid point.
"""

import numpy as np
from arch import arch_model
from config.backfill_config import BackfillConfig
from tqdm import tqdm

# Load data
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]

# Training data
train_start, train_end = BackfillConfig.get_train_indices()
train_surfaces = vol_surf_data[train_start:train_end]

# Backfill period
backfill_start, backfill_end = BackfillConfig.get_backfill_indices()
horizon = BackfillConfig.backfill_horizon
num_windows = (backfill_end - backfill_start - horizon) // horizon

# Storage
predictions = np.zeros((num_windows, horizon, 5, 5))
ground_truth = []

# Fit GARCH for each grid point
print("Fitting GARCH models for each grid point...")
for i in tqdm(range(5)):
    for j in range(5):
        # Training data for this grid point
        train_series = train_surfaces[:, i, j] * 100  # Scale to percentages

        # Fit GARCH(1,1)
        model = arch_model(train_series, vol='GARCH', p=1, q=1)
        fitted = model.fit(disp='off')

        # Generate forecasts for each window
        for window_idx in range(num_windows):
            start_day = backfill_start + window_idx * horizon

            # Forecast horizon steps
            forecast = fitted.forecast(horizon=horizon, start=start_day)
            predictions[window_idx, :, i, j] = forecast.mean.values[-1, :] / 100

# Ground truth
for i in range(num_windows):
    start_day = backfill_start + i * horizon
    ground_truth.append(vol_surf_data[start_day:start_day + horizon])
ground_truth = np.array(ground_truth)

# Compute RMSE
rmse_1d = np.sqrt(np.mean((predictions[:, 0, :, :] - ground_truth[:, 0, :, :]) ** 2))
rmse_30d = np.sqrt(np.mean((predictions[:, -1, :, :] - ground_truth[:, -1, :, :]) ** 2))

print(f"\nGARCH Baseline:")
print(f"  1-day RMSE:  {rmse_1d:.6f}")
print(f"  30-day RMSE: {rmse_30d:.6f}")

# Save
np.savez("baselines/garch_predictions.npz",
         predictions=predictions,
         ground_truth=ground_truth,
         rmse_1d=rmse_1d,
         rmse_30d=rmse_30d)
```

### 6.3 Comparison Table

**Add to** `evaluate_backfill.py`:

```python
print("\n" + "=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)

# Load baseline results
hist_mean = np.load("baselines/historical_mean_predictions.npz")
garch = np.load("baselines/garch_predictions.npz")

# VAE results (already computed)
vae_rmse_1d = results[0]['rmse']
vae_rmse_30d = results[3]['rmse']
vae_ci_viol_1d = results[0]['ci_violations']
vae_ci_viol_30d = results[3]['ci_violations']
vae_dir_30d = results[3]['direction_acc']

# Format table
print(f"\n{'Method':<20} {'1-day RMSE':<12} {'30-day RMSE':<12} {'CI Viol(30d)':<12} {'Direction(30d)':<15}")
print("-" * 80)
print(f"{'Historical Mean':<20} {hist_mean['rmse_1d']:<12.6f} {hist_mean['rmse_30d']:<12.6f} {'N/A':<12} {'N/A':<15}")
print(f"{'GARCH(1,1)':<20} {garch['rmse_1d']:<12.6f} {garch['rmse_30d']:<12.6f} {'N/A':<12} {'N/A':<15}")
print(f"{'VAE (Ours)':<20} {vae_rmse_1d:<12.6f} {vae_rmse_30d:<12.6f} {f'{vae_ci_viol_30d:.1f}%':<12} {f'{vae_dir_30d:.1f}%':<15}")
print("-" * 80)

# Improvement
hist_improve = (hist_mean['rmse_30d'] - vae_rmse_30d) / hist_mean['rmse_30d'] * 100
garch_improve = (garch['rmse_30d'] - vae_rmse_30d) / garch['rmse_30d'] * 100

print(f"\nVAE Improvement over baselines (30-day RMSE):")
print(f"  vs Historical Mean: {hist_improve:+.1f}%")
print(f"  vs GARCH:           {garch_improve:+.1f}%")
```

---

## Summary of Changes

### Files to Create (8 new files, ~1150 lines total):

1. **config/backfill_config.py** - Configuration (~80 lines)
2. **train_backfill_model.py** - Training script (~150 lines)
3. **generate_backfill_sequences.py** - Generation (~130 lines)
4. **evaluate_backfill.py** - Metrics + cointegration (~250 lines)
5. **visualize_backfill.py** - Plots (~80 lines)
6. **baselines/historical_mean.py** - Baseline 1 (~50 lines)
7. **baselines/garch_forecast.py** - Baseline 2 (~100 lines)
8. **comparison_table.py** - Summary (~50 lines, or add to evaluate_backfill.py)

### Files to Modify (2 files, ~200 lines added):

1. **vae/cvae_with_mem_randomized.py**:
   - Add `generate_autoregressive_sequence()` (~40 lines)
   - Add `_update_context()` (~20 lines)
   - Add `train_step_multihorizon()` (~80 lines)
   - Add `_extract_horizon_batch()` (~40 lines)
   - **Total**: ~180 lines added

2. **vae/utils.py**:
   - Add `train_with_scheduled_sampling()` (~40 lines)

### No Changes Needed:
- Encoder/decoder architecture ✓
- Quantile loss function ✓
- Context encoder ✓
- Data loading classes ✓
- All existing training infrastructure ✓

---

## Expected Results (Conservative MVP Estimates)

**Performance targets**:
- 1-day RMSE: 0.018-0.025 (competitive with current 0.030)
- 30-day RMSE: 0.040-0.055 (acceptable for first MVP)
- CI violations: 15-22% at 30 days (improvement over baseline, not perfect)
- Direction accuracy: 52-56% at 30 days (slight edge over random 50%)
- Cointegration: Likely preserved (rank should match)

**Success criteria**:
1. ✓ Can generate 30-day sequences autoregressively
2. ✓ Multi-horizon loss improves long-term accuracy vs single-step
3. ✓ Training on 1-3 years produces reasonable results
4. ✓ Beats naive baselines (historical mean, GARCH)
5. ✓ Cointegration structure preserved

**What we learn**:
- Error accumulation rate in autoregressive mode
- Multi-horizon loss effectiveness
- Limited data training feasibility
- Whether current architecture sufficient or needs extensions

---

## Timeline: 6 weeks

- **Week 1**: Add autoregressive generation methods (~180 lines in CVAEMemRand)
- **Week 2**: Add multi-horizon loss and scheduled sampling (~120 lines in CVAEMemRand + utils)
- **Week 3**: Config + training script (~230 lines)
- **Week 4**: Generation script (~130 lines)
- **Week 5**: Evaluation + visualization (~330 lines)
- **Week 6**: Baselines + comparison (~200 lines)

**Total**: ~1350 lines of new/modified code

---

## Next Steps After MVP

**If MVP shows promise** (beats baselines, <20% CI violations):
1. Increase context length (5 → 30 days)
2. Add data augmentation (GAN-based)
3. Try conformal calibration for CI
4. Add VECM hybrid post-processing

**If MVP struggles** (doesn't beat baselines):
1. Debug: Check error accumulation pattern
2. Try direct multi-step models (no autoregression)
3. Simplify: Maybe 7-day horizon instead of 30
4. Consider whether architecture needs more capacity

---

## Key Risks & Mitigations

**Risk 1**: Error accumulation makes 30-day forecasts poor
- **Mitigation**: Start with 7-14 days, gradually increase
- **Fallback**: Direct multi-step models (separate model per horizon)

**Risk 2**: Multi-horizon loss destabilizes training
- **Mitigation**: Use conservative weights {1: 1.0, 7: 0.8, 14: 0.6, 30: 0.4}
- **Fallback**: Train only on [1, 7, 14] first, add 30 later

**Risk 3**: Limited data (1-3 years) causes overfitting
- **Mitigation**: Strong regularization (dropout 0.3), early stopping
- **Fallback**: Use 3 years minimum, add data augmentation if needed

**Risk 4**: Scheduled sampling causes training collapse
- **Mitigation**: Conservative schedule (epochs 200-400), monitor validation
- **Fallback**: Skip scheduled sampling, just use multi-horizon loss

---

## Success Metrics Summary

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| 1-day RMSE | < 0.025 | < 0.020 |
| 30-day RMSE | < 0.050 | < 0.045 |
| 30-day CI violations | < 20% | < 15% |
| 30-day direction accuracy | > 52% | > 55% |
| Cointegration rank | Preserved | Preserved |
| vs Historical Mean (30d) | Beat by >10% | Beat by >20% |
| vs GARCH (30d) | Competitive | Beat by >5% |

---

**End of MVP Plan**
