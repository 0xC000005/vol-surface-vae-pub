#!/usr/bin/env python3
"""
Hypothesis 2: Loss Function Bias (Quantile Loss Mechanics)

Question: Does pinball loss inherently favor mean reversion at long horizons?

Method:
1. Load validation predictions and ground truth
2. Compare two prediction strategies:
   - Strategy A: Context-dependent predictions (high epistemic uncertainty)
   - Strategy B: Mean reversion predictions (low epistemic uncertainty)
3. Measure which strategy minimizes quantile loss

If loss_B < loss_A: Pinball loss favors mean reversion (need different loss)
If loss_A < loss_B: Loss is fine, problem is elsewhere
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "backfill" / "backfill_context60_best.pt"
DATA_PATH = PROJECT_ROOT / "data" / "vol_surface_with_ret.npz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "context60_baseline" / "analysis" / "mean_reversion_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HYPOTHESIS 2: LOSS FUNCTION BIAS ANALYSIS")
print("=" * 80)
print()

# Pinball loss function (from quantile regression)
def pinball_loss(predictions, targets, quantile):
    """
    Pinball loss for quantile regression.

    Args:
        predictions: (N,) predicted quantile values
        targets: (N,) ground truth values
        quantile: float in [0, 1], e.g., 0.05, 0.5, 0.95

    Returns:
        loss: scalar
    """
    errors = targets - predictions
    loss = torch.maximum((quantile - 1) * errors, quantile * errors)
    return torch.mean(loss)

# Load ground truth data
print("Loading ground truth data...")
data = np.load(DATA_PATH)
surfaces = data["surface"]  # (N, 5, 5)
atm_values = surfaces[:, 2, 2]  # (N,) ATM point
print(f"Loaded {len(surfaces)} days of data")
print()

# Define test set (last 20% of data)
train_size = int(0.8 * len(atm_values))
test_atm = atm_values[train_size:]
train_atm = atm_values[:train_size]

print(f"Train set: {len(train_atm)} days")
print(f"Test set: {len(test_atm)} days")
print()

# Compute historical mean (from training data only)
historical_mean = np.mean(train_atm)
print(f"Historical mean IV (train): {historical_mean:.4f}")
print()

# Extract test sequences with context
context_len = 60
horizon = 90

max_start_idx = len(test_atm) - context_len - horizon
print(f"Creating {max_start_idx} test sequences (context={context_len}, horizon={horizon})")
print()

# Build test dataset
context_endpoints = []
targets_h1 = []
targets_h30 = []
targets_h90 = []

for start_idx in range(max_start_idx):
    context_end_idx = start_idx + context_len

    context_endpoint = test_atm[context_end_idx - 1]
    target_h1 = test_atm[context_end_idx]           # Day 1
    target_h30 = test_atm[context_end_idx + 29]     # Day 30
    target_h90 = test_atm[context_end_idx + 89]     # Day 90

    context_endpoints.append(context_endpoint)
    targets_h1.append(target_h1)
    targets_h30.append(target_h30)
    targets_h90.append(target_h90)

context_endpoints = torch.tensor(context_endpoints, dtype=torch.float32)
targets_h1 = torch.tensor(targets_h1, dtype=torch.float32)
targets_h30 = torch.tensor(targets_h30, dtype=torch.float32)
targets_h90 = torch.tensor(targets_h90, dtype=torch.float32)

print(f"Created {len(context_endpoints)} test sequences")
print()

# Define two prediction strategies
print("=" * 80)
print("STRATEGY COMPARISON")
print("=" * 80)
print()

results = []

for horizon_name, targets in [("H=1", targets_h1), ("H=30", targets_h30), ("H=90", targets_h90)]:
    print(f"{horizon_name}:")
    print()

    # Strategy A: Context-dependent predictions
    # Use linear model: pred = α + β * context_endpoint
    # Fit on this test set (oracle strategy - best linear predictor)
    X = context_endpoints.numpy().reshape(-1, 1)
    y = targets.numpy()

    # Simple linear regression
    X_mean = X.mean()
    y_mean = y.mean()
    beta = ((X - X_mean) * (y - y_mean)).sum() / ((X - X_mean) ** 2).sum()
    alpha = y_mean - beta * X_mean

    pred_A = torch.tensor(alpha + beta * context_endpoints.numpy(), dtype=torch.float32)

    # Strategy B: Mean reversion predictions
    # Always predict historical mean (constant prediction)
    pred_B = torch.full_like(targets, historical_mean)

    # Compute pinball losses for p05, p50, p95
    quantiles = [0.05, 0.50, 0.95]
    quantile_names = ["p05", "p50", "p95"]

    for q, q_name in zip(quantiles, quantile_names):
        # Adjust predictions for quantiles (simple approximation)
        # For Strategy A: scale by quantile
        # For Strategy B: use historical quantile

        if q_name == "p50":
            # Median predictions (use as-is)
            pred_A_q = pred_A
            pred_B_q = pred_B
        elif q_name == "p05":
            # Lower quantile: subtract some spread
            # Use empirical quantile spread from training data
            train_std = np.std(train_atm)
            pred_A_q = pred_A - 1.645 * train_std
            pred_B_q = pred_B - 1.645 * train_std
        else:  # p95
            # Upper quantile: add some spread
            train_std = np.std(train_atm)
            pred_A_q = pred_A + 1.645 * train_std
            pred_B_q = pred_B + 1.645 * train_std

        loss_A = pinball_loss(pred_A_q, targets, q).item()
        loss_B = pinball_loss(pred_B_q, targets, q).item()

        # Winner
        winner = "Strategy A" if loss_A < loss_B else "Strategy B"
        improvement = abs(loss_A - loss_B) / max(loss_A, loss_B) * 100

        print(f"  {q_name} (quantile={q}):")
        print(f"    Strategy A (context-dependent): {loss_A:.6f}")
        print(f"    Strategy B (mean reversion):    {loss_B:.6f}")
        print(f"    Winner: {winner} ({improvement:.1f}% better)")
        print()

        results.append({
            "horizon": horizon_name,
            "quantile": q_name,
            "loss_A": loss_A,
            "loss_B": loss_B,
            "winner": winner,
            "improvement_pct": improvement,
        })

# Summary across all horizons and quantiles
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

# Count wins
wins_A = sum(1 for r in results if r["winner"] == "Strategy A")
wins_B = sum(1 for r in results if r["winner"] == "Strategy B")

print(f"Strategy A wins: {wins_A}/{len(results)}")
print(f"Strategy B wins: {wins_B}/{len(results)}")
print()

# Average losses
avg_loss_A = np.mean([r["loss_A"] for r in results])
avg_loss_B = np.mean([r["loss_B"] for r in results])

print(f"Average loss Strategy A: {avg_loss_A:.6f}")
print(f"Average loss Strategy B: {avg_loss_B:.6f}")
print()

# Check if mean reversion is favored at long horizons
h90_results = [r for r in results if r["horizon"] == "H=90"]
h90_wins_B = sum(1 for r in h90_results if r["winner"] == "Strategy B")

print(f"H=90 results:")
print(f"  Strategy B (mean reversion) wins: {h90_wins_B}/{len(h90_results)}")
print()

# Visualization: Loss comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, horizon_name in enumerate(["H=1", "H=30", "H=90"]):
    ax = axes[i]

    horizon_results = [r for r in results if r["horizon"] == horizon_name]
    quantile_names = [r["quantile"] for r in horizon_results]
    losses_A = [r["loss_A"] for r in horizon_results]
    losses_B = [r["loss_B"] for r in horizon_results]

    x_pos = np.arange(len(quantile_names))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, losses_A, width, label='Strategy A (context-dep)', color='blue', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, losses_B, width, label='Strategy B (mean rev)', color='red', alpha=0.7)

    ax.set_xlabel('Quantile', fontsize=11)
    ax.set_ylabel('Pinball Loss', fontsize=11)
    ax.set_title(f'{horizon_name}', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(quantile_names)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'loss_function_bias_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'loss_function_bias_comparison.png'}")
print()

# Verdict
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

if h90_wins_B >= 2:  # If mean reversion wins at least 2/3 quantiles at H=90
    print("✅ HYPOTHESIS CONFIRMED: Loss function favors mean reversion at long horizons")
    print()
    print("IMPLICATION:")
    print("  - Pinball loss penalizes context-dependent predictions too much")
    print("  - Model learns to predict mean to minimize loss")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Add diversity regularization to loss function")
    print("  2. Use weighted pinball loss (lower weight at day-90)")
    print("  3. Consider alternative loss: Continuous Ranked Probability Score (CRPS)")
    print("  4. Add explicit penalty for predicting near historical mean")
else:
    print("❌ HYPOTHESIS REJECTED: Loss function does NOT inherently favor mean reversion")
    print()
    print("IMPLICATION:")
    print("  - Context-dependent predictions minimize loss better")
    print("  - Problem is NOT with loss function choice")
    print()
    print("RECOMMENDED NEXT STEPS:")
    print("  1. Investigate decoder horizon sensitivity (H4)")
    print("  2. Check latent information bottleneck (H3)")
    print("  3. Investigate model capacity or architecture issues")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
