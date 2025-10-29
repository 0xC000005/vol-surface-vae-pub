"""
Diagnose why teacher forcing generation has high CI violations despite quantile training.

Compares:
1. Training/test pinball loss (from model training)
2. Generation pinball loss (computed on generated surfaces vs ground truth)

This helps identify if the issue is:
- Training problem: High train/test loss → model didn't learn quantiles properly
- Generation problem: Low train/test loss but high generation loss → inference issue
- Calibration problem: Low losses but still high CI violations → need recalibration
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from vae.cvae_with_mem_randomized import CVAEMemRand, QuantileLoss
import os

print("=" * 80)
print("QUANTILE REGRESSION DIAGNOSIS: GENERATION vs TRAINING LOSS")
print("=" * 80)

# Configuration
torch.set_default_dtype(torch.float64)
quantiles = [0.05, 0.5, 0.95]
quantile_loss_fn = QuantileLoss(quantiles=quantiles)

# Load data
print("\n1. Loading ground truth data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)

print(f"   Total data: {vol_surf_data.shape[0]} days")
print(f"   Train: days 0-3999")
print(f"   Valid: days 4000-4999")
print(f"   Test: days 5000-{vol_surf_data.shape[0]-1}")

# Model configurations
base_folder = "test_spx/quantile_regression"
models = [
    {"name": "no_ex", "use_ex": False},
    {"name": "ex_no_loss", "use_ex": True},
    {"name": "ex_loss", "use_ex": True},
]

# Load training/test losses from results.csv
print(f"\n2. Loading training/test losses from results.csv...")
results_csv_path = f"{base_folder}/results.csv"
if os.path.exists(results_csv_path):
    results_df = pd.read_csv(results_csv_path)
    print(f"   Found results.csv with columns: {list(results_df.columns)}")
else:
    print(f"   ⚠ results.csv not found, will proceed without training losses")
    results_df = None

results = []

for model_info in models:
    model_name = model_info["name"]
    use_ex = model_info["use_ex"]

    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    # Load trained model
    model_path = f"{base_folder}/{model_name}.pt"
    if not os.path.exists(model_path):
        print(f"   ⚠ Model not found: {model_path}")
        continue

    print(f"\n3. Loading trained model config...")
    model_data = torch.load(model_path, weights_only=False)
    model_config = model_data["model_config"]

    print(f"   Model config: latent_dim={model_config['latent_dim']}, "
          f"mem_hidden={model_config['mem_hidden']}, "
          f"use_quantile_regression={model_config.get('use_quantile_regression', False)}")

    # Extract training/test losses from results.csv
    train_loss = None
    test_loss = None
    test_re_loss = None
    test_kl_loss = None

    if results_df is not None:
        # Try both with and without .pt extension
        model_row = results_df[results_df['fn'] == f"{model_name}.pt"]
        if len(model_row) == 0:
            model_row = results_df[results_df['fn'] == model_name]

        if len(model_row) > 0:
            test_loss = model_row['test_loss'].values[0]
            test_re_loss = model_row['test_re_surface'].values[0]
            test_kl_loss = model_row['test_kl_loss'].values[0]
            print(f"   Training losses from results.csv:")
            print(f"     - Test loss: {test_loss:.6f}")
            print(f"     - Test RE loss (pinball): {test_re_loss:.6f}")
            print(f"     - Test KL loss: {test_kl_loss:.6f}")
        else:
            print(f"   ⚠ Model {model_name} not found in results.csv")

    # Load generated surfaces
    gen_path = f"{base_folder}/{model_name}_quantile_mle_gen5.npz"
    if not os.path.exists(gen_path):
        print(f"   ⚠ Generated surfaces not found: {gen_path}")
        continue

    print(f"\n4. Loading generated surfaces...")
    gen_data = np.load(gen_path)

    # Check if quantiles are stored separately or as a single array
    if "surfaces_p05" in gen_data and "surfaces_p50" in gen_data and "surfaces_p95" in gen_data:
        # Quantiles stored separately: (days, H, W) each
        p05 = gen_data["surfaces_p05"]
        p50 = gen_data["surfaces_p50"]
        p95 = gen_data["surfaces_p95"]
        print(f"   Generated surfaces (separate quantiles):")
        print(f"     - p05 shape: {p05.shape}")
        print(f"     - p50 shape: {p50.shape}")
        print(f"     - p95 shape: {p95.shape}")

        # Stack into (days, 3, H, W)
        gen_surfaces = np.stack([p05, p50, p95], axis=1)
        print(f"   Stacked shape: {gen_surfaces.shape}")

    elif "surfaces" in gen_data:
        # Quantile format: (days, samples, 3, H, W) or (days, samples, H, W) or (days, 3, H, W)
        gen_surfaces = gen_data["surfaces"]
        print(f"   Generated surfaces shape: {gen_surfaces.shape}")

        if gen_surfaces.ndim == 5 and gen_surfaces.shape[2] == 3:
            # Quantile outputs: (days, samples, 3, H, W)
            # For MLE generation, samples should be 1
            if gen_surfaces.shape[1] == 1:
                gen_surfaces = gen_surfaces[:, 0, :, :, :]  # (days, 3, H, W)
            else:
                print(f"   ⚠ Expected 1 sample for MLE, got {gen_surfaces.shape[1]}")
                gen_surfaces = gen_surfaces[:, 0, :, :, :]
        elif gen_surfaces.ndim == 4:
            if gen_surfaces.shape[1] == 3:
                # Already (days, 3, H, W)
                pass
            elif gen_surfaces.shape[1] == 1:
                # (days, 1, H, W) - standard MSE output, not quantile
                print(f"   ⚠ This appears to be a standard MSE model output, not quantile")
                continue
            else:
                print(f"   ⚠ Unexpected shape: {gen_surfaces.shape}")
                continue
        else:
            print(f"   ⚠ Unexpected shape: {gen_surfaces.shape}")
            continue
    else:
        print(f"   ⚠ Could not find surfaces in generated data")
        print(f"   Available keys: {list(gen_data.keys())}")
        continue

    # Generation starts at day 5 (context_len=5), so ground truth starts at day 5
    start_day = 5
    num_gen_days = gen_surfaces.shape[0]
    ground_truth = vol_surf_data[start_day:start_day+num_gen_days]

    print(f"   Ground truth shape: {ground_truth.shape}")
    print(f"   Generation covers days {start_day} to {start_day+num_gen_days-1}")

    # Determine which split the generated days fall into
    test_start = 5000
    valid_start = 4000

    # Calculate how many days fall into each split
    gen_days_train = max(0, min(valid_start, start_day + num_gen_days) - start_day)
    gen_days_valid = max(0, min(test_start, start_day + num_gen_days) - max(start_day, valid_start))
    gen_days_test = max(0, (start_day + num_gen_days) - max(start_day, test_start))

    print(f"   Generated days by split:")
    print(f"     - Train: {gen_days_train} days")
    print(f"     - Valid: {gen_days_valid} days")
    print(f"     - Test: {gen_days_test} days")

    # Calculate pinball loss on generated surfaces
    print(f"\n5. Calculating pinball loss on generated surfaces...")

    # Convert to tensors
    gen_surfaces_tensor = torch.tensor(gen_surfaces, dtype=torch.float64)  # (days, 3, H, W)
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float64)  # (days, H, W)

    # Reshape for loss calculation:
    # Loss function expects: preds (B, T, num_quantiles, H, W), target (B, T, H, W)
    # We have: preds (days, 3, H, W), target (days, H, W)
    # Add T dimension: preds (days, 1, 3, H, W), target (days, 1, H, W)
    gen_surfaces_tensor = gen_surfaces_tensor.unsqueeze(1)  # (days, 1, 3, H, W)
    # But the quantile dimension needs to be at index 2, which it already is!
    # Actually, current shape is (days, 1, 3, H, W) - correct!
    # Wait, let me recheck: we stacked as [p05, p50, p95] at axis=1, so shape is (days, 3, H, W)
    # After unsqueeze(1): (days, 1, 3, H, W) - quantile is at index 2 ✓

    ground_truth_tensor = ground_truth_tensor.unsqueeze(1)  # (days, 1, H, W)

    # Calculate pinball loss - NOTE: loss function signature is (preds, target)
    generation_pinball_loss = quantile_loss_fn(gen_surfaces_tensor, ground_truth_tensor)
    generation_pinball_loss_value = generation_pinball_loss.item()

    print(f"   Generation pinball loss (all days): {generation_pinball_loss_value:.6f}")

    # Calculate pinball loss for each split separately
    if gen_days_train > 0:
        train_gen_loss = quantile_loss_fn(
            gen_surfaces_tensor[:gen_days_train],
            ground_truth_tensor[:gen_days_train]
        ).item()
        print(f"   Generation pinball loss (train days): {train_gen_loss:.6f}")
    else:
        train_gen_loss = None

    if gen_days_valid > 0:
        valid_start_idx = gen_days_train
        valid_end_idx = gen_days_train + gen_days_valid
        valid_gen_loss = quantile_loss_fn(
            gen_surfaces_tensor[valid_start_idx:valid_end_idx],
            ground_truth_tensor[valid_start_idx:valid_end_idx]
        ).item()
        print(f"   Generation pinball loss (valid days): {valid_gen_loss:.6f}")
    else:
        valid_gen_loss = None

    if gen_days_test > 0:
        test_start_idx = gen_days_train + gen_days_valid
        test_gen_loss = quantile_loss_fn(
            gen_surfaces_tensor[test_start_idx:],
            ground_truth_tensor[test_start_idx:]
        ).item()
        print(f"   Generation pinball loss (test days): {test_gen_loss:.6f}")
    else:
        test_gen_loss = None

    # Calculate per-quantile losses for more detailed analysis
    print(f"\n6. Per-quantile analysis...")
    per_quantile_losses = []
    for q_idx, quantile in enumerate(quantiles):
        # Extract quantile prediction: shape (days, 1, 3, H, W) -> (days, 1, H, W)
        q_pred = gen_surfaces_tensor[:, :, q_idx, :, :]  # (days, 1, H, W)
        error = ground_truth_tensor - q_pred  # (days, 1, H, W)
        # Pinball loss for this quantile
        q_loss = torch.where(error >= 0, quantile * error, (quantile - 1) * error).mean()
        per_quantile_losses.append(q_loss.item())
        print(f"   Quantile {quantile:.2f} loss: {q_loss.item():.6f}")

    # Store results
    results.append({
        "model": model_name,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_re_loss": test_re_loss,
        "test_kl_loss": test_kl_loss,
        "generation_loss_all": generation_pinball_loss_value,
        "generation_loss_train": train_gen_loss,
        "generation_loss_valid": valid_gen_loss,
        "generation_loss_test": test_gen_loss,
        "q05_loss": per_quantile_losses[0],
        "q50_loss": per_quantile_losses[1],
        "q95_loss": per_quantile_losses[2],
        "gen_days_train": gen_days_train,
        "gen_days_valid": gen_days_valid,
        "gen_days_test": gen_days_test,
    })

# Create summary table
print(f"\n{'='*80}")
print("SUMMARY: TRAINING vs GENERATION LOSS COMPARISON")
print(f"{'='*80}")

if len(results) == 0:
    print("\n⚠ No results to display - check that models and generated surfaces exist")
    exit(1)

df = pd.DataFrame(results)
print("\nKey Metrics Comparison:")
print(df[["model", "test_re_loss", "generation_loss_all", "generation_loss_test"]].to_string(index=False))

print("\n\nFull Results:")
print(df.to_string(index=False))

# Calculate ratios to identify discrepancies
print(f"\n{'='*80}")
print("DIAGNOSIS")
print(f"{'='*80}")

for _, row in df.iterrows():
    print(f"\n{row['model'].upper()}:")
    print(f"  Test RE loss (training): {row['test_re_loss']:.6f}")

    # Use train days for comparison since generated days don't cover test set
    comparison_loss = row['generation_loss_test'] if row['generation_loss_test'] is not None else row['generation_loss_train']
    comparison_label = "test days" if row['generation_loss_test'] is not None else "train days"

    if comparison_loss is not None:
        print(f"  Generation loss ({comparison_label}): {comparison_loss:.6f}")
    else:
        print(f"  Generation loss: No data available")

    if row['test_re_loss'] is not None and comparison_loss is not None:
        ratio = comparison_loss / row['test_re_loss']
        print(f"  Ratio (gen/{comparison_label}/test): {ratio:.2f}x")

        if ratio < 1.2:
            print(f"  ✓ Losses are similar - model is consistent between training and generation")
        elif ratio < 2.0:
            print(f"  ⚠ Generation loss is moderately higher - possible minor distribution shift")
        else:
            print(f"  ✗ Generation loss is much higher - significant problem!")
            print(f"    Possible causes:")
            print(f"    - Distribution shift between training and generation")
            print(f"    - Bug in generation code")
            print(f"    - Model not properly learning quantiles")

    print(f"\n  Per-quantile losses:")
    print(f"    - p05: {row['q05_loss']:.6f}")
    print(f"    - p50: {row['q50_loss']:.6f}")
    print(f"    - p95: {row['q95_loss']:.6f}")

    # Check if quantiles are ordered correctly
    if row['q05_loss'] < row['q50_loss'] < row['q95_loss']:
        print(f"  ✓ Quantile losses are ordered correctly")
    else:
        print(f"  ⚠ Quantile losses are NOT ordered as expected")

# Save results
output_file = f"{base_folder}/generation_vs_training_loss.csv"
df.to_csv(output_file, index=False)
print(f"\n\nResults saved to: {output_file}")

# Create visualization
print(f"\n{'='*80}")
print("CREATING VISUALIZATIONS")
print(f"{'='*80}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training vs Generation loss comparison
ax = axes[0]
x = np.arange(len(results))
width = 0.35

test_losses = [r['test_re_loss'] for r in results]
# Use train days for generation since test days aren't available
gen_losses = [r['generation_loss_test'] if r['generation_loss_test'] is not None else r['generation_loss_train'] for r in results]
model_names = [r['model'] for r in results]

ax.bar(x - width/2, test_losses, width, label='Test Loss (During Training)', alpha=0.8, color='steelblue')
ax.bar(x + width/2, gen_losses, width, label='Generation Loss (Train Days)', alpha=0.8, color='coral')

ax.set_xlabel('Model')
ax.set_ylabel('Pinball Loss')
ax.set_title('Training vs Generation Pinball Loss\n(Lower is better)')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Per-quantile losses
ax = axes[1]
q_labels = ['p05', 'p50', 'p95']
for i, model_name in enumerate(model_names):
    q_losses = [results[i]['q05_loss'], results[i]['q50_loss'], results[i]['q95_loss']]
    ax.plot(q_labels, q_losses, marker='o', label=model_name, linewidth=2, markersize=8)

ax.set_xlabel('Quantile')
ax.set_ylabel('Pinball Loss')
ax.set_title('Per-Quantile Pinball Loss on Generated Surfaces\n(Lower is better)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plot_file = f"{base_folder}/generation_vs_training_loss.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {plot_file}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
print("""
If generation loss ≈ test loss:
  → Model is working correctly during generation
  → High CI violations are due to CALIBRATION issues, not generation bugs
  → Solution: Post-hoc calibration (conformal prediction) or loss reweighting

If generation loss >> test loss:
  → Something is wrong with the generation process
  → Possible issues: distribution shift, bug in generation code, or model collapse
  → Solution: Debug generation code, check for distribution shift

Next steps based on diagnosis above.
""")
