"""
Test if CI width increases progressively within a 30-day generation sequence.

Question: Does the model know that predictions further from context should be more uncertain?
Expected: CI at day 1 < CI at day 30
"""
import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("PROGRESSIVE CI WIDENING TEST")
print("=" * 80)
print()
print("Question: Within a 30-day sequence, does CI widen as we move away from context?")
print("Expected: CI(day 1) < CI(day 30)")
print()

# ============================================================================
# Load Model
# ============================================================================

print("Loading model...")
model_file = "models_backfill/backfill_16yr.pt"
model_data = torch.load(model_file, weights_only=False)
model_config = model_data["model_config"]

model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
print("✓ Model loaded")

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]

# Concatenate extra features
ex_data = np.concatenate([
    ret_data[..., np.newaxis],
    skew_data[..., np.newaxis],
    slope_data[..., np.newaxis]
], axis=-1)

print("✓ Data loaded")
print()

# ============================================================================
# Generate 30-Day Sequence
# ============================================================================

context_len = model_config['context_len']
horizon = 30

print(f"Generating full 30-day sequences...")
print(f"  Context length: {context_len}")
print(f"  Horizon: {horizon}")
print()

# Temporarily change model horizon
original_horizon = model.horizon
model.horizon = horizon

# Test on a few examples from training set
test_indices = [2500, 3000, 3500, 4000]  # Different regimes
results = []

with torch.no_grad():
    for day_idx in test_indices:
        # Create full sequence: context + target
        full_seq_len = context_len + horizon

        # Extract data
        surface_seq = vol_surf_data[day_idx - context_len : day_idx + horizon]  # (C+H, 5, 5)
        ex_seq = ex_data[day_idx - context_len : day_idx + horizon]  # (C+H, 3)

        # Create input batch
        x = {
            "surface": torch.from_numpy(surface_seq).unsqueeze(0),  # (1, C+H, 5, 5)
            "ex_feats": torch.from_numpy(ex_seq).unsqueeze(0)  # (1, C+H, 3)
        }

        # Forward pass - get posterior sample
        surf_recon, ex_recon, z_mean, z_logvar, z = model.forward(x)

        # surf_recon shape: (1, H, 3, 5, 5)
        # Extract: (H, 3, 5, 5)
        surf_recon = surf_recon.cpu().numpy()[0]

        # Extract quantiles for each day
        p05 = surf_recon[:, 0, :, :]  # (H, 5, 5)
        p50 = surf_recon[:, 1, :, :]  # (H, 5, 5)
        p95 = surf_recon[:, 2, :, :]  # (H, 5, 5)

        # Compute CI width for each day
        ci_width_per_day = np.mean(p95 - p05, axis=(1, 2))  # (H,)

        results.append({
            'day_idx': day_idx,
            'ci_widths': ci_width_per_day
        })

        # Print summary for this example
        ci_day1 = ci_width_per_day[0]
        ci_day30 = ci_width_per_day[-1]
        growth = (ci_day30 / ci_day1 - 1) * 100

        print(f"Example at index {day_idx}:")
        print(f"  CI width day 1:  {ci_day1:.3f}")
        print(f"  CI width day 30: {ci_day30:.3f}")
        print(f"  Growth: {growth:+.1f}%")
        print()

# Restore original horizon
model.horizon = original_horizon

# ============================================================================
# Aggregate Analysis
# ============================================================================

print("=" * 80)
print("AGGREGATE ANALYSIS")
print("=" * 80)
print()

# Average CI width progression across all examples
all_ci_widths = np.array([r['ci_widths'] for r in results])  # (N_examples, H)
avg_ci_widths = np.mean(all_ci_widths, axis=0)  # (H,)

print("Average CI width by day:")
print(f"  Day 1:  {avg_ci_widths[0]:.3f}")
print(f"  Day 7:  {avg_ci_widths[6]:.3f}")
print(f"  Day 14: {avg_ci_widths[13]:.3f}")
print(f"  Day 30: {avg_ci_widths[29]:.3f}")
print()

avg_growth = (avg_ci_widths[-1] / avg_ci_widths[0] - 1) * 100
print(f"Average growth from day 1 to day 30: {avg_growth:+.1f}%")
print()

# Check if monotonically increasing
is_monotonic = np.all(np.diff(avg_ci_widths) >= 0)
print(f"Is average CI width monotonically increasing? {is_monotonic}")
print()

# Correlation between day number and CI width
days = np.arange(1, horizon + 1)
correlation = np.corrcoef(days, avg_ci_widths)[0, 1]
print(f"Correlation between day number and CI width: {correlation:.3f}")
print()

# ============================================================================
# Verdict
# ============================================================================

print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

if avg_growth > 15:
    verdict = "✓ YES - Model strongly understands temporal distance"
    explanation = "CI width increases significantly with horizon"
elif avg_growth > 5:
    verdict = "⚠ PARTIAL - Model somewhat understands temporal distance"
    explanation = "CI width increases moderately with horizon"
else:
    verdict = "✗ NO - Model does not understand temporal distance"
    explanation = "CI width does not increase meaningfully with horizon"

print(verdict)
print(f"  {explanation}")
print(f"  Average growth: {avg_growth:+.1f}%")
print(f"  Correlation with day number: {correlation:.3f}")
print()

if not is_monotonic:
    print("⚠ Warning: CI width is not monotonically increasing")
    print("  Model has learned temporal uncertainty but inconsistently")
print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
