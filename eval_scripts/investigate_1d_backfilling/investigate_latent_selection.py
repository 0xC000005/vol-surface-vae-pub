"""
Investigation: Test Latent Selection Fix for Realistic Backfilling.

Compares 4 scenarios:
1. Oracle (S1): z[T] from real AMZN encoding (upper bound)
2. Mixed 80/20 (S2): Random mix (mimics training)
3. Realistic Original (S3): z[T-1] from masked encoding (CURRENT - BROKEN)
4. Realistic Fixed (S4): z[T] from masked encoding (PROPOSED FIX)

Hypothesis: S4 should leverage MSFT/SP500 information at T+1, improving performance.
"""

import numpy as np
import torch
from vae.cvae_1d_with_mem_randomized import CVAE1DMemRand
from vae.utils import set_seeds
import os
from tqdm import tqdm

# Configuration
DATA_FILE = "data/stock_returns_multifeature.npz"
MODEL_PATH = "models_1d_backfilling/backfill_model.pt"
OUTPUT_FILE = "models_1d_backfilling/latent_selection_investigation.npz"
CONTEXT_LEN = 5
SEED = 42

print("=" * 80)
print("INVESTIGATING LATENT SELECTION - 4 SCENARIOS")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Context length: {CONTEXT_LEN}")
print(f"Output: {OUTPUT_FILE}")
print()

# Set random seed
set_seeds(SEED)
torch.set_default_dtype(torch.float64)

# Load data
print("Loading data...")
data = np.load(DATA_FILE)
all_features = data["all_features"]  # (N, 12)
dates = data["dates"]
print(f"  Total samples: {len(all_features)}")
print()

# Test set split
TRAIN_END = 4000
VALID_END = 5000
test_start = VALID_END

all_test = all_features[test_start:]
dates_test = dates[test_start:]

print(f"Test set: {len(all_test)} samples (start index: {test_start})")
print()

# Load model
print("Loading model...")
model_data = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = CVAE1DMemRand(model_data["model_config"])
model.load_weights(dict_to_load=model_data)
model.eval()
print(f"  Model loaded: latent_dim={model.config['latent_dim']}")
print()


def generate_scenario_1_oracle(model, all_features_full, day, ctx_len):
    """S1: Oracle - z[T] from real AMZN encoding."""
    start_idx = day - ctx_len
    end_idx = day + 1

    target_seq = torch.from_numpy(all_features_full[start_idx:end_idx]).unsqueeze(0).to(model.config['device'])

    # Encode with REAL data
    x = {"target": target_seq}
    z_mean, z_log_var, _ = model.encoder(x)

    # Use latent at T
    z_mean_t = z_mean[:, ctx_len, :]
    z_logvar_t = z_log_var[:, ctx_len, :]

    # Context
    context = {"target": target_seq[:, :ctx_len, :]}

    # Generate
    prediction = model.get_prediction_with_latent(context, z_mean_t, z_logvar_t)
    prediction_amzn = prediction[0, :, 0]  # (3,)

    return prediction_amzn.detach().cpu().numpy(), z_mean_t.detach().cpu().numpy(), z_logvar_t.detach().cpu().numpy()


def generate_scenario_2_mixed(model, all_features_full, day, ctx_len, oracle_prob=0.8):
    """S2: Mixed 80/20 - Random oracle or realistic."""
    if np.random.random() < oracle_prob:
        pred, z_mean, z_logvar = generate_scenario_1_oracle(model, all_features_full, day, ctx_len)
        return pred, z_mean, z_logvar
    else:
        pred, z_mean, z_logvar = generate_scenario_3_realistic_original(model, all_features_full, day, ctx_len)
        return pred, z_mean, z_logvar


def generate_scenario_3_realistic_original(model, all_features_full, day, ctx_len):
    """S3: Realistic Original - z[T-1] from masked encoding (CURRENT IMPLEMENTATION - BROKEN)."""
    start_idx = day - ctx_len
    end_idx = day + 1

    target_seq = torch.from_numpy(all_features_full[start_idx:end_idx]).unsqueeze(0).to(model.config['device'])

    # Forward-fill AMZN (channel 0) at T+1
    target_masked = target_seq.clone()
    target_masked[:, ctx_len:, 0:1] = target_seq[:, ctx_len-1:ctx_len, 0:1]

    # Encode with masked data
    x_masked = {"target": target_masked}
    z_mean, z_log_var, _ = model.encoder(x_masked)

    # BUG: Use PREVIOUS timestep latent (T-1)
    z_mean_prev = z_mean[:, ctx_len-1, :]  # ← WRONG! Misses MSFT/SP500 at T+1
    z_logvar_prev = z_log_var[:, ctx_len-1, :]

    # Context
    context = {"target": target_seq[:, :ctx_len, :]}

    # Generate
    prediction = model.get_prediction_with_latent(context, z_mean_prev, z_logvar_prev)
    prediction_amzn = prediction[0, :, 0]

    return prediction_amzn.detach().cpu().numpy(), z_mean_prev.detach().cpu().numpy(), z_logvar_prev.detach().cpu().numpy()


def generate_scenario_4_realistic_fixed(model, all_features_full, day, ctx_len):
    """S4: Realistic Fixed - z[T] from masked encoding (PROPOSED FIX)."""
    start_idx = day - ctx_len
    end_idx = day + 1

    target_seq = torch.from_numpy(all_features_full[start_idx:end_idx]).unsqueeze(0).to(model.config['device'])

    # Forward-fill AMZN (channel 0) at T+1
    target_masked = target_seq.clone()
    target_masked[:, ctx_len:, 0:1] = target_seq[:, ctx_len-1:ctx_len, 0:1]

    # Encode with masked data
    x_masked = {"target": target_masked}
    z_mean, z_log_var, _ = model.encoder(x_masked)

    # FIX: Use CURRENT timestep latent (T) which saw MSFT[T+1] and SP500[T+1]
    z_mean_current = z_mean[:, ctx_len, :]  # ← FIXED! Has MSFT/SP500 info
    z_logvar_current = z_log_var[:, ctx_len, :]

    # Context
    context = {"target": target_seq[:, :ctx_len, :]}

    # Generate
    prediction = model.get_prediction_with_latent(context, z_mean_current, z_logvar_current)
    prediction_amzn = prediction[0, :, 0]

    return prediction_amzn.detach().cpu().numpy(), z_mean_current.detach().cpu().numpy(), z_logvar_current.detach().cpu().numpy()


# Generate predictions
print("=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

first_day = test_start + CONTEXT_LEN
num_predictions = len(all_test) - CONTEXT_LEN

print(f"First prediction day: {first_day}")
print(f"Number of predictions: {num_predictions}")
print()

# Storage
actuals = []
dates_pred = []

s1_preds = {"p05": [], "p50": [], "p95": [], "z_mean": [], "z_logvar": []}
s2_preds = {"p05": [], "p50": [], "p95": [], "z_mean": [], "z_logvar": []}
s3_preds = {"p05": [], "p50": [], "p95": [], "z_mean": [], "z_logvar": []}
s4_preds = {"p05": [], "p50": [], "p95": [], "z_mean": [], "z_logvar": []}

for i, day in enumerate(tqdm(range(first_day, len(all_features)), desc="Generating")):
    actual = all_features[day, 0]
    actuals.append(actual)
    dates_pred.append(dates[day])

    # S1: Oracle
    pred_s1, z_mean_s1, z_logvar_s1 = generate_scenario_1_oracle(model, all_features, day, CONTEXT_LEN)
    s1_preds["p05"].append(pred_s1[0])
    s1_preds["p50"].append(pred_s1[1])
    s1_preds["p95"].append(pred_s1[2])
    s1_preds["z_mean"].append(z_mean_s1)
    s1_preds["z_logvar"].append(z_logvar_s1)

    # S2: Mixed
    pred_s2, z_mean_s2, z_logvar_s2 = generate_scenario_2_mixed(model, all_features, day, CONTEXT_LEN, oracle_prob=0.8)
    s2_preds["p05"].append(pred_s2[0])
    s2_preds["p50"].append(pred_s2[1])
    s2_preds["p95"].append(pred_s2[2])
    s2_preds["z_mean"].append(z_mean_s2)
    s2_preds["z_logvar"].append(z_logvar_s2)

    # S3: Realistic Original (BROKEN)
    pred_s3, z_mean_s3, z_logvar_s3 = generate_scenario_3_realistic_original(model, all_features, day, CONTEXT_LEN)
    s3_preds["p05"].append(pred_s3[0])
    s3_preds["p50"].append(pred_s3[1])
    s3_preds["p95"].append(pred_s3[2])
    s3_preds["z_mean"].append(z_mean_s3)
    s3_preds["z_logvar"].append(z_logvar_s3)

    # S4: Realistic Fixed (PROPOSED)
    pred_s4, z_mean_s4, z_logvar_s4 = generate_scenario_4_realistic_fixed(model, all_features, day, CONTEXT_LEN)
    s4_preds["p05"].append(pred_s4[0])
    s4_preds["p50"].append(pred_s4[1])
    s4_preds["p95"].append(pred_s4[2])
    s4_preds["z_mean"].append(z_mean_s4)
    s4_preds["z_logvar"].append(z_logvar_s4)

print()

# Convert to arrays
actuals = np.array(actuals)
dates_pred = np.array(dates_pred)

for key in s1_preds:
    if key in ["z_mean", "z_logvar"]:
        s1_preds[key] = np.concatenate(s1_preds[key], axis=0)  # (N, latent_dim)
        s2_preds[key] = np.concatenate(s2_preds[key], axis=0)
        s3_preds[key] = np.concatenate(s3_preds[key], axis=0)
        s4_preds[key] = np.concatenate(s4_preds[key], axis=0)
    else:
        s1_preds[key] = np.array(s1_preds[key])
        s2_preds[key] = np.array(s2_preds[key])
        s3_preds[key] = np.array(s3_preds[key])
        s4_preds[key] = np.array(s4_preds[key])

# Save
np.savez(
    OUTPUT_FILE,
    dates=dates_pred,
    actuals=actuals,
    # S1
    s1_p05=s1_preds["p05"], s1_p50=s1_preds["p50"], s1_p95=s1_preds["p95"],
    s1_z_mean=s1_preds["z_mean"], s1_z_logvar=s1_preds["z_logvar"],
    # S2
    s2_p05=s2_preds["p05"], s2_p50=s2_preds["p50"], s2_p95=s2_preds["p95"],
    s2_z_mean=s2_preds["z_mean"], s2_z_logvar=s2_preds["z_logvar"],
    # S3
    s3_p05=s3_preds["p05"], s3_p50=s3_preds["p50"], s3_p95=s3_preds["p95"],
    s3_z_mean=s3_preds["z_mean"], s3_z_logvar=s3_preds["z_logvar"],
    # S4
    s4_p05=s4_preds["p05"], s4_p50=s4_preds["p50"], s4_p95=s4_preds["p95"],
    s4_z_mean=s4_preds["z_mean"], s4_z_logvar=s4_preds["z_logvar"],
)

print(f"Saved to: {OUTPUT_FILE}")
print()

# Quick comparison
print("=" * 80)
print("QUICK COMPARISON")
print("=" * 80)
print()

scenarios = [
    ("S1: Oracle (z[T] from real)", s1_preds),
    ("S2: Mixed 80/20", s2_preds),
    ("S3: Realistic Original (z[T-1])", s3_preds),
    ("S4: Realistic Fixed (z[T])", s4_preds),
]

for name, preds in scenarios:
    p05, p50, p95 = preds["p05"], preds["p50"], preds["p95"]

    rmse = np.sqrt(np.mean((p50 - actuals) ** 2))
    mae = np.mean(np.abs(p50 - actuals))

    pred_signs = np.sign(p50)
    actual_signs = np.sign(actuals)
    direction_acc = np.mean(pred_signs == actual_signs) * 100

    violations = (actuals < p05) | (actuals > p95)
    ci_violation_rate = np.mean(violations) * 100

    mean_ci_width = np.mean(p95 - p05)

    print(f"{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Direction Acc: {direction_acc:.2f}%")
    print(f"  CI Violations: {ci_violation_rate:.2f}%")
    print(f"  Mean CI Width: {mean_ci_width:.4f}")
    print()

print("=" * 80)
print("KEY COMPARISON: S3 (BROKEN) vs S4 (FIXED)")
print("=" * 80)
print()

rmse_s3 = np.sqrt(np.mean((s3_preds["p50"] - actuals) ** 2))
rmse_s4 = np.sqrt(np.mean((s4_preds["p50"] - actuals) ** 2))
rmse_improvement = ((rmse_s3 - rmse_s4) / rmse_s3) * 100

direction_s3 = np.mean(np.sign(s3_preds["p50"]) == np.sign(actuals)) * 100
direction_s4 = np.mean(np.sign(s4_preds["p50"]) == np.sign(actuals)) * 100
direction_improvement = direction_s4 - direction_s3

ci_s3 = np.mean((actuals < s3_preds["p05"]) | (actuals > s3_preds["p95"])) * 100
ci_s4 = np.mean((actuals < s4_preds["p05"]) | (actuals > s4_preds["p95"])) * 100
ci_improvement = ci_s3 - ci_s4

print(f"RMSE improvement: {rmse_improvement:+.2f}% ({rmse_s3:.4f} → {rmse_s4:.4f})")
print(f"Direction accuracy improvement: {direction_improvement:+.2f}% ({direction_s3:.2f}% → {direction_s4:.2f}%)")
print(f"CI violations improvement: {ci_improvement:+.2f}% ({ci_s3:.2f}% → {ci_s4:.2f}%)")
print()

if direction_s4 > 52:
    print("✓ FIX SUCCESSFUL: Direction accuracy > 52% shows predictive signal!")
elif direction_s4 > direction_s3 + 2:
    print("✓ FIX HELPS: Modest improvement, MSFT/SP500 have some signal")
else:
    print("✗ FIX INEFFECTIVE: Latent may have collapsed or MSFT/SP500 uncorrelated")
print()

print("Next step: python analyze_latent_investigation.py")
