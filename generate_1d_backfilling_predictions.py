"""
Generate 1D Backfilling Predictions with 3 Latent Selection Scenarios.

Tests 3 different ways to select the latent variable for prediction:

1. **Scenario 1 (Oracle)**: Use true latent from encoding with real AMZN at T+1
   - Encode: [AMZN[0:T+1], MSFT[0:T+1], SP500[0:T+1]] (all real data)
   - Use: z_mean[:, T] and z_logvar[:, T]
   - Performance: Upper bound (has future information)

2. **Scenario 2 (Mixed 80/20)**: Randomly use oracle (80%) or masked (20%)
   - Mimics training distribution
   - Tests if mixed training helps

3. **Scenario 3 (Realistic Backfilling)**: Forward-fill AMZN at T+1, use previous latent
   - Encode: [AMZN_fwd_fill[0:T+1], MSFT[0:T+1], SP500[0:T+1]]
   - Use: z_mean[:, T-1] and z_logvar[:, T-1] (previous timestep)
   - Performance: Production scenario

All scenarios sample from N(z_mean, exp(z_logvar)) using reparameterization.
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
OUTPUT_FILE = "models_1d_backfilling/backfill_predictions_ctx5.npz"
CONTEXT_LEN = 5  # Number of historical days to use
SEED = 42  # For reproducibility

print("=" * 80)
print("GENERATING 1D BACKFILLING PREDICTIONS - 3 SCENARIOS")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Data: {DATA_FILE}")
print(f"Context length: {CONTEXT_LEN}")
print(f"Output: {OUTPUT_FILE}")
print()

# Set random seed
set_seeds(SEED)
torch.set_default_dtype(torch.float64)

# Check files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun: python train_1d_backfilling_model.py")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data not found: {DATA_FILE}\nRun: python prepare_stock_data_multifeature.py")

# Load data
print("Loading data...")
data = np.load(DATA_FILE)
amzn_return = data["amzn_return"]  # (N, 1)
extra_features = data["extra_features"]  # (N, 11)
dates = data["dates"]

print(f"  Total samples: {len(amzn_return)}")
print(f"  AMZN return shape: {amzn_return.shape}")
print(f"  Extra features shape: {extra_features.shape}")
print()

# Test set split
TRAIN_END = 4000
VALID_END = 5000
test_start = VALID_END

amzn_test = amzn_return[test_start:]
extra_test = extra_features[test_start:]
dates_test = dates[test_start:]

print(f"Test set:")
print(f"  Start index: {test_start}")
print(f"  Size: {len(amzn_test)}")
print(f"  First prediction at index: {test_start + CONTEXT_LEN} (need {CONTEXT_LEN} days context)")
print()

# Load model
print("Loading model...")
model_data = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = CVAE1DMemRand(model_data["model_config"])
model.load_weights(dict_to_load=model_data)
model.eval()

print(f"  Model loaded successfully")
print(f"  Config: latent_dim={model.config['latent_dim']}, ex_feats_dim={model.config['ex_feats_dim']}")
print()


def generate_scenario_1_oracle(model, amzn_full, extra_full, day, ctx_len):
    """
    Scenario 1 (Oracle): Use true latent from encoding with real AMZN at T+1.

    Args:
        model: Trained CVAE model
        amzn_full: Full AMZN data (N, 1)
        extra_full: Full extra features (N, 11)
        day: Current day index (prediction for day)
        ctx_len: Context length

    Returns:
        prediction: (3,) array [p05, p50, p95]
    """
    # Extract full sequence [0:day+1] including target at day
    start_idx = day - ctx_len
    end_idx = day + 1

    target_seq = torch.from_numpy(amzn_full[start_idx:end_idx]).unsqueeze(0)  # (1, T+1, 1)
    extra_seq = torch.from_numpy(extra_full[start_idx:end_idx]).unsqueeze(0)  # (1, T+1, 11)

    # Encode with REAL data at T+1
    x = {"target": target_seq, "ex_feats": extra_seq}
    z_mean, z_log_var, _ = model.encoder(x)

    # Use latent at T+1 (index ctx_len)
    z_mean_t = z_mean[:, ctx_len, :]  # (1, latent_dim)
    z_logvar_t = z_log_var[:, ctx_len, :]  # (1, latent_dim)

    # Context (historical data only)
    context = {
        "target": target_seq[:, :ctx_len, :],
        "ex_feats": extra_seq[:, :ctx_len, :]
    }

    # Generate prediction with oracle latent
    prediction = model.get_prediction_with_latent(context, z_mean_t, z_logvar_t)

    return prediction.squeeze(0).cpu().numpy()  # (3,)


def generate_scenario_2_mixed(model, amzn_full, extra_full, day, ctx_len, oracle_prob=0.8):
    """
    Scenario 2 (Mixed 80/20): Randomly use oracle or forward-masked latent.

    Args:
        model: Trained CVAE model
        amzn_full: Full AMZN data (N, 1)
        extra_full: Full extra features (N, 11)
        day: Current day index (prediction for day)
        ctx_len: Context length
        oracle_prob: Probability of using oracle latent (default: 0.8)

    Returns:
        prediction: (3,) array [p05, p50, p95]
    """
    # Randomly decide: oracle or masked
    if np.random.random() < oracle_prob:
        # Use oracle (scenario 1)
        return generate_scenario_1_oracle(model, amzn_full, extra_full, day, ctx_len)
    else:
        # Use forward-masked (scenario 3)
        return generate_scenario_3_realistic(model, amzn_full, extra_full, day, ctx_len)


def generate_scenario_3_realistic(model, amzn_full, extra_full, day, ctx_len):
    """
    Scenario 3 (Realistic Backfilling): Forward-fill AMZN at T+1, use previous latent.

    Args:
        model: Trained CVAE model
        amzn_full: Full AMZN data (N, 1)
        extra_full: Full extra features (N, 11)
        day: Current day index (prediction for day)
        ctx_len: Context length

    Returns:
        prediction: (3,) array [p05, p50, p95]
    """
    # Extract sequence [0:day+1]
    start_idx = day - ctx_len
    end_idx = day + 1

    target_seq = torch.from_numpy(amzn_full[start_idx:end_idx]).unsqueeze(0)  # (1, T+1, 1)
    extra_seq = torch.from_numpy(extra_full[start_idx:end_idx]).unsqueeze(0)  # (1, T+1, 11)

    # Forward-fill AMZN at T+1: repeat last known value
    target_masked = torch.cat([
        target_seq[:, :ctx_len, :],  # Real AMZN [0:T]
        target_seq[:, ctx_len-1:ctx_len, :]  # Repeat AMZN[T] (forward fill)
    ], dim=1)  # (1, T+1, 1)

    # Encode with masked target but REAL extra features
    x_masked = {"target": target_masked, "ex_feats": extra_seq}
    z_mean, z_log_var, _ = model.encoder(x_masked)

    # Use PREVIOUS timestep latent (index ctx_len-1, not ctx_len)
    z_mean_prev = z_mean[:, ctx_len-1, :]  # (1, latent_dim)
    z_logvar_prev = z_log_var[:, ctx_len-1, :]  # (1, latent_dim)

    # Context (historical data only)
    context = {
        "target": target_seq[:, :ctx_len, :],  # Real historical AMZN
        "ex_feats": extra_seq[:, :ctx_len, :]
    }

    # Generate prediction with previous-day latent
    prediction = model.get_prediction_with_latent(context, z_mean_prev, z_logvar_prev)

    return prediction.squeeze(0).cpu().numpy()  # (3,)


# Generate predictions for all test days
print("=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

# First valid day for prediction (need ctx_len days of history)
first_day = test_start + CONTEXT_LEN
num_predictions = len(amzn_test) - CONTEXT_LEN

print(f"First prediction day: {first_day}")
print(f"Number of predictions: {num_predictions}")
print()

# Storage arrays
actuals = []
dates_pred = []

s1_p05 = []
s1_p50 = []
s1_p95 = []

s2_p05 = []
s2_p50 = []
s2_p95 = []

s3_p05 = []
s3_p50 = []
s3_p95 = []

# Generate predictions
for i, day in enumerate(tqdm(range(first_day, len(amzn_return)), desc="Generating predictions")):
    # Actual value
    actual = amzn_return[day, 0]
    actuals.append(actual)
    dates_pred.append(dates[day])

    # Scenario 1: Oracle
    pred_s1 = generate_scenario_1_oracle(model, amzn_return, extra_features, day, CONTEXT_LEN)
    s1_p05.append(pred_s1[0])
    s1_p50.append(pred_s1[1])
    s1_p95.append(pred_s1[2])

    # Scenario 2: Mixed 80/20
    pred_s2 = generate_scenario_2_mixed(model, amzn_return, extra_features, day, CONTEXT_LEN, oracle_prob=0.8)
    s2_p05.append(pred_s2[0])
    s2_p50.append(pred_s2[1])
    s2_p95.append(pred_s2[2])

    # Scenario 3: Realistic Backfilling
    pred_s3 = generate_scenario_3_realistic(model, amzn_return, extra_features, day, CONTEXT_LEN)
    s3_p05.append(pred_s3[0])
    s3_p50.append(pred_s3[1])
    s3_p95.append(pred_s3[2])

print()
print("=" * 80)
print("SAVING PREDICTIONS")
print("=" * 80)

# Convert to numpy arrays
actuals = np.array(actuals)
dates_pred = np.array(dates_pred)

s1_p05 = np.array(s1_p05)
s1_p50 = np.array(s1_p50)
s1_p95 = np.array(s1_p95)

s2_p05 = np.array(s2_p05)
s2_p50 = np.array(s2_p50)
s2_p95 = np.array(s2_p95)

s3_p05 = np.array(s3_p05)
s3_p50 = np.array(s3_p50)
s3_p95 = np.array(s3_p95)

# Save to NPZ
np.savez(
    OUTPUT_FILE,
    dates=dates_pred,
    actuals=actuals,
    # Scenario 1: Oracle
    s1_p05=s1_p05,
    s1_p50=s1_p50,
    s1_p95=s1_p95,
    # Scenario 2: Mixed 80/20
    s2_p05=s2_p05,
    s2_p50=s2_p50,
    s2_p95=s2_p95,
    # Scenario 3: Realistic Backfilling
    s3_p05=s3_p05,
    s3_p50=s3_p50,
    s3_p95=s3_p95,
)

print(f"Predictions saved to: {OUTPUT_FILE}")
print()

# Quick stats
print("Quick statistics:")
print("-" * 80)
print(f"Total predictions: {len(actuals)}")
print(f"Actuals: mean={actuals.mean():.4f}, std={actuals.std():.4f}")
print()
print(f"Scenario 1 (Oracle):")
print(f"  p50: mean={s1_p50.mean():.4f}, std={s1_p50.std():.4f}")
print(f"  CI width: mean={(s1_p95 - s1_p05).mean():.4f}")
print()
print(f"Scenario 2 (Mixed 80/20):")
print(f"  p50: mean={s2_p50.mean():.4f}, std={s2_p50.std():.4f}")
print(f"  CI width: mean={(s2_p95 - s2_p05).mean():.4f}")
print()
print(f"Scenario 3 (Realistic Backfilling):")
print(f"  p50: mean={s3_p50.mean():.4f}, std={s3_p50.std():.4f}")
print(f"  CI width: mean={(s3_p95 - s3_p05).mean():.4f}")
print()

print("=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)
print()
print("Next step:")
print("  Run: python evaluate_1d_backfilling_model.py")
