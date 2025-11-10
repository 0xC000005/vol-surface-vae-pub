"""
Generate 1D Backfilling IN-SAMPLE Predictions (2008-2010).

⚠️ WARNING: These are IN-SAMPLE predictions on TRAINING DATA.
The model was trained on this data, so these predictions are NOT for evaluation.
Use only for visualization and understanding model behavior on historical periods.

Tests 3 different ways to select the latent variable for prediction:

1. **Scenario 1 (Oracle)**: Use true latent from encoding with real AMZN at T+1
2. **Scenario 2 (Mixed 80/20)**: Randomly use oracle (80%) or masked (20%)
3. **Scenario 3 (Realistic Backfilling)**: Forward-fill AMZN at T+1, use previous latent

Target period: 2008-2010 (includes 2008 financial crisis)
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
OUTPUT_FILE = "models_1d_backfilling/backfill_predictions_insample.npz"
CONTEXT_LEN = 5  # Number of historical days to use
SEED = 42  # For reproducibility

# In-sample date range (2008-2010)
INSAMPLE_START = 2000  # ~2008-01-16
INSAMPLE_END = 2700    # ~2010-10-26

print("=" * 80)
print("GENERATING 1D BACKFILLING IN-SAMPLE PREDICTIONS")
print("⚠️  WARNING: IN-SAMPLE PREDICTIONS ON TRAINING DATA")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Data: {DATA_FILE}")
print(f"Context length: {CONTEXT_LEN}")
print(f"Date range: Indices {INSAMPLE_START}-{INSAMPLE_END} (~2008-2010)")
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
all_features = data["all_features"]  # (N, 12) - Unified target: 3 stocks × 4 features
dates = data["dates"]

print(f"  Total samples: {len(all_features)}")
print(f"  All features shape: {all_features.shape}")
print(f"  Features: 12D (AMZN×4 + MSFT×4 + SP500×4)")
print()

# In-sample subset
print(f"In-sample subset:")
print(f"  Start index: {INSAMPLE_START} ({dates[INSAMPLE_START]})")
print(f"  End index: {INSAMPLE_END} ({dates[INSAMPLE_END]})")
print(f"  Size: {INSAMPLE_END - INSAMPLE_START}")
print(f"  First prediction at index: {INSAMPLE_START + CONTEXT_LEN} (need {CONTEXT_LEN} days context)")
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


def generate_scenario_1_oracle(model, all_features_full, day, ctx_len):
    """
    Scenario 1 (Oracle): Use true latent from encoding with real data at T+1.

    Args:
        model: Trained CVAE model
        all_features_full: Full features (N, 12) - All stock features
        day: Current day index (prediction for day)
        ctx_len: Context length

    Returns:
        prediction: (3,) array [p05, p50, p95] for AMZN return (channel 0)
    """
    # Extract full sequence [0:day+1] including target at day
    start_idx = day - ctx_len
    end_idx = day + 1

    target_seq = torch.from_numpy(all_features_full[start_idx:end_idx]).unsqueeze(0).to(model.config['device'])  # (1, T+1, 12)

    # Encode with REAL data at T+1
    x = {"target": target_seq}
    z_mean, z_log_var, _ = model.encoder(x)

    # Use latent at T+1 (index ctx_len)
    z_mean_t = z_mean[:, ctx_len, :]  # (1, latent_dim)
    z_logvar_t = z_log_var[:, ctx_len, :]  # (1, latent_dim)

    # Context (historical data only)
    context = {"target": target_seq[:, :ctx_len, :]}

    # Generate prediction with oracle latent
    prediction = model.get_prediction_with_latent(context, z_mean_t, z_logvar_t)
    # prediction shape: (1, num_quantiles, target_dim) = (1, 3, 12)

    # Extract channel 0 (AMZN return)
    prediction_amzn = prediction[0, :, 0]  # (3,) - [p05, p50, p95] for AMZN

    return prediction_amzn.cpu().numpy()  # (3,)


def generate_scenario_2_mixed(model, all_features_full, day, ctx_len, oracle_prob=0.8):
    """
    Scenario 2 (Mixed 80/20): Randomly use oracle or forward-masked latent.

    Args:
        model: Trained CVAE model
        all_features_full: Full features (N, 12) - All stock features
        day: Current day index (prediction for day)
        ctx_len: Context length
        oracle_prob: Probability of using oracle latent (default: 0.8)

    Returns:
        prediction: (3,) array [p05, p50, p95] for AMZN return (channel 0)
    """
    # Randomly decide: oracle or masked
    if np.random.random() < oracle_prob:
        # Use oracle (scenario 1)
        return generate_scenario_1_oracle(model, all_features_full, day, ctx_len)
    else:
        # Use forward-masked (scenario 3)
        return generate_scenario_3_realistic(model, all_features_full, day, ctx_len)


def generate_scenario_3_realistic(model, all_features_full, day, ctx_len):
    """
    Scenario 3 (Realistic Backfilling): Forward-fill AMZN (channel 0) at T+1, use previous latent.

    Args:
        model: Trained CVAE model
        all_features_full: Full features (N, 12) - All stock features
        day: Current day index (prediction for day)
        ctx_len: Context length

    Returns:
        prediction: (3,) array [p05, p50, p95] for AMZN return (channel 0)
    """
    # Extract sequence [0:day+1]
    start_idx = day - ctx_len
    end_idx = day + 1

    target_seq = torch.from_numpy(all_features_full[start_idx:end_idx]).unsqueeze(0).to(model.config['device'])  # (1, T+1, 12)

    # Forward-fill AMZN (channel 0) at T+1: repeat last known value
    # Other channels (MSFT, SP500) remain unmasked
    target_masked = target_seq.clone()
    target_masked[:, ctx_len:, 0:1] = target_seq[:, ctx_len-1:ctx_len, 0:1]  # Forward-fill channel 0 only, preserve dimensions

    # Encode with masked data
    x_masked = {"target": target_masked}
    z_mean, z_log_var, _ = model.encoder(x_masked)

    # Use PREVIOUS timestep latent (index ctx_len-1, not ctx_len)
    z_mean_prev = z_mean[:, ctx_len-1, :]  # (1, latent_dim)
    z_logvar_prev = z_log_var[:, ctx_len-1, :]  # (1, latent_dim)

    # Context (historical data only)
    context = {"target": target_seq[:, :ctx_len, :]}  # Real historical data

    # Generate prediction with previous-day latent
    prediction = model.get_prediction_with_latent(context, z_mean_prev, z_logvar_prev)
    # prediction shape: (1, num_quantiles, target_dim) = (1, 3, 12)

    # Extract channel 0 (AMZN return)
    prediction_amzn = prediction[0, :, 0]  # (3,) - [p05, p50, p95] for AMZN

    return prediction_amzn.cpu().numpy()  # (3,)


# Generate predictions for in-sample period
print("=" * 80)
print("GENERATING IN-SAMPLE PREDICTIONS (2008-2010)")
print("=" * 80)

# First valid day for prediction (need ctx_len days of history)
first_day = INSAMPLE_START + CONTEXT_LEN
num_predictions = INSAMPLE_END - first_day

print(f"First prediction day: {first_day} ({dates[first_day]})")
print(f"Last prediction day: {INSAMPLE_END - 1} ({dates[INSAMPLE_END - 1]})")
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
for i, day in enumerate(tqdm(range(first_day, INSAMPLE_END), desc="Generating predictions")):
    # Actual value (AMZN return is channel 0)
    actual = all_features[day, 0]
    actuals.append(actual)
    dates_pred.append(dates[day])

    # Scenario 1: Oracle
    pred_s1 = generate_scenario_1_oracle(model, all_features, day, CONTEXT_LEN)
    s1_p05.append(pred_s1[0])
    s1_p50.append(pred_s1[1])
    s1_p95.append(pred_s1[2])

    # Scenario 2: Mixed 80/20
    pred_s2 = generate_scenario_2_mixed(model, all_features, day, CONTEXT_LEN, oracle_prob=0.8)
    s2_p05.append(pred_s2[0])
    s2_p50.append(pred_s2[1])
    s2_p95.append(pred_s2[2])

    # Scenario 3: Realistic Backfilling
    pred_s3 = generate_scenario_3_realistic(model, all_features, day, CONTEXT_LEN)
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
print(f"Date range: {dates_pred[0]} to {dates_pred[-1]}")
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
print("⚠️  REMINDER: These are IN-SAMPLE predictions (model trained on this data)")
print("Next step:")
print("  Run: python visualize_1d_backfilling_predictions_insample.py")
