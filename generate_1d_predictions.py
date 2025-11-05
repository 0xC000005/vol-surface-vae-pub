"""
Generate predictions from trained 1D VAE models using teacher forcing.

For each day in the test set:
- Use context (past 5-10 days)
- Generate prediction for next day
- Produce both stochastic (1000 samples) and MLE (deterministic) predictions
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vae.cvae_1d_with_mem_randomized import CVAE1DMemRand
import os

# Configuration
DATA_FILE = "data/stock_returns.npz"
MODELS_DIR = "models_1d"
OUTPUT_DIR = "predictions_1d"
CONTEXT_LEN = 5
NUM_SAMPLES = 1000  # For stochastic generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test set configuration
TRAIN_END = 4000
VALID_END = 5000

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING 1D VAE PREDICTIONS (TEACHER FORCING)")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Context length: {CONTEXT_LEN}")
print(f"Stochastic samples: {NUM_SAMPLES}")
print()

# Load data
print("Loading data...")
data = np.load(DATA_FILE)
amzn_returns = data["amzn_returns"]
sp500_returns = data["sp500_returns"]
msft_returns = data["msft_returns"]
cond_sp500 = data["cond_sp500"]
cond_msft = data["cond_msft"]
cond_both = data["cond_both"]
dates = pd.to_datetime(data["dates"])

# Extract test set
amzn_test = amzn_returns[VALID_END:]
sp500_test = cond_sp500[VALID_END:]
msft_test = cond_msft[VALID_END:]
both_test = cond_both[VALID_END:]
dates_test = dates[VALID_END:]

# Need context + test samples
# Context starts at VALID_END - CONTEXT_LEN
amzn_full = amzn_returns[VALID_END - CONTEXT_LEN:]
sp500_full = cond_sp500[VALID_END - CONTEXT_LEN:]
msft_full = cond_msft[VALID_END - CONTEXT_LEN:]
both_full = cond_both[VALID_END - CONTEXT_LEN:]

num_test_days = len(amzn_test)

print(f"Test set: {num_test_days} days")
print(f"Date range: {dates_test[0].strftime('%Y-%m-%d')} to {dates_test[-1].strftime('%Y-%m-%d')}")
print()

# Model configurations
models = [
    {
        "name": "amzn_only",
        "file": "amzn_only.pt",
        "description": "Amazon only",
        "has_cond": False,
        "cond_data": None,
    },
    {
        "name": "amzn_sp500",
        "file": "amzn_sp500_no_loss.pt",
        "description": "Amazon + SP500",
        "has_cond": True,
        "cond_data": sp500_full,
    },
    {
        "name": "amzn_msft",
        "file": "amzn_msft_no_loss.pt",
        "description": "Amazon + MSFT",
        "has_cond": True,
        "cond_data": msft_full,
    },
    {
        "name": "amzn_both",
        "file": "amzn_both_no_loss.pt",
        "description": "Amazon + SP500 + MSFT",
        "has_cond": True,
        "cond_data": both_full,
    },
]


def load_model(model_path):
    """Load trained model."""
    print(f"  Loading: {model_path}")
    model_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_config = model_data["model_config"]
    model = CVAE1DMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()
    return model, model_config


def generate_predictions(model, target_data, cond_data, num_days, context_len, num_samples):
    """
    Generate predictions using teacher forcing.

    For each day t:
    - Context: [t-C, ..., t-1]
    - Predict: day t
    - Use actual data for next iteration

    Returns:
        stochastic_preds: (num_days, num_samples)
        mle_preds: (num_days,)
    """
    stochastic_preds = []
    mle_preds = []

    for day_idx in tqdm(range(num_days), desc="  Generating"):
        # Extract context window
        start_idx = day_idx
        end_idx = day_idx + context_len

        # Prepare context batch
        ctx_target = torch.tensor(
            target_data[start_idx:end_idx, np.newaxis],  # (C, 1)
            dtype=torch.float64,
            device=DEVICE
        ).unsqueeze(0)  # (1, C, 1)

        ctx_dict = {"target": ctx_target}

        if cond_data is not None:
            ctx_cond = torch.tensor(
                cond_data[start_idx:end_idx],  # (C, K)
                dtype=torch.float64,
                device=DEVICE
            ).unsqueeze(0)  # (1, C, K)
            ctx_dict["cond_feats"] = ctx_cond

        # Generate stochastic predictions
        with torch.no_grad():
            stoch_pred = model.get_prediction_given_context(
                ctx_dict,
                num_samples=num_samples,
                use_mean=False
            )  # (1, num_samples, 1)

        # Generate MLE prediction
        with torch.no_grad():
            mle_pred = model.get_prediction_given_context(
                ctx_dict,
                num_samples=1,
                use_mean=True
            )  # (1, 1, 1)

        # Extract predictions
        stoch_pred = stoch_pred[0, :, 0].cpu().numpy()  # (num_samples,)
        mle_pred = mle_pred[0, 0, 0].cpu().item()

        stochastic_preds.append(stoch_pred)
        mle_preds.append(mle_pred)

    stochastic_preds = np.array(stochastic_preds)  # (num_days, num_samples)
    mle_preds = np.array(mle_preds)  # (num_days,)

    return stochastic_preds, mle_preds


# Generate predictions for all models
for model_dict in models:
    print("=" * 80)
    print(f"Processing: {model_dict['description']}")
    print("=" * 80)

    model_path = os.path.join(MODELS_DIR, model_dict["file"])

    # Load model
    model, model_config = load_model(model_path)

    # Generate predictions
    stochastic_preds, mle_preds = generate_predictions(
        model,
        amzn_full,
        model_dict["cond_data"],
        num_test_days,
        CONTEXT_LEN,
        NUM_SAMPLES
    )

    print(f"  Stochastic predictions shape: {stochastic_preds.shape}")
    print(f"  MLE predictions shape: {mle_preds.shape}")

    # Save predictions
    output_file = os.path.join(OUTPUT_DIR, f"{model_dict['name']}_predictions.npz")
    np.savez(
        output_file,
        stochastic=stochastic_preds,
        mle=mle_preds,
        ground_truth=amzn_test,
        dates=dates_test.values,
    )
    print(f"  Saved: {output_file}")
    print()

print("=" * 80)
print("PREDICTION GENERATION COMPLETE")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated predictions:")
for model_dict in models:
    print(f"  - {model_dict['name']}_predictions.npz")
print()
print("Next steps:")
print("  1. Run: python visualize_1d_predictions.py")
print("  2. Run: python evaluate_1d_models.py")
