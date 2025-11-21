"""
Marginal Distribution Comparison for Quantile Models with Encoded Latents.

Tests whether sampling from the TRUE posterior z ~ N(z_mean, z_log_var) for each day
produces correct marginal distributions when pooled across all days.

Key difference from original script:
- Uses encoded latents (ground truth z_mean, z_log_var) instead of prior N(0,1)
- Quantile decoder outputs (p05, p50, p95) - focuses on p50 for comparison
- Tests: "Does the decoder produce correct marginals when given true posterior?"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
from vae.cvae_with_mem_randomized import CVAEMemRand

# Configuration
MODEL_DIR = "test_spx/quantile_regression"
OUTPUT_DIR = "results/2024_1213/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
CONTEXT_LEN = 5
START_DAY = 2008  # 2008-01-02 (financial crisis period)
NUM_DAYS = 757    # Through 2010-12-31
NUM_SAMPLES_PER_DAY = 100  # Number of times to sample from encoded distribution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid points to visualize (same as teacher forcing)
GRID_POINTS = {
    "ATM 3-Month": (1, 2),   # moneyness=1.0, TTM=3 months
    "ATM 1-Year": (3, 2),    # moneyness=1.0, TTM=1 year
    "OTM Put 1-Year": (3, 0) # moneyness=0.7, TTM=1 year
}

# Model configurations
MODELS = {
    "no_ex": {
        "name": "No EX\n(Surface Only)",
        "color": "#1f77b4",
        "path": f"{MODEL_DIR}/no_ex.pt"
    },
    "ex_no_loss": {
        "name": "EX No Loss\n(+Features)",
        "color": "#ff7f0e",
        "path": f"{MODEL_DIR}/ex_no_loss.pt"
    },
    "ex_loss": {
        "name": "EX Loss\n(+Features+Loss)",
        "color": "#2ca02c",
        "path": f"{MODEL_DIR}/ex_loss.pt"
    }
}

print("="*80)
print("MARGINAL DISTRIBUTION COMPARISON: QUANTILE MODELS WITH ENCODED LATENTS")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Samples per day: {NUM_SAMPLES_PER_DAY}")
print(f"Context length: {CONTEXT_LEN}")
print()


def load_model(model_path):
    """Load trained quantile model."""
    print(f"  Loading: {model_path}")
    model_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_config = model_data["model_config"]
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()
    return model, model_config


def load_data():
    """Load 2008-2010 period dataset."""
    print("Loading ground truth data...")
    data = np.load("data/vol_surface_with_ret.npz")

    # Load surfaces - load NUM_DAYS starting from START_DAY (need CONTEXT_LEN extra for first prediction)
    surfaces = data["surface"][START_DAY:START_DAY + CONTEXT_LEN + NUM_DAYS]

    # Load extra features if available
    if "ret" in data and "skews" in data and "slopes" in data:
        ret = data["ret"][START_DAY:START_DAY + CONTEXT_LEN + NUM_DAYS]
        skews = data["skews"][START_DAY:START_DAY + CONTEXT_LEN + NUM_DAYS]
        slopes = data["slopes"][START_DAY:START_DAY + CONTEXT_LEN + NUM_DAYS]
        ex_data = np.stack([ret, skews, slopes], axis=1)
    else:
        ex_data = None

    # Load dates
    dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    dates = pd.to_datetime(dates_df["date"].values[START_DAY:START_DAY + CONTEXT_LEN + NUM_DAYS])

    print(f"  Surface shape: {surfaces.shape}")
    if ex_data is not None:
        print(f"  Extra features shape: {ex_data.shape}")
    print(f"  Time period: {dates[0]} to {dates[-1]}")
    print(f"  Total days (including context): {len(dates)}")
    print(f"  Prediction days: {NUM_DAYS}")

    return surfaces, ex_data, dates


def generate_from_encoded_distribution(model, data_batch, context_len, num_samples, model_has_ex_feats):
    """
    Generate multiple samples by sampling from encoded z ~ N(z_mean, z_log_var).

    Returns:
        predictions: (num_samples, 3, H, W) - quantile predictions (p05, p50, p95)
    """
    with torch.no_grad():
        # Encode full sequence to get z_mean and z_log_var
        full_input = {
            "surface": data_batch["surface"].unsqueeze(0).to(DEVICE)
        }
        if model_has_ex_feats and "ex_feats" in data_batch:
            full_input["ex_feats"] = data_batch["ex_feats"].unsqueeze(0).to(DEVICE)

        # Get latent encoding
        z_mean, z_log_var, _ = model.encoder(full_input)

        # Extract context for context encoder
        ctx_input = {
            "surface": data_batch["surface"][:context_len].unsqueeze(0).to(DEVICE)
        }
        if model_has_ex_feats and "ex_feats" in data_batch:
            ctx_input["ex_feats"] = data_batch["ex_feats"][:context_len].unsqueeze(0).to(DEVICE)

        # Get context embedding (this will be reused for all samples)
        ctx_embedding = model.ctx_encoder(ctx_input)

        # Prepare context embedding for decoder (pad with zeros for future timestep)
        B = 1
        T = context_len + 1
        ctx_embedding_dim = ctx_embedding.shape[2]
        ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim)).to(DEVICE)
        ctx_embedding_padded[:, :context_len, :] = ctx_embedding

        # Generate multiple samples
        predictions = []
        for _ in range(num_samples):
            # Sample from encoded distribution: z ~ N(z_mean, exp(0.5 * z_log_var))
            std = torch.exp(0.5 * z_log_var)
            eps = torch.randn_like(std)
            z_sampled = z_mean + eps * std

            # Prepare decoder input
            decoder_input = torch.cat([ctx_embedding_padded, z_sampled], dim=-1)

            # Decode
            if model_has_ex_feats:
                decoded_surface, _ = model.decoder(decoder_input)
            else:
                decoded_surface = model.decoder(decoder_input)

            # Extract future timestep prediction: (1, T, 3, H, W) -> (3, H, W)
            prediction = decoded_surface[0, context_len, :, :, :]
            predictions.append(prediction.cpu().numpy())

        predictions = np.array(predictions)  # (num_samples, 3, H, W)

        return predictions


def generate_all_predictions(model, surfaces, ex_data, num_samples, model_has_ex_feats):
    """
    Generate predictions for NUM_DAYS in the dataset.

    Returns:
        all_predictions: (NUM_DAYS, num_samples, 3, 5, 5)
    """
    all_predictions = []

    print(f"  Generating predictions for {NUM_DAYS} days...")

    surface_tensor = torch.tensor(surfaces, dtype=torch.float32)
    if ex_data is not None:
        ex_tensor = torch.tensor(ex_data, dtype=torch.float32)

    for day_idx in tqdm(range(NUM_DAYS), desc="  Progress"):
        # Prepare batch for this day
        start_idx = day_idx
        end_idx = day_idx + CONTEXT_LEN + 1

        batch = {
            "surface": surface_tensor[start_idx:end_idx]
        }

        if model_has_ex_feats and ex_data is not None:
            batch["ex_feats"] = ex_tensor[start_idx:end_idx]

        # Generate samples from encoded distribution
        predictions = generate_from_encoded_distribution(
            model, batch, CONTEXT_LEN, num_samples, model_has_ex_feats
        )

        all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)  # (NUM_DAYS, num_samples, 3, 5, 5)

    return all_predictions


print("Loading models and generating predictions...")
print()

# Load data
surfaces, ex_data, dates = load_data()
ground_truth_surfaces = surfaces[CONTEXT_LEN:]  # Ground truth for comparison (NUM_DAYS values)

# Align dates (predictions start at day CONTEXT_LEN)
dates_aligned = dates[CONTEXT_LEN:]

# Generate predictions for all models
model_predictions = {}

for model_key, model_info in MODELS.items():
    print(f"Processing: {model_key}")

    # Load model
    model, model_config = load_model(model_info["path"])
    model_has_ex_feats = model_config["ex_feats_dim"] > 0

    # Generate predictions
    predictions = generate_all_predictions(
        model, surfaces, ex_data, NUM_SAMPLES_PER_DAY, model_has_ex_feats
    )

    model_predictions[model_key] = predictions

    print(f"  Predictions shape: {predictions.shape}")
    print()

print("="*80)
print("Creating visualization...")
print("="*80)

# Create Figure: Implied Vol - 3 rows (grid points) x 3 columns (models)
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
total_samples = NUM_DAYS * NUM_SAMPLES_PER_DAY

fig.suptitle(
    "Marginal Distribution Comparison: Quantile Models with Encoded Latents (2008-2010)\n" +
    f"Historical: {NUM_DAYS} days | Model: {NUM_DAYS} days × {NUM_SAMPLES_PER_DAY} samples from encoded z ~ N(μ, σ²) = {total_samples/1e3:.1f}K samples\n" +
    f"Time Period: {dates_aligned[0].strftime('%Y-%m-%d')} to {dates_aligned[-1].strftime('%Y-%m-%d')}",
    fontsize=14, fontweight='bold'
)

grid_point_names = list(GRID_POINTS.keys())
model_keys = list(MODELS.keys())

for row_idx, (grid_name, (grid_row, grid_col)) in enumerate(GRID_POINTS.items()):
    # Extract ground truth for this grid point
    gt_values = ground_truth_surfaces[:, grid_row, grid_col]

    for col_idx, model_key in enumerate(model_keys):
        ax = axes[row_idx, col_idx]

        # Extract model predictions for this grid point
        # Focus on p50 (median) quantile: index 1 of the 3 quantiles
        # Shape: (num_days, num_samples, 3, 5, 5) -> extract quantile 1 (p50)
        model_p50 = model_predictions[model_key][:, :, 1, grid_row, grid_col]
        stoch_values = model_p50.flatten()  # Flatten to 1D array

        print(f"  {grid_name} - {model_key}: GT shape={gt_values.shape}, Model p50 shape={stoch_values.shape}")

        # Determine common bins for both histograms
        all_values = np.concatenate([gt_values, stoch_values])
        bins = np.linspace(all_values.min(), all_values.max(), 50)

        # Plot histograms
        ax.hist(gt_values, bins=bins, alpha=0.6, color='black',
                label='Historical', density=True, edgecolor='black', linewidth=0.5)

        model_color = MODELS[model_key]["color"]
        ax.hist(stoch_values, bins=bins, alpha=0.5, color=model_color,
                label='Model p50 (Median)', density=True, edgecolor=model_color, linewidth=0.5)

        # Title and labels
        if row_idx == 0:
            ax.set_title(MODELS[model_key]["name"], fontsize=12, fontweight='bold')

        if col_idx == 0:
            ax.set_ylabel(f'{grid_name}\nDensity', fontsize=10, fontweight='bold')

        if row_idx == 2:
            ax.set_xlabel('Implied Volatility', fontsize=10)

        # Legend (only for first subplot)
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='upper right', fontsize=9)

        ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = f"{OUTPUT_DIR}/marginal_distribution_quantile_encoded_2008_2010.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_file}")
plt.close()

print("\n" + "="*80)
print("MARGINAL DISTRIBUTION ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated file: {output_file}")
print(f"\nData summary:")
print(f"  - Time period: {dates_aligned[0].strftime('%Y-%m-%d')} to {dates_aligned[-1].strftime('%Y-%m-%d')}")
print(f"  - Total days: {NUM_DAYS}")
print(f"  - Historical samples: {NUM_DAYS}")
print(f"  - Model samples per day: {NUM_SAMPLES_PER_DAY}")
print(f"  - Total model samples: {total_samples:,}")
print(f"\nKey insights to look for:")
print(f"  - Do model p50 (median) distributions match historical distributions?")
print(f"  - Are the means and spreads similar?")
print(f"  - Does sampling from encoded z ~ N(μ, σ²) produce realistic marginals?")
print(f"  - Are there systematic biases (too narrow/wide, shifted)?")
print(f"\nInterpretation:")
print(f"  - Good match: Decoder correctly maps latent space to surface space")
print(f"  - Poor match despite good conditional performance (2008-2010 test):")
print(f"    → Suggests encoded distribution doesn't span full historical range")
print(f"    → May indicate prior mismatch or insufficient latent capacity")
