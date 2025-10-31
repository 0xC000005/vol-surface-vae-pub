"""
Verify MONOTONIC quantile model reconstruction using encoded latents (z_mean).

This script tests the NEW monotonic quantile models on 2008-2010 financial crisis
period using the EXACT latent mean from encoding the full sequence. This isolates
decoder performance by removing encoder uncertainty.

Goal:
1. Verify p50 predictions track ground truth
2. Verify ground truth falls within [p05, p95] CIs at ~10% violation rate
3. Verify ZERO quantile crossings (p05 <= p50 <= p95) - enforced by softplus
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from vae.cvae_with_mem_randomized import CVAEMemRand

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_DAY = 2008  # 2008-01-02 (financial crisis period)
NUM_DAYS = 757    # Through 2010-12-31 (full 2008-2010 period)
CONTEXT_LEN = 5   # Context length for generation

# Model paths - MONOTONIC QUANTILE MODELS
MODEL_PATHS = {
    "no_ex": "test_spx/quantile_regression_monotone/no_ex.pt",
    "ex_no_loss": "test_spx/quantile_regression_monotone/ex_no_loss.pt",
    "ex_loss": "test_spx/quantile_regression_monotone/ex_loss.pt",
}

# Data path
DATA_PATH = "data/vol_surface_with_ret.npz"


def load_model(model_path):
    """Load trained quantile model."""
    print(f"Loading model: {model_path}")
    model_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_config = model_data["model_config"]
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()
    return model


def load_data(start_day, num_days, context_len):
    """Load training data for verification."""
    print(f"Loading data from day {start_day} to {start_day + num_days}")
    data = np.load(DATA_PATH)

    # Need context_len days before start_day
    surface = data["surface"][start_day - context_len : start_day + num_days]

    result = {"surface": surface}

    # Construct ex_data from ret, skews, slopes (if available)
    if "ret" in data and "skews" in data and "slopes" in data:
        ret = data["ret"][start_day - context_len : start_day + num_days]
        skews = data["skews"][start_day - context_len : start_day + num_days]
        slopes = data["slopes"][start_day - context_len : start_day + num_days]
        # Stack into (N, 3) array
        ex_data = np.stack([ret, skews, slopes], axis=1)
        result["ex_data"] = ex_data

    return result


def generate_with_encoded_latents(model, data_batch, context_len, model_has_ex_feats):
    """
    Generate predictions using z_mean from encoding the full sequence.

    This simulates "perfect" encoder knowledge - we encode the actual target
    and use its z_mean instead of sampling z~N(0,1).

    Args:
        model: Trained CVAEMemRand model
        data_batch: Dict with "surface" and optionally "ex_feats"
        context_len: Number of context days (C)
        model_has_ex_feats: Whether model was trained with extra features

    Returns:
        predictions: (num_quantiles, H, W) - quantile surfaces [p05, p50, p95]
    """
    with torch.no_grad():
        # Encode full sequence (context + target) to get z_mean
        full_input = {
            "surface": data_batch["surface"].unsqueeze(0).to(DEVICE)
        }
        if model_has_ex_feats and "ex_feats" in data_batch:
            full_input["ex_feats"] = data_batch["ex_feats"].unsqueeze(0).to(DEVICE)

        # Get latent encoding
        z_mean, z_log_var, z_sampled = model.encoder(full_input)

        # Extract context for context encoder
        ctx_input = {
            "surface": data_batch["surface"][:context_len].unsqueeze(0).to(DEVICE)
        }
        if model_has_ex_feats and "ex_feats" in data_batch:
            ctx_input["ex_feats"] = data_batch["ex_feats"][:context_len].unsqueeze(0).to(DEVICE)

        # Get context embedding
        ctx_embedding = model.ctx_encoder(ctx_input)  # (1, C, ctx_dim)

        # Prepare decoder input
        B = 1
        T = context_len + 1
        ctx_embedding_dim = ctx_embedding.shape[2]
        ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim)).to(DEVICE)
        ctx_embedding_padded[:, :context_len, :] = ctx_embedding

        # Use z_mean (not sampled z) for all timesteps
        decoder_input = torch.cat([ctx_embedding_padded, z_mean], dim=-1)

        # Decode
        if model_has_ex_feats:
            decoded_surface, decoded_ex_feat = model.decoder(decoder_input)
        else:
            decoded_surface = model.decoder(decoder_input)

        # Extract future timestep prediction: (1, T, 3, H, W) -> (3, H, W)
        prediction = decoded_surface[0, context_len, :, :, :]

        return prediction.cpu().numpy()


def calculate_metrics(predictions, ground_truth):
    """
    Calculate verification metrics.

    Args:
        predictions: (num_days, 3, H, W) - [p05, p50, p95]
        ground_truth: (num_days, H, W)

    Returns:
        dict with metrics
    """
    num_days, H, W = ground_truth.shape

    # Extract quantiles
    p05 = predictions[:, 0, :, :]
    p50 = predictions[:, 1, :, :]
    p95 = predictions[:, 2, :, :]

    # Flatten for overall metrics
    p50_flat = p50.flatten()
    gt_flat = ground_truth.flatten()
    p05_flat = p05.flatten()
    p95_flat = p95.flatten()

    # Point forecast metrics (p50)
    rmse = np.sqrt(np.mean((p50_flat - gt_flat) ** 2))
    mae = np.mean(np.abs(p50_flat - gt_flat))
    r2 = r2_score(gt_flat, p50_flat)

    # CI calibration metrics
    below_p05 = np.mean(gt_flat < p05_flat)
    above_p95 = np.mean(gt_flat > p95_flat)
    ci_violations = below_p05 + above_p95

    # CI width
    ci_width = np.mean(p95_flat - p05_flat)

    # Check quantile ordering (should be p05 <= p50 <= p95)
    # For MONOTONIC models, this should be 0.00%
    quantile_crossings = np.mean((p05 > p50) | (p50 > p95))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ci_violations": ci_violations,
        "below_p05": below_p05,
        "above_p95": above_p95,
        "ci_width": ci_width,
        "quantile_crossings": quantile_crossings,
    }


def generate_predictions_for_all_days(model, data, context_len, num_days, model_has_ex_feats):
    """
    Generate predictions for all days in the sequence.

    Args:
        model: Trained model
        data: Dict with "surface" and optionally "ex_data"
        context_len: Number of context days
        num_days: Number of days to predict
        model_has_ex_feats: Whether model uses extra features

    Returns:
        predictions: (num_days, 3, H, W)
        ground_truth: (num_days, H, W)
    """
    predictions = []
    ground_truth = []

    surface = torch.tensor(data["surface"], dtype=torch.float32)

    if model_has_ex_feats and "ex_data" in data:
        ex_feats = torch.tensor(data["ex_data"], dtype=torch.float32)
    else:
        ex_feats = None

    for day_idx in range(num_days):
        # Prepare batch for this day
        start_idx = day_idx
        end_idx = day_idx + context_len + 1

        batch = {
            "surface": surface[start_idx:end_idx]  # (C+1, H, W)
        }

        if ex_feats is not None:
            batch["ex_feats"] = ex_feats[start_idx:end_idx]  # (C+1, n_feats)

        # Generate prediction using z_mean
        pred = generate_with_encoded_latents(model, batch, context_len, model_has_ex_feats)

        # Ground truth is the last day in the sequence
        gt = surface[end_idx - 1].numpy()

        predictions.append(pred)
        ground_truth.append(gt)

    predictions = np.array(predictions)  # (num_days, 3, H, W)
    ground_truth = np.array(ground_truth)  # (num_days, H, W)

    return predictions, ground_truth


def plot_verification(all_predictions, all_ground_truth, model_names):
    """
    Create 3x3 grid visualization: 3 models × 3 grid points.

    Args:
        all_predictions: Dict of {model_name: (num_days, 3, H, W)}
        all_ground_truth: (num_days, H, W) - same for all models
        model_names: List of model names
    """
    # Select 3 representative grid points
    grid_points = [
        (2, 2, "ATM 3M"),      # Center - At the money, 3 month
        (2, 4, "ATM 1Y"),      # Center - At the money, 1 year
        (0, 4, "OTM Put 1Y"),  # Low moneyness, 1 year
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for col, model_name in enumerate(model_names):
        predictions = all_predictions[model_name]

        for row, (grid_i, grid_j, grid_label) in enumerate(grid_points):
            ax = axes[row, col]

            # Extract time series for this grid point
            gt = all_ground_truth[:, grid_i, grid_j]
            p05 = predictions[:, 0, grid_i, grid_j]
            p50 = predictions[:, 1, grid_i, grid_j]
            p95 = predictions[:, 2, grid_i, grid_j]

            # Find CI violations
            violations = (gt < p05) | (gt > p95)

            # Find quantile crossings (should be 0 for monotonic models)
            crossings = (p05 > p50) | (p50 > p95)

            # Plot
            days = np.arange(len(gt))
            ax.fill_between(days, p05, p95, alpha=0.3, color='blue', label='90% CI')
            ax.plot(days, gt, 'k-', linewidth=1.5, label='Ground Truth')
            ax.plot(days, p50, 'b-', linewidth=1.5, label='p50 (Median)')

            # Highlight violations
            if np.any(violations):
                ax.scatter(days[violations], gt[violations],
                          color='red', s=30, zorder=5, label='CI Violations')

            # Calculate metrics for this grid point
            rmse = np.sqrt(np.mean((p50 - gt) ** 2))
            r2 = r2_score(gt, p50)
            violation_rate = np.mean(violations)
            crossing_rate = np.mean(crossings)

            # Add metrics text box
            metrics_text = (f'RMSE: {rmse:.4f}\nR²: {r2:.4f}\n'
                          f'Violations: {violation_rate*100:.1f}%\n'
                          f'Crossings: {crossing_rate*100:.2f}%')
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Labels and title
            if row == 0:
                ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{grid_label}\nImplied Vol', fontsize=10)
            if row == 2:
                ax.set_xlabel('Day Index', fontsize=10)

            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=8)

            ax.grid(True, alpha=0.3)

    plt.suptitle('MONOTONIC Quantile Models: Reconstruction Verification (z_mean) - 2008-2010 Financial Crisis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reconstruction_verification_2008_2010_monotone.png', dpi=150, bbox_inches='tight')
    print("Saved: reconstruction_verification_2008_2010_monotone.png")
    plt.close()


def print_summary(model_name, metrics):
    """Print formatted metrics summary."""
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Point Forecast (p50):")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"\nCI Calibration (90% CI):")
    print(f"  Total Violations: {metrics['ci_violations']*100:.2f}% (target: ~10%)")
    print(f"  Below p05:        {metrics['below_p05']*100:.2f}% (target: ~5%)")
    print(f"  Above p95:        {metrics['above_p95']*100:.2f}% (target: ~5%)")
    print(f"  Mean CI Width:    {metrics['ci_width']:.6f}")
    print(f"\nQuantile Ordering (MONOTONIC):")
    print(f"  Crossing Rate:    {metrics['quantile_crossings']*100:.4f}% (should be 0.00%)")
    if metrics['quantile_crossings'] == 0:
        print(f"  ✓ PERFECT MONOTONICITY - Zero crossings detected!")
    elif metrics['quantile_crossings'] < 0.01:
        print(f"  ✓ EXCELLENT - Near-zero crossings")
    else:
        print(f"  ✗ WARNING - Unexpected crossings in monotonic model!")


def main():
    print("="*60)
    print("MONOTONIC QUANTILE MODEL - VERIFICATION")
    print("Using Encoded Latents (z_mean)")
    print("Period: 2008-2010 Financial Crisis")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Verification window: Days {START_DAY}-{START_DAY + NUM_DAYS}")
    print(f"Context length: {CONTEXT_LEN}")
    print()

    # Load data
    data = load_data(START_DAY, NUM_DAYS, CONTEXT_LEN)
    print(f"Surface shape: {data['surface'].shape}")
    if "ex_data" in data:
        print(f"Extra features shape: {data['ex_data'].shape}")

    # Store results
    all_predictions = {}
    all_metrics = {}

    # Process each model
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        # Load model
        model = load_model(model_path)
        model_has_ex_feats = model.config["ex_feats_dim"] > 0

        # Generate predictions
        print(f"Generating predictions using z_mean...")
        predictions, ground_truth = generate_predictions_for_all_days(
            model, data, CONTEXT_LEN, NUM_DAYS, model_has_ex_feats
        )

        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth)

        # Store results
        all_predictions[model_name] = predictions
        all_metrics[model_name] = metrics

        # Print summary
        print_summary(model_name, metrics)

    # Create visualization
    print(f"\n{'='*60}")
    print("Creating visualization...")
    print(f"{'='*60}")
    plot_verification(all_predictions, ground_truth, list(MODEL_PATHS.keys()))

    # Final summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'RMSE':<10} {'R²':<10} {'CI Viol %':<12} {'Crossings %':<12}")
    print("-"*60)
    for model_name in MODEL_PATHS.keys():
        m = all_metrics[model_name]
        print(f"{model_name:<15} {m['rmse']:<10.6f} {m['r2']:<10.4f} "
              f"{m['ci_violations']*100:<12.2f} {m['quantile_crossings']*100:<12.4f}")

    print("\nVerification complete!")
    print("\nExpected results for MONOTONIC models:")
    print("  - Crossing rate: 0.00% (enforced by softplus transformation)")
    print("  - CI violations: Similar to baseline (~5-11% with z_mean)")


if __name__ == "__main__":
    main()
