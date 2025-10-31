"""
Interactive Plotly visualization of MONOTONIC quantile model reconstruction using encoded latents (z_mean).

Generates an interactive HTML plot for the 2008-2010 financial crisis period.
Tests the new monotonic quantile models with softplus reparameterization.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        ctx_embedding = model.ctx_encoder(ctx_input)

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
    """Calculate verification metrics including quantile crossing detection."""
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

    # Quantile crossing detection (should be 0% for monotonic models)
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
    """Generate predictions for all days in the sequence."""
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
            "surface": surface[start_idx:end_idx]
        }

        if ex_feats is not None:
            batch["ex_feats"] = ex_feats[start_idx:end_idx]

        # Generate prediction using z_mean
        pred = generate_with_encoded_latents(model, batch, context_len, model_has_ex_feats)

        # Ground truth is the last day in the sequence
        gt = surface[end_idx - 1].numpy()

        predictions.append(pred)
        ground_truth.append(gt)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    return predictions, ground_truth


def create_interactive_plot(all_predictions, all_ground_truth, model_names):
    """
    Create interactive Plotly 3x3 grid visualization.
    """
    # Select 3 representative grid points
    grid_points = [
        (2, 2, "ATM 3M"),
        (2, 4, "ATM 1Y"),
        (0, 4, "OTM Put 1Y"),
    ]

    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"{model}" for model in model_names] * 3,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
        row_titles=[gp[2] for gp in grid_points],
    )

    days = np.arange(len(all_ground_truth))

    for col, model_name in enumerate(model_names):
        predictions = all_predictions[model_name]

        for row, (grid_i, grid_j, grid_label) in enumerate(grid_points):
            # Extract time series for this grid point
            gt = all_ground_truth[:, grid_i, grid_j]
            p05 = predictions[:, 0, grid_i, grid_j]
            p50 = predictions[:, 1, grid_i, grid_j]
            p95 = predictions[:, 2, grid_i, grid_j]

            # Find CI violations
            violations = (gt < p05) | (gt > p95)
            violations_idx = np.where(violations)[0]

            # Find quantile crossings (should be 0 for monotonic models)
            crossings = (p05 > p50) | (p50 > p95)
            crossing_rate = np.mean(crossings)

            # Calculate metrics for this grid point
            rmse = np.sqrt(np.mean((p50 - gt) ** 2))
            r2 = r2_score(gt, p50)
            violation_rate = np.mean(violations)

            # Add CI band (shaded area)
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p95,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=row+1, col=col+1
            )

            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p05,
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(68, 114, 196, 0.3)',
                    fill='tonexty',
                    name='90% CI' if row == 0 and col == 0 else None,
                    showlegend=(row == 0 and col == 0),
                    hovertemplate='Day: %{x}<br>p05: %{y:.4f}<extra></extra>',
                ),
                row=row+1, col=col+1
            )

            # Add ground truth line
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=gt,
                    mode='lines',
                    line=dict(color='black', width=1.5),
                    name='Ground Truth' if row == 0 and col == 0 else None,
                    showlegend=(row == 0 and col == 0),
                    hovertemplate='Day: %{x}<br>Truth: %{y:.4f}<extra></extra>',
                ),
                row=row+1, col=col+1
            )

            # Add p50 line
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p50,
                    mode='lines',
                    line=dict(color='rgb(68, 114, 196)', width=1.5),
                    name='p50 (Median)' if row == 0 and col == 0 else None,
                    showlegend=(row == 0 and col == 0),
                    hovertemplate='Day: %{x}<br>p50: %{y:.4f}<extra></extra>',
                ),
                row=row+1, col=col+1
            )

            # Add violation markers
            if len(violations_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=days[violations_idx],
                        y=gt[violations_idx],
                        mode='markers',
                        marker=dict(color='red', size=6, symbol='circle'),
                        name='CI Violations' if row == 0 and col == 0 else None,
                        showlegend=(row == 0 and col == 0),
                        hovertemplate='Day: %{x}<br>Violation: %{y:.4f}<extra></extra>',
                    ),
                    row=row+1, col=col+1
                )

            # Add metrics as text trace
            metrics_text = (f'RMSE: {rmse:.4f}<br>R²: {r2:.4f}<br>'
                          f'Violations: {violation_rate*100:.1f}%<br>'
                          f'Crossings: {crossing_rate*100:.2f}%')

            # Calculate position in data coordinates
            x_pos = days[int(len(days) * 0.05)]
            y_range = gt.max() - gt.min()
            y_pos = gt.max() - y_range * 0.05

            fig.add_trace(
                go.Scatter(
                    x=[x_pos],
                    y=[y_pos],
                    mode='text',
                    text=[metrics_text],
                    textposition='top right',
                    textfont=dict(size=9, color='black'),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=row+1, col=col+1
            )

            # Update axes labels
            if col == 0:
                fig.update_yaxes(title_text="Implied Vol", row=row+1, col=col+1)
            if row == 2:
                fig.update_xaxes(title_text="Day Index", row=row+1, col=col+1)

    # Update layout
    fig.update_layout(
        title_text='MONOTONIC Quantile Models - Reconstruction Verification (z_mean) - 2008-2010 Crisis<br><sub>Interactive - Zoom, Pan, Hover | Crossings should be 0.00%</sub>',
        height=1200,
        width=1800,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
        ),
    )

    # Update all axes to have gridlines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')

    return fig


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
        print(f"  ✓ PERFECT MONOTONICITY!")
    elif metrics['quantile_crossings'] < 0.01:
        print(f"  ✓ EXCELLENT")
    else:
        print(f"  ✗ WARNING - Unexpected crossings!")


def main():
    print("="*60)
    print("MONOTONIC QUANTILE MODELS - INTERACTIVE VERIFICATION")
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

    # Create interactive plot
    print(f"\n{'='*60}")
    print("Creating interactive Plotly visualization...")
    print(f"{'='*60}")
    fig = create_interactive_plot(all_predictions, ground_truth, list(MODEL_PATHS.keys()))

    # Save as HTML
    output_file = 'reconstruction_verification_2008_2010_interactive_monotone.html'
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    print(f"\nTo view:")
    print(f"  1. On server: firefox {output_file}")
    print(f"  2. Or copy to local machine and open in browser")

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
