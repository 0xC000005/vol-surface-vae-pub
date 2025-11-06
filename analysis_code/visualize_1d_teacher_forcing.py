"""
Teacher Forcing Visualization for 1D Stock Return Predictions.

2×4 grid:
- Row 1: Ground truth latent (encodes full sequence)
- Row 2: Context-only latent (realistic generation)
- Columns: 4 models (amzn_only, amzn_sp500_no_loss, amzn_msft_no_loss, amzn_both_no_loss)

For MSE models: Generates 1000 samples per day, computes empirical quantiles.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error
from vae.cvae_1d_with_mem_randomized import CVAE1DMemRand
import os

# Set default dtype to match training
torch.set_default_dtype(torch.float64)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "data/stock_returns.npz"
OUTPUT_DIR = "plots_1d"
CONTEXT_LEN = 5
NUM_SAMPLES = 1000  # Generate 1000 samples to compute quantiles

# Test set configuration
TRAIN_END = 4000
VALID_END = 5000
START_DAY = VALID_END
NUM_DAYS = 500  # Visualize first 500 days of test set

# Model paths (4 models)
MODEL_PATHS = {
    "amzn_only": "models_1d/amzn_only.pt",
    "amzn_sp500_no_loss": "models_1d/amzn_sp500_no_loss.pt",
    "amzn_msft_no_loss": "models_1d/amzn_msft_no_loss.pt",
    "amzn_both_no_loss": "models_1d/amzn_both_no_loss.pt",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_path):
    """Load trained 1D VAE model."""
    model_data = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_config = model_data["model_config"]
    model = CVAE1DMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()
    return model, model_config


def load_data(start_day, num_days, context_len):
    """Load data for visualization."""
    data = np.load(DATA_FILE)

    # Need context_len days before start_day
    amzn_returns = data["amzn_returns"][start_day - context_len : start_day + num_days]
    cond_sp500 = data["cond_sp500"][start_day - context_len : start_day + num_days]
    cond_msft = data["cond_msft"][start_day - context_len : start_day + num_days]
    cond_both = data["cond_both"][start_day - context_len : start_day + num_days]

    return {
        "target": amzn_returns,
        "cond_sp500": cond_sp500,
        "cond_msft": cond_msft,
        "cond_both": cond_both,
    }


def generate_with_encoded_latents_sampling(model, data_batch, context_len, num_samples):
    """
    Generate predictions using ground truth latent encoding.

    Encodes FULL sequence, uses z_mean for context, samples z~N(0,1) for future.
    Follows pattern from generate_surfaces.py for MSE models.
    """
    with torch.no_grad():
        # Encode full sequence (context + target) to get z_mean
        z_mean, z_log_var, _ = model.encoder(data_batch)

        # Extract context for context encoder
        ctx_input = {}
        for key in data_batch:
            if key == "target":
                ctx_input[key] = data_batch[key][:, :context_len, :]
            elif key == "cond_feats":
                ctx_input[key] = data_batch[key][:, :context_len, :]

        # Get context embedding
        ctx_embeddings = model.ctx_encoder(ctx_input)

        # Prepare decoder input
        B = 1
        T = context_len + 1
        ctx_dim = ctx_embeddings.shape[2]
        latent_dim = z_mean.shape[2]

        ctx_padded = torch.zeros((B, T, ctx_dim), device=DEVICE, dtype=z_mean.dtype)
        ctx_padded[:, :context_len, :] = ctx_embeddings

        # Generate multiple samples
        all_predictions = []
        for _ in range(num_samples):
            # Sample z ~ N(0, 1) for future timestep (standard VAE generation)
            z_future = torch.randn((B, 1, latent_dim), device=DEVICE, dtype=z_mean.dtype)

            # Construct full z: use z_mean for context, sampled z for future
            z_full = torch.zeros((B, T, latent_dim), device=DEVICE, dtype=z_mean.dtype)
            z_full[:, :context_len, :] = z_mean[:, :context_len, :]
            z_full[:, context_len:, :] = z_future

            # Decode
            decoder_input = torch.cat([ctx_padded, z_full], dim=-1)
            decoded_target, _ = model.decoder(decoder_input)

            # Extract future prediction
            prediction = decoded_target[:, context_len:, :]  # (1, 1, 1)
            all_predictions.append(prediction[0, 0, 0].item())

        return np.array(all_predictions)  # (num_samples,)


def generate_with_context_only_sampling(model, data_batch, context_len, num_samples):
    """
    Generate predictions using context-only latent encoding.

    Encodes ONLY context, uses z_mean for context, samples z~N(0,1) for future.
    Follows pattern from verify_reconstruction_2008_2010_context_only.py.
    """
    with torch.no_grad():
        # Extract context only
        ctx_input = {}
        for key in data_batch:
            if key == "target":
                ctx_input[key] = data_batch[key][:, :context_len, :]
            elif key == "cond_feats":
                ctx_input[key] = data_batch[key][:, :context_len, :]

        # Encode context only
        z_mean, z_log_var, _ = model.encoder(ctx_input)

        # Get context embedding
        ctx_embeddings = model.ctx_encoder(ctx_input)

        # Prepare for decoder
        B = 1
        T = context_len + 1
        ctx_dim = ctx_embeddings.shape[2]
        latent_dim = z_mean.shape[2]

        ctx_padded = torch.zeros((B, T, ctx_dim), device=DEVICE, dtype=z_mean.dtype)
        ctx_padded[:, :context_len, :] = ctx_embeddings

        # Generate multiple samples
        all_predictions = []
        for _ in range(num_samples):
            # Sample z ~ N(0, 1) for future timestep
            z_future = torch.randn((B, 1, latent_dim), device=DEVICE, dtype=z_mean.dtype)

            # Construct full z: use z_mean for context, sampled z for future
            z_full = torch.zeros((B, T, latent_dim), device=DEVICE, dtype=z_mean.dtype)
            z_full[:, :context_len, :] = z_mean
            z_full[:, context_len:, :] = z_future

            # Decode
            decoder_input = torch.cat([ctx_padded, z_full], dim=-1)
            decoded_target, _ = model.decoder(decoder_input)

            # Extract future prediction
            prediction = decoded_target[:, context_len:, :]  # (1, 1, 1)
            all_predictions.append(prediction[0, 0, 0].item())

        return np.array(all_predictions)  # (num_samples,)


def generate_predictions_for_all_days(model, data, context_len, num_days, cond_key, use_context_only, num_samples):
    """Generate predictions for all days."""
    predictions_all_samples = []  # (num_days, num_samples)
    ground_truth = []

    target = torch.tensor(data["target"], dtype=torch.float64)

    if cond_key is not None:
        cond_feats = torch.tensor(data[cond_key], dtype=torch.float64)
    else:
        cond_feats = None

    print(f"  Generating predictions...")
    for day_idx in range(num_days):
        if day_idx % 100 == 0:
            print(f"    Day {day_idx}/{num_days}")

        # Prepare batch for this day
        start_idx = day_idx
        end_idx = day_idx + context_len + 1

        batch = {
            "target": target[start_idx:end_idx].unsqueeze(0).unsqueeze(-1).to(DEVICE)  # (1, T, 1)
        }

        if cond_feats is not None:
            batch["cond_feats"] = cond_feats[start_idx:end_idx].unsqueeze(0).to(DEVICE)  # (1, T, K)

        # Generate samples
        if use_context_only:
            samples = generate_with_context_only_sampling(model, batch, context_len, num_samples)
        else:
            samples = generate_with_encoded_latents_sampling(model, batch, context_len, num_samples)

        # Ground truth
        gt = target[end_idx - 1].item()

        predictions_all_samples.append(samples)
        ground_truth.append(gt)

    predictions_all_samples = np.array(predictions_all_samples)  # (num_days, num_samples)
    ground_truth = np.array(ground_truth)

    # Compute quantiles from samples (empirical percentiles)
    p05 = np.percentile(predictions_all_samples, 5, axis=1)
    p50 = np.percentile(predictions_all_samples, 50, axis=1)
    p95 = np.percentile(predictions_all_samples, 95, axis=1)

    predictions_quantiles = np.stack([p05, p50, p95], axis=1)  # (num_days, 3)

    return predictions_quantiles, ground_truth


def calculate_metrics(predictions, ground_truth):
    """Calculate metrics from quantile predictions."""
    p05 = predictions[:, 0]
    p50 = predictions[:, 1]
    p95 = predictions[:, 2]

    rmse = np.sqrt(mean_squared_error(ground_truth, p50))
    r2 = r2_score(ground_truth, p50)

    violations = (ground_truth < p05) | (ground_truth > p95)
    ci_violations = np.mean(violations)
    below_p05 = np.mean(ground_truth < p05)
    above_p95 = np.mean(ground_truth > p95)
    ci_width = np.mean(p95 - p05)

    return {
        "rmse": rmse,
        "r2": r2,
        "ci_violations": ci_violations,
        "below_p05": below_p05,
        "above_p95": above_p95,
        "ci_width": ci_width,
    }


def create_plotly_visualization(all_predictions_gt, all_predictions_ctx, ground_truth, model_names):
    """
    Create 2×4 Plotly grid.
    Row 1: Ground truth latent (4 models)
    Row 2: Context-only latent (4 models)
    """
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{m}" for m in model_names] * 2,
        vertical_spacing=0.10,
        horizontal_spacing=0.05,
        row_titles=["Ground Truth Latent", "Context-Only Latent"],
    )

    days = np.arange(len(ground_truth))

    for row, (all_preds, method_name) in enumerate([(all_predictions_gt, "GT"), (all_predictions_ctx, "Ctx")]):
        for col, model_name in enumerate(model_names):
            predictions = all_preds[model_name]
            gt = ground_truth

            p05 = predictions[:, 0]
            p50 = predictions[:, 1]
            p95 = predictions[:, 2]

            # Find violations
            violations = (gt < p05) | (gt > p95)
            violations_idx = np.where(violations)[0]

            # CI band
            fig.add_trace(
                go.Scatter(
                    x=days, y=p95, mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip',
                ),
                row=row+1, col=col+1
            )

            fig.add_trace(
                go.Scatter(
                    x=days, y=p05, mode='lines', line=dict(width=0),
                    fillcolor='rgba(68, 114, 196, 0.3)', fill='tonexty',
                    name='90% CI' if (row==0 and col==0) else None,
                    showlegend=(row==0 and col==0),
                ),
                row=row+1, col=col+1
            )

            # Ground truth
            fig.add_trace(
                go.Scatter(
                    x=days, y=gt, mode='lines', line=dict(color='black', width=1.5),
                    name='Ground Truth' if (row==0 and col==0) else None,
                    showlegend=(row==0 and col==0),
                ),
                row=row+1, col=col+1
            )

            # p50
            fig.add_trace(
                go.Scatter(
                    x=days, y=p50, mode='lines', line=dict(color='rgb(68, 114, 196)', width=1.5),
                    name='p50' if (row==0 and col==0) else None,
                    showlegend=(row==0 and col==0),
                ),
                row=row+1, col=col+1
            )

            # Violations
            if len(violations_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=days[violations_idx], y=gt[violations_idx],
                        mode='markers', marker=dict(color='red', size=4),
                        name='Violations' if (row==0 and col==0) else None,
                        showlegend=(row==0 and col==0),
                    ),
                    row=row+1, col=col+1
                )

            # Metrics
            metrics = calculate_metrics(predictions, gt)
            metrics_text = f'R²={metrics["r2"]:.3f}<br>Viol={metrics["ci_violations"]*100:.1f}%'

            fig.add_trace(
                go.Scatter(
                    x=[days[int(len(days)*0.05)]], y=[gt.max()],
                    mode='text', text=[metrics_text],
                    textfont=dict(size=9), showlegend=False, hoverinfo='skip',
                ),
                row=row+1, col=col+1
            )

            # Axes
            if row == 1:
                fig.update_xaxes(title_text="Day", row=row+1, col=col+1)
            if col == 0:
                fig.update_yaxes(title_text="Return", row=row+1, col=col+1)

    fig.update_layout(
        title_text='1D Teacher Forcing: 2 Methods × 4 Models',
        height=800, width=2000,
        hovermode='closest',
        showlegend=True,
    )

    return fig


def create_matplotlib_visualization(all_predictions_gt, all_predictions_ctx, ground_truth, model_names):
    """Create 2×4 matplotlib grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    days = np.arange(len(ground_truth))

    for row, (all_preds, method_name) in enumerate([
        (all_predictions_gt, "Ground Truth Latent"),
        (all_predictions_ctx, "Context-Only Latent")
    ]):
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]
            predictions = all_preds[model_name]
            gt = ground_truth

            p05 = predictions[:, 0]
            p50 = predictions[:, 1]
            p95 = predictions[:, 2]

            # CI band
            ax.fill_between(days, p05, p95, alpha=0.3, color='skyblue', label='90% CI' if (row==0 and col==0) else None)

            # Lines
            ax.plot(days, gt, 'k-', linewidth=1, label='Truth' if (row==0 and col==0) else None)
            ax.plot(days, p50, 'b-', linewidth=1, label='p50' if (row==0 and col==0) else None)

            # Violations
            violations = (gt < p05) | (gt > p95)
            violations_idx = np.where(violations)[0]
            if len(violations_idx) > 0:
                ax.scatter(days[violations_idx], gt[violations_idx], color='red', s=10, zorder=5,
                          label='Violations' if (row==0 and col==0) else None)

            # Metrics
            metrics = calculate_metrics(predictions, gt)
            text = f'R²={metrics["r2"]:.3f}\nViol={metrics["ci_violations"]*100:.1f}%'
            ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Labels
            if row == 1:
                ax.set_xlabel('Day')
            if col == 0:
                ax.set_ylabel(f'{method_name}\nReturn')
            ax.set_title(model_name, fontsize=10)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig


def main():
    print("="*80)
    print("1D TEACHER FORCING VISUALIZATION (2×4 Grid)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Days: {START_DAY} to {START_DAY + NUM_DAYS}")
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Samples per day: {NUM_SAMPLES}")
    print()

    # Load data
    data = load_data(START_DAY, NUM_DAYS, CONTEXT_LEN)

    # Conditioning keys for each model
    cond_keys = {
        "amzn_only": None,
        "amzn_sp500_no_loss": "cond_sp500",
        "amzn_msft_no_loss": "cond_msft",
        "amzn_both_no_loss": "cond_both",
    }

    # Generate predictions for both methods
    all_predictions_gt = {}
    all_predictions_ctx = {}
    ground_truth = None

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        # Load model
        print(f"  Loading model...")
        model, model_config = load_model(model_path)
        cond_key = cond_keys[model_name]

        # Ground truth latent
        print(f"  Method 1: Ground truth latent")
        preds_gt, gt = generate_predictions_for_all_days(
            model, data, CONTEXT_LEN, NUM_DAYS, cond_key, use_context_only=False, num_samples=NUM_SAMPLES
        )
        all_predictions_gt[model_name] = preds_gt
        if ground_truth is None:
            ground_truth = gt

        # Context-only latent
        print(f"  Method 2: Context-only latent")
        preds_ctx, _ = generate_predictions_for_all_days(
            model, data, CONTEXT_LEN, NUM_DAYS, cond_key, use_context_only=True, num_samples=NUM_SAMPLES
        )
        all_predictions_ctx[model_name] = preds_ctx

        # Print metrics
        metrics_gt = calculate_metrics(preds_gt, ground_truth)
        metrics_ctx = calculate_metrics(preds_ctx, ground_truth)
        print(f"  GT:  R²={metrics_gt['r2']:.4f}, Viol={metrics_gt['ci_violations']*100:.2f}%")
        print(f"  Ctx: R²={metrics_ctx['r2']:.4f}, Viol={metrics_ctx['ci_violations']*100:.2f}%")

    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")

    model_names = list(MODEL_PATHS.keys())

    # Plotly
    fig_plotly = create_plotly_visualization(all_predictions_gt, all_predictions_ctx, ground_truth, model_names)
    output_html = f'{OUTPUT_DIR}/teacher_forcing_1d_2x4.html'
    fig_plotly.write_html(output_html)
    print(f"  Saved: {output_html}")

    # Matplotlib
    fig_matplotlib = create_matplotlib_visualization(all_predictions_gt, all_predictions_ctx, ground_truth, model_names)
    output_png = f'{OUTPUT_DIR}/teacher_forcing_1d_2x4.png'
    fig_matplotlib.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_png}")
    plt.close()

    print(f"\nDone!")


if __name__ == "__main__":
    main()
