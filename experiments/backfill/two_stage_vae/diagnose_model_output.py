"""
Diagnose what model output positions actually represent.

Key question: Is output[-1] a reconstruction of input[-1] or a prediction of the next step?
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP


def load_vae(device: str = "cuda"):
    """Load the Student-t VAE model."""
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    if Path(vae_path).exists():
        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        config = vae_ckpt["model_config"]
        config["device"] = device
        vae = CVAETwoStageStudentTMLP(config)
        vae.load_state_dict(vae_ckpt["model_state_dict"])
        vae = vae.to(device)
        vae.eval()
        return vae, config
    raise FileNotFoundError("No VAE model found")


def main():
    print("=" * 70)
    print("DIAGNOSIS: What Do Model Output Positions Represent?")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, config = load_vae(device)

    context_len = 30

    # Pick a point
    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100

    # Get context
    context = log_returns[start_idx:start_idx + context_len]  # (30, 5, 5)
    next_step = log_returns[start_idx + context_len]  # The NEXT step after context

    print(f"\nContext shape: {context.shape}")
    print(f"Next step (target for prediction): {next_step.shape}")
    print(f"Next step ATM value: {next_step[2,2]:.6f}")

    # Method 1: Use forward() - what does output[-1] represent?
    print("\n" + "=" * 70)
    print("Method 1: forward() - Reconstruction Mode")
    print("=" * 70)

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    batch = {"surface": ctx_tensor}

    with torch.no_grad():
        ctx_emb = vae.ctx_encoder(batch)
        z_mean, z_logvar, z = vae.main_encoder(batch)
        mean, _, _, _ = vae.decoder(ctx_emb, z_mean, sample=False)

    output = mean[0].cpu().numpy()  # (30, 5, 5)

    print(f"\nInput shape: {context.shape}")
    print(f"Output shape: {output.shape}")

    # Compare output positions to input positions
    print("\nCorrelation of output[t] with input[t] (reconstruction):")
    for t in [0, 14, 28, 29]:
        corr = np.corrcoef(output[t].flatten(), context[t].flatten())[0, 1]
        print(f"  t={t}: {corr:.4f}")

    print("\nCorrelation of output[-1] with different targets:")
    print(f"  output[-1] vs input[-1] (reconstruction): {np.corrcoef(output[-1].flatten(), context[-1].flatten())[0, 1]:.4f}")
    print(f"  output[-1] vs next_step (prediction):     {np.corrcoef(output[-1].flatten(), next_step.flatten())[0, 1]:.4f}")

    # Method 2: Use get_surface_given_conditions() - actual prediction
    print("\n" + "=" * 70)
    print("Method 2: get_surface_given_conditions() - Prediction Mode")
    print("=" * 70)

    with torch.no_grad():
        result = vae.get_surface_given_conditions(
            {"surface": ctx_tensor},
            horizon=1,
            sample_from_decoder=False
        )
        pred_mean = result[0][0, 0].cpu().numpy()  # First horizon step

    print(f"\nPrediction (horizon=1) ATM value: {pred_mean[2,2]:.6f}")
    print(f"Actual next step ATM value:       {next_step[2,2]:.6f}")
    print(f"Correlation with next step:       {np.corrcoef(pred_mean.flatten(), next_step.flatten())[0, 1]:.4f}")

    # Critical insight
    print("\n" + "=" * 70)
    print("CRITICAL INSIGHT")
    print("=" * 70)
    print("""
The chaining script uses output[-1] from forward(), which is a RECONSTRUCTION
of input[-1], NOT a prediction of the next step!

This is why:
1. output[-1] has high correlation with input[-1] (reconstruction works)
2. output[-1] has LOW/ZERO correlation with next_step (not designed for prediction)
3. Predictions are near zero (reconstructing log-returns which are ~0)

The CORRECT approach for chaining is to use get_surface_given_conditions()
with horizon=1 to get actual predictions.
""")

    # Compare the two approaches
    print("=" * 70)
    print("COMPARISON: Reconstruction vs Prediction")
    print("=" * 70)

    print(f"\nforward() output[-1] (WRONG for chaining):")
    print(f"  Value: {output[-1, 2, 2]:.6f}")
    print(f"  This is: reconstruction of input[-1] = {context[-1, 2, 2]:.6f}")

    print(f"\nget_surface_given_conditions() (CORRECT for chaining):")
    print(f"  Value: {pred_mean[2, 2]:.6f}")
    print(f"  Target: next step = {next_step[2, 2]:.6f}")

    # Test over multiple samples to see if prediction has any signal
    print("\n" + "=" * 70)
    print("Multi-Sample Test: Does Prediction Have Any Signal?")
    print("=" * 70)

    n_test = 100
    pred_values = []
    actual_values = []

    for i in range(n_test):
        idx = train_end + i
        ctx = log_returns[idx:idx + context_len]
        target = log_returns[idx + context_len]

        ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            result = vae.get_surface_given_conditions(
                {"surface": ctx_t},
                horizon=1,
                sample_from_decoder=False
            )
            pred = result[0][0, 0].cpu().numpy()

        pred_values.append(pred[2, 2])
        actual_values.append(target[2, 2])

    pred_values = np.array(pred_values)
    actual_values = np.array(actual_values)

    corr = np.corrcoef(pred_values, actual_values)[0, 1]
    print(f"\nPrediction correlation over {n_test} samples: {corr:.4f}")
    print(f"Prediction std: {pred_values.std():.6f}")
    print(f"Actual std:     {actual_values.std():.6f}")

    # Check if model is using predictors or just prior
    print("\n" + "=" * 70)
    print("Model Configuration Check")
    print("=" * 70)
    print(f"horizon in config: {config.get('horizon', 'not set')}")
    print(f"max_horizon in config: {config.get('max_horizon', 'not set')}")

    # The model was trained as autoencoder - does it have predictors?
    has_predictors = hasattr(vae, 'latent_predictor') and vae.latent_predictor is not None
    print(f"Has latent predictor: {has_predictors}")


if __name__ == "__main__":
    main()
