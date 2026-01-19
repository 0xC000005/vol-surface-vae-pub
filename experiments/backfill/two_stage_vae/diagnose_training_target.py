"""
Diagnose the training target issue.

Key question: Is the model trained to reconstruct x[t] or predict x[t+1]?

Expected finding: The model is trained to RECONSTRUCT x[t] at position t,
NOT predict x[t+1]. This explains why chaining fails - the model was
never trained for prediction.

Solution: For proper autoregressive chaining, we need to either:
1. Train a prediction model (target is shifted by 1)
2. Use the model differently (feed context, take horizon output)
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
    print("DIAGNOSIS: Training Target (Reconstruction vs Prediction)")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, config = load_vae(device)

    print("\nModel Configuration:")
    print(f"  context_len: {config.get('context_len')}")
    print(f"  horizon: {config.get('horizon')}")
    print(f"  loss_mode: {config.get('loss_mode')}")

    # Test with full 60-step sequence (like training)
    print("\n" + "=" * 70)
    print("Test 1: Full 60-step sequence (like training)")
    print("=" * 70)

    train_end = int(len(log_returns) * 0.7)
    start_idx = train_end + 100

    # Get 60-step sequence
    seq_len = 60
    sequence = log_returns[start_idx:start_idx + seq_len]  # (60, 5, 5)

    seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    batch = {"surface": seq_tensor}

    with torch.no_grad():
        ctx_emb = vae.ctx_encoder(batch)
        z_mean, _, _ = vae.main_encoder(batch)
        mean, _, _, _ = vae.decoder(ctx_emb, z_mean, sample=False)

    output = mean[0].cpu().numpy()  # (60, 5, 5)
    input_seq = sequence

    print(f"\nInput shape: {input_seq.shape}")
    print(f"Output shape: {output.shape}")

    # For horizon positions (30-59), check if output matches input at same position or shifted
    print("\nHorizon positions (30-59) correlation analysis (ATM):")
    print(f"{'Position':<10} | {'out[t] vs in[t]':>15} | {'out[t] vs in[t+1]':>17} | {'out[t] vs in[t-1]':>17}")
    print("-" * 70)

    for t in [30, 35, 40, 45, 50, 55, 58]:
        corr_same = np.corrcoef(output[t].flatten(), input_seq[t].flatten())[0, 1]
        if t + 1 < seq_len:
            corr_next = np.corrcoef(output[t].flatten(), input_seq[t+1].flatten())[0, 1]
        else:
            corr_next = float('nan')
        corr_prev = np.corrcoef(output[t].flatten(), input_seq[t-1].flatten())[0, 1]
        print(f"t={t:<7} | {corr_same:>15.4f} | {corr_next:>17.4f} | {corr_prev:>17.4f}")

    # MSE analysis: which target gives lower error?
    print("\n" + "=" * 70)
    print("MSE Analysis: Which target does output match?")
    print("=" * 70)

    # Horizon positions only
    horizon_output = output[30:]  # (30, 5, 5)
    target_same = input_seq[30:]  # input at same position (reconstruction)
    target_shifted = input_seq[31:]  # input at next position (prediction)

    mse_same = np.mean((horizon_output[:-1] - target_same[:-1])**2)
    mse_shifted = np.mean((horizon_output[:-1] - target_shifted)**2)

    print(f"\nMSE(output[30:59], input[30:59]) [reconstruction]: {mse_same:.6f}")
    print(f"MSE(output[30:59], input[31:60]) [prediction]:     {mse_shifted:.6f}")

    if mse_same < mse_shifted:
        print("\n>>> Model is trained for RECONSTRUCTION (same position)")
    else:
        print("\n>>> Model is trained for PREDICTION (next position)")

    # Test 2: What happens with 30-step context only?
    print("\n" + "=" * 70)
    print("Test 2: 30-step context only (inference scenario)")
    print("=" * 70)

    context = log_returns[start_idx:start_idx + 30]  # (30, 5, 5)
    next_step = log_returns[start_idx + 30]  # The actual next step

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    batch = {"surface": ctx_tensor}

    with torch.no_grad():
        ctx_emb = vae.ctx_encoder(batch)
        z_mean, _, _ = vae.main_encoder(batch)
        mean, _, _, _ = vae.decoder(ctx_emb, z_mean, sample=False)

    output = mean[0].cpu().numpy()  # (30, 5, 5)

    print(f"\nContext shape: {context.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Next step shape: {next_step.shape}")

    print("\nAt position 29 (last context position):")
    print(f"  output[29] ATM: {output[29, 2, 2]:.6f}")
    print(f"  input[29] ATM:  {context[29, 2, 2]:.6f}")
    print(f"  next_step ATM:  {next_step[2, 2]:.6f}")

    corr_same = np.corrcoef(output[29].flatten(), context[29].flatten())[0, 1]
    corr_next = np.corrcoef(output[29].flatten(), next_step.flatten())[0, 1]
    print(f"\nCorrelation analysis:")
    print(f"  output[29] vs input[29] (reconstruction): {corr_same:.4f}")
    print(f"  output[29] vs next_step (prediction):     {corr_next:.4f}")

    # KEY INSIGHT
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print("""
The model is trained to RECONSTRUCT x[t] at position t, NOT predict x[t+1].

Training setup:
- Input: 60-step sequence [x_0, ..., x_59]
- Output: 60-step reconstruction [y_0, ..., y_59]
- Loss: MSE(y[30:60], x[30:60]) - horizon positions only

At position 30:
- ctx_emb[30] = f(x_0:29) - causal, doesn't see x_30
- z[30] = g(x_0:30) - NON-CAUSAL, SEES x_30!
- Output y[30] is trained to match x[30] (reconstruction)

The main encoder z[t] provides information about x[t], allowing
reconstruction even though ctx_emb[t] is causal.

For autoregressive chaining, we're treating output[-1] as a PREDICTION
of the next step, but the model was trained to RECONSTRUCT the current step.

The weak correlation (~0.3) with the next step is just due to autocorrelation
in the data (consecutive log-returns are ~-0.1 correlated).

SOLUTION OPTIONS:
1. Train a proper prediction model (shift targets by 1 in loss)
2. Train with causal z as well (not just causal ctx_emb)
3. Use the full 60-step forward pass and take position 30 as prediction of x_31
   (but this still uses non-causal z, so it's still reconstruction)

The fundamental issue: The model architecture allows z[t] to see x[t],
making reconstruction trivial but prediction impossible.
""")


if __name__ == "__main__":
    main()
