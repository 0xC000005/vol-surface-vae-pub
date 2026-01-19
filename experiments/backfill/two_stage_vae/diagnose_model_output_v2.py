"""
Diagnose model output correlations in detail.

Key finding: output[-1] is NEGATIVELY correlated with input[-1]
but POSITIVELY correlated with next_step. Why?
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
    print("DIAGNOSIS V2: Understanding Negative Reconstruction Correlation")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, config = load_vae(device)

    context_len = 30
    train_end = int(len(surfaces) * 0.7)

    # Test over multiple samples
    print("\n" + "=" * 70)
    print("Testing correlations over 200 samples")
    print("=" * 70)

    n_test = 200

    # Store values for each position
    output_vals = {t: [] for t in range(context_len)}
    input_vals = {t: [] for t in range(context_len)}
    next_vals = []

    for i in range(n_test):
        idx = train_end + i
        context = log_returns[idx:idx + context_len]
        next_step = log_returns[idx + context_len]

        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            ctx_emb = vae.ctx_encoder({"surface": ctx_tensor})
            z_mean, _, _ = vae.main_encoder({"surface": ctx_tensor})
            mean, _, _, _ = vae.decoder(ctx_emb, z_mean, sample=False)

        output = mean[0].cpu().numpy()

        for t in range(context_len):
            output_vals[t].append(output[t, 2, 2])  # ATM only
            input_vals[t].append(context[t, 2, 2])

        next_vals.append(next_step[2, 2])

    # Convert to arrays
    for t in range(context_len):
        output_vals[t] = np.array(output_vals[t])
        input_vals[t] = np.array(input_vals[t])
    next_vals = np.array(next_vals)

    # Compute correlations
    print("\nCorrelations (ATM grid point only):")
    print(f"{'Position':<10} | {'out[t] vs in[t]':>15} | {'out[t] vs in[t-1]':>17} | {'out[t] vs next':>14}")
    print("-" * 65)

    for t in [0, 10, 20, 28, 29]:
        corr_same = np.corrcoef(output_vals[t], input_vals[t])[0, 1]
        if t > 0:
            corr_prev = np.corrcoef(output_vals[t], input_vals[t-1])[0, 1]
        else:
            corr_prev = float('nan')
        corr_next = np.corrcoef(output_vals[t], next_vals)[0, 1]
        print(f"t={t:<7} | {corr_same:>15.4f} | {corr_prev:>17.4f} | {corr_next:>14.4f}")

    # Key question: What is output[-1] actually predicting?
    print("\n" + "=" * 70)
    print("What does output[-1] correlate with?")
    print("=" * 70)

    # Test correlation with various targets
    targets = {
        "input[-1]": input_vals[29],
        "input[-2]": input_vals[28],
        "next_step": next_vals,
    }

    for name, target in targets.items():
        corr = np.corrcoef(output_vals[29], target)[0, 1]
        print(f"output[-1] vs {name:<15}: {corr:.4f}")

    # Check if it's predicting negative of input (mean-reversion?)
    print("\n" + "=" * 70)
    print("Checking Mean-Reversion Hypothesis")
    print("=" * 70)

    # If model learned mean-reversion: output = -input (predicting reversal)
    corr_neg_input = np.corrcoef(output_vals[29], -input_vals[29])[0, 1]
    print(f"output[-1] vs -input[-1]: {corr_neg_input:.4f}")

    # Actually, check if input[-1] ≈ -next_step (mean reversion in data)
    corr_reversal = np.corrcoef(input_vals[29], next_vals)[0, 1]
    print(f"input[-1] vs next_step (data autocorr): {corr_reversal:.4f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics (ATM)")
    print("=" * 70)
    print(f"output[-1] mean: {output_vals[29].mean():.6f}, std: {output_vals[29].std():.6f}")
    print(f"input[-1]  mean: {input_vals[29].mean():.6f}, std: {input_vals[29].std():.6f}")
    print(f"next_step  mean: {next_vals.mean():.6f}, std: {next_vals.std():.6f}")

    # Check actual values
    print("\n" + "=" * 70)
    print("Sample Values (first 5 samples)")
    print("=" * 70)
    print(f"{'Sample':<8} | {'output[-1]':>12} | {'input[-1]':>12} | {'next_step':>12}")
    print("-" * 55)
    for i in range(5):
        print(f"{i:<8} | {output_vals[29][i]:>12.6f} | {input_vals[29][i]:>12.6f} | {next_vals[i]:>12.6f}")

    # INSIGHT: Check if the model is outputting a SCALED version
    print("\n" + "=" * 70)
    print("Scale Analysis")
    print("=" * 70)

    # Regression: output = a * input + b
    from scipy import stats
    slope, intercept, r_val, p_val, std_err = stats.linregress(input_vals[29], output_vals[29])
    print(f"Regression output[-1] = a * input[-1] + b:")
    print(f"  slope (a): {slope:.4f}")
    print(f"  intercept (b): {intercept:.6f}")
    print(f"  R²: {r_val**2:.4f}")

    # If slope is negative, model learned to predict reversal
    if slope < 0:
        print(f"\n>>> Model learned MEAN REVERSION: output ≈ {slope:.2f} * input")
        print(f"    This explains negative correlation!")

    # Check context encoder causality
    print("\n" + "=" * 70)
    print("Context Encoder Causality Check")
    print("=" * 70)
    print("""
The context encoder is CAUSAL: ctx_emb[t] = f(x_{0:t-1})
This means ctx_emb[29] only sees x_0 to x_28 (not x_29).

But the main encoder sees x_29: z[29] = g(x_{0:29})

So at position 29:
- ctx_emb[29] doesn't know about input[29]
- z[29] knows about input[29]

If output[29] is negatively correlated with input[29],
the model might be using z[29] to predict the REVERSAL of x[29],
i.e., predicting that tomorrow's return will be opposite of today's.
""")


if __name__ == "__main__":
    main()
