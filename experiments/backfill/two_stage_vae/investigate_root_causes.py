"""
Investigate Root Causes of Marginal Distribution Issues

1. Student-t nu parameter - Is it producing fat tails?
2. Predictor bias - Is z prediction systematically biased?
3. Variance decomposition - Where does excess variance come from?
4. Autocorrelation - Is temporal dependence captured?

Usage:
    python experiments/backfill/two_stage_vae/investigate_root_causes.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP, GT_KURTOSIS_MLP, GT_NU_MLP
from vae.predictors import LatentPredictor


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_models(device: str = "cuda"):
    """Load the Student-t VAE and trained predictor."""
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    pred_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    pred_ckpt = torch.load(pred_path, map_location=device, weights_only=False)

    predictor = LatentPredictor(config)
    predictor.load_state_dict(pred_ckpt["predictor_state_dict"])
    predictor = predictor.to(device)
    predictor.eval()

    return vae, predictor, config


def create_dataloader(log_returns, context_len, batch_size):
    """Create DataLoader from log-returns."""
    N = len(log_returns)
    sequences = []
    for i in range(N - context_len - 1):
        seq = log_returns[i:i + context_len + 1]
        sequences.append(seq)
    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# =============================================================================
# Investigation 1: Student-t Nu Parameter
# =============================================================================

def investigate_nu_parameter(vae):
    """Check Student-t degrees of freedom parameter."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 1: STUDENT-T NU PARAMETER")
    print("=" * 70)

    # Get nu values from decoder
    nu = vae.decoder.nu.cpu().numpy()
    nu_grid = nu.reshape(5, 5)

    print("\nGT Kurtosis (excess, Fisher definition):")
    print(GT_KURTOSIS_MLP.round(2))

    print("\nComputed Nu values (fixed, from GT kurtosis):")
    print(GT_NU_MLP.reshape(5, 5).round(2))

    print("\nActual Nu in model:")
    print(nu_grid.round(2))

    # Theoretical kurtosis from nu
    theoretical_kurt = np.where(nu > 4, 6.0 / (nu - 4), np.inf)
    theoretical_kurt_grid = theoretical_kurt.reshape(5, 5)

    print("\nTheoretical excess kurtosis from nu (should match GT):")
    print(theoretical_kurt_grid.round(2))

    # Key statistics
    print("\n--- Summary ---")
    print(f"Nu range: [{nu.min():.2f}, {nu.max():.2f}]")
    print(f"Nu mean: {nu.mean():.2f}")

    # For nu=4.1, kurtosis = 6/(4.1-4) = 60
    # For nu=5, kurtosis = 6/(5-4) = 6
    # For nu=10, kurtosis = 6/(10-4) = 1

    if nu.min() > 30:
        print("WARNING: Nu > 30 means distribution is approximately Gaussian!")
    elif nu.min() < 5:
        print("Nu values are low enough for heavy tails (good)")
    else:
        print(f"Nu values suggest moderate tails")

    return nu


# =============================================================================
# Investigation 2: Predictor Bias
# =============================================================================

def investigate_predictor_bias(vae, predictor, val_loader, config, context_len=20, device="cuda"):
    """Check if predictor z prediction is systematically biased."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 2: PREDICTOR BIAS")
    print("=" * 70)

    z_diffs = []
    z_targets = []
    z_preds = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            B, T = surface.shape[:2]

            if T <= context_len:
                continue

            batch = {"surface": surface}

            # Get target z from encoder (sees full sequence)
            z_target, z_logvar_target, _ = vae.main_encoder(batch)
            z_target_future = z_target[:, context_len:]  # (B, horizon, latent_dim)

            # Get predicted z from predictor (sees context only)
            context = surface[:, :context_len]
            horizon = T - context_len
            z_pred, z_logvar_pred = predictor(context, horizon=horizon)

            z_diffs.append((z_pred - z_target_future).cpu())
            z_targets.append(z_target_future.cpu())
            z_preds.append(z_pred.cpu())

    z_diffs = torch.cat(z_diffs, dim=0)  # (N, horizon, latent_dim)
    z_targets = torch.cat(z_targets, dim=0)
    z_preds = torch.cat(z_preds, dim=0)

    # Mean bias per dimension
    mean_bias = z_diffs.mean(dim=(0, 1))  # (latent_dim,)
    std_bias = z_diffs.std(dim=(0, 1))

    print(f"\nZ prediction bias (mean over all samples and horizons):")
    print(f"  Mean bias per dim: {mean_bias.numpy().round(4)}")
    print(f"  Std of diff per dim: {std_bias.numpy().round(4)}")

    # Overall statistics
    print(f"\n  Overall mean bias: {z_diffs.mean().item():.4f}")
    print(f"  Overall std of diff: {z_diffs.std().item():.4f}")

    # Check if bias is consistent (t-test)
    t_stat, p_val = stats.ttest_1samp(z_diffs.flatten().numpy(), 0)
    print(f"\n  T-test for bias != 0: t={t_stat:.2f}, p={p_val:.4e}")
    if p_val < 0.05:
        print("  => SIGNIFICANT BIAS detected!")
    else:
        print("  => No significant bias")

    # Z variance comparison
    z_target_var = z_targets.var(dim=(0, 1))
    z_pred_var = z_preds.var(dim=(0, 1))

    print(f"\nZ variance comparison:")
    print(f"  z_target var per dim: {z_target_var.numpy().round(4)}")
    print(f"  z_pred var per dim: {z_pred_var.numpy().round(4)}")
    print(f"  Ratio (pred/target): {(z_pred_var / z_target_var).numpy().round(2)}")

    return {
        "mean_bias": mean_bias.numpy(),
        "std_bias": std_bias.numpy(),
        "z_target_var": z_target_var.numpy(),
        "z_pred_var": z_pred_var.numpy(),
    }


# =============================================================================
# Investigation 3: Variance Decomposition
# =============================================================================

def investigate_variance_sources(vae, predictor, log_returns, device="cuda",
                                  context_len=20, horizon=30, n_samples=100):
    """Decompose variance into z-sampling vs decoder-sampling components."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: VARIANCE SOURCE DECOMPOSITION")
    print("=" * 70)

    # Pick a random context
    np.random.seed(42)
    start_idx = np.random.randint(0, len(log_returns) - context_len - horizon)
    context = log_returns[start_idx:start_idx + context_len]
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get context embedding
        ctx_emb_context = vae.ctx_encoder({"surface": context_tensor})
        z_ctx_mean, z_ctx_logvar, _ = vae.main_encoder({"surface": context_tensor})
        z_future_mean, z_future_logvar = predictor(context_tensor, horizon=horizon)

        # === Test 1: Fix z, vary decoder sampling ===
        print("\n--- Variance from DECODER sampling only (z fixed at mean) ---")
        decoder_only_samples = []
        for _ in range(n_samples):
            # Use z = z_mean (no sampling)
            z_ctx = z_ctx_mean
            z_future = z_future_mean

            ctx_dim = ctx_emb_context.shape[-1]
            ctx_emb_future = torch.zeros(1, horizon, ctx_dim, device=device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)
            z = torch.cat([z_ctx, z_future], dim=1)

            # Decode with sampling
            _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)
            decoder_only_samples.append(sample[0, context_len:, 2, 2].cpu().numpy())

        decoder_only_samples = np.array(decoder_only_samples)  # (n_samples, horizon)
        decoder_var = decoder_only_samples.var(axis=0)

        print(f"  Decoder variance at h=1: {decoder_var[0]:.6f}")
        print(f"  Decoder variance at h=30: {decoder_var[-1]:.6f}")

        # === Test 2: Vary z, fix decoder (use mean) ===
        print("\n--- Variance from Z sampling only (decoder at mean) ---")
        z_only_samples = []
        for _ in range(n_samples):
            # Sample z
            z_ctx = z_ctx_mean + torch.exp(0.5 * z_ctx_logvar) * torch.randn_like(z_ctx_mean)
            z_future = z_future_mean + torch.exp(0.5 * z_future_logvar) * torch.randn_like(z_future_mean)

            ctx_dim = ctx_emb_context.shape[-1]
            ctx_emb_future = torch.zeros(1, horizon, ctx_dim, device=device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)
            z = torch.cat([z_ctx, z_future], dim=1)

            # Decode WITHOUT sampling (mean only)
            mean, _, _, _ = vae.decoder(ctx_emb, z, sample=False)
            z_only_samples.append(mean[0, context_len:, 2, 2].cpu().numpy())

        z_only_samples = np.array(z_only_samples)
        z_var = z_only_samples.var(axis=0)

        print(f"  Z variance at h=1: {z_var[0]:.6f}")
        print(f"  Z variance at h=30: {z_var[-1]:.6f}")

        # === Test 3: Both sources ===
        print("\n--- Total variance (both z and decoder sampling) ---")
        total_samples = []
        for _ in range(n_samples):
            z_ctx = z_ctx_mean + torch.exp(0.5 * z_ctx_logvar) * torch.randn_like(z_ctx_mean)
            z_future = z_future_mean + torch.exp(0.5 * z_future_logvar) * torch.randn_like(z_future_mean)

            ctx_dim = ctx_emb_context.shape[-1]
            ctx_emb_future = torch.zeros(1, horizon, ctx_dim, device=device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)
            z = torch.cat([z_ctx, z_future], dim=1)

            _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)
            total_samples.append(sample[0, context_len:, 2, 2].cpu().numpy())

        total_samples = np.array(total_samples)
        total_var = total_samples.var(axis=0)

        print(f"  Total variance at h=1: {total_var[0]:.6f}")
        print(f"  Total variance at h=30: {total_var[-1]:.6f}")

        # === Summary ===
        print("\n--- Variance Decomposition ---")
        print(f"  Decoder contribution: {decoder_var.mean() / total_var.mean() * 100:.1f}%")
        print(f"  Z contribution: {z_var.mean() / total_var.mean() * 100:.1f}%")

        # Check predictor logvar
        print("\n--- Predictor z_logvar statistics ---")
        pred_std = torch.exp(0.5 * z_future_logvar).cpu().numpy()
        print(f"  Predictor z std (from logvar): mean={pred_std.mean():.4f}, range=[{pred_std.min():.4f}, {pred_std.max():.4f}]")

        encoder_std = torch.exp(0.5 * z_ctx_logvar).cpu().numpy()
        print(f"  Encoder z std (from logvar): mean={encoder_std.mean():.4f}, range=[{encoder_std.min():.4f}, {encoder_std.max():.4f}]")

    return {
        "decoder_var": decoder_var,
        "z_var": z_var,
        "total_var": total_var,
    }


# =============================================================================
# Investigation 4: Autocorrelation Structure
# =============================================================================

def investigate_autocorrelation(vae, predictor, log_returns, surfaces, device="cuda",
                                 context_len=20, horizon=30, n_contexts=50, n_samples=10):
    """Compare autocorrelation of generated vs GT sequences."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 4: AUTOCORRELATION STRUCTURE")
    print("=" * 70)

    # GT ACF: compute from all overlapping windows
    atm_returns = log_returns[:, 2, 2]

    # Compute GT ACF
    gt_acf = []
    for lag in range(horizon):
        if lag == 0:
            gt_acf.append(1.0)
        else:
            corr = np.corrcoef(atm_returns[:-lag], atm_returns[lag:])[0, 1]
            gt_acf.append(corr)
    gt_acf = np.array(gt_acf)

    print(f"\nGT ACF (ATM IV log-returns):")
    print(f"  Lag 1: {gt_acf[1]:.4f}")
    print(f"  Lag 5: {gt_acf[5]:.4f}")
    print(f"  Lag 10: {gt_acf[10]:.4f}")

    # Model ACF: generate sequences and compute within-sequence ACF
    np.random.seed(42)
    max_start = len(log_returns) - context_len - horizon
    sampled_starts = np.random.choice(max_start, size=n_contexts, replace=False)

    all_model_acfs = []

    with torch.no_grad():
        for start_idx in sampled_starts:
            context = log_returns[start_idx:start_idx + context_len]
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

            ctx_emb_context = vae.ctx_encoder({"surface": context_tensor})
            z_ctx_mean, z_ctx_logvar, _ = vae.main_encoder({"surface": context_tensor})
            z_future_mean, z_future_logvar = predictor(context_tensor, horizon=horizon)

            for _ in range(n_samples):
                z_ctx = z_ctx_mean + torch.exp(0.5 * z_ctx_logvar) * torch.randn_like(z_ctx_mean)
                z_future = z_future_mean + torch.exp(0.5 * z_future_logvar) * torch.randn_like(z_future_mean)

                ctx_dim = ctx_emb_context.shape[-1]
                ctx_emb_future = torch.zeros(1, horizon, ctx_dim, device=device)
                ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)
                z = torch.cat([z_ctx, z_future], dim=1)

                _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)
                returns = sample[0, context_len:, 2, 2].cpu().numpy()

                # Compute ACF for this sequence
                seq_acf = []
                for lag in range(horizon):
                    if lag == 0:
                        seq_acf.append(1.0)
                    elif lag < len(returns):
                        corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                        seq_acf.append(corr if not np.isnan(corr) else 0)
                    else:
                        seq_acf.append(0)
                all_model_acfs.append(seq_acf)

    model_acfs = np.array(all_model_acfs)
    model_acf_mean = model_acfs.mean(axis=0)
    model_acf_std = model_acfs.std(axis=0)

    print(f"\nModel ACF (mean over {n_contexts * n_samples} sequences):")
    print(f"  Lag 1: {model_acf_mean[1]:.4f} +/- {model_acf_std[1]:.4f}")
    print(f"  Lag 5: {model_acf_mean[5]:.4f} +/- {model_acf_std[5]:.4f}")
    print(f"  Lag 10: {model_acf_mean[10]:.4f} +/- {model_acf_std[10]:.4f}")

    # Comparison
    print("\n--- ACF Comparison ---")
    for lag in [1, 5, 10, 20]:
        if lag < horizon:
            diff = model_acf_mean[lag] - gt_acf[lag]
            print(f"  Lag {lag}: GT={gt_acf[lag]:.4f}, Model={model_acf_mean[lag]:.4f}, Diff={diff:+.4f}")

    return {
        "gt_acf": gt_acf,
        "model_acf_mean": model_acf_mean,
        "model_acf_std": model_acf_std,
    }


# =============================================================================
# Bonus: Check actual decoder output distribution
# =============================================================================

def check_decoder_distribution(vae, predictor, log_returns, device="cuda",
                                context_len=20, n_samples=10000):
    """Check the actual distribution of decoder outputs vs theoretical Student-t."""
    print("\n" + "=" * 70)
    print("BONUS: DECODER OUTPUT DISTRIBUTION CHECK")
    print("=" * 70)

    # Pick a context
    np.random.seed(42)
    start_idx = 1000
    context = log_returns[start_idx:start_idx + context_len]
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = vae.ctx_encoder({"surface": context_tensor})
        z_mean, z_logvar, _ = vae.main_encoder({"surface": context_tensor})

        # Sample many times from decoder with fixed z
        samples = []
        for _ in range(n_samples):
            _, sample, _, _ = vae.decoder(ctx_emb, z_mean, sample=True)
            samples.append(sample[0, -1, 2, 2].cpu().numpy())

        samples = np.array(samples)

    # Compute statistics
    emp_mean = samples.mean()
    emp_std = samples.std()
    emp_skew = stats.skew(samples)
    emp_kurt = stats.kurtosis(samples)

    # Theoretical Student-t with nu from decoder
    nu_atm = vae.decoder.nu[12].item()  # ATM position is grid[2,2] = index 12

    print(f"\nATM grid point decoder output distribution ({n_samples} samples):")
    print(f"  Nu (degrees of freedom): {nu_atm:.2f}")
    print(f"  Empirical mean: {emp_mean:.6f}")
    print(f"  Empirical std: {emp_std:.6f}")
    print(f"  Empirical skewness: {emp_skew:.4f}")
    print(f"  Empirical kurtosis: {emp_kurt:.4f}")

    # Theoretical Student-t kurtosis
    if nu_atm > 4:
        theoretical_kurt = 6.0 / (nu_atm - 4)
        print(f"\n  Theoretical Student-t kurtosis (nu={nu_atm:.2f}): {theoretical_kurt:.4f}")
        print(f"  Kurtosis ratio (empirical/theoretical): {emp_kurt / theoretical_kurt:.2f}")

    # Normality test
    _, p_normal = stats.normaltest(samples)
    print(f"\n  Normality test p-value: {p_normal:.4e}")
    if p_normal < 0.05:
        print("  => Distribution is NOT normal (good - should have fat tails)")
    else:
        print("  => Distribution appears NORMAL (bad - tails should be heavier)")

    return {
        "samples": samples,
        "emp_kurt": emp_kurt,
        "nu_atm": nu_atm,
    }


def main():
    print("=" * 70)
    print("ROOT CAUSE INVESTIGATION")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("\nLoading models...")
    vae, predictor, config = load_models(device)
    print("Models loaded.")

    context_len = 20
    batch_size = 64

    # Create validation loader
    val_start = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)
    val_data = log_returns[val_start:val_end]
    val_loader = create_dataloader(val_data, context_len, batch_size)

    # Run investigations
    nu_results = investigate_nu_parameter(vae)

    bias_results = investigate_predictor_bias(
        vae, predictor, val_loader, config, context_len, device
    )

    variance_results = investigate_variance_sources(
        vae, predictor, log_returns, device, context_len
    )

    acf_results = investigate_autocorrelation(
        vae, predictor, log_returns, surfaces, device, context_len
    )

    decoder_results = check_decoder_distribution(
        vae, predictor, log_returns, device, context_len
    )

    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("ROOT CAUSE SUMMARY")
    print("=" * 70)

    print("\n1. Student-t Nu Parameter:")
    print(f"   Nu range: [{nu_results.min():.2f}, {nu_results.max():.2f}]")
    if nu_results.min() < 10:
        print("   => Nu is LOW ENOUGH for fat tails")
    else:
        print("   => Nu may be TOO HIGH (near Gaussian)")

    print("\n2. Predictor Bias:")
    print(f"   Mean bias: {bias_results['mean_bias'].mean():.4f}")
    if abs(bias_results['mean_bias'].mean()) > 0.1:
        print("   => SIGNIFICANT BIAS in z prediction")
    else:
        print("   => Bias is small")

    print("\n3. Variance Sources:")
    dec_contrib = variance_results['decoder_var'].mean() / variance_results['total_var'].mean()
    z_contrib = variance_results['z_var'].mean() / variance_results['total_var'].mean()
    print(f"   Decoder contribution: {dec_contrib*100:.1f}%")
    print(f"   Z contribution: {z_contrib*100:.1f}%")

    print("\n4. Autocorrelation:")
    gt_acf1 = acf_results['gt_acf'][1]
    model_acf1 = acf_results['model_acf_mean'][1]
    print(f"   GT ACF(1): {gt_acf1:.4f}")
    print(f"   Model ACF(1): {model_acf1:.4f}")
    if abs(model_acf1 - gt_acf1) > 0.1:
        print("   => ACF MISMATCH - temporal structure not captured")
    else:
        print("   => ACF matches reasonably")

    print("\n5. Decoder Distribution:")
    print(f"   Empirical kurtosis: {decoder_results['emp_kurt']:.4f}")
    if decoder_results['nu_atm'] > 4:
        theoretical = 6.0 / (decoder_results['nu_atm'] - 4)
        ratio = decoder_results['emp_kurt'] / theoretical
        print(f"   Theoretical kurtosis: {theoretical:.4f}")
        print(f"   Ratio: {ratio:.2f}")
        if ratio < 0.5:
            print("   => DECODER NOT PRODUCING FAT TAILS (key issue!)")
        else:
            print("   => Decoder is producing expected tails")


if __name__ == "__main__":
    main()
