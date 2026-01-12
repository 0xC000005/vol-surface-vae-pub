"""
Comprehensive Analysis of Two-Stage VAE (Oracle Case)

Runs 6 analysis sections to thoroughly evaluate the VAE under oracle conditions
(encoder sees full sequence including target).

Sections:
1. Latent Space Analysis - z dimension utilization, information content, trajectory
2. Decoder Expressivity Analysis - FiLM contribution, bottleneck analysis
3. Distribution Shape Diagnostics - kurtosis, skewness, tail ratios
4. Spatial Correlation Analysis - 25x25 correlation, eigenvalue spectrum
5. Temporal Dynamics Analysis - autocorrelation, mean-reversion, roughness
6. Failure Mode Catalog - bias, worst cases, crisis periods

Usage:
    python experiments/backfill/two_stage_vae/analyze_oracle_comprehensive.py

Output:
    results/two_stage_analysis/
    ├── latent_space_analysis.json
    ├── decoder_expressivity.json
    ├── shape_diagnostics.json
    ├── spatial_correlation.json
    ├── temporal_dynamics.json
    ├── failure_modes.json
    ├── qq_plots_grid.png
    ├── kurtosis_heatmap.png
    ├── correlation_matrices.png
    └── comprehensive_report.md
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.two_stage_config import TwoStageConfig
from experiments.backfill.two_stage_vae.exp_low_rank_cov import (
    CVAETwoStageLowRankCov,
    to_log_returns,
    create_dataloader,
)


# ============================================================================
# Setup and Data Loading
# ============================================================================


def setup_paths():
    """Create output directory and return paths."""
    output_dir = Path("results/two_stage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_data():
    """Load volatility surface data and compute log-returns."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)
    return surfaces, log_returns, log_surfaces


def load_model():
    """Load the best trained model with its saved config."""
    model_path = Path("models/backfill/two_stage/low_rank_cov/diagonal_only_best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_data = torch.load(model_path, map_location="cpu", weights_only=False)
    config = model_data["model_config"]

    # Ensure device is set
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    model = CVAETwoStageLowRankCov(config)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()
    return model, config


def extract_oracle_samples(model, log_returns, config, n_sequences=500, n_samples=100):
    """Extract oracle z and samples from the model.

    Oracle case: encoder sees full sequence including target.
    """
    device = config["device"]
    model = model.to(device)
    model.eval()

    context_len = config["context_len"]
    seq_len = context_len + 1  # Context + 1 prediction step

    # Storage for analysis
    z_means = []
    z_logvars = []
    z_samples = []
    oracle_preds = []
    gt_targets = []

    # Sample sequences evenly across the dataset
    total_seqs = len(log_returns) - seq_len
    indices = np.linspace(0, total_seqs - 1, n_sequences, dtype=int)

    with torch.no_grad():
        for idx in indices:
            seq = log_returns[idx : idx + seq_len]
            batch = torch.tensor(seq[None], dtype=torch.float32).to(device)
            batch_dict = {"surface": batch}

            # Encode (oracle - encoder sees full sequence)
            ctx_emb = model.ctx_encoder(batch_dict)
            z_mean, z_logvar, z = model.main_encoder(batch_dict)

            z_means.append(z_mean.cpu().numpy())
            z_logvars.append(z_logvar.cpu().numpy())
            z_samples.append(z.cpu().numpy())

            # Generate samples
            samples_batch = []
            z_std = torch.exp(0.5 * z_logvar)
            for _ in range(n_samples):
                eps = torch.randn_like(z_std)
                z_samp = z_mean + z_std * eps
                mean_pred, sample_pred = model.decoder(ctx_emb, z_samp, sample=True)
                # Get prediction for the last timestep (target)
                samples_batch.append(sample_pred[0, -1].cpu().numpy())

            oracle_preds.append(np.array(samples_batch))
            gt_targets.append(seq[-1])  # Ground truth target

    return {
        "z_means": np.array(z_means),  # (n_seq, 1, T, latent_dim)
        "z_logvars": np.array(z_logvars),
        "z_samples": np.array(z_samples),
        "oracle_preds": np.array(oracle_preds),  # (n_seq, n_samples, 5, 5)
        "gt_targets": np.array(gt_targets),  # (n_seq, 5, 5)
    }


# ============================================================================
# Section 1: Latent Space Analysis
# ============================================================================


def analyze_latent_space(oracle_data, log_returns, output_dir):
    """Section 1: Analyze z dimension utilization, information content, trajectory."""
    print("\n=== Section 1: Latent Space Analysis ===")

    z_means = oracle_data["z_means"]  # (n_seq, 1, T, latent_dim)
    z_logvars = oracle_data["z_logvars"]
    z_samples = oracle_data["z_samples"]
    gt_targets = oracle_data["gt_targets"]

    # Flatten: (n_seq * T, latent_dim)
    z_mean_flat = z_means.reshape(-1, z_means.shape[-1])
    z_logvar_flat = z_logvars.reshape(-1, z_logvars.shape[-1])
    z_sample_flat = z_samples.reshape(-1, z_samples.shape[-1])
    latent_dim = z_mean_flat.shape[-1]

    results = {"latent_dim": latent_dim}

    # 1.1 z Dimension Utilization
    print("  1.1 z Dimension Utilization...")

    # z_logvar per dimension (identify collapsed dims where logvar << -4)
    mean_logvar = z_logvar_flat.mean(axis=0)
    collapsed_dims = np.sum(mean_logvar < -4)

    # z_mean variance per dimension (which dims carry signal)
    mean_variance = z_mean_flat.var(axis=0)

    # KL contribution per dimension
    z_var = np.exp(z_logvar_flat)
    kl_per_dim = 0.5 * (z_var + z_mean_flat**2 - 1 - z_logvar_flat)
    mean_kl_per_dim = kl_per_dim.mean(axis=0)

    # Dimension activation: % samples where |z| > 0.5
    activation_rate = (np.abs(z_sample_flat) > 0.5).mean(axis=0)

    results["dimension_utilization"] = {
        "mean_logvar_per_dim": mean_logvar.tolist(),
        "collapsed_dims_count": int(collapsed_dims),
        "collapsed_threshold": -4.0,
        "mean_variance_per_dim": mean_variance.tolist(),
        "total_variance": float(mean_variance.sum()),
        "kl_per_dim": mean_kl_per_dim.tolist(),
        "total_kl": float(mean_kl_per_dim.sum()),
        "activation_rate_per_dim": activation_rate.tolist(),
        "avg_activation_rate": float(activation_rate.mean()),
    }

    # 1.2 z Information Content (Linear Probe)
    print("  1.2 z Information Content (Linear Probe)...")

    # Prepare features from targets
    # ATM IV = target[2,2], skew = target[0,2] - target[4,2], slope = target[2,0] - target[2,4]
    atm_iv = gt_targets[:, 2, 2]
    skew = gt_targets[:, 0, 2] - gt_targets[:, 4, 2]
    slope = gt_targets[:, 2, 0] - gt_targets[:, 2, 4]
    vix_proxy = gt_targets.mean(axis=(1, 2))  # Average across grid

    # Use z_mean at last timestep for prediction
    z_for_probe = z_means[:, 0, -1, :]  # (n_seq, latent_dim)

    # Train linear probes
    r2_scores = {}
    for name, target in [
        ("atm_iv", atm_iv),
        ("skew", skew),
        ("slope", slope),
        ("vix_proxy", vix_proxy),
    ]:
        reg = LinearRegression()
        reg.fit(z_for_probe, target)
        r2 = reg.score(z_for_probe, target)
        r2_scores[name] = float(r2)

    results["information_content"] = {
        "r2_scores": r2_scores,
        "interpretation": (
            "High R² means z captures feature; "
            "Low R² means information lost during encoding"
        ),
    }

    # 1.3 z Trajectory Smoothness
    print("  1.3 z Trajectory Smoothness...")

    # Reshape to (n_seq, T, latent_dim)
    z_mean_seq = z_means[:, 0]  # (n_seq, T, latent_dim)
    T = z_mean_seq.shape[1]

    if T > 1:
        # Autocorrelation at lag 1
        z_t = z_mean_seq[:, :-1].reshape(-1, latent_dim)
        z_t1 = z_mean_seq[:, 1:].reshape(-1, latent_dim)

        autocorr = []
        for d in range(latent_dim):
            if z_t[:, d].std() > 1e-6 and z_t1[:, d].std() > 1e-6:
                corr = np.corrcoef(z_t[:, d], z_t1[:, d])[0, 1]
                autocorr.append(corr if np.isfinite(corr) else 0.0)
            else:
                autocorr.append(0.0)

        # First-difference variance
        z_diff = np.diff(z_mean_seq, axis=1)
        diff_var = z_diff.var(axis=(0, 1))

        # Cosine similarity between adjacent timesteps
        z_norm = z_mean_seq / (
            np.linalg.norm(z_mean_seq, axis=-1, keepdims=True) + 1e-8
        )
        cos_sim = (z_norm[:, :-1] * z_norm[:, 1:]).sum(axis=-1).mean()

        results["trajectory_smoothness"] = {
            "autocorr_lag1_per_dim": autocorr,
            "mean_autocorr_lag1": float(np.mean(autocorr)),
            "diff_variance_per_dim": diff_var.tolist(),
            "total_diff_variance": float(diff_var.sum()),
            "mean_cosine_similarity": float(cos_sim),
        }
    else:
        results["trajectory_smoothness"] = {
            "note": "T=1, cannot compute temporal metrics"
        }

    # 1.4 PCA of z Space
    print("  1.4 PCA of z Space...")

    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_sample_flat)

    results["pca_analysis"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained": float(sum(pca.explained_variance_ratio_)),
        "components_shape": list(pca.components_.shape),
        "z_pca_mean": z_pca.mean(axis=0).tolist(),
        "z_pca_std": z_pca.std(axis=0).tolist(),
    }

    # Save results
    with open(output_dir / "latent_space_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'latent_space_analysis.json'}")
    return results


# ============================================================================
# Section 2: Decoder Expressivity Analysis
# ============================================================================


def analyze_decoder_expressivity(model, oracle_data, log_returns, config, output_dir):
    """Section 2: Analyze decoder capacity and FiLM contribution."""
    print("\n=== Section 2: Decoder Expressivity Analysis ===")

    device = config["device"]
    model = model.to(device)
    model.eval()

    context_len = config["context_len"]
    seq_len = context_len + 1

    results = {}

    # 2.1 Oracle z Reconstruction Test
    print("  2.1 Oracle z Reconstruction Test...")

    oracle_mse_list = []
    zero_z_mse_list = []

    n_test = min(200, len(log_returns) - seq_len)
    indices = np.linspace(0, len(log_returns) - seq_len - 1, n_test, dtype=int)

    with torch.no_grad():
        for idx in indices:
            seq = log_returns[idx : idx + seq_len]
            batch = torch.tensor(seq[None], dtype=torch.float32).to(device)
            batch_dict = {"surface": batch}
            target = batch[:, 1:]  # (1, T, 5, 5)

            # Oracle z (posterior mean)
            ctx_emb = model.ctx_encoder(batch_dict)
            z_mean, z_logvar, _ = model.main_encoder(batch_dict)

            # Decode with oracle z
            oracle_pred, _ = model.decoder(ctx_emb, z_mean, sample=False)
            oracle_pred = oracle_pred[:, :-1]  # Align with target
            oracle_mse = F.mse_loss(oracle_pred, target).item()
            oracle_mse_list.append(oracle_mse)

            # Decode with zero z
            zero_z = torch.zeros_like(z_mean)
            zero_pred, _ = model.decoder(ctx_emb, zero_z, sample=False)
            zero_pred = zero_pred[:, :-1]
            zero_mse = F.mse_loss(zero_pred, target).item()
            zero_z_mse_list.append(zero_mse)

    oracle_mse_mean = float(np.mean(oracle_mse_list))
    zero_z_mse_mean = float(np.mean(zero_z_mse_list))
    z_contribution = zero_z_mse_mean - oracle_mse_mean

    results["oracle_reconstruction"] = {
        "oracle_mse_mean": oracle_mse_mean,
        "oracle_mse_std": float(np.std(oracle_mse_list)),
        "zero_z_mse_mean": zero_z_mse_mean,
        "zero_z_mse_std": float(np.std(zero_z_mse_list)),
        "z_contribution": z_contribution,
        "z_contribution_percent": (
            z_contribution / zero_z_mse_mean * 100 if zero_z_mse_mean > 0 else 0
        ),
        "interpretation": (
            "z_contribution = zero_z_mse - oracle_mse. "
            "Higher means z carries more information."
        ),
    }

    # 2.2 FiLM Contribution Analysis (requires model modification for full analysis)
    print("  2.2 FiLM Contribution Analysis...")

    # Get gamma and beta statistics
    gamma_weights = model.decoder.mean_decoder.gamma_net.weight.detach().cpu().numpy()
    beta_weights = model.decoder.mean_decoder.beta_net.weight.detach().cpu().numpy()

    results["film_analysis"] = {
        "gamma_weight_norm": float(np.linalg.norm(gamma_weights)),
        "beta_weight_norm": float(np.linalg.norm(beta_weights)),
        "gamma_weight_mean": float(gamma_weights.mean()),
        "gamma_weight_std": float(gamma_weights.std()),
        "beta_weight_mean": float(beta_weights.mean()),
        "beta_weight_std": float(beta_weights.std()),
        "note": (
            "Full FiLM ablation requires model modification. "
            "These are weight statistics only."
        ),
    }

    # 2.3 Decoder Architecture Summary
    print("  2.3 Decoder Architecture Summary...")

    latent_dim = config.get("latent_dim", 16)
    mem_hidden = config.get("mem_hidden", 64)

    results["architecture"] = {
        "latent_dim": latent_dim,
        "mem_hidden": mem_hidden,
        "input_dim": latent_dim,  # z_only decoder
        "bottleneck_ratio": f"{latent_dim}→{mem_hidden}",
        "bottleneck_severity": (
            "SEVERE" if mem_hidden < latent_dim else
            "MODERATE" if mem_hidden == latent_dim else
            "OK"
        ),
    }

    # 2.4 Per-Grid Reconstruction Error
    print("  2.4 Per-Grid Reconstruction Error...")

    gt_targets = oracle_data["gt_targets"]
    oracle_preds = oracle_data["oracle_preds"]

    # Mean of oracle samples
    oracle_means = oracle_preds.mean(axis=1)  # (n_seq, 5, 5)

    # MSE per grid point
    per_grid_mse = ((oracle_means - gt_targets) ** 2).mean(axis=0)  # (5, 5)

    # Bias per grid point
    per_grid_bias = (oracle_means - gt_targets).mean(axis=0)  # (5, 5)

    results["per_grid_error"] = {
        "mse_per_grid": per_grid_mse.tolist(),
        "mse_min": float(per_grid_mse.min()),
        "mse_max": float(per_grid_mse.max()),
        "mse_range_ratio": float(per_grid_mse.max() / (per_grid_mse.min() + 1e-8)),
        "bias_per_grid": per_grid_bias.tolist(),
        "bias_min": float(per_grid_bias.min()),
        "bias_max": float(per_grid_bias.max()),
        "corner_indices": [(0, 0), (0, 4), (4, 0), (4, 4)],
        "corner_mse": [
            float(per_grid_mse[0, 0]),
            float(per_grid_mse[0, 4]),
            float(per_grid_mse[4, 0]),
            float(per_grid_mse[4, 4]),
        ],
        "atm_mse": float(per_grid_mse[2, 2]),
    }

    # Save results
    with open(output_dir / "decoder_expressivity.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'decoder_expressivity.json'}")
    return results


# ============================================================================
# Section 3: Distribution Shape Diagnostics
# ============================================================================


def analyze_distribution_shape(oracle_data, log_returns, output_dir):
    """Section 3: Kurtosis, skewness, tail ratios per grid point."""
    print("\n=== Section 3: Distribution Shape Diagnostics ===")

    gt_targets = oracle_data["gt_targets"]  # (n_seq, 5, 5)
    oracle_preds = oracle_data["oracle_preds"]  # (n_seq, n_samples, 5, 5)

    results = {}

    # 3.1 Per-Grid Shape Statistics
    print("  3.1 Per-Grid Shape Statistics...")

    gt_kurtosis = np.zeros((5, 5))
    oracle_kurtosis = np.zeros((5, 5))
    gt_skewness = np.zeros((5, 5))
    oracle_skewness = np.zeros((5, 5))
    tail_ratios = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_vals = gt_targets[:, i, j]
            oracle_vals = oracle_preds[:, :, i, j].flatten()

            # Kurtosis (excess kurtosis)
            gt_kurtosis[i, j] = stats.kurtosis(gt_vals)
            oracle_kurtosis[i, j] = stats.kurtosis(oracle_vals)

            # Skewness
            gt_skewness[i, j] = stats.skew(gt_vals)
            oracle_skewness[i, j] = stats.skew(oracle_vals)

            # Tail ratio: P(|x| > 2*std) oracle / GT
            gt_std = gt_vals.std()
            oracle_std = oracle_vals.std()
            gt_tail = (np.abs(gt_vals) > 2 * gt_std).mean()
            oracle_tail = (np.abs(oracle_vals) > 2 * oracle_std).mean()
            tail_ratios[i, j] = oracle_tail / (gt_tail + 1e-8)

    kurtosis_recovery = oracle_kurtosis / (gt_kurtosis + 1e-8)
    skewness_match = np.sign(oracle_skewness) == np.sign(gt_skewness)

    results["per_grid_stats"] = {
        "gt_kurtosis": gt_kurtosis.tolist(),
        "oracle_kurtosis": oracle_kurtosis.tolist(),
        "kurtosis_recovery_ratio": kurtosis_recovery.tolist(),
        "mean_kurtosis_recovery": float(np.mean(kurtosis_recovery)),
        "gt_skewness": gt_skewness.tolist(),
        "oracle_skewness": oracle_skewness.tolist(),
        "skewness_sign_match": skewness_match.tolist(),
        "skewness_match_rate": float(skewness_match.mean()),
        "tail_ratios_2sigma": tail_ratios.tolist(),
        "mean_tail_ratio": float(np.mean(tail_ratios)),
    }

    # 3.2 ATM point detailed stats
    print("  3.2 ATM Detailed Statistics...")

    atm_gt = gt_targets[:, 2, 2]
    atm_oracle = oracle_preds[:, :, 2, 2].flatten()

    results["atm_detailed"] = {
        "gt_mean": float(atm_gt.mean()),
        "gt_std": float(atm_gt.std()),
        "gt_kurtosis": float(stats.kurtosis(atm_gt)),
        "gt_skewness": float(stats.skew(atm_gt)),
        "oracle_mean": float(atm_oracle.mean()),
        "oracle_std": float(atm_oracle.std()),
        "oracle_kurtosis": float(stats.kurtosis(atm_oracle)),
        "oracle_skewness": float(stats.skew(atm_oracle)),
        "kurtosis_ratio": float(stats.kurtosis(atm_oracle) / (stats.kurtosis(atm_gt) + 1e-8)),
    }

    # 3.3 Create Kurtosis Heatmap
    print("  3.3 Creating Kurtosis Heatmap...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # GT Kurtosis
    im0 = axes[0].imshow(gt_kurtosis, cmap="viridis", aspect="auto")
    axes[0].set_title("GT Kurtosis")
    axes[0].set_xlabel("Maturity")
    axes[0].set_ylabel("Moneyness")
    plt.colorbar(im0, ax=axes[0])

    # Oracle Kurtosis
    im1 = axes[1].imshow(oracle_kurtosis, cmap="viridis", aspect="auto")
    axes[1].set_title("Oracle Kurtosis")
    axes[1].set_xlabel("Maturity")
    plt.colorbar(im1, ax=axes[1])

    # Recovery Ratio
    im2 = axes[2].imshow(
        np.clip(kurtosis_recovery, 0, 1), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
    )
    axes[2].set_title("Kurtosis Recovery (Oracle/GT)")
    axes[2].set_xlabel("Maturity")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / "kurtosis_heatmap.png", dpi=150)
    plt.close()

    # 3.4 Create Q-Q Plots Grid
    print("  3.4 Creating Q-Q Plots Grid...")

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            gt_vals = np.sort(gt_targets[:, i, j])
            oracle_vals = np.sort(oracle_preds[:, :, i, j].flatten())

            # Subsample oracle for plotting
            oracle_subsample = oracle_vals[:: max(1, len(oracle_vals) // len(gt_vals))]
            min_len = min(len(gt_vals), len(oracle_subsample))

            ax.scatter(gt_vals[:min_len], oracle_subsample[:min_len], alpha=0.5, s=5)
            ax.plot(
                [gt_vals.min(), gt_vals.max()],
                [gt_vals.min(), gt_vals.max()],
                "r--",
                lw=1,
            )
            ax.set_title(f"({i},{j})", fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Q-Q Plots: Oracle vs GT per Grid Point", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "qq_plots_grid.png", dpi=150)
    plt.close()

    # 3.5 Tail Probability Analysis
    print("  3.5 Tail Probability Analysis...")

    # For ATM point, compute tail probabilities
    atm_gt = gt_targets[:, 2, 2]
    atm_oracle = oracle_preds[:, :, 2, 2].flatten()

    gt_std = atm_gt.std()
    oracle_std = atm_oracle.std()

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    tail_analysis = {"thresholds": thresholds, "gt_tail_prob": [], "oracle_tail_prob": []}

    for thresh in thresholds:
        gt_prob = (np.abs(atm_gt) > thresh * gt_std).mean()
        oracle_prob = (np.abs(atm_oracle) > thresh * oracle_std).mean()
        tail_analysis["gt_tail_prob"].append(float(gt_prob))
        tail_analysis["oracle_tail_prob"].append(float(oracle_prob))

    # Positive vs Negative tail asymmetry at 2-sigma
    gt_pos_tail = (atm_gt > 2 * gt_std).sum()
    gt_neg_tail = (atm_gt < -2 * gt_std).sum()
    oracle_pos_tail = (atm_oracle > 2 * oracle_std).sum()
    oracle_neg_tail = (atm_oracle < -2 * oracle_std).sum()

    tail_analysis["gt_pos_neg_ratio_2sigma"] = float(
        gt_pos_tail / (gt_neg_tail + 1) if gt_neg_tail > 0 else gt_pos_tail
    )
    tail_analysis["oracle_pos_neg_ratio_2sigma"] = float(
        oracle_pos_tail / (oracle_neg_tail + 1) if oracle_neg_tail > 0 else oracle_pos_tail
    )

    results["tail_analysis"] = tail_analysis

    # Save results
    with open(output_dir / "shape_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'shape_diagnostics.json'}")
    return results


# ============================================================================
# Section 4: Spatial Correlation Analysis
# ============================================================================


def analyze_spatial_correlation(model, oracle_data, log_returns, config, output_dir):
    """Section 4: Cross-grid correlation and eigenvalue spectrum."""
    print("\n=== Section 4: Spatial Correlation Analysis ===")

    gt_targets = oracle_data["gt_targets"]  # (n_seq, 5, 5)
    oracle_preds = oracle_data["oracle_preds"]  # (n_seq, n_samples, 5, 5)

    results = {}

    # 4.1 Full 25x25 Correlation Matrix
    print("  4.1 Full 25x25 Correlation Matrix...")

    # Flatten to (n_seq, 25)
    gt_flat = gt_targets.reshape(-1, 25)

    # For oracle, use mean of samples
    oracle_means = oracle_preds.mean(axis=1).reshape(-1, 25)

    gt_corr = np.corrcoef(gt_flat.T)  # (25, 25)
    oracle_corr = np.corrcoef(oracle_means.T)  # (25, 25)

    # Frobenius norm of difference
    frob_norm = np.linalg.norm(oracle_corr - gt_corr)

    # Mean absolute correlation (excluding diagonal)
    gt_mean_abs_corr = np.abs(gt_corr[~np.eye(25, dtype=bool)]).mean()
    oracle_mean_abs_corr = np.abs(oracle_corr[~np.eye(25, dtype=bool)]).mean()

    # Key pairs: ATM (12) vs OTM (0), ATM (12) vs ITM (24)
    atm_idx = 12  # (2,2) flattened
    otm_idx = 0  # (0,0) flattened
    itm_idx = 24  # (4,4) flattened

    results["correlation_matrix"] = {
        "frobenius_norm_diff": float(frob_norm),
        "gt_mean_abs_corr": float(gt_mean_abs_corr),
        "oracle_mean_abs_corr": float(oracle_mean_abs_corr),
        "correlation_preserved": float(oracle_mean_abs_corr / (gt_mean_abs_corr + 1e-8)),
        "gt_atm_otm_corr": float(gt_corr[atm_idx, otm_idx]),
        "oracle_atm_otm_corr": float(oracle_corr[atm_idx, otm_idx]),
        "gt_atm_itm_corr": float(gt_corr[atm_idx, itm_idx]),
        "oracle_atm_itm_corr": float(oracle_corr[atm_idx, itm_idx]),
    }

    # 4.2 Eigenvalue Spectrum
    print("  4.2 Eigenvalue Spectrum...")

    gt_eigenvals = np.linalg.eigvalsh(gt_corr)[::-1]  # Sorted descending
    oracle_eigenvals = np.linalg.eigvalsh(oracle_corr)[::-1]

    # Effective rank (eigenvalues > 0.1)
    gt_eff_rank = np.sum(gt_eigenvals > 0.1)
    oracle_eff_rank = np.sum(oracle_eigenvals > 0.1)

    results["eigenvalue_spectrum"] = {
        "gt_eigenvalues": gt_eigenvals.tolist(),
        "oracle_eigenvalues": oracle_eigenvals.tolist(),
        "gt_effective_rank": int(gt_eff_rank),
        "oracle_effective_rank": int(oracle_eff_rank),
        "gt_top_eigenvalue": float(gt_eigenvals[0]),
        "oracle_top_eigenvalue": float(oracle_eigenvals[0]),
        "gt_top3_variance_explained": float(sum(gt_eigenvals[:3]) / sum(gt_eigenvals)),
        "oracle_top3_variance_explained": float(
            sum(oracle_eigenvals[:3]) / sum(oracle_eigenvals)
        ),
    }

    # 4.3 Adjacent Grid Correlation
    print("  4.3 Adjacent Grid Correlation...")

    def get_adj_corr(corr_matrix, direction):
        """Get correlations between adjacent grid points."""
        adj_corrs = []
        for i in range(5):
            for j in range(5):
                idx1 = i * 5 + j
                if direction == "horizontal" and j < 4:
                    idx2 = i * 5 + (j + 1)
                    adj_corrs.append(corr_matrix[idx1, idx2])
                elif direction == "vertical" and i < 4:
                    idx2 = (i + 1) * 5 + j
                    adj_corrs.append(corr_matrix[idx1, idx2])
        return np.array(adj_corrs)

    gt_h_adj = get_adj_corr(gt_corr, "horizontal")
    gt_v_adj = get_adj_corr(gt_corr, "vertical")
    oracle_h_adj = get_adj_corr(oracle_corr, "horizontal")
    oracle_v_adj = get_adj_corr(oracle_corr, "vertical")

    results["adjacent_correlation"] = {
        "gt_horizontal_mean": float(gt_h_adj.mean()),
        "oracle_horizontal_mean": float(oracle_h_adj.mean()),
        "gt_vertical_mean": float(gt_v_adj.mean()),
        "oracle_vertical_mean": float(oracle_v_adj.mean()),
        "horizontal_preserved": float(oracle_h_adj.mean() / (gt_h_adj.mean() + 1e-8)),
        "vertical_preserved": float(oracle_v_adj.mean() / (gt_v_adj.mean() + 1e-8)),
    }

    # 4.4 Learned Covariance Analysis
    print("  4.4 Learned Covariance Analysis...")

    device = config["device"]
    model = model.to(device)
    learned_cov = model.decoder.get_covariance_matrix().detach().cpu().numpy()

    # Convert to correlation
    d = np.sqrt(np.diag(learned_cov))
    learned_corr = learned_cov / (d[:, None] * d[None, :] + 1e-8)

    results["learned_covariance"] = {
        "diagonal_variance": np.diag(learned_cov).tolist(),
        "diagonal_var_min": float(np.diag(learned_cov).min()),
        "diagonal_var_max": float(np.diag(learned_cov).max()),
        "learned_vs_gt_corr_frob": float(np.linalg.norm(learned_corr - gt_corr)),
        "learned_mean_abs_corr": float(
            np.abs(learned_corr[~np.eye(25, dtype=bool)]).mean()
        ),
    }

    # 4.5 Create Correlation Matrix Visualization
    print("  4.5 Creating Correlation Matrix Visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(gt_corr, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title("GT Correlation (25x25)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(oracle_corr, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_title("Oracle Sample Correlation (25x25)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(learned_corr, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_title("Learned Covariance Correlation (25x25)")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrices.png", dpi=150)
    plt.close()

    # Save results
    with open(output_dir / "spatial_correlation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'spatial_correlation.json'}")
    return results


# ============================================================================
# Section 5: Temporal Dynamics Analysis
# ============================================================================


def analyze_temporal_dynamics(log_returns, oracle_data, output_dir):
    """Section 5: Autocorrelation, mean-reversion, path roughness."""
    print("\n=== Section 5: Temporal Dynamics Analysis ===")

    gt_targets = oracle_data["gt_targets"]  # (n_seq, 5, 5)
    oracle_preds = oracle_data["oracle_preds"]  # (n_seq, n_samples, 5, 5)

    # Use ATM point for temporal analysis
    gt_atm = log_returns[:, 2, 2]  # Full time series
    oracle_atm_mean = oracle_preds[:, :, 2, 2].mean(axis=1)  # (n_seq,)

    results = {}

    # 5.1 Autocorrelation Structure
    print("  5.1 Autocorrelation Structure...")

    lags = [1, 5, 10, 20, 60]
    gt_acf = []
    oracle_acf = []

    for lag in lags:
        if len(gt_atm) > lag:
            gt_corr = np.corrcoef(gt_atm[:-lag], gt_atm[lag:])[0, 1]
            gt_acf.append(float(gt_corr) if np.isfinite(gt_corr) else 0.0)
        else:
            gt_acf.append(0.0)

        if len(oracle_atm_mean) > lag:
            oracle_corr = np.corrcoef(oracle_atm_mean[:-lag], oracle_atm_mean[lag:])[
                0, 1
            ]
            oracle_acf.append(float(oracle_corr) if np.isfinite(oracle_corr) else 0.0)
        else:
            oracle_acf.append(0.0)

    results["autocorrelation"] = {
        "lags": lags,
        "gt_acf": gt_acf,
        "oracle_acf": oracle_acf,
        "acf_mse": float(np.mean((np.array(gt_acf) - np.array(oracle_acf)) ** 2)),
    }

    # 5.2 Mean-Reversion Speed (AR(1) fit)
    print("  5.2 Mean-Reversion Speed...")

    # Fit AR(1): x_t = alpha + beta * x_{t-1}
    from sklearn.linear_model import LinearRegression

    X_gt = gt_atm[:-1].reshape(-1, 1)
    y_gt = gt_atm[1:]
    ar1_gt = LinearRegression()
    ar1_gt.fit(X_gt, y_gt)
    gt_beta = ar1_gt.coef_[0]
    gt_mean_reversion = 1 - gt_beta

    X_oracle = oracle_atm_mean[:-1].reshape(-1, 1)
    y_oracle = oracle_atm_mean[1:]
    ar1_oracle = LinearRegression()
    ar1_oracle.fit(X_oracle, y_oracle)
    oracle_beta = ar1_oracle.coef_[0]
    oracle_mean_reversion = 1 - oracle_beta

    results["mean_reversion"] = {
        "gt_ar1_beta": float(gt_beta),
        "gt_mean_reversion_speed": float(gt_mean_reversion),
        "oracle_ar1_beta": float(oracle_beta),
        "oracle_mean_reversion_speed": float(oracle_mean_reversion),
        "interpretation": "Higher speed = faster mean reversion. beta < 1 = mean-reverting.",
    }

    # 5.3 Path Roughness
    print("  5.3 Path Roughness...")

    gt_roughness = np.var(np.diff(gt_atm)) / (np.var(gt_atm) + 1e-8)

    # For oracle, compute roughness of the sequence of means
    oracle_roughness = np.var(np.diff(oracle_atm_mean)) / (
        np.var(oracle_atm_mean) + 1e-8
    )

    results["path_roughness"] = {
        "gt_roughness": float(gt_roughness),
        "oracle_roughness": float(oracle_roughness),
        "roughness_ratio": float(oracle_roughness / (gt_roughness + 1e-8)),
        "interpretation": "Ratio > 1 means oracle paths are rougher than GT.",
    }

    # 5.4 Variance Ratio Test
    print("  5.4 Variance Ratio Test...")

    horizons = [1, 5, 10, 20, 30]
    gt_vr = []
    oracle_vr = []

    for k in horizons:
        if len(gt_atm) > k:
            # VR(k) = Var(x_t - x_{t-k}) / (k * Var(x_t - x_{t-1}))
            gt_diff_k = gt_atm[k:] - gt_atm[:-k]
            gt_diff_1 = np.diff(gt_atm)
            vr_gt = np.var(gt_diff_k) / (k * np.var(gt_diff_1) + 1e-8)
            gt_vr.append(float(vr_gt))
        else:
            gt_vr.append(1.0)

        if len(oracle_atm_mean) > k:
            oracle_diff_k = oracle_atm_mean[k:] - oracle_atm_mean[:-k]
            oracle_diff_1 = np.diff(oracle_atm_mean)
            vr_oracle = np.var(oracle_diff_k) / (k * np.var(oracle_diff_1) + 1e-8)
            oracle_vr.append(float(vr_oracle))
        else:
            oracle_vr.append(1.0)

    results["variance_ratio"] = {
        "horizons": horizons,
        "gt_variance_ratio": gt_vr,
        "oracle_variance_ratio": oracle_vr,
        "interpretation": (
            "VR=1: random walk. VR<1: mean-reversion. VR>1: momentum."
        ),
    }

    # Save results
    with open(output_dir / "temporal_dynamics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'temporal_dynamics.json'}")
    return results


# ============================================================================
# Section 6: Failure Mode Catalog
# ============================================================================


def analyze_failure_modes(oracle_data, log_returns, log_surfaces, output_dir):
    """Section 6: Systematic bias, worst cases, crisis periods."""
    print("\n=== Section 6: Failure Mode Catalog ===")

    gt_targets = oracle_data["gt_targets"]  # (n_seq, 5, 5)
    oracle_preds = oracle_data["oracle_preds"]  # (n_seq, n_samples, 5, 5)
    oracle_means = oracle_preds.mean(axis=1)  # (n_seq, 5, 5)

    results = {}

    # 6.1 Systematic Bias Analysis
    print("  6.1 Systematic Bias Analysis...")

    # Per grid point bias
    bias_per_grid = (oracle_means - gt_targets).mean(axis=0)  # (5, 5)

    # Overall bias
    overall_bias = (oracle_means - gt_targets).mean()

    results["systematic_bias"] = {
        "bias_per_grid": bias_per_grid.tolist(),
        "overall_bias": float(overall_bias),
        "bias_min": float(bias_per_grid.min()),
        "bias_max": float(bias_per_grid.max()),
        "atm_bias": float(bias_per_grid[2, 2]),
        "note": "Negative bias = systematic underestimation",
    }

    # 6.2 Worst-Case Reconstructions
    print("  6.2 Worst-Case Reconstructions...")

    # MSE per sequence
    mse_per_seq = ((oracle_means - gt_targets) ** 2).mean(axis=(1, 2))  # (n_seq,)

    worst_indices = np.argsort(mse_per_seq)[-20:]
    best_indices = np.argsort(mse_per_seq)[:20]

    results["worst_cases"] = {
        "worst_20_mse": mse_per_seq[worst_indices].tolist(),
        "worst_20_indices": worst_indices.tolist(),
        "best_20_mse": mse_per_seq[best_indices].tolist(),
        "best_20_indices": best_indices.tolist(),
        "mse_mean": float(mse_per_seq.mean()),
        "mse_std": float(mse_per_seq.std()),
        "mse_p95": float(np.percentile(mse_per_seq, 95)),
        "mse_p99": float(np.percentile(mse_per_seq, 99)),
    }

    # 6.3 Analyze Worst Cases Pattern
    print("  6.3 Analyzing Worst Case Patterns...")

    # Look at GT values for worst cases
    worst_gt = gt_targets[worst_indices]
    best_gt = gt_targets[best_indices]

    results["worst_case_patterns"] = {
        "worst_gt_mean": float(worst_gt.mean()),
        "worst_gt_std": float(worst_gt.std()),
        "best_gt_mean": float(best_gt.mean()),
        "best_gt_std": float(best_gt.std()),
        "worst_gt_abs_mean": float(np.abs(worst_gt).mean()),
        "best_gt_abs_mean": float(np.abs(best_gt).mean()),
        "interpretation": (
            "If worst_gt_abs_mean >> best_gt_abs_mean, "
            "model fails on extreme moves"
        ),
    }

    # 6.4 Extreme Move Detection
    print("  6.4 Extreme Move Detection...")

    # Days with |log-return| > 3*std at ATM
    atm_gt = gt_targets[:, 2, 2]
    atm_std = atm_gt.std()
    extreme_mask = np.abs(atm_gt) > 3 * atm_std

    n_extreme = extreme_mask.sum()
    n_normal = (~extreme_mask).sum()

    if n_extreme > 0 and n_normal > 0:
        extreme_mse = mse_per_seq[extreme_mask].mean()
        normal_mse = mse_per_seq[~extreme_mask].mean()

        results["extreme_moves"] = {
            "n_extreme_days": int(n_extreme),
            "n_normal_days": int(n_normal),
            "extreme_mse_mean": float(extreme_mse),
            "normal_mse_mean": float(normal_mse),
            "extreme_vs_normal_ratio": float(extreme_mse / (normal_mse + 1e-8)),
            "extreme_threshold_sigma": 3.0,
        }
    else:
        results["extreme_moves"] = {
            "n_extreme_days": int(n_extreme),
            "n_normal_days": int(n_normal),
            "note": "Not enough extreme days for comparison",
        }

    # 6.5 CI Coverage Analysis
    print("  6.5 CI Coverage Analysis...")

    # 90% CI coverage per grid point
    violations_per_grid = np.zeros((5, 5))
    ci_width_per_grid = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_vals = gt_targets[:, i, j]
            oracle_samples = oracle_preds[:, :, i, j]  # (n_seq, n_samples)

            # 5th and 95th percentiles from oracle samples
            p5 = np.percentile(oracle_samples, 5, axis=1)
            p95 = np.percentile(oracle_samples, 95, axis=1)

            # Violations: GT outside [p5, p95]
            violations = (gt_vals < p5) | (gt_vals > p95)
            violations_per_grid[i, j] = violations.mean()
            ci_width_per_grid[i, j] = (p95 - p5).mean()

    results["ci_coverage"] = {
        "violations_per_grid": violations_per_grid.tolist(),
        "mean_violation_rate": float(violations_per_grid.mean()),
        "target_violation_rate": 0.10,
        "ci_width_per_grid": ci_width_per_grid.tolist(),
        "atm_violation_rate": float(violations_per_grid[2, 2]),
        "atm_ci_width": float(ci_width_per_grid[2, 2]),
    }

    # Save results
    with open(output_dir / "failure_modes.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'failure_modes.json'}")
    return results


# ============================================================================
# Generate Comprehensive Report
# ============================================================================


def generate_report(output_dir, all_results):
    """Generate markdown summary report."""
    print("\n=== Generating Comprehensive Report ===")

    report = [
        "# Two-Stage VAE Oracle Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
        "## Executive Summary\n",
    ]

    # Latent Space
    ls = all_results.get("latent_space", {})
    du = ls.get("dimension_utilization", {})
    report.extend([
        "### 1. Latent Space Analysis\n",
        f"- **Latent Dimension**: {ls.get('latent_dim', 'N/A')}",
        f"- **Collapsed Dimensions**: {du.get('collapsed_dims_count', 'N/A')} (threshold: logvar < -4)",
        f"- **Total KL**: {du.get('total_kl', 'N/A'):.4f}",
        f"- **Avg Activation Rate**: {du.get('avg_activation_rate', 'N/A'):.2%}",
        "",
        "**Information Content (R² from z → features):**",
    ])
    ic = ls.get("information_content", {}).get("r2_scores", {})
    for k, v in ic.items():
        report.append(f"  - {k}: {v:.4f}")
    report.append("")

    # Decoder Expressivity
    de = all_results.get("decoder_expressivity", {})
    oracle_recon = de.get("oracle_reconstruction", {})
    report.extend([
        "### 2. Decoder Expressivity Analysis\n",
        f"- **Oracle MSE**: {oracle_recon.get('oracle_mse_mean', 'N/A'):.6f}",
        f"- **Zero-z MSE**: {oracle_recon.get('zero_z_mse_mean', 'N/A'):.6f}",
        f"- **z Contribution**: {oracle_recon.get('z_contribution_percent', 'N/A'):.1f}%",
        "",
        "**Architecture:**",
    ])
    arch = de.get("architecture", {})
    report.extend([
        f"  - Bottleneck: {arch.get('bottleneck_ratio', 'N/A')}",
        f"  - Severity: {arch.get('bottleneck_severity', 'N/A')}",
        "",
    ])

    # Distribution Shape
    ds = all_results.get("distribution_shape", {})
    atm = ds.get("atm_detailed", {})
    report.extend([
        "### 3. Distribution Shape Diagnostics\n",
        "**ATM Point (2,2):**",
        f"- GT Kurtosis: {atm.get('gt_kurtosis', 'N/A'):.2f} | Oracle: {atm.get('oracle_kurtosis', 'N/A'):.2f}",
        f"- Kurtosis Recovery: {atm.get('kurtosis_ratio', 'N/A'):.2%}",
        f"- GT Skewness: {atm.get('gt_skewness', 'N/A'):.3f} | Oracle: {atm.get('oracle_skewness', 'N/A'):.3f}",
        "",
    ])
    pgs = ds.get("per_grid_stats", {})
    report.extend([
        f"- Mean Kurtosis Recovery: {pgs.get('mean_kurtosis_recovery', 'N/A'):.2%}",
        f"- Skewness Sign Match Rate: {pgs.get('skewness_match_rate', 'N/A'):.2%}",
        "",
    ])

    # Spatial Correlation
    sc = all_results.get("spatial_correlation", {})
    cm = sc.get("correlation_matrix", {})
    report.extend([
        "### 4. Spatial Correlation Analysis\n",
        f"- **Frobenius Norm Diff**: {cm.get('frobenius_norm_diff', 'N/A'):.2f}",
        f"- GT Mean Abs Corr: {cm.get('gt_mean_abs_corr', 'N/A'):.3f}",
        f"- Oracle Mean Abs Corr: {cm.get('oracle_mean_abs_corr', 'N/A'):.3f}",
        f"- Correlation Preserved: {cm.get('correlation_preserved', 'N/A'):.2%}",
        "",
    ])
    es = sc.get("eigenvalue_spectrum", {})
    report.extend([
        f"- GT Effective Rank: {es.get('gt_effective_rank', 'N/A')}",
        f"- Oracle Effective Rank: {es.get('oracle_effective_rank', 'N/A')}",
        "",
    ])

    # Temporal Dynamics
    td = all_results.get("temporal_dynamics", {})
    mr = td.get("mean_reversion", {})
    pr = td.get("path_roughness", {})
    report.extend([
        "### 5. Temporal Dynamics Analysis\n",
        f"- GT Mean-Reversion Speed: {mr.get('gt_mean_reversion_speed', 'N/A'):.4f}",
        f"- Oracle Mean-Reversion Speed: {mr.get('oracle_mean_reversion_speed', 'N/A'):.4f}",
        f"- GT Path Roughness: {pr.get('gt_roughness', 'N/A'):.4f}",
        f"- Oracle Path Roughness: {pr.get('oracle_roughness', 'N/A'):.4f}",
        f"- Roughness Ratio: {pr.get('roughness_ratio', 'N/A'):.2f}x",
        "",
    ])

    # Failure Modes
    fm = all_results.get("failure_modes", {})
    sb = fm.get("systematic_bias", {})
    ci = fm.get("ci_coverage", {})
    report.extend([
        "### 6. Failure Mode Catalog\n",
        f"- **Overall Bias**: {sb.get('overall_bias', 'N/A'):.6f}",
        f"- **ATM Bias**: {sb.get('atm_bias', 'N/A'):.6f}",
        f"- **Mean CI Violation Rate**: {ci.get('mean_violation_rate', 'N/A'):.2%}",
        f"- **Target**: 10%",
        "",
    ])
    em = fm.get("extreme_moves", {})
    if em.get("n_extreme_days", 0) > 0:
        report.extend([
            f"- Extreme Days MSE: {em.get('extreme_mse_mean', 'N/A'):.6f}",
            f"- Normal Days MSE: {em.get('normal_mse_mean', 'N/A'):.6f}",
            f"- Extreme/Normal Ratio: {em.get('extreme_vs_normal_ratio', 'N/A'):.2f}x",
            "",
        ])

    # Summary Table
    report.extend([
        "---\n",
        "## Summary Metrics Table\n",
        "| Metric | GT | Oracle | Gap |",
        "|--------|----|----|-----|",
    ])

    # Add key metrics
    if atm:
        report.append(
            f"| Kurtosis (ATM) | {atm.get('gt_kurtosis', 0):.1f} | "
            f"{atm.get('oracle_kurtosis', 0):.1f} | "
            f"{atm.get('kurtosis_ratio', 0):.1%} |"
        )
        report.append(
            f"| Skewness (ATM) | {atm.get('gt_skewness', 0):.2f} | "
            f"{atm.get('oracle_skewness', 0):.2f} | - |"
        )

    if cm:
        report.append(
            f"| Cross-grid Corr | {cm.get('gt_mean_abs_corr', 0):.3f} | "
            f"{cm.get('oracle_mean_abs_corr', 0):.3f} | "
            f"{cm.get('correlation_preserved', 0):.1%} |"
        )

    if pr:
        report.append(
            f"| Path Roughness | {pr.get('gt_roughness', 0):.4f} | "
            f"{pr.get('oracle_roughness', 0):.4f} | "
            f"{pr.get('roughness_ratio', 0):.2f}x |"
        )

    report.extend([
        "",
        "---\n",
        "## Recommendations\n",
        "1. **Increase decoder capacity**: Current 8-dim LSTM bottleneck limits z expressivity",
        "2. **Address kurtosis gap**: Consider Student-t decoder with fixed nu from GT",
        "3. **Improve correlation**: Low-rank covariance helps but rank may need increase",
        "4. **Reduce bias**: Investigate systematic negative bias in predictions",
        "",
    ])

    # Write report
    report_path = output_dir / "comprehensive_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"  Report saved to {report_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run comprehensive analysis."""
    print("=" * 70)
    print("Two-Stage VAE Oracle Case: Comprehensive Analysis")
    print("=" * 70)

    # Setup
    output_dir = setup_paths()
    print(f"\nOutput directory: {output_dir}")

    # Load data
    print("\nLoading data...")
    surfaces, log_returns, log_surfaces = load_data()
    print(f"  Surfaces shape: {surfaces.shape}")
    print(f"  Log-returns shape: {log_returns.shape}")

    # Load model (config is loaded from checkpoint)
    print("\nLoading model...")
    model, config = load_model()
    print(f"  Model loaded successfully")
    print(f"  Using device: {config['device']}")
    print(f"  latent_dim: {config.get('latent_dim', 'N/A')}")
    print(f"  context_len: {config.get('context_len', 'N/A')}")

    # Extract oracle samples
    print("\nExtracting oracle samples...")
    oracle_data = extract_oracle_samples(
        model, log_returns, config, n_sequences=500, n_samples=100
    )
    print(f"  z_means shape: {oracle_data['z_means'].shape}")
    print(f"  oracle_preds shape: {oracle_data['oracle_preds'].shape}")
    print(f"  gt_targets shape: {oracle_data['gt_targets'].shape}")

    # Run all analysis sections
    all_results = {}

    all_results["latent_space"] = analyze_latent_space(oracle_data, log_returns, output_dir)

    all_results["decoder_expressivity"] = analyze_decoder_expressivity(
        model, oracle_data, log_returns, config, output_dir
    )

    all_results["distribution_shape"] = analyze_distribution_shape(
        oracle_data, log_returns, output_dir
    )

    all_results["spatial_correlation"] = analyze_spatial_correlation(
        model, oracle_data, log_returns, config, output_dir
    )

    all_results["temporal_dynamics"] = analyze_temporal_dynamics(
        log_returns, oracle_data, output_dir
    )

    all_results["failure_modes"] = analyze_failure_modes(
        oracle_data, log_returns, log_surfaces, output_dir
    )

    # Generate comprehensive report
    generate_report(output_dir, all_results)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
