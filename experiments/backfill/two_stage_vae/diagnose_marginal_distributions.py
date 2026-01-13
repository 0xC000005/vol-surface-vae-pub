"""
Diagnose Marginal Distributions: Risk Awareness and Unconditional Match

Tests:
1. Conditional Marginal Sensitivity (Risk Awareness)
   - Does CI width change with market regime?
   - Expected: Crisis > High Vol > Normal > Low Vol

2. Unconditional Marginal Comparison
   - Pool all model predictions vs all GT 30-day windows
   - Compare mean, std, skew, kurtosis at each horizon

3. Mean Trajectory Analysis
   - Why do sample means show no trend?

Usage:
    python experiments/backfill/two_stage_vae/diagnose_marginal_distributions.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictor


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_models(device: str = "cuda"):
    """Load the Student-t VAE and trained predictor."""
    # Load VAE
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load predictor
    pred_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    pred_ckpt = torch.load(pred_path, map_location=device, weights_only=False)

    predictor = LatentPredictor(config)
    predictor.load_state_dict(pred_ckpt["predictor_state_dict"])
    predictor = predictor.to(device)
    predictor.eval()

    return vae, predictor, config


def categorize_by_regime(surfaces: np.ndarray, context_len: int = 20):
    """
    Categorize indices by ATM IV level at the END of context window.

    Returns dict: regime_name -> list of valid start indices
    """
    # ATM IV is at grid position [2, 2]
    atm_iv = surfaces[:, 2, 2]

    regimes = {
        "Low Vol (<15%)": [],
        "Normal (15-25%)": [],
        "High Vol (25-40%)": [],
        "Crisis (>40%)": [],
    }

    # Valid start indices: need context_len + horizon days after
    max_start = len(surfaces) - context_len - 30

    for start_idx in range(max_start):
        # IV at end of context window
        context_end_idx = start_idx + context_len - 1
        iv = atm_iv[context_end_idx]

        if iv < 0.15:
            regimes["Low Vol (<15%)"].append(start_idx)
        elif iv < 0.25:
            regimes["Normal (15-25%)"].append(start_idx)
        elif iv < 0.40:
            regimes["High Vol (25-40%)"].append(start_idx)
        else:
            regimes["Crisis (>40%)"].append(start_idx)

    return regimes


def generate_trajectories_predictor(vae, predictor, log_returns, start_idx,
                                     context_len=20, horizon=30, n_samples=50, device="cuda"):
    """
    Generate trajectories using Prior Predictor mode.
    Returns log-returns (not levels) for each sample at each horizon.
    """
    context = log_returns[start_idx:start_idx + context_len]
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    sample_returns = np.zeros((n_samples, horizon))

    with torch.no_grad():
        # Get context embedding
        ctx_emb_context = vae.ctx_encoder({"surface": context_tensor})

        # Get z for context from encoder
        z_ctx_mean, z_ctx_logvar, _ = vae.main_encoder({"surface": context_tensor})

        # Get predicted z for future from predictor
        z_future_mean, z_future_logvar = predictor(context_tensor, horizon=horizon)

        for s in range(n_samples):
            # Sample z for context
            z_ctx = z_ctx_mean + torch.exp(0.5 * z_ctx_logvar) * torch.randn_like(z_ctx_mean)

            # Sample z for each future step from predictor
            z_future = z_future_mean + torch.exp(0.5 * z_future_logvar) * torch.randn_like(z_future_mean)

            for h in range(horizon):
                # Build sequence up to this horizon
                ctx_dim = ctx_emb_context.shape[-1]
                ctx_emb_future = torch.zeros(1, h + 1, ctx_dim, device=device)
                ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

                z = torch.cat([z_ctx, z_future[:, :h+1]], dim=1)

                # Decode
                _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)
                return_h = sample[0, -1, 2, 2].cpu().numpy()
                sample_returns[s, h] = return_h

    return sample_returns


def test_conditional_sensitivity(vae, predictor, log_returns, surfaces,
                                  n_contexts_per_regime=30, n_samples=50,
                                  context_len=20, horizon=30, device="cuda"):
    """
    Test 1: Does CI width change with market regime?
    """
    print("\n" + "=" * 70)
    print("TEST 1: CONDITIONAL MARGINAL SENSITIVITY (RISK AWARENESS)")
    print("=" * 70)

    regimes = categorize_by_regime(surfaces, context_len)

    # Print regime counts
    print("\nRegime distribution:")
    for regime, indices in regimes.items():
        print(f"  {regime}: {len(indices)} contexts")

    results = {}

    for regime, indices in regimes.items():
        if len(indices) < n_contexts_per_regime:
            print(f"\nSkipping {regime}: only {len(indices)} contexts available")
            continue

        print(f"\nProcessing {regime}...")

        # Sample contexts
        np.random.seed(42)  # Reproducibility
        sampled_indices = np.random.choice(indices, size=min(n_contexts_per_regime, len(indices)), replace=False)

        all_returns = []

        for i, start_idx in enumerate(sampled_indices):
            if (i + 1) % 10 == 0:
                print(f"  Context {i+1}/{len(sampled_indices)}")

            returns = generate_trajectories_predictor(
                vae, predictor, log_returns, start_idx,
                context_len=context_len, horizon=horizon,
                n_samples=n_samples, device=device
            )
            all_returns.append(returns)

        all_returns = np.array(all_returns)  # (n_contexts, n_samples, horizon)

        # Compute CI width at each horizon
        ci_widths = []
        for h in range(horizon):
            # Pool all samples at this horizon
            returns_h = all_returns[:, :, h].flatten()  # (n_contexts * n_samples,)
            p05 = np.percentile(returns_h, 5)
            p95 = np.percentile(returns_h, 95)
            ci_widths.append(p95 - p05)

        # Compute mean, std of trajectories
        mean_returns = all_returns.mean(axis=(0, 1))  # (horizon,)
        std_returns = all_returns.std(axis=(0, 1))  # (horizon,)

        results[regime] = {
            "ci_widths": np.array(ci_widths),
            "mean_returns": mean_returns,
            "std_returns": std_returns,
            "all_returns": all_returns,
            "n_contexts": len(sampled_indices),
        }

        print(f"  CI width at h=1: {ci_widths[0]:.4f}")
        print(f"  CI width at h=30: {ci_widths[-1]:.4f}")
        print(f"  Mean return at h=30: {mean_returns[-1]:.4f}")

    return results


def test_unconditional_marginal(vae, predictor, log_returns, surfaces,
                                 n_contexts=100, n_samples=20,
                                 context_len=20, horizon=30, device="cuda"):
    """
    Test 2: Does unconditional marginal match GT?
    """
    print("\n" + "=" * 70)
    print("TEST 2: UNCONDITIONAL MARGINAL COMPARISON")
    print("=" * 70)

    # === GT: Stack all 30-day rolling windows ===
    print("\nComputing GT unconditional marginal...")
    gt_returns = {}
    n_windows = len(log_returns) - horizon - context_len

    for h in range(1, horizon + 1):
        gt_returns[h] = []

    # ATM IV log-returns
    atm_log_returns = log_returns[:, 2, 2]

    for t in range(context_len, len(atm_log_returns) - horizon):
        for h in range(1, horizon + 1):
            # Cumulative return from t to t+h
            cum_return = atm_log_returns[t:t+h].sum()
            gt_returns[h].append(cum_return)

    gt_stats = {}
    for h in [1, 7, 14, 30]:
        arr = np.array(gt_returns[h])
        gt_stats[h] = {
            "mean": arr.mean(),
            "std": arr.std(),
            "skew": stats.skew(arr),
            "kurtosis": stats.kurtosis(arr),
            "data": arr,
        }

    print(f"  GT windows: {len(gt_returns[1])}")

    # === Model: Sample random contexts ===
    print(f"\nComputing Model unconditional marginal ({n_contexts} contexts x {n_samples} samples)...")

    np.random.seed(42)
    max_start = len(log_returns) - context_len - horizon
    sampled_starts = np.random.choice(max_start, size=n_contexts, replace=False)

    model_returns = {h: [] for h in range(1, horizon + 1)}

    for i, start_idx in enumerate(sampled_starts):
        if (i + 1) % 20 == 0:
            print(f"  Context {i+1}/{n_contexts}")

        returns = generate_trajectories_predictor(
            vae, predictor, log_returns, start_idx,
            context_len=context_len, horizon=horizon,
            n_samples=n_samples, device=device
        )

        # Accumulate returns to get cumulative returns
        cum_returns = np.cumsum(returns, axis=1)

        for h in range(1, horizon + 1):
            model_returns[h].extend(cum_returns[:, h-1].tolist())

    model_stats = {}
    for h in [1, 7, 14, 30]:
        arr = np.array(model_returns[h])
        model_stats[h] = {
            "mean": arr.mean(),
            "std": arr.std(),
            "skew": stats.skew(arr),
            "kurtosis": stats.kurtosis(arr),
            "data": arr,
        }

    # === Print comparison ===
    print("\n" + "-" * 70)
    print("UNCONDITIONAL MARGINAL COMPARISON (ATM IV cumulative log-returns)")
    print("-" * 70)
    print(f"{'Horizon':<10} {'Metric':<10} {'GT':<12} {'Model':<12} {'Diff':<12}")
    print("-" * 70)

    for h in [1, 7, 14, 30]:
        for metric in ["mean", "std", "skew", "kurtosis"]:
            gt_val = gt_stats[h][metric]
            model_val = model_stats[h][metric]
            diff = model_val - gt_val
            print(f"{h:<10} {metric:<10} {gt_val:<12.4f} {model_val:<12.4f} {diff:<+12.4f}")
        print("-" * 70)

    return gt_stats, model_stats


def plot_conditional_sensitivity(results, output_path):
    """Plot CI width by regime."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "Low Vol (<15%)": "#27ae60",
        "Normal (15-25%)": "#3498db",
        "High Vol (25-40%)": "#f39c12",
        "Crisis (>40%)": "#e74c3c",
    }

    horizon = 30
    days = np.arange(1, horizon + 1)

    # Plot 1: CI width by horizon
    ax = axes[0]
    for regime, data in results.items():
        ax.plot(days, data["ci_widths"], label=regime, color=colors[regime], linewidth=2)

    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("90% CI Width (log-return)")
    ax.set_title("CI Width by Market Regime\n(Should increase with volatility if risk-aware)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: CI width at h=30 by regime (bar chart)
    ax = axes[1]
    regime_names = list(results.keys())
    ci_30 = [results[r]["ci_widths"][-1] for r in regime_names]
    bar_colors = [colors[r] for r in regime_names]

    bars = ax.bar(range(len(regime_names)), ci_30, color=bar_colors)
    ax.set_xticks(range(len(regime_names)))
    ax.set_xticklabels([r.split()[0] for r in regime_names], rotation=15)
    ax.set_ylabel("90% CI Width at h=30")
    ax.set_title("CI Width at 30-Day Horizon by Regime")
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, ci_30):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_unconditional_comparison(gt_stats, model_stats, output_path):
    """Plot GT vs Model unconditional marginals."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    horizons = [1, 7, 14, 30]

    # Row 1: Histogram comparison
    for i, h in enumerate(horizons):
        ax = axes[0, i]
        gt_data = gt_stats[h]["data"]
        model_data = model_stats[h]["data"]

        # Compute common bins
        all_data = np.concatenate([gt_data, model_data])
        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)

        ax.hist(gt_data, bins=bins, alpha=0.5, density=True, label="GT", color="black")
        ax.hist(model_data, bins=bins, alpha=0.5, density=True, label="Model", color="blue")

        ax.set_xlabel("Cumulative Log-Return")
        ax.set_ylabel("Density")
        ax.set_title(f"Horizon {h} days")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Q-Q plots
    for i, h in enumerate(horizons):
        ax = axes[1, i]
        gt_data = gt_stats[h]["data"]
        model_data = model_stats[h]["data"]

        # Q-Q plot: GT quantiles vs Model quantiles
        percentiles = np.linspace(1, 99, 99)
        gt_quantiles = np.percentile(gt_data, percentiles)
        model_quantiles = np.percentile(model_data, percentiles)

        ax.scatter(gt_quantiles, model_quantiles, alpha=0.5, s=20)
        lims = [min(gt_quantiles.min(), model_quantiles.min()),
                max(gt_quantiles.max(), model_quantiles.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label="y=x")

        ax.set_xlabel("GT Quantiles")
        ax.set_ylabel("Model Quantiles")
        ax.set_title(f"Q-Q Plot (h={h})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Unconditional Marginal: GT vs Model\n(ATM IV Cumulative Log-Returns)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_mean_trajectory(results, output_path):
    """Test 3: Analyze why sample means show no trend."""
    print("\n" + "=" * 70)
    print("TEST 3: MEAN TRAJECTORY ANALYSIS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    horizon = 30
    days = np.arange(1, horizon + 1)

    colors = {
        "Low Vol (<15%)": "#27ae60",
        "Normal (15-25%)": "#3498db",
        "High Vol (25-40%)": "#f39c12",
        "Crisis (>40%)": "#e74c3c",
    }

    # Plot 1: Mean trajectory by regime
    ax = axes[0]
    for regime, data in results.items():
        mean_cum = np.cumsum(data["mean_returns"])
        ax.plot(days, mean_cum, label=regime, color=colors[regime], linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean Cumulative Log-Return")
    ax.set_title("Mean Trajectory by Regime\n(Should differ if model captures regime trends)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Std of mean prediction across contexts
    ax = axes[1]
    for regime, data in results.items():
        # Compute mean trajectory for each context
        all_returns = data["all_returns"]  # (n_contexts, n_samples, horizon)
        context_means = all_returns.mean(axis=1)  # (n_contexts, horizon)
        std_of_means = context_means.std(axis=0)  # (horizon,)

        ax.plot(days, std_of_means, label=regime, color=colors[regime], linewidth=2)

    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Std of Mean Prediction")
    ax.set_title("Std of Mean Trajectory Across Contexts\n(Higher = more context-dependent mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print statistics
    print("\nMean cumulative return at h=30 by regime:")
    for regime, data in results.items():
        mean_cum_30 = np.cumsum(data["mean_returns"])[-1]
        print(f"  {regime}: {mean_cum_30:.4f}")


def main():
    print("=" * 70)
    print("MARGINAL DISTRIBUTION DIAGNOSTICS")
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

    output_dir = Path("models/backfill/two_stage/prior_network/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Conditional Sensitivity
    cond_results = test_conditional_sensitivity(
        vae, predictor, log_returns, surfaces,
        n_contexts_per_regime=30, n_samples=50,
        device=device
    )

    plot_conditional_sensitivity(
        cond_results,
        output_dir / "conditional_sensitivity.png"
    )

    # Test 2: Unconditional Marginal
    gt_stats, model_stats = test_unconditional_marginal(
        vae, predictor, log_returns, surfaces,
        n_contexts=100, n_samples=20,
        device=device
    )

    plot_unconditional_comparison(
        gt_stats, model_stats,
        output_dir / "unconditional_comparison.png"
    )

    # Test 3: Mean Trajectory Analysis
    analyze_mean_trajectory(
        cond_results,
        output_dir / "mean_trajectory_analysis.png"
    )

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Risk awareness check
    if len(cond_results) >= 2:
        ci_widths_30 = {r: d["ci_widths"][-1] for r, d in cond_results.items()}
        sorted_regimes = sorted(ci_widths_30.items(), key=lambda x: x[1])
        print("\nRisk Awareness (CI width at h=30):")
        for regime, width in sorted_regimes:
            print(f"  {regime}: {width:.4f}")

        # Check if ordering is correct
        expected_order = ["Low Vol (<15%)", "Normal (15-25%)", "High Vol (25-40%)", "Crisis (>40%)"]
        actual_order = [r for r, _ in sorted_regimes]
        is_risk_aware = all(
            expected_order.index(actual_order[i]) <= expected_order.index(actual_order[i+1])
            for i in range(len(actual_order)-1)
            if actual_order[i] in expected_order and actual_order[i+1] in expected_order
        )
        print(f"\n  Risk-aware ordering: {'YES' if is_risk_aware else 'NO'}")
        print(f"  Expected: Low Vol < Normal < High Vol < Crisis")
        print(f"  Actual: {' < '.join([r.split()[0] for r, _ in sorted_regimes])}")

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
