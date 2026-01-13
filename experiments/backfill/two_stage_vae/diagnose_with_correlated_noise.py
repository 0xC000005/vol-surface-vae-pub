"""
Diagnostic Plots with Correlated Decoder Noise

Generate the same diagnostic plots as before to verify if issues are resolved:
1. Conditional Fan Chart (50 sample trajectories per regime)
2. Conditional Sensitivity Mean Trajectory Analysis
3. Unconditional Marginal Comparison

Usage:
    python experiments/backfill/two_stage_vae/diagnose_with_correlated_noise.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictorCov


def to_log_returns(surfaces: np.ndarray):
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def build_mean_reverting_cholesky(T: int, rho: float, device: str = "cuda"):
    """Build Cholesky factor for correlation structure."""
    idx = torch.arange(T, device=device, dtype=torch.float32)
    diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    Sigma = torch.pow(torch.tensor(rho, device=device), diff)
    Sigma = Sigma + 1e-4 * torch.eye(T, device=device)
    try:
        L = torch.linalg.cholesky(Sigma)
    except RuntimeError:
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        eigvals = torch.clamp(eigvals, min=1e-4)
        Sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        L = torch.linalg.cholesky(Sigma)
    return L


def decode_with_correlated_noise(decoder, z, rho_noise: float, device: str = "cuda"):
    """Decode z with correlated noise."""
    B, T, _ = z.shape

    z_flat = z.view(B * T, -1)
    mean = decoder.mean_net(z_flat).view(B, T, 5, 5)

    z_pooled = z.mean(dim=1)
    factor_flat = decoder.factor_net(z_pooled)
    factor = factor_flat.view(B, 25, decoder.rank)

    log_diag = decoder.log_diag_net(z_pooled)
    log_diag = torch.clamp(log_diag, min=-10, max=2)

    if abs(rho_noise) < 0.01:
        eps_rank = torch.randn(B, T, decoder.rank, device=device)
        eps_diag = torch.randn(B, T, 25, device=device)
    else:
        L = build_mean_reverting_cholesky(T, rho_noise, device)
        eps_rank_iid = torch.randn(B, T, decoder.rank, device=device)
        eps_rank = torch.einsum('ij,bjr->bir', L, eps_rank_iid)
        eps_diag_iid = torch.randn(B, T, 25, device=device)
        eps_diag = torch.einsum('ij,bjk->bik', L, eps_diag_iid)

    correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)
    diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
    independent_gauss = diag_std * eps_diag
    total_gauss = correlated_gauss + independent_gauss

    chi2_samples = torch.zeros(B, T, 25, device=device)
    for i in range(25):
        nu_i = decoder.nu[i].item()
        alpha = nu_i / 2
        beta = nu_i / 2
        gamma_samples = torch._standard_gamma(torch.full((B, T), alpha, device=device)) / beta
        chi2_samples[:, :, i] = gamma_samples

    student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
    total_t = total_gauss * student_t_factor
    samples = mean + total_t.view(B, T, 5, 5)

    return mean, samples


def load_models(device: str = "cuda"):
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    ar1_path = "models/backfill/two_stage/prior_network_ar1/prior_network_ar1_best.pt"
    ar1_ckpt = torch.load(ar1_path, map_location=device, weights_only=False)

    ar1_config = config.copy()
    ar1_config["init_rho"] = 0.8
    ar1_config["init_sigma"] = 1.0
    ar1_predictor = LatentPredictorCov(ar1_config)
    ar1_predictor.load_state_dict(ar1_ckpt["predictor_state_dict"])
    ar1_predictor = ar1_predictor.to(device)
    ar1_predictor.eval()

    return vae, ar1_predictor, config


def get_regime_indices(surfaces):
    """Get indices for different market regimes based on ATM IV level."""
    atm_iv = surfaces[:, 2, 2]

    # Define regimes by IV percentiles
    p25 = np.percentile(atm_iv, 25)
    p50 = np.percentile(atm_iv, 50)
    p75 = np.percentile(atm_iv, 75)
    p95 = np.percentile(atm_iv, 95)

    regimes = {
        "Low Vol": np.where(atm_iv < p25)[0],
        "Normal": np.where((atm_iv >= p25) & (atm_iv < p75))[0],
        "High Vol": np.where((atm_iv >= p75) & (atm_iv < p95))[0],
        "Crisis": np.where(atm_iv >= p95)[0],
    }
    return regimes


def generate_trajectories(vae, predictor, log_returns, surfaces, start_idx,
                          context_len=20, horizon=30, n_samples=50,
                          rho_noise=-0.2, device="cuda"):
    """Generate trajectories with correlated decoder noise."""
    context = log_returns[start_idx:start_idx + context_len]
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]
    gt_returns = log_returns[start_idx + context_len:start_idx + context_len + horizon, 2, 2]
    gt_iv = surfaces[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    sample_returns = np.zeros((n_samples, horizon))
    sample_iv = np.zeros((n_samples, horizon))

    with torch.no_grad():
        z_mean, _ = predictor(context_tensor, horizon=horizon)
        z_samples = predictor.sample_z(z_mean, n_samples=n_samples)

        for s in range(n_samples):
            z = z_samples[s]
            _, samples = decode_with_correlated_noise(
                vae.decoder, z, rho_noise=rho_noise, device=device
            )
            returns = samples[0, :, 2, 2].cpu().numpy()
            sample_returns[s] = returns

            # Convert to IV levels
            for h in range(horizon):
                if h == 0:
                    sample_iv[s, h] = initial_iv * np.exp(returns[h])
                else:
                    sample_iv[s, h] = sample_iv[s, h-1] * np.exp(returns[h])

    return gt_returns, gt_iv, sample_returns, sample_iv, initial_iv


def plot_fan_chart(ax, gt_iv, sample_iv, initial_iv, title, color):
    """Plot a single fan chart panel."""
    horizon = len(gt_iv)
    days = np.arange(1, horizon + 1)

    # Plot sample paths
    for s in range(min(50, sample_iv.shape[0])):
        ax.plot(days, sample_iv[s], color=color, alpha=0.15, linewidth=0.5)

    # Compute CI bands
    p05 = np.percentile(sample_iv, 5, axis=0)
    p50 = np.percentile(sample_iv, 50, axis=0)
    p95 = np.percentile(sample_iv, 95, axis=0)

    ax.fill_between(days, p05, p95, color=color, alpha=0.2, label='90% CI')
    ax.plot(days, p50, color=color, linewidth=2, label='Median')
    ax.plot(days, gt_iv, 'k-', linewidth=2.5, label='Ground Truth')
    ax.axhline(y=initial_iv, color='gray', linestyle='--', alpha=0.5)

    # Check violations
    violations = (gt_iv < p05) | (gt_iv > p95)
    violation_days = days[violations]
    violation_ivs = gt_iv[violations]
    if len(violation_days) > 0:
        ax.scatter(violation_days, violation_ivs, c='red', s=40, zorder=5, marker='x', linewidths=2)

    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('ATM IV')
    violation_pct = violations.mean() * 100
    ax.set_title(f'{title}\n(Violations: {violation_pct:.0f}%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return violation_pct


def main():
    print("=" * 70)
    print("Diagnostic Plots with Correlated Decoder Noise (rho=-0.2)")
    print("=" * 70)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    vae, predictor, config = load_models(device)

    context_len = 20
    horizon = 30
    n_samples = 50
    rho_noise = -0.2  # Optimal for mean-reverting ACF

    regimes = get_regime_indices(surfaces)

    save_dir = Path("models/backfill/two_stage/prior_network_ar1/diagnostics")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PLOT 1: Conditional Fan Charts (50 samples per regime)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PLOT 1: Conditional Fan Charts")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {
        "Low Vol": "#2ecc71",
        "Normal": "#3498db",
        "High Vol": "#e67e22",
        "Crisis": "#e74c3c",
    }

    violations_by_regime = {}

    for i, (regime_name, regime_indices) in enumerate(regimes.items()):
        # Filter valid indices
        valid_indices = [idx for idx in regime_indices
                        if idx >= 100 and idx < len(log_returns) - context_len - horizon - 10]

        if len(valid_indices) < 5:
            print(f"  {regime_name}: Not enough samples")
            continue

        # Pick representative index
        start_idx = valid_indices[len(valid_indices) // 2]

        print(f"  Generating {regime_name} (idx={start_idx})...")

        gt_ret, gt_iv, sample_ret, sample_iv, initial_iv = generate_trajectories(
            vae, predictor, log_returns, surfaces, start_idx,
            context_len=context_len, horizon=horizon, n_samples=n_samples,
            rho_noise=rho_noise, device=device
        )

        viol_pct = plot_fan_chart(
            axes[i], gt_iv, sample_iv, initial_iv,
            f"{regime_name}", colors[regime_name]
        )
        violations_by_regime[regime_name] = viol_pct
        print(f"    Violations: {viol_pct:.0f}%")

    plt.suptitle(
        f'Conditional Fan Charts with Correlated Noise (rho={rho_noise})\n'
        f'50 Samples per Regime | 30-Day Horizon',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_dir / "fan_chart_correlated.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {save_dir / 'fan_chart_correlated.png'}")
    plt.close()

    # ========================================================================
    # PLOT 2: Conditional Sensitivity - Mean Trajectory Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("PLOT 2: Conditional Sensitivity - Mean Trajectories")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate many samples per regime to compute mean trajectory
    n_windows = 30
    n_samples_per_window = 20

    regime_stats = {}

    for i, (regime_name, regime_indices) in enumerate(regimes.items()):
        valid_indices = [idx for idx in regime_indices
                        if idx >= 100 and idx < len(log_returns) - context_len - horizon - 10]

        if len(valid_indices) < n_windows:
            print(f"  {regime_name}: Not enough windows")
            continue

        # Sample windows from this regime
        window_indices = np.random.choice(valid_indices, min(n_windows, len(valid_indices)), replace=False)

        all_gt_returns = []
        all_model_returns = []
        all_ci_widths = []

        for start_idx in window_indices:
            gt_ret, gt_iv, sample_ret, sample_iv, initial_iv = generate_trajectories(
                vae, predictor, log_returns, surfaces, start_idx,
                context_len=context_len, horizon=horizon, n_samples=n_samples_per_window,
                rho_noise=rho_noise, device=device
            )

            all_gt_returns.append(gt_ret)
            all_model_returns.append(sample_ret)

            # CI width at each horizon
            ci_width = np.percentile(sample_ret, 95, axis=0) - np.percentile(sample_ret, 5, axis=0)
            all_ci_widths.append(ci_width)

        all_gt_returns = np.array(all_gt_returns)
        all_model_returns = np.array(all_model_returns)
        all_ci_widths = np.array(all_ci_widths)

        # Compute statistics
        gt_cumsum = np.cumsum(all_gt_returns, axis=1)
        model_cumsum = np.cumsum(all_model_returns.mean(axis=1), axis=1)

        gt_mean = gt_cumsum.mean(axis=0)
        model_mean = model_cumsum.mean(axis=0)
        ci_width_mean = all_ci_widths.mean(axis=0)

        regime_stats[regime_name] = {
            "gt_mean": gt_mean,
            "model_mean": model_mean,
            "ci_width_mean": ci_width_mean,
        }

        ax = axes.flatten()[i]
        days = np.arange(1, horizon + 1)

        ax.plot(days, gt_mean, 'k-', linewidth=2, label='GT Mean')
        ax.plot(days, model_mean, color=colors[regime_name], linewidth=2, label='Model Mean')
        ax.fill_between(days, model_mean - ci_width_mean/2, model_mean + ci_width_mean/2,
                       color=colors[regime_name], alpha=0.2, label='Mean CI Width')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Horizon')
        ax.set_ylabel('Cumulative Return')
        ax.set_title(f'{regime_name}\nMean CI Width: {ci_width_mean.mean():.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"  {regime_name}: Mean CI Width = {ci_width_mean.mean():.4f}")

    plt.suptitle(
        f'Conditional Sensitivity: Mean Trajectories by Regime\n'
        f'Correlated Noise (rho={rho_noise})',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_dir / "mean_trajectory_correlated.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {save_dir / 'mean_trajectory_correlated.png'}")
    plt.close()

    # Check if CI width varies by regime (risk awareness)
    print("\n  Risk Awareness Check (CI Width by Regime):")
    ci_widths = {k: v["ci_width_mean"].mean() for k, v in regime_stats.items()}
    for regime, width in sorted(ci_widths.items(), key=lambda x: x[1]):
        print(f"    {regime}: {width:.4f}")

    if ci_widths.get("Crisis", 0) > ci_widths.get("Low Vol", float('inf')):
        print("  --> PASS: Crisis has wider CI than Low Vol (risk-aware)")
    else:
        print("  --> ISSUE: CI width not increasing with risk")

    # ========================================================================
    # PLOT 3: Unconditional Marginal Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("PLOT 3: Unconditional Marginal Comparison")
    print("=" * 70)

    # Generate samples from many windows (pooled across regimes)
    n_windows_total = 200
    n_samples_per = 10

    all_indices = np.arange(100, len(log_returns) - context_len - horizon - 10)
    sample_indices = np.random.choice(all_indices, n_windows_total, replace=False)

    gt_pooled = []
    model_pooled = []

    print("  Generating pooled samples...")
    for start_idx in sample_indices:
        gt_ret, _, sample_ret, _, _ = generate_trajectories(
            vae, predictor, log_returns, surfaces, start_idx,
            context_len=context_len, horizon=horizon, n_samples=n_samples_per,
            rho_noise=rho_noise, device=device
        )
        gt_pooled.append(gt_ret)
        model_pooled.extend(sample_ret)

    gt_pooled = np.array(gt_pooled)
    model_pooled = np.array(model_pooled)

    # Compute cumulative returns at h=30
    gt_cumsum_h30 = np.sum(gt_pooled, axis=1)
    model_cumsum_h30 = np.sum(model_pooled, axis=1)

    # Compute statistics
    gt_stats = {
        "mean": np.mean(gt_cumsum_h30),
        "std": np.std(gt_cumsum_h30),
        "skew": stats.skew(gt_cumsum_h30),
        "kurt": stats.kurtosis(gt_cumsum_h30, fisher=True),
    }
    model_stats = {
        "mean": np.mean(model_cumsum_h30),
        "std": np.std(model_cumsum_h30),
        "skew": stats.skew(model_cumsum_h30),
        "kurt": stats.kurtosis(model_cumsum_h30, fisher=True),
    }

    # ACF comparison
    gt_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in gt_pooled]
    model_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in model_pooled]
    gt_acf = np.mean([a for a in gt_acfs if not np.isnan(a)])
    model_acf = np.mean([a for a in model_acfs if not np.isnan(a)])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Distribution comparison
    axes[0, 0].hist(gt_cumsum_h30, bins=50, alpha=0.5, label='GT', density=True)
    axes[0, 0].hist(model_cumsum_h30, bins=50, alpha=0.5, label='Model', density=True)
    axes[0, 0].set_xlabel('Cumulative Return (h=30)')
    axes[0, 0].set_title(f'Distribution Comparison\nGT: k={gt_stats["kurt"]:.1f}, Model: k={model_stats["kurt"]:.1f}')
    axes[0, 0].legend()

    # Q-Q plot
    n_q = 100
    quantiles = np.linspace(0.01, 0.99, n_q)
    gt_q = np.quantile(gt_cumsum_h30, quantiles)
    model_q = np.quantile(model_cumsum_h30, quantiles)

    axes[0, 1].scatter(gt_q, model_q, alpha=0.5, s=20)
    min_val, max_val = min(gt_q.min(), model_q.min()), max(gt_q.max(), model_q.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    axes[0, 1].set_xlabel('GT Quantiles')
    axes[0, 1].set_ylabel('Model Quantiles')
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].legend()

    # ACF histogram
    axes[0, 2].hist(gt_acfs, bins=30, alpha=0.5, label=f'GT (mean={gt_acf:.3f})', density=True)
    axes[0, 2].hist(model_acfs, bins=30, alpha=0.5, label=f'Model (mean={model_acf:.3f})', density=True)
    axes[0, 2].axvline(gt_acf, color='blue', linestyle='--')
    axes[0, 2].axvline(model_acf, color='orange', linestyle='--')
    axes[0, 2].set_xlabel('ACF(1)')
    axes[0, 2].set_title(f'ACF Distribution\nGT: {gt_acf:.3f}, Model: {model_acf:.3f}')
    axes[0, 2].legend()

    # Statistics comparison bar chart
    metrics = ['Mean', 'Std', 'Skew', 'Kurtosis', 'ACF(1)']
    gt_vals = [gt_stats['mean'], gt_stats['std'], gt_stats['skew'], gt_stats['kurt'], gt_acf]
    model_vals = [model_stats['mean'], model_stats['std'], model_stats['skew'], model_stats['kurt'], model_acf]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 0].bar(x - width/2, gt_vals, width, label='GT')
    axes[1, 0].bar(x + width/2, model_vals, width, label='Model')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].set_title('Statistics Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Per-horizon std comparison
    gt_std_by_h = np.std(np.cumsum(gt_pooled, axis=1), axis=0)
    model_std_by_h = np.std(np.cumsum(model_pooled, axis=1), axis=0)

    axes[1, 1].plot(range(1, horizon+1), gt_std_by_h, 'b-', linewidth=2, label='GT')
    axes[1, 1].plot(range(1, horizon+1), model_std_by_h, 'orange', linewidth=2, label='Model')
    axes[1, 1].set_xlabel('Horizon')
    axes[1, 1].set_ylabel('Std of Cumulative Return')
    axes[1, 1].set_title('Variance Scaling by Horizon')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Summary table
    axes[1, 2].axis('off')
    summary_text = f"""
UNCONDITIONAL MARGINAL COMPARISON
(Correlated Noise rho={rho_noise})

                    GT          Model       Gap
Mean:           {gt_stats['mean']:8.4f}    {model_stats['mean']:8.4f}    {abs(gt_stats['mean']-model_stats['mean']):.4f}
Std:            {gt_stats['std']:8.4f}    {model_stats['std']:8.4f}    {abs(gt_stats['std']-model_stats['std']):.4f}
Skewness:       {gt_stats['skew']:8.2f}    {model_stats['skew']:8.2f}    {abs(gt_stats['skew']-model_stats['skew']):.2f}
Kurtosis:       {gt_stats['kurt']:8.2f}    {model_stats['kurt']:8.2f}    {abs(gt_stats['kurt']-model_stats['kurt']):.2f}
ACF(1):         {gt_acf:8.4f}    {model_acf:8.4f}    {abs(gt_acf-model_acf):.4f}

ACF Recovery: {(1 - abs(gt_acf - model_acf)/abs(gt_acf))*100:.1f}%
"""
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, family='monospace', verticalalignment='center')

    plt.suptitle(
        f'Unconditional Marginal Comparison\n'
        f'Correlated Noise (rho={rho_noise}) | {n_windows_total} windows x {n_samples_per} samples',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_dir / "unconditional_comparison_correlated.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {save_dir / 'unconditional_comparison_correlated.png'}")
    plt.close()

    # Print final summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"\n1. FAN CHART VIOLATIONS:")
    for regime, viol in violations_by_regime.items():
        status = "OK" if viol <= 20 else "HIGH"
        print(f"   {regime}: {viol:.0f}% [{status}]")

    print(f"\n2. RISK AWARENESS (CI Width by Regime):")
    for regime, width in sorted(ci_widths.items(), key=lambda x: x[1]):
        print(f"   {regime}: {width:.4f}")

    print(f"\n3. UNCONDITIONAL STATISTICS:")
    print(f"   {'Metric':<12} {'GT':<10} {'Model':<10} {'Status'}")
    print(f"   {'-'*42}")

    acf_ok = abs(gt_acf - model_acf) < 0.05
    print(f"   {'ACF(1)':<12} {gt_acf:<10.4f} {model_acf:<10.4f} {'OK' if acf_ok else 'GAP'}")

    kurt_ok = abs(gt_stats['kurt'] - model_stats['kurt']) < 2
    print(f"   {'Kurtosis':<12} {gt_stats['kurt']:<10.2f} {model_stats['kurt']:<10.2f} {'OK' if kurt_ok else 'GAP'}")

    std_ok = abs(gt_stats['std'] - model_stats['std']) / gt_stats['std'] < 0.5
    print(f"   {'Std':<12} {gt_stats['std']:<10.4f} {model_stats['std']:<10.4f} {'OK' if std_ok else 'HIGH'}")

    acf_recovery = (1 - abs(gt_acf - model_acf)/abs(gt_acf))*100
    print(f"\n   ** ACF Recovery: {acf_recovery:.1f}% **")

    return {
        "violations": violations_by_regime,
        "ci_widths": ci_widths,
        "gt_stats": gt_stats,
        "model_stats": model_stats,
        "gt_acf": gt_acf,
        "model_acf": model_acf,
        "acf_recovery": acf_recovery,
    }


if __name__ == "__main__":
    results = main()
