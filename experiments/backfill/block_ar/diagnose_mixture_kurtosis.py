"""
Diagnostic: Validate Mixture-of-Regimes Kurtosis Hypothesis.

Tests whether pooled kurtosis comes from mixing calm/turb regimes with different
spread, not from individual-regime non-Gaussianity. If confirmed, hierarchical
sampling (regime-dependent spread) can produce kurtosis without exp().

Uses 89q model (direct IV, best distributional metrics) on test data.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_mixture_kurtosis.py \
        --model_path models/backfill/afcrps_89q_direct_iv/best_coverage_model.pt \
        --n_samples 50 --max_batches 13 --device cuda
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def classify_regime(history: np.ndarray, turb_pct: float = 80.0):
    """Classify windows into calm/turb using vol-of-vol.

    Args:
        history: (N, T_hist, 5, 5) denormalized IV [0, 1]
        turb_pct: percentile threshold for turbulent (top 20% = 80)

    Returns:
        turb_mask: (N,) bool, True = turbulent
        calm_mask: (N,) bool, True = calm (bottom 20%)
    """
    mean_iv = history.mean(axis=(-1, -2))  # (N, T)
    daily_chg = np.diff(mean_iv, axis=1)   # (N, T-1)
    vov = daily_chg.std(axis=1)            # (N,)
    turb_thresh = np.percentile(vov, turb_pct)
    calm_thresh = np.percentile(vov, 100 - turb_pct)
    return vov > turb_thresh, vov < calm_thresh


def compute_kurtosis(x):
    """Compute excess kurtosis (Gaussian = 0) from array."""
    x = x.ravel()
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    m = x.mean()
    m2 = ((x - m) ** 2).mean()
    m4 = ((x - m) ** 4).mean()
    if m2 < 1e-12:
        return np.nan
    return m4 / m2**2 - 3.0  # excess kurtosis (Gaussian = 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/block_ar/mixture_kurtosis_diagnosis")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    sp_cfg = {k: v for k, v in checkpoint["config"].items()
              if k in SinglePassConfig.__dataclass_fields__}
    sp_config = SinglePassConfig(**sp_cfg)
    model = SinglePassBlockAR(sp_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load test data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_dataset = VolSurfaceDataset(
        surfaces, sp_config.history_len, sp_config.future_len,
        start_idx=4540,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test set: {len(test_dataset)} windows")

    # Generate samples
    print("Generating samples...")
    all_samples, all_gt, all_history = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= args.max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))
            samples = model.sample(history, n_samples=args.n_samples)
            history_denorm = denormalize_iv(history)
            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_history.append(history_denorm.cpu().numpy())
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{args.max_batches}")

    samples = np.concatenate(all_samples, axis=0)    # (N, S, T, 5, 5)
    gt = np.concatenate(all_gt, axis=0)               # (N, T, 5, 5)
    history_arr = np.concatenate(all_history, axis=0)  # (N, T_hist, 5, 5)
    N = samples.shape[0]
    print(f"Generated: {N} windows, {args.n_samples} samples each")

    # Regime classification
    turb_mask, calm_mask = classify_regime(history_arr)
    n_calm, n_turb = calm_mask.sum(), turb_mask.sum()
    print(f"Regimes: {n_calm} calm, {n_turb} turb, {N - n_calm - n_turb} middle")

    # ================================================================
    # TEST 1: GT kurtosis decomposition
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: GT Kurtosis Decomposition")
    print("=" * 60)

    gt_changes = np.diff(gt, axis=1)  # (N, T-1, 5, 5)
    gt_calm_changes = gt_changes[calm_mask]
    gt_turb_changes = gt_changes[turb_mask]
    gt_all_changes = gt_changes

    kurt_calm = compute_kurtosis(gt_calm_changes)
    kurt_turb = compute_kurtosis(gt_turb_changes)
    kurt_pooled = compute_kurtosis(gt_all_changes)
    print(f"  Calm kurtosis (excess):   {kurt_calm:.2f}")
    print(f"  Turb kurtosis (excess):   {kurt_turb:.2f}")
    print(f"  Pooled kurtosis (excess): {kurt_pooled:.2f}")
    print(f"  Calm std:  {gt_calm_changes.std():.5f}")
    print(f"  Turb std:  {gt_turb_changes.std():.5f}")
    print(f"  Ratio:     {gt_turb_changes.std() / gt_calm_changes.std():.2f}x")

    if kurt_calm < 5.0 and kurt_turb < 5.0 and kurt_pooled > 5.0:
        print("  → CONFIRMED: Pooled kurtosis from regime mixing, not per-regime tails")
    elif kurt_calm < kurt_pooled and kurt_turb < kurt_pooled:
        print("  → PARTIAL: Pooled > per-regime, but per-regime kurtosis also elevated")
    else:
        print("  → REJECTED: Per-regime kurtosis already matches pooled")

    # ================================================================
    # TEST 2: Model ensemble shape per regime
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Model Ensemble Shape Per Regime")
    print("=" * 60)

    ens_mean = samples.mean(axis=1)  # (N, T, 5, 5)
    ens_std = samples.std(axis=1)    # (N, T, 5, 5)

    # Normalized deviations: (sample - mean) / std for each member
    deviations = (samples - ens_mean[:, None]) / np.clip(ens_std[:, None], 1e-8, None)

    dev_calm = deviations[calm_mask].ravel()
    dev_turb = deviations[turb_mask].ravel()

    # Subsample for speed
    rng = np.random.RandomState(42)
    if len(dev_calm) > 500000:
        dev_calm = rng.choice(dev_calm, 500000, replace=False)
    if len(dev_turb) > 500000:
        dev_turb = rng.choice(dev_turb, 500000, replace=False)

    kurt_dev_calm = compute_kurtosis(dev_calm)
    kurt_dev_turb = compute_kurtosis(dev_turb)

    # Shapiro-Wilk on subsamples (max 5000)
    sw_calm = stats.shapiro(rng.choice(dev_calm, min(5000, len(dev_calm)), replace=False))
    sw_turb = stats.shapiro(rng.choice(dev_turb, min(5000, len(dev_turb)), replace=False))

    print(f"  Calm normalized dev kurtosis: {kurt_dev_calm:.2f} (Gaussian=0)")
    print(f"  Turb normalized dev kurtosis: {kurt_dev_turb:.2f} (Gaussian=0)")
    print(f"  Calm Shapiro-Wilk: W={sw_calm.statistic:.4f}, p={sw_calm.pvalue:.4f}")
    print(f"  Turb Shapiro-Wilk: W={sw_turb.statistic:.4f}, p={sw_turb.pvalue:.4f}")

    if abs(kurt_dev_calm) < 1.0 and abs(kurt_dev_turb) < 1.0:
        print("  → CONFIRMED: Ensemble IS approximately Gaussian in both regimes")
    else:
        print("  → NOT GAUSSIAN: Ensemble has non-Gaussian shape")

    # ================================================================
    # TEST 3: Width ratio measurement
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Width Ratio (Ensemble Std Calm vs Turb)")
    print("=" * 60)

    # Per-window ensemble std, averaged over cells and time
    window_std = ens_std.mean(axis=(1, 2, 3))  # (N,)
    calm_std = window_std[calm_mask].mean()
    turb_std = window_std[turb_mask].mean()
    model_ratio = turb_std / calm_std

    # GT realized absolute daily change std per window
    gt_window_std = np.abs(gt_changes).mean(axis=(1, 2, 3)).std()  # rough measure
    gt_calm_std = np.abs(gt_calm_changes).mean(axis=(1, 2, 3))
    gt_turb_std = np.abs(gt_turb_changes).mean(axis=(1, 2, 3))
    gt_ratio = gt_turb_std.mean() / gt_calm_std.mean()

    print(f"  Model ensemble std:  calm={calm_std:.5f}, turb={turb_std:.5f}, ratio={model_ratio:.2f}x")
    print(f"  GT |daily change|:   calm={gt_calm_std.mean():.5f}, turb={gt_turb_std.mean():.5f}, ratio={gt_ratio:.2f}x")
    print(f"  Gap: model {model_ratio:.2f}x vs GT {gt_ratio:.2f}x")

    if model_ratio < 1.3 and gt_ratio > 1.5:
        print("  → CONFIRMED: Model is regime-blind (ratio ~1), GT has ~2x turb/calm")
    else:
        print(f"  → Model ratio {model_ratio:.2f}x, GT ratio {gt_ratio:.2f}x")

    # ================================================================
    # TEST 4: Synthetic mixture kurtosis
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Synthetic Mixture Kurtosis")
    print("=" * 60)

    # Member daily changes per regime
    member_changes = np.diff(samples, axis=2)  # (N, S, T-1, 5, 5)
    calm_member_changes = member_changes[calm_mask].ravel()
    turb_member_changes = member_changes[turb_mask].ravel()

    # Subsample for tractability
    n_sub = min(500000, len(calm_member_changes), len(turb_member_changes))
    calm_sub = rng.choice(calm_member_changes, n_sub, replace=False)
    turb_sub = rng.choice(turb_member_changes, n_sub, replace=False)

    # Current model: mix with correct proportions (~80% calm, 20% turb)
    calm_frac = n_calm / (n_calm + n_turb)
    n_calm_mix = int(n_sub * calm_frac)
    n_turb_mix = n_sub - n_calm_mix
    current_mix = np.concatenate([calm_sub[:n_calm_mix], turb_sub[:n_turb_mix]])
    kurt_current = compute_kurtosis(current_mix)

    # Scaled turb: multiply turb by (GT_ratio / model_ratio) to match GT regime spread
    scale_factor = gt_ratio / max(model_ratio, 0.01)
    turb_scaled = turb_sub * scale_factor
    scaled_mix = np.concatenate([calm_sub[:n_calm_mix], turb_scaled[:n_turb_mix]])
    kurt_scaled = compute_kurtosis(scaled_mix)

    # Also try fixed 2x scaling
    turb_2x = turb_sub * 2.0
    mix_2x = np.concatenate([calm_sub[:n_calm_mix], turb_2x[:n_turb_mix]])
    kurt_2x = compute_kurtosis(mix_2x)

    print(f"  Calm frac: {calm_frac:.1%}, Turb frac: {1-calm_frac:.1%}")
    print(f"  Current model (no scaling):     kurtosis = {kurt_current:.2f}")
    print(f"  GT-matched scaling ({scale_factor:.2f}x turb): kurtosis = {kurt_scaled:.2f}")
    print(f"  Fixed 2x turb scaling:          kurtosis = {kurt_2x:.2f}")
    print(f"  Calm-only kurtosis:             {compute_kurtosis(calm_sub):.2f}")
    print(f"  Turb-only kurtosis:             {compute_kurtosis(turb_sub):.2f}")

    if kurt_current < 2.0 and kurt_scaled > 3.0:
        print("  → CONFIRMED: Regime-dependent scaling creates kurtosis from Gaussian components")
    elif kurt_scaled > kurt_current + 1.0:
        print("  → PARTIAL: Scaling increases kurtosis significantly")
    else:
        print("  → REJECTED: Scaling doesn't meaningfully change kurtosis")

    # ================================================================
    # FIGURE: 4-panel diagnostic
    # ================================================================
    print("\nGenerating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: GT daily change histograms
    ax = axes[0, 0]
    bins = np.linspace(-0.03, 0.03, 80)
    ax.hist(gt_calm_changes.ravel(), bins=bins, alpha=0.5, density=True,
            label=f'Calm (kurt={kurt_calm:.1f})', color='blue')
    ax.hist(gt_turb_changes.ravel(), bins=bins, alpha=0.5, density=True,
            label=f'Turb (kurt={kurt_turb:.1f})', color='red')
    ax.hist(gt_all_changes.ravel(), bins=bins, alpha=0.3, density=True,
            label=f'Pooled (kurt={kurt_pooled:.1f})', color='gray',
            histtype='step', linewidth=2)
    ax.set_title('Test 1: GT Daily Changes by Regime')
    ax.set_xlabel('Daily IV Change')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.03, 0.03)

    # Panel 2: Normalized ensemble deviations
    ax = axes[0, 1]
    bins_z = np.linspace(-4, 4, 80)
    ax.hist(dev_calm, bins=bins_z, alpha=0.5, density=True,
            label=f'Calm (kurt={kurt_dev_calm:.2f})', color='blue')
    ax.hist(dev_turb, bins=bins_z, alpha=0.5, density=True,
            label=f'Turb (kurt={kurt_dev_turb:.2f})', color='red')
    # Overlay standard normal
    x_norm = np.linspace(-4, 4, 200)
    ax.plot(x_norm, stats.norm.pdf(x_norm), 'k--', lw=2, label='N(0,1)')
    ax.set_title('Test 2: Normalized Ensemble Deviations')
    ax.set_xlabel('(sample - mean) / std')
    ax.legend(fontsize=8)

    # Panel 3: Ensemble std boxplot
    ax = axes[1, 0]
    calm_stds = ens_std[calm_mask].mean(axis=(1, 2, 3))
    turb_stds = ens_std[turb_mask].mean(axis=(1, 2, 3))
    bp = ax.boxplot([calm_stds, turb_stds], labels=['Calm', 'Turb'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax.set_title(f'Test 3: Ensemble Std (ratio={model_ratio:.2f}x, GT={gt_ratio:.2f}x)')
    ax.set_ylabel('Mean ensemble std')

    # Panel 4: Synthetic mixture
    ax = axes[1, 1]
    bins_mix = np.linspace(-0.03, 0.03, 80)
    ax.hist(current_mix, bins=bins_mix, alpha=0.5, density=True,
            label=f'Current (kurt={kurt_current:.1f})', color='gray')
    ax.hist(scaled_mix, bins=bins_mix, alpha=0.5, density=True,
            label=f'Scaled {scale_factor:.1f}x (kurt={kurt_scaled:.1f})', color='orange')
    ax.hist(mix_2x, bins=bins_mix, alpha=0.3, density=True,
            label=f'2x turb (kurt={kurt_2x:.1f})', color='green',
            histtype='step', linewidth=2)
    ax.set_title('Test 4: Synthetic Mixture (Calm + Scaled Turb)')
    ax.set_xlabel('Member Daily IV Change')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.03, 0.03)

    plt.tight_layout()
    fig_path = f"{args.output_dir}/mixture_kurtosis_diagnosis.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  GT pooled excess kurtosis:      {kurt_pooled:.2f}")
    print(f"  GT calm/turb excess kurtosis:    {kurt_calm:.2f} / {kurt_turb:.2f}")
    print(f"  GT turb/calm std ratio:          {gt_ratio:.2f}x")
    print(f"  Model turb/calm std ratio:       {model_ratio:.2f}x")
    print(f"  Ensemble shape (both regimes):   ~Gaussian (kurt={kurt_dev_calm:.2f}/{kurt_dev_turb:.2f})")
    print(f"  Current mixture kurtosis:        {kurt_current:.2f}")
    print(f"  GT-scaled mixture kurtosis:      {kurt_scaled:.2f}")
    print(f"  2x-scaled mixture kurtosis:      {kurt_2x:.2f}")

    hypothesis = (
        abs(kurt_dev_calm) < 1.5 and abs(kurt_dev_turb) < 1.5  # Gaussian per regime
        and model_ratio < 1.5  # regime-blind spread
        and gt_ratio > 1.5     # GT has regime-dependent spread
        and kurt_scaled > kurt_current + 1.0  # scaling creates kurtosis
    )
    print(f"\n  Mixture-of-regimes hypothesis: {'CONFIRMED' if hypothesis else 'INCONCLUSIVE'}")
    if hypothesis:
        print("  → Regime-dependent spread (hierarchical sampling) should produce kurtosis")
        print("     without needing exp(). The model's ensemble is Gaussian per-regime,")
        print("     but the pooled distribution needs turb windows to have wider spread.")


if __name__ == "__main__":
    main()
