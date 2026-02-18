"""
Regime-conditional calibration diagnostic.

Tests whether the model produces different conditional distributions for different
regimes. A well-calibrated conditional model should:
  - Produce wider CIs after volatile history
  - Produce narrower CIs after calm history
  - Maintain correct coverage in BOTH regimes (not just on average)

If CI widths are similar across regimes, the model is learning the marginal
distribution with a mean shift — not truly conditional forecasting.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from diffusion.block_ar.block_ar_ddpm import BlockARConfig, ConditionalBlockARDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def classify_regimes(histories: np.ndarray, method: str = "iv_volatility") -> np.ndarray:
    """Classify each sample's history as volatile (True) or calm (False).

    Args:
        histories: (N, T, 5, 5) in [0, 1] denormalized IV surfaces
        method: classification method

    Returns:
        is_volatile: (N,) boolean array — True = volatile regime
    """
    # Compute day-to-day IV changes across the history window
    iv_changes = np.diff(histories, axis=1)  # (N, T-1, 5, 5)

    # Realized volatility of IV changes per sample (std of all changes)
    per_sample_vol = iv_changes.reshape(histories.shape[0], -1).std(axis=1)  # (N,)

    # Median split
    median_vol = np.median(per_sample_vol)
    is_volatile = per_sample_vol >= median_vol

    return is_volatile, per_sample_vol, median_vol


def compute_regime_metrics(
    gt: np.ndarray,
    gen: np.ndarray,
    is_volatile: np.ndarray,
    ci_level: float = 0.9,
    horizons: list = None,
):
    """Compute CI coverage and width separately for volatile and calm regimes.

    Args:
        gt: (N, T, 5, 5) ground truth in [0, 1]
        gen: (N, n_samples, T, 5, 5) generated samples in [0, 1]
        is_volatile: (N,) boolean
        ci_level: confidence level
        horizons: list of horizons to evaluate

    Returns:
        dict with per-regime metrics
    """
    if horizons is None:
        horizons = [1, 7, 14, 30]

    alpha = (1 - ci_level) / 2

    results = {}
    for regime_name, mask in [("volatile", is_volatile), ("calm", ~is_volatile)]:
        gt_r = gt[mask]        # (N_r, T, 5, 5)
        gen_r = gen[mask]      # (N_r, n_samples, T, 5, 5)
        n_r = mask.sum()

        # Overall CI
        lower = np.quantile(gen_r, alpha, axis=1)        # (N_r, T, 5, 5)
        upper = np.quantile(gen_r, 1 - alpha, axis=1)    # (N_r, T, 5, 5)
        covered = (gt_r >= lower) & (gt_r <= upper)
        width = upper - lower

        # Per-horizon
        horizon_metrics = {}
        for h in horizons:
            h_idx = h - 1
            if h_idx < gt_r.shape[1]:
                lower_h = np.quantile(gen_r[:, :, h_idx], alpha, axis=1)
                upper_h = np.quantile(gen_r[:, :, h_idx], 1 - alpha, axis=1)
                covered_h = (gt_r[:, h_idx] >= lower_h) & (gt_r[:, h_idx] <= upper_h)
                width_h = upper_h - lower_h
                horizon_metrics[h] = {
                    "coverage": float(covered_h.mean()),
                    "mean_width": float(width_h.mean()),
                    "median_width": float(np.median(width_h)),
                }

        results[regime_name] = {
            "n_samples": int(n_r),
            "overall_coverage": float(covered.mean()),
            "mean_width": float(width.mean()),
            "median_width": float(np.median(width)),
            "per_horizon": horizon_metrics,
        }

    return results


def compute_calibration_per_regime(
    gt: np.ndarray,
    gen: np.ndarray,
    is_volatile: np.ndarray,
):
    """Compute full calibration curves per regime."""
    levels = np.linspace(0.1, 0.95, 18)
    results = {}

    for regime_name, mask in [("volatile", is_volatile), ("calm", ~is_volatile)]:
        gt_r = gt[mask]
        gen_r = gen[mask]

        nominal = []
        empirical = []
        widths = []
        for p in levels:
            alpha = (1 - p) / 2
            lower = np.quantile(gen_r, alpha, axis=1)
            upper = np.quantile(gen_r, 1 - alpha, axis=1)
            covered = (gt_r >= lower) & (gt_r <= upper)
            nominal.append(float(p))
            empirical.append(float(covered.mean()))
            widths.append(float((upper - lower).mean()))

        cal_error = float(np.mean(np.abs(np.array(nominal) - np.array(empirical))))
        results[regime_name] = {
            "nominal": nominal,
            "empirical": empirical,
            "widths": widths,
            "calibration_error": cal_error,
        }

    return results


def generate_samples_with_history(model, dataloader, n_samples=10, max_batches=10, device="cuda"):
    """Generate samples and also return raw histories for regime classification."""
    model.eval()
    all_history = []
    all_gt = []
    all_gen = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            history_denorm = denormalize_iv(history)
            future_gt_denorm = denormalize_iv(future_gt)
            samples = model.sample_batched(history, n_samples=n_samples)

            all_history.append(history_denorm.cpu().numpy())
            all_gt.append(future_gt_denorm.cpu().numpy())
            all_gen.append(samples.cpu().numpy())

            print(f"  Batch {batch_idx+1}/{max_batches}")

    histories = np.concatenate(all_history, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    gen = np.concatenate(all_gen, axis=0)
    return histories, gt, gen


def main():
    parser = argparse.ArgumentParser(description="Regime-conditional calibration diagnostic")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/block_ar_dual_path/best_coverage_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_ema", action="store_true", default=True)
    args = parser.parse_args()

    # --- Load model ---
    print("=" * 70)
    print("REGIME-CONDITIONAL CALIBRATION DIAGNOSTIC")
    print("=" * 70)

    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, weights_only=False, map_location=args.device)
    config = checkpoint["config"]
    if isinstance(config, dict):
        config = BlockARConfig(**config)

    model = ConditionalBlockARDDPM(config).to(args.device)
    if args.no_ema:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        ema = checkpoint.get("ema_params", {})
        if ema:
            state = model.state_dict()
            for name in ema:
                if name in state:
                    state[name] = ema[name]
            model.load_state_dict(state)
    model.eval()

    # --- Load data ---
    data = np.load(args.data_path)
    surfaces = data["surface"]
    val_dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4040, end_idx=4540)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # --- Generate samples ---
    print(f"\nGenerating {args.n_samples} samples per window ({args.max_batches} batches)...")
    histories, gt, gen = generate_samples_with_history(
        model, val_loader, args.n_samples, args.max_batches, args.device
    )
    print(f"  Total windows: {gt.shape[0]}")

    # --- Classify regimes ---
    print("\n--- Regime Classification (median split on history IV volatility) ---\n")
    is_volatile, per_sample_vol, median_vol = classify_regimes(histories)
    n_volatile = is_volatile.sum()
    n_calm = (~is_volatile).sum()
    print(f"  Median IV volatility: {median_vol:.6f}")
    print(f"  Volatile samples: {n_volatile} (vol >= {median_vol:.6f})")
    print(f"  Calm samples:     {n_calm} (vol < {median_vol:.6f})")
    print(f"  Vol range: [{per_sample_vol.min():.6f}, {per_sample_vol.max():.6f}]")
    print(f"  Volatile mean vol: {per_sample_vol[is_volatile].mean():.6f}")
    print(f"  Calm mean vol:     {per_sample_vol[~is_volatile].mean():.6f}")

    # --- Ground truth regime differences ---
    print("\n--- Ground Truth Regime Differences ---\n")
    gt_vol_changes = np.diff(gt[is_volatile], axis=1)
    gt_calm_changes = np.diff(gt[~is_volatile], axis=1)
    print(f"  GT future std (volatile): {gt_vol_changes.std():.6f}")
    print(f"  GT future std (calm):     {gt_calm_changes.std():.6f}")
    print(f"  GT vol/calm std ratio:    {gt_vol_changes.std() / gt_calm_changes.std():.3f}")

    # --- Per-regime CI metrics ---
    print("\n--- Per-Regime CI Metrics (90% CI) ---\n")
    regime_metrics = compute_regime_metrics(gt, gen, is_volatile, ci_level=0.9)

    vol_m = regime_metrics["volatile"]
    calm_m = regime_metrics["calm"]

    print(f"  {'Metric':<25s}  {'Volatile':>10s}  {'Calm':>10s}  {'Ratio (V/C)':>12s}  {'Verdict':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

    # Overall
    width_ratio = vol_m["mean_width"] / calm_m["mean_width"] if calm_m["mean_width"] > 0 else float("inf")
    verdict = "ADAPTIVE" if width_ratio > 1.15 else "FLAT"
    print(f"  {'Overall coverage':<25s}  {vol_m['overall_coverage']:>9.1%}  {calm_m['overall_coverage']:>9.1%}  {'':>12s}  {'':>10s}")
    print(f"  {'Overall CI width':<25s}  {vol_m['mean_width']:>10.4f}  {calm_m['mean_width']:>10.4f}  {width_ratio:>11.3f}x  {verdict:>10s}")

    # Per-horizon
    print()
    for h in [1, 7, 14, 30]:
        if h in vol_m["per_horizon"] and h in calm_m["per_horizon"]:
            vh = vol_m["per_horizon"][h]
            ch = calm_m["per_horizon"][h]
            wr = vh["mean_width"] / ch["mean_width"] if ch["mean_width"] > 0 else float("inf")
            verdict = "ADAPTIVE" if wr > 1.15 else "FLAT"
            print(f"  h={h:<3d} coverage           {vh['coverage']:>9.1%}  {ch['coverage']:>9.1%}  {'':>12s}  {'':>10s}")
            print(f"  h={h:<3d} CI width            {vh['mean_width']:>10.4f}  {ch['mean_width']:>10.4f}  {wr:>11.3f}x  {verdict:>10s}")

    # --- Calibration per regime ---
    print("\n--- Calibration Curves per Regime ---\n")
    cal = compute_calibration_per_regime(gt, gen, is_volatile)

    print(f"  {'Nominal':>8s}  {'Vol Emp':>8s}  {'Vol Width':>10s}  {'Calm Emp':>8s}  {'Calm Width':>10s}  {'Width Ratio':>12s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*12}")

    for i in range(len(cal["volatile"]["nominal"])):
        nom = cal["volatile"]["nominal"][i]
        ve = cal["volatile"]["empirical"][i]
        vw = cal["volatile"]["widths"][i]
        ce = cal["calm"]["empirical"][i]
        cw = cal["calm"]["widths"][i]
        wr = vw / cw if cw > 0 else float("inf")
        print(f"  {nom:>7.0%}  {ve:>7.1%}  {vw:>10.4f}  {ce:>7.1%}  {cw:>10.4f}  {wr:>11.3f}x")

    print(f"\n  Volatile calibration error: {cal['volatile']['calibration_error']:.4f}")
    print(f"  Calm calibration error:     {cal['calm']['calibration_error']:.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    gt_ratio = gt_vol_changes.std() / gt_calm_changes.std()
    print(f"\n  Ground truth vol/calm std ratio:    {gt_ratio:.3f}x")
    print(f"  Model CI width vol/calm ratio:      {width_ratio:.3f}x")

    if width_ratio > 1.15:
        print(f"\n  RESULT: Model IS regime-adaptive (width ratio {width_ratio:.3f}x)")
        if width_ratio > 0.8 * gt_ratio:
            print(f"  Model captures >{80:.0f}% of regime difference")
        else:
            pct = width_ratio / gt_ratio * 100
            print(f"  But only captures {pct:.0f}% of ground truth regime difference")
    else:
        print(f"\n  RESULT: Model is NOT regime-adaptive (width ratio {width_ratio:.3f}x)")
        print(f"  The model produces similar CI widths regardless of regime.")
        print(f"  This means it learned the MARGINAL distribution + mean shift,")
        print(f"  not a truly conditional distribution.")

    coverage_gap = abs(vol_m["overall_coverage"] - calm_m["overall_coverage"])
    if coverage_gap > 0.05:
        worse_regime = "volatile" if vol_m["overall_coverage"] < calm_m["overall_coverage"] else "calm"
        print(f"\n  WARNING: Coverage gap of {coverage_gap:.1%} between regimes")
        print(f"  Model is UNDER-covering in {worse_regime} regime")
        print(f"  Aggregate {(vol_m['overall_coverage'] * n_volatile + calm_m['overall_coverage'] * n_calm) / (n_volatile + n_calm):.1%} coverage hides this problem")
    else:
        print(f"\n  Coverage gap between regimes: {coverage_gap:.1%} (acceptable)")


if __name__ == "__main__":
    main()
