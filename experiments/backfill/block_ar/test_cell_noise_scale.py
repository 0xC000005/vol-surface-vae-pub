"""
Experiment 46: Per-cell posterior noise scaling at inference.

Computes per-cell normalized target std from training data and uses it
to scale the posterior noise in the reverse diffusion process. Cells
with higher target variance get more noise → wider CIs.

This is an INFERENCE-ONLY experiment (no retraining) to test whether
per-cell noise modulation improves coverage. If it works, we'll
train a model with heteroscedastic forward noise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffusion.block_ar.block_ar_ddpm import (
    ConditionalBlockARDDPM,
    BlockARConfig,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def compute_cell_noise_scale(surfaces, config_dict, power=1.0, clamp_min=0.5, clamp_max=2.0):
    """Compute per-cell noise scale from training data statistics.

    Computes the std of normalized targets (in vol_scaled space) per cell,
    normalizes by the median cell std, and clamps to [clamp_min, clamp_max].

    Returns:
        cell_noise_scale: (5, 5) tensor
    """
    train_ds = VolSurfaceDataset(surfaces, start_idx=0, end_idx=4540,
                                  history_len=30, future_len=30)
    loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

    eps_iv = 1e-4
    global_mean_vol = config_dict.get('global_mean_vol', 0.006)
    vol_scale_min = config_dict.get('vol_scale_min', 0.5)
    vol_scale_max = config_dict.get('vol_scale_max', 2.0)

    all_targets = []
    for batch in loader:
        history = batch['history']
        future = batch['future']

        baseline = denormalize_iv(history[:, -1:]).mean(dim=1).clamp(min=0.01).unsqueeze(1)

        past_abs = denormalize_iv(history)
        mean_iv = past_abs.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vol = daily_chg.std(dim=1, keepdim=True)
        vol_scale = (vol / global_mean_vol).clamp(vol_scale_min, vol_scale_max)
        vol_scale = vol_scale.unsqueeze(-1).unsqueeze(-1)

        target_abs = denormalize_iv(future).clamp(min=eps_iv, max=1 - eps_iv)
        log_ratio = torch.log(target_abs / baseline)
        target_norm = log_ratio / vol_scale

        all_targets.append(target_norm.numpy())

    all_targets = np.concatenate(all_targets, axis=0)  # (N, 30, 5, 5)

    # Per-cell std across all windows and horizons
    cell_std = all_targets.std(axis=(0, 1))  # (5, 5)

    # Normalize by median so median cell gets scale=1.0
    median_std = np.median(cell_std)
    cell_scale = (cell_std / median_std) ** power

    # Clamp to moderate range
    cell_scale = np.clip(cell_scale, clamp_min, clamp_max)

    print(f"Cell noise scale (power={power}, clamp=[{clamp_min}, {clamp_max}]):")
    for r in range(5):
        print(f"  {' '.join(f'{cell_scale[r,c]:.3f}' for c in range(5))}")
    print(f"  Range: [{cell_scale.min():.3f}, {cell_scale.max():.3f}], Median: {np.median(cell_scale):.3f}")

    return torch.tensor(cell_scale, dtype=torch.float32)


def load_model(model_path, device, **config_overrides):
    """Load a Block-AR model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    for k, v in config_overrides.items():
        config_dict[k] = v
    config = BlockARConfig(**config_dict)
    model = ConditionalBlockARDDPM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, config_dict


def evaluate_coverage(model, test_loader, n_samples, max_batches, device,
                      max_global_residual=0):
    """Generate samples and evaluate per-cell coverage."""
    all_samples = []
    all_gt = []
    all_history = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            samples = model.sample_batched(
                history, n_samples=n_samples, max_residual=0,
                max_global_residual=max_global_residual,
            )
            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_history.append(denormalize_iv(history).cpu().numpy())

    samples = np.concatenate(all_samples, axis=0)  # (N, n_samples, 30, 5, 5)
    gt = np.concatenate(all_gt, axis=0)              # (N, 30, 5, 5)
    hist = np.concatenate(all_history, axis=0)        # (N, 30, 5, 5)

    # Overall coverage
    alpha = 0.05
    lower = np.quantile(samples, alpha, axis=1)
    upper = np.quantile(samples, 1 - alpha, axis=1)
    covered = (gt >= lower) & (gt <= upper)
    overall = covered.mean() * 100

    # Regime split
    daily_changes = np.diff(hist, axis=1)
    vol_of_vol = daily_changes.std(axis=(1, 2, 3))
    q20 = np.percentile(vol_of_vol, 20)
    q80 = np.percentile(vol_of_vol, 80)
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80

    horizons = [(0, 1), (6, 7), (13, 14), (29, 30)]

    under_count = 0
    over_count = 0
    results = {}

    for regime, mask, regime_name in [("calm", calm_mask, "calm"), ("turb", turb_mask, "turb")]:
        results[regime] = {}
        for h_idx, h_name in horizons:
            cov_cells = covered[mask][:, h_idx]  # (n_regime, 5, 5)
            cell_cov = cov_cells.mean(axis=0) * 100
            n_under = (cell_cov < 70).sum()
            n_over = (cell_cov > 95).sum()
            under_count += n_under
            over_count += n_over

            results[regime][str(h_name)] = {
                "grid": cell_cov.tolist(),
                "worst": float(cell_cov.min()),
                "best": float(cell_cov.max()),
            }

    # Kurtosis (aggregate)
    daily_changes_model = np.diff(samples[:, :, :, :, :], axis=2)  # (N, S, 29, 5, 5)
    daily_changes_gt = np.diff(gt, axis=1)  # (N, 29, 5, 5)
    from scipy.stats import kurtosis
    model_kurtosis = kurtosis(daily_changes_model.flatten())
    gt_kurtosis = kurtosis(daily_changes_gt.flatten())
    kurtosis_ratio = model_kurtosis / gt_kurtosis if gt_kurtosis > 0 else 0

    return {
        "overall_coverage": overall,
        "under_70": int(under_count),
        "over_95": int(over_count),
        "combined": int(under_count + over_count),
        "kurtosis_ratio": kurtosis_ratio,
        "kurtosis_model": model_kurtosis,
        "kurtosis_gt": gt_kurtosis,
        "n_calm": int(calm_mask.sum()),
        "n_turb": int(turb_mask.sum()),
        "regime_results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--power", type=float, default=0.5,
                        help="Power for cell_noise_scale (0=uniform, 1=full)")
    parser.add_argument("--clamp_min", type=float, default=0.5)
    parser.add_argument("--clamp_max", type=float, default=2.0)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--max_global_residual", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/block_ar/exp46_cell_noise")
    parser.add_argument("--no_ema", action="store_true")
    args = parser.parse_args()

    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model, config_dict = load_model(args.model_path, device)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    # Compute cell noise scale from training data
    cell_noise_scale = compute_cell_noise_scale(
        surfaces, config_dict,
        power=args.power,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
    )

    # Set on model for inference
    model._cell_noise_scale = cell_noise_scale
    print(f"\nCell noise scale set on model (power={args.power})")

    # Test data
    test_ds = VolSurfaceDataset(surfaces, start_idx=4540, end_idx=len(surfaces),
                                 history_len=30, future_len=30)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Evaluate
    print(f"\nEvaluating: {args.max_batches} batches, {args.n_samples} samples...")
    results = evaluate_coverage(model, test_loader, args.n_samples,
                                 args.max_batches, device,
                                 args.max_global_residual)

    print(f"\n=== Results ===")
    print(f"Overall coverage: {results['overall_coverage']:.1f}%")
    print(f"Under 70%: {results['under_70']}")
    print(f"Over 95%: {results['over_95']}")
    print(f"Combined: {results['combined']}")
    print(f"Kurtosis ratio: {results['kurtosis_ratio']:.3f}")
    print(f"Regimes: {results['n_calm']} calm, {results['n_turb']} turb")

    for regime_name in ['calm', 'turb']:
        print(f"\n  {regime_name}:")
        for h in ['1', '7', '14', '30']:
            r = results['regime_results'][regime_name][h]
            print(f"    h={h:>2}: worst={r['worst']:.1f}% best={r['best']:.1f}%")

    # Also run baseline (no cell noise scale) for comparison
    print("\n=== Baseline (no cell noise scale) ===")
    model._cell_noise_scale = None
    baseline_results = evaluate_coverage(model, test_loader, args.n_samples,
                                          args.max_batches, device,
                                          args.max_global_residual)
    print(f"Overall coverage: {baseline_results['overall_coverage']:.1f}%")
    print(f"Under 70%: {baseline_results['under_70']}")
    print(f"Over 95%: {baseline_results['over_95']}")
    print(f"Combined: {baseline_results['combined']}")
    print(f"Kurtosis ratio: {baseline_results['kurtosis_ratio']:.3f}")

    # Save
    # Convert numpy types for JSON
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    output = to_native({
        "model": args.model_path,
        "power": args.power,
        "clamp": [args.clamp_min, args.clamp_max],
        "cell_noise_scale": cell_noise_scale.tolist(),
        "with_cell_noise": results,
        "baseline": baseline_results,
    })
    output_path = f"{args.output_dir}/results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
