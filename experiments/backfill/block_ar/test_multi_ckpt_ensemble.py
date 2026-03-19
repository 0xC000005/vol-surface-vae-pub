#!/usr/bin/env python
"""
Multi-Checkpoint Ensemble Evaluation.

Loads multiple model checkpoints, generates samples from each, combines
into one ensemble, and runs the full 8-suite test battery.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/test_multi_ckpt_ensemble.py \
        --models models/backfill/afcrps_108a/best_model.pt \
                 models/backfill/afcrps_99m_v2/best_model.pt \
                 models/backfill/afcrps_111b/best_model.pt \
        --samples_per_model 17 \
        --noise_dist_override 0:gaussian \
        --max_batches 20 --device cuda \
        --output_dir results/block_ar/E3_multi_ckpt_30d
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig
from diffusion.block_ar.block_ar_ddpm import denormalize_iv, BlockARConfig
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.block_ar.test_block_ar_requirements import (
    run_surface_validity_tests,
    run_ci_coverage_tests,
    run_time_series_tests,
    run_block_ar_tests,
    run_cointegration_tests,
    run_regime_coverage_tests,
    run_distributional_fidelity_tests,
    print_summary,
    convert_to_serializable,
    plot_calibration_curve,
    plot_acf_comparison,
    visualize_generated_paths,
    plot_growing_uncertainty,
    hash_file,
)


def load_single_pass_model(model_path: str, device: str,
                           noise_dist_override: str = None) -> SinglePassBlockAR:
    """Load a SinglePassBlockAR model from checkpoint.

    Args:
        model_path: Path to checkpoint .pt file
        device: 'cuda' or 'cpu'
        noise_dist_override: If set, override the noise_dist config
            (e.g. 'gaussian' to use Gaussian inference for a Student-t model)

    Returns:
        model on device, in eval mode
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg_dict = {k: v for k, v in ckpt["config"].items()
                if k in SinglePassConfig.__dataclass_fields__}
    config = SinglePassConfig(**cfg_dict)

    if noise_dist_override:
        print(f"    noise_dist override: {config.noise_dist} -> {noise_dist_override}")
        config.noise_dist = noise_dist_override

    model = SinglePassBlockAR(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Loaded epoch {epoch}, {n_params:,} params, "
          f"ar_frame={config.ar_frame}, noise_dist={config.noise_dist}")
    return model


def generate_samples_from_model(
    model: SinglePassBlockAR,
    test_loader: DataLoader,
    n_samples: int,
    max_batches: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate samples from a single model for all test windows.

    Returns:
        samples: (N, n_samples, T, 5, 5) in [0, 1]
        ground_truth: (N, T, 5, 5) in [0, 1]
        history: (N, H, 5, 5) in [0, 1]
    """
    all_samples = []
    all_gt = []
    all_history = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="  Generating", total=max_batches, leave=False)
        ):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))
            extra_hist = batch.get("history_returns")
            if extra_hist is not None:
                extra_hist = extra_hist.to(device)

            samples = model.sample_batched(
                history, n_samples=n_samples, max_residual=20,
                extra_hist=extra_hist,
            )

            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_history.append(denormalize_iv(history).cpu().numpy())

    return (
        np.concatenate(all_samples, axis=0),
        np.concatenate(all_gt, axis=0),
        np.concatenate(all_history, axis=0),
    )


def generate_conditionality_samples(
    models: List[SinglePassBlockAR],
    test_loader: DataLoader,
    n_samples_per_model: int,
    max_batches: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate conditional + unconditional ensemble samples for conditionality test.

    For each batch, generates n_samples_per_model from each model for both
    conditional (real history) and unconditional (zero history) cases,
    then concatenates across models.

    Returns:
        cond_samples: (N, total_samples, T, 5, 5)
        uncond_samples: (N, total_samples, T, 5, 5)
        ground_truth: (N, T, 5, 5)
        history_denorm: (N, H, 5, 5)
    """
    all_cond = []
    all_uncond = []
    all_gt = []
    all_hist = []

    MAX_UNCOND_BATCHES = 5

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="  Conditionality ensemble", total=max_batches, leave=False)
        ):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            batch_cond_parts = []
            batch_uncond_parts = []

            for model in models:
                model.eval()
                extra_hist = batch.get("history_returns")
                if extra_hist is not None:
                    extra_hist = extra_hist.to(device)

                cond = model.sample_batched(
                    history, n_samples=n_samples_per_model, max_residual=20,
                    extra_hist=extra_hist,
                )
                batch_cond_parts.append(cond.cpu().numpy())

                if batch_idx < MAX_UNCOND_BATCHES:
                    zero_history = torch.zeros_like(history)
                    uncond = model.sample_batched(
                        zero_history, n_samples=n_samples_per_model, max_residual=20,
                    )
                    batch_uncond_parts.append(uncond.cpu().numpy())

            all_cond.append(np.concatenate(batch_cond_parts, axis=1))
            if batch_uncond_parts:
                all_uncond.append(np.concatenate(batch_uncond_parts, axis=1))
            all_gt.append(future_gt.cpu().numpy())
            all_hist.append(denormalize_iv(history).cpu().numpy())

    cond_samples = np.concatenate(all_cond, axis=0)
    uncond_samples = np.concatenate(all_uncond, axis=0) if all_uncond else None
    ground_truth = np.concatenate(all_gt, axis=0)
    history_arr = np.concatenate(all_hist, axis=0)

    return cond_samples, uncond_samples, ground_truth, history_arr


def run_ensemble_conditionality(
    models: List[SinglePassBlockAR],
    test_loader: DataLoader,
    n_samples_per_model: int,
    max_batches: int,
    device: str,
) -> Dict:
    """Run conditionality test using multi-model ensemble.

    Adapts the logic from run_conditionality_tests() but generates
    samples from all models combined.
    """
    print("\n" + "=" * 60)
    print("TEST SUITE 3: CONDITIONALITY (ENSEMBLE)")
    print("=" * 60)

    cond_all, uncond_all, gt_all, hist_all = generate_conditionality_samples(
        models, test_loader, n_samples_per_model,
        min(max_batches, 15), device,
    )

    N = cond_all.shape[0]
    T = min(cond_all.shape[2], gt_all.shape[1])
    cond_all = cond_all[:, :, :T]
    gt_all = gt_all[:, :T]

    # Conditional metrics
    cond_lower = np.quantile(cond_all, 0.05, axis=1)
    cond_upper = np.quantile(cond_all, 0.95, axis=1)
    cond_width = (cond_upper - cond_lower).mean()
    cond_median = np.median(cond_all, axis=1)
    cond_mae = np.abs(cond_median - gt_all).mean()

    # Per-cell conditional MAE
    per_cell_cond_mae = np.abs(cond_median - gt_all).mean(axis=(0, 1))  # (5, 5)

    # Unconditional metrics
    if uncond_all is not None:
        uncond_all = uncond_all[:, :, :T]
        # Only first MAX_UNCOND_BATCHES windows were sampled unconditionally
        n_uncond = uncond_all.shape[0]
        uncond_lower = np.quantile(uncond_all, 0.05, axis=1)
        uncond_upper = np.quantile(uncond_all, 0.95, axis=1)
        uncond_width = (uncond_upper - uncond_lower).mean()
        uncond_median = np.median(uncond_all, axis=1)
        uncond_mae = np.abs(uncond_median - gt_all[:n_uncond, :T]).mean()
        # Per-cell uncond MAE
        per_cell_uncond_mae = np.abs(uncond_median - gt_all[:n_uncond, :T]).mean(axis=(0, 1))
    else:
        uncond_width = cond_width
        uncond_mae = cond_mae
        per_cell_uncond_mae = per_cell_cond_mae

    width_ratio = cond_width / uncond_width if uncond_width > 0 else 1.0
    mae_reduction_pct = (uncond_mae - cond_mae) / uncond_mae * 100 if uncond_mae > 0 else 0.0
    mae_pass = mae_reduction_pct > 5.0

    # Per-cell MAE reduction
    per_cell_mae_reduction = (per_cell_uncond_mae - per_cell_cond_mae) / per_cell_uncond_mae * 100
    worst_cell_mae_reduction = per_cell_mae_reduction.min()
    worst_cell_mae_pass = worst_cell_mae_reduction > -15.0

    print(f"  Width ratio (cond/uncond): {width_ratio:.3f} (informational)")
    print(f"    Cond width:   {cond_width:.4f}")
    print(f"    Uncond width: {uncond_width:.4f}")
    print(f"  MAE reduction: {mae_reduction_pct:.1f}% "
          f"(target >5%) {'PASS' if mae_pass else 'FAIL'}")
    print(f"    Cond MAE:   {cond_mae:.4f}")
    print(f"    Uncond MAE: {uncond_mae:.4f}")
    print(f"  Worst cell MAE red: {worst_cell_mae_reduction:.1f}% "
          f"{'PASS' if worst_cell_mae_pass else 'FAIL'}")

    # Turb/calm width ratio from history volatility-of-volatility
    mean_iv_hist = hist_all.mean(axis=(2, 3))  # (N, H)
    daily_ch = np.diff(mean_iv_hist, axis=1)
    vov = daily_ch.std(axis=1)  # (N,)
    q20 = np.percentile(vov, 20)
    q80 = np.percentile(vov, 80)
    calm_mask = vov <= q20
    turb_mask = vov >= q80

    calm_width = (cond_upper[calm_mask] - cond_lower[calm_mask]).mean() if calm_mask.any() else cond_width
    turb_width = (cond_upper[turb_mask] - cond_lower[turb_mask]).mean() if turb_mask.any() else cond_width
    turb_calm_ratio = turb_width / calm_width if calm_width > 0 else 1.0
    turb_calm_pass = turb_calm_ratio > 1.15

    print(f"  Turb/Calm ratio: {turb_calm_ratio:.3f} "
          f"(target >1.15) {'PASS' if turb_calm_pass else 'FAIL'}")
    print(f"    Calm width: {calm_width:.4f} ({calm_mask.sum()} windows)")
    print(f"    Turb width: {turb_width:.4f} ({turb_mask.sum()} windows)")

    # Growing uncertainty
    var_per_time = cond_all.var(axis=1).mean(axis=(0, 2, 3))  # (T,)
    horizon_keys = [1, 10, 20, 30]
    avg_horizon_var = {}
    for h in horizon_keys:
        if h <= T:
            avg_horizon_var[h] = float(var_per_time[h - 1])

    valid_keys = sorted(avg_horizon_var.keys())
    monotonic = all(
        avg_horizon_var[valid_keys[i]] < avg_horizon_var[valid_keys[i + 1]]
        for i in range(len(valid_keys) - 1)
    )
    print(f"  Growing uncertainty: {'PASS' if monotonic else 'FAIL'} (informational)")

    overall_pass = turb_calm_pass and mae_pass and worst_cell_mae_pass

    return {
        'width_ratio': float(width_ratio),
        'cond_width': float(cond_width),
        'uncond_width': float(uncond_width),
        'mae_reduction_pct': float(mae_reduction_pct),
        'mae_pass': mae_pass,
        'turb_calm_ratio': float(turb_calm_ratio),
        'turb_calm_pass': turb_calm_pass,
        'worst_cell_mae_reduction': float(worst_cell_mae_reduction),
        'worst_cell_mae_pass': worst_cell_mae_pass,
        'growing_uncertainty_monotonic': monotonic,
        'per_cell_mae_reduction': per_cell_mae_reduction.tolist(),
        'avg_horizon_var': avg_horizon_var,
        'pass': overall_pass,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Checkpoint Ensemble Evaluation (full 8-suite test)"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to model checkpoints",
    )
    parser.add_argument(
        "--samples_per_model", type=int, default=17,
        help="Samples to generate per model (total = n_models * this)",
    )
    parser.add_argument(
        "--noise_dist_override", nargs="*", default=[],
        help="Override noise_dist for specific models. Format: idx:dist "
             "(e.g. 0:gaussian means model 0 uses gaussian inference)",
    )
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir", type=str,
        default="results/block_ar/E3_multi_ckpt_30d",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Parse noise_dist overrides
    noise_overrides = {}
    for item in args.noise_dist_override:
        idx_str, dist = item.split(":")
        noise_overrides[int(idx_str)] = dist

    n_models = len(args.models)
    total_samples = n_models * args.samples_per_model

    # Header
    print("=" * 60)
    print("Multi-Checkpoint Ensemble Evaluation")
    print("=" * 60)
    print(f"Models:          {n_models}")
    for i, p in enumerate(args.models):
        override_note = f" [noise_dist -> {noise_overrides[i]}]" if i in noise_overrides else ""
        print(f"  [{i}] {p}{override_note}")
    print(f"Samples/model:   {args.samples_per_model}")
    print(f"Total ensemble:  {total_samples}")
    print(f"Max batches:     {args.max_batches}")
    print(f"Device:          {device}")
    print(f"Output:          {output_dir}")
    print("=" * 60)

    # =====================================================================
    # Load test data
    # =====================================================================
    config = get_default_config()
    print("\nLoading test data...")
    data = np.load(config.data_path)
    surfaces = data["surface"]
    returns = data["ret"] if "ret" in data else None

    # Check if any model uses extra_features
    all_extra_features = []
    for mp in args.models:
        ckpt = torch.load(mp, map_location="cpu", weights_only=False)
        all_extra_features.append(ckpt["config"].get("extra_features", 0))
    max_extra = max(all_extra_features)
    model_returns = data["ret"] if (max_extra > 0 and "ret" in data) else None
    return_scale = 0.05  # default

    test_dataset = VolSurfaceDataset(
        surfaces,
        config.history_len,
        config.future_len,
        start_idx=config.test_start,
        returns=model_returns,
        return_scale=return_scale,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"  Test set: {len(test_dataset)} windows")

    # =====================================================================
    # Load all models
    # =====================================================================
    print("\nLoading models...")
    models = []
    for i, model_path in enumerate(args.models):
        print(f"  [{i}] {model_path}")
        override = noise_overrides.get(i, None)
        model = load_single_pass_model(model_path, device, noise_dist_override=override)
        models.append(model)

    # =====================================================================
    # Generate samples from each model
    # =====================================================================
    print(f"\nGenerating {args.samples_per_model} samples per model...")
    all_model_samples = []
    ground_truth = None
    history_arr = None

    for i, model in enumerate(models):
        print(f"\n  Model [{i}]: {args.models[i]}")
        samples_i, gt_i, hist_i = generate_samples_from_model(
            model, test_loader, args.samples_per_model,
            args.max_batches, device,
        )
        print(f"    -> samples: {samples_i.shape}")
        all_model_samples.append(samples_i)
        if ground_truth is None:
            ground_truth = gt_i
            history_arr = hist_i

    # Combine samples along the ensemble dimension
    cond_samples = np.concatenate(all_model_samples, axis=1)
    print(f"\nEnsemble samples: {cond_samples.shape}")
    print(f"Ground truth: {ground_truth.shape}")
    print(f"History: {history_arr.shape}")

    # Save raw ensemble samples for reproducibility
    npz_path = f"{output_dir}/ensemble_samples.npz"
    np.savez_compressed(
        npz_path,
        cond_samples=cond_samples,
        ground_truth=ground_truth,
        history=history_arr,
    )
    print(f"Saved ensemble samples to {npz_path}")

    # =====================================================================
    # Run all 8 test suites
    # =====================================================================
    results = {}

    # Test Suite 1: Surface Validity
    results['surface'] = run_surface_validity_tests(cond_samples, ground_truth)

    # Test Suite 2: CI Coverage
    results['coverage'] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Test Suite 3: Conditionality (needs model access for uncond samples)
    cond_test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    results['conditionality'] = run_ensemble_conditionality(
        models, cond_test_loader, args.samples_per_model,
        args.max_batches, device,
    )

    # Test Suite 4: Time Series Properties
    results['time_series'] = run_time_series_tests(cond_samples, ground_truth)

    # Test Suite 5: Block-AR Specific
    results['block_ar'] = run_block_ar_tests(cond_samples, block_size=10)

    # Test Suite 6: Cointegration
    if returns is not None:
        results['cointegration'] = run_cointegration_tests(
            cond_samples, ground_truth,
            returns=returns,
            test_start=config.test_start,
            history_len=config.history_len,
            future_len=config.future_len,
        )
    else:
        print("\n  Skipping cointegration test (no returns data)")

    # Test Suite 7: Three-Layer Regime Coverage
    results['regime_coverage'] = run_regime_coverage_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Test Suite 8: Distributional Fidelity
    results['distributional'] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history_arr,
    )

    # =====================================================================
    # Summary
    # =====================================================================
    all_pass = print_summary(results)

    # =====================================================================
    # Per-model comparison
    # =====================================================================
    print("\n" + "=" * 60)
    print("PER-MODEL INDIVIDUAL COVERAGE")
    print("=" * 60)
    for i, model_samples in enumerate(all_model_samples):
        lower_i = np.quantile(model_samples, 0.05, axis=1)
        upper_i = np.quantile(model_samples, 0.95, axis=1)
        cov_i = ((ground_truth >= lower_i) & (ground_truth <= upper_i)).mean()
        print(f"  Model [{i}] {Path(args.models[i]).parent.name}: "
              f"90% CI = {cov_i:.1%} ({model_samples.shape[1]} samples)")

    # =====================================================================
    # Visualizations
    # =====================================================================
    print("\nGenerating visualizations...")
    try:
        plot_calibration_curve(
            results['coverage'], f"{output_dir}/calibration_curve.png"
        )
        plot_acf_comparison(
            results['time_series']['acf'], f"{output_dir}/acf_comparison.png"
        )
        visualize_generated_paths(
            cond_samples, ground_truth, idx=0,
            output_path=f"{output_dir}/path_visualization.png",
        )
        plot_growing_uncertainty(
            cond_samples, f"{output_dir}/uncertainty_growth.png"
        )
    except Exception as e:
        print(f"  Visualization error (non-fatal): {e}")

    # =====================================================================
    # Eval provenance
    # =====================================================================
    results['eval_config'] = {
        'ensemble': True,
        'n_models': n_models,
        'model_paths': [str(Path(p).resolve()) for p in args.models],
        'model_hashes': [hash_file(p) for p in args.models],
        'samples_per_model': args.samples_per_model,
        'total_samples': total_samples,
        'noise_dist_overrides': {str(k): v for k, v in noise_overrides.items()},
        'max_batches': args.max_batches,
        'device': device,
    }

    # =====================================================================
    # Save results JSON
    # =====================================================================
    results_serializable = convert_to_serializable(results)
    json_path = f"{output_dir}/summary.json"
    with open(json_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"All outputs saved to: {output_dir}")

    # Clean up GPU memory
    for m in models:
        del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
