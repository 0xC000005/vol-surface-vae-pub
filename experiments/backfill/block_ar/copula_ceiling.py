#!/usr/bin/env python
"""
H0: Gaussian Copula Ceiling Measurement (RC14)

Estimate GT Gaussian copula from training daily changes, generate samples
from 146b, apply ECC (Ensemble Copula Coupling) reordering, evaluate with
full 9-suite test battery.

This is a CEILING measurement: if Suite 9 passes with perfect copula,
then cross-cell correlation is the bottleneck (proceed to H1a/H1b).
If Suite 9 still fails, marginals are broken (proceed to H2).

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/copula_ceiling.py \
        --model_path models/backfill/afcrps_146b/best_model.pt \
        --no_ema --max_batches 20 --n_samples 50 \
        --output_dir results/block_ar/151a_copula_30d --device cuda
"""

import argparse
import json
import hashlib
import sys
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig, normalize_iv, denormalize_iv,
)
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_surface_validity_tests,
    run_ci_coverage_tests,
    run_conditionality_tests,
    run_time_series_tests,
    run_block_ar_tests,
    run_cointegration_tests,
    run_regime_coverage_tests,
    run_distributional_fidelity_tests,
    run_cross_cell_correlation_tests,
    print_summary,
    convert_to_serializable,
    hash_file,
    generate_all_samples,
)


def estimate_gaussian_copula(surfaces, train_end=4540):
    """Estimate 25x25 Spearman rank correlation from training daily changes."""
    train = surfaces[:train_end]
    changes = np.diff(train, axis=0)  # (N-1, 5, 5)
    flat = changes.reshape(-1, 25)    # (N-1, 25)
    corr, _ = spearmanr(flat)         # (25, 25)
    # Ensure positive-definite (numerical)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize diagonal to 1
    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / np.outer(d, d)
    L = np.linalg.cholesky(corr_pd)
    print(f"  Copula estimated from {len(flat)} daily changes")
    print(f"  Mean off-diag correlation: {corr[np.triu_indices(25, k=1)].mean():.3f}")
    print(f"  Copula effective rank: {np.exp(-np.sum((eigvals/eigvals.sum()) * np.log(eigvals/eigvals.sum() + 1e-10))):.2f}")
    return corr_pd, L


def apply_ecc_reorder(samples, L, seed=42):
    """Apply Ensemble Copula Coupling to reorder ensemble members.

    For each (window, timestep), reorder the K members across 25 cells
    so that cross-cell correlation matches the Gaussian copula L @ L.T.
    Marginals are preserved exactly (same K values per cell).

    Args:
        samples: (N, K, T, 5, 5) in [0,1]
        L: (25, 25) Cholesky factor of GT copula
    Returns:
        reordered: (N, K, T, 5, 5) with GT cross-cell correlation
    """
    N, K, T, H, W = samples.shape
    n_cells = H * W
    rng = np.random.RandomState(seed)

    # Reshape: merge N and T for vectorized processing
    # (N, K, T, 5, 5) → (N, K, T, 25) → (N*T, K, 25)
    flat = samples.reshape(N, K, T, n_cells)
    flat = flat.transpose(0, 2, 1, 3).reshape(N * T, K, n_cells)  # (N*T, K, 25)

    # Sort each cell's K values
    sorted_idx = np.argsort(flat, axis=1)
    sorted_vals = np.take_along_axis(flat, sorted_idx, axis=1)  # (N*T, K, 25)

    # Draw templates from N(0, Sigma) using Cholesky
    z = rng.randn(N * T, K, n_cells)    # (N*T, K, 25)
    templates = z @ L.T                  # (N*T, K, 25)

    # Rank templates per cell: rank 0..K-1
    template_ranks = np.argsort(np.argsort(templates, axis=1), axis=1)  # (N*T, K, 25)

    # Reassign: member k, cell j = sorted_vals[template_ranks[k, j], j]
    reordered = np.take_along_axis(sorted_vals, template_ranks, axis=1)  # (N*T, K, 25)

    # Reshape back: (N*T, K, 25) → (N, T, K, 25) → (N, K, T, 5, 5)
    reordered = reordered.reshape(N, T, K, n_cells).transpose(0, 2, 1, 3)
    reordered = reordered.reshape(N, K, T, H, W)

    # Verify marginals preserved
    orig_means = samples.mean(axis=1)  # (N, T, 5, 5)
    reord_means = reordered.mean(axis=1)
    max_diff = np.abs(orig_means - reord_means).max()
    print(f"  Marginal preservation check: max mean diff = {max_diff:.6f}")

    return reordered


def main():
    parser = argparse.ArgumentParser(description="H0: Gaussian Copula Ceiling")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--max_residual", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H0: GAUSSIAN COPULA CEILING MEASUREMENT")
    print("=" * 60)
    print(f"Model:       {args.model_path}")
    print(f"Samples:     {args.n_samples}")
    print(f"Max batches: {args.max_batches}")

    # ── Step 1: Estimate GT copula ──
    print("\n── Estimating GT Gaussian copula ──")
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]
    returns = data["ret"] if "ret" in data else None
    copula_corr, L = estimate_gaussian_copula(surfaces, train_end=config.test_start)

    # ── Step 2: Load model ──
    print("\n── Loading model ──")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    sp_cfg = {k: v for k, v in checkpoint["config"].items()
              if k in SinglePassConfig.__dataclass_fields__}
    sp_config = SinglePassConfig(**sp_cfg)
    model = SinglePassBlockAR(sp_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"  Loaded epoch {checkpoint.get('epoch', '?')}")

    # ── Step 3: Load test data ──
    _cfg = model.config
    extra_features = getattr(_cfg, "extra_features", 0)
    return_scale = getattr(_cfg, "return_scale", 0.05)
    model_returns = data["ret"] if (extra_features > 0 and "ret" in data) else None

    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start,
        returns=model_returns, return_scale=return_scale,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0)
    print(f"  Test windows: {len(test_dataset)}")

    # ── Step 4: Set seeds & generate samples ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n── Generating samples (original) ──")
    cond_samples, ground_truth, history_arr = generate_all_samples(
        model, test_loader, n_samples=args.n_samples,
        max_batches=args.max_batches, max_residual=args.max_residual,
        device=device,
    )
    print(f"  Samples: {cond_samples.shape}")

    # ── Step 5: Apply ECC copula reordering ──
    print("\n── Applying Gaussian copula (ECC) reordering ──")
    reordered = apply_ecc_reorder(cond_samples, L, seed=args.seed)

    # Quick pre-check: Suite 9 metric before and after
    from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
        run_cross_cell_correlation_tests as _s9,
    )
    print("\n── Pre-check: Suite 9 before vs after copula ──")
    print("BEFORE copula:")
    s9_before = _s9(cond_samples, ground_truth)
    print("AFTER copula:")
    s9_after = _s9(reordered, ground_truth)
    print(f"\n  rank_ratio: {s9_before['rank_ratio']:.3f} → {s9_after['rank_ratio']:.3f}")
    print(f"  corr_ratio: {s9_before['corr_ratio']:.3f} → {s9_after['corr_ratio']:.3f}")
    print(f"  Suite 9: {'PASS' if s9_before['overall_pass'] else 'FAIL'} → "
          f"{'PASS' if s9_after['overall_pass'] else 'FAIL'}")

    # ── Step 6: Run full 9-suite evaluation on reordered samples ──
    print("\n" + "=" * 60)
    print("FULL 9-SUITE EVALUATION ON COPULA-REORDERED SAMPLES")
    print("=" * 60)

    results = {}
    results['surface'] = run_surface_validity_tests(reordered, ground_truth)
    results['coverage'] = run_ci_coverage_tests(reordered, ground_truth)

    # Suite 3: conditionality — uses model directly, NOT reordered samples
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    cond_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0)
    results['conditionality'] = run_conditionality_tests(
        model, cond_loader, n_samples=args.n_samples,
        max_batches=min(args.max_batches, 15),
        max_residual=args.max_residual, device=device,
    )

    results['time_series'] = run_time_series_tests(reordered, ground_truth)
    results['block_ar'] = run_block_ar_tests(reordered, block_size=config.block_size)

    if returns is not None:
        results['cointegration'] = run_cointegration_tests(
            reordered, ground_truth, returns=returns,
            test_start=config.test_start,
            history_len=config.history_len, future_len=config.future_len,
        )

    results['regime_coverage'] = run_regime_coverage_tests(
        reordered, ground_truth, history_arr,
    )
    results['distributional'] = run_distributional_fidelity_tests(
        reordered, ground_truth, history_arr,
    )
    results['cross_cell_correlation'] = run_cross_cell_correlation_tests(
        reordered, ground_truth,
    )

    # ── Summary ──
    print_summary(results)

    # ── Save ──
    results['copula_diagnostic'] = {
        'copula_type': 'gaussian_spearman',
        'copula_mean_offdiag': float(copula_corr[np.triu_indices(25, k=1)].mean()),
        'suite9_before_copula': convert_to_serializable(s9_before),
        'suite9_after_copula': convert_to_serializable(s9_after),
        'marginal_preservation_max_diff': float(
            np.abs(cond_samples.mean(1) - reordered.mean(1)).max()
        ),
    }
    results['eval_config'] = {
        'checkpoint_path': str(Path(args.model_path).resolve()),
        'checkpoint_hash': hash_file(args.model_path),
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'n_samples': args.n_samples,
        'max_batches': args.max_batches,
        'copula_reorder': True,
        'experiment_type': 'H0_copula_ceiling',
    }

    results_ser = convert_to_serializable(results)
    json_path = f"{args.output_dir}/summary.json"
    with open(json_path, 'w') as f:
        json.dump(results_ser, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
