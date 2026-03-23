#!/usr/bin/env python3
"""RC11-H0: Inference-time noise scaling probe.

Monkey-patches 144b's _sample_noise to scale z by beta, then runs the full
test suite. Tests whether CI bottleneck is amplitude or structure.

Kill condition: beta=3.0 gives CI<80% → rank-1 prevents coverage.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig, normalize_iv


def run_probe(model_path, beta, output_dir, device, max_batches=20, n_samples=50, seed=42):
    """Run test suite with noise scaled by beta."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    sp_cfg = {k: v for k, v in checkpoint["config"].items()
              if k in SinglePassConfig.__dataclass_fields__}
    config = SinglePassConfig(**sp_cfg)
    model = SinglePassBlockAR(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval()

    # Monkey-patch _sample_noise to scale by beta
    if beta != 1.0:
        original_fn = model._sample_noise
        def scaled_noise(B, dev):
            z = original_fn(B, dev)
            return z * beta
        model._sample_noise = scaled_noise
        print(f"  Noise scaling: beta={beta}")

    # Load data
    from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
        VolSurfaceDataset, generate_all_samples, get_default_config,
        run_surface_validity_tests, run_ci_coverage_tests,
        run_conditionality_tests, run_time_series_tests, run_block_ar_tests,
        run_cointegration_tests, run_regime_coverage_tests,
        run_distributional_fidelity_tests, run_cross_cell_correlation_tests,
    )
    from torch.utils.data import DataLoader

    test_config = get_default_config()
    data = np.load(test_config.data_path)
    surfaces = data["surface"]
    returns = data["ret"] if "ret" in data else None

    extra_features = getattr(config, "extra_features", 0)
    return_scale = getattr(config, "return_scale", 0.05)
    model_returns = data["ret"] if (extra_features > 0 and "ret" in data) else None

    test_dataset = VolSurfaceDataset(
        surfaces, test_config.history_len, test_config.future_len,
        start_idx=test_config.test_start,
        returns=model_returns, return_scale=return_scale,
    )
    test_loader = DataLoader(test_dataset, batch_size=test_config.batch_size,
                             shuffle=False, num_workers=0)

    # Generate samples
    print(f"\n{'='*60}")
    print(f"RC11-H0: beta={beta} noise scaling probe")
    print(f"{'='*60}")
    cond_samples, ground_truth, history_arr = generate_all_samples(
        model, test_loader, n_samples=n_samples, max_batches=max_batches,
        max_residual=20, device=device,
    )
    print(f"  Samples: {cond_samples.shape}, GT: {ground_truth.shape}")

    # Run all 9 test suites
    results = {}

    # Suite 1: Surface Validity
    results["surface"] = run_surface_validity_tests(cond_samples, ground_truth)

    # Suite 2: CI Coverage
    results["coverage"] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Suite 3: Conditionality (needs model for unconditioned samples)
    cond_test_loader = DataLoader(test_dataset, batch_size=test_config.batch_size,
                                  shuffle=False, num_workers=0)
    results["conditionality"] = run_conditionality_tests(
        model, cond_test_loader, n_samples=n_samples,
        max_batches=min(max_batches, 15), device=device,
    )

    # Suite 4: Time Series
    results["time_series"] = run_time_series_tests(cond_samples, ground_truth)

    # Suite 5: Block-AR
    block_size = getattr(config, "block_size", 10)
    results["block_ar"] = run_block_ar_tests(cond_samples, block_size=block_size)

    # Suite 6: Cointegration
    results["cointegration"] = run_cointegration_tests(
        cond_samples, ground_truth, returns=returns,
        test_start=test_config.test_start, history_len=test_config.history_len,
    )

    # Suite 7: Regime Coverage
    results["regime_coverage"] = run_regime_coverage_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Suite 8: Distributional
    results["distributional"] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Suite 9: Cross-cell Correlation
    results["cross_cell_correlation"] = run_cross_cell_correlation_tests(
        cond_samples, ground_truth,
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Extract key metrics (keys may be float or string depending on source)
    cov_overall = results["coverage"]["overall"]
    ci_90 = (cov_overall.get(0.9) or cov_overall.get("0.9")) * 100
    cov_h1 = results["coverage"]["per_horizon"].get(1) or results["coverage"]["per_horizon"].get("1")
    ci_h1 = (cov_h1.get(0.9) or cov_h1.get("0.9")) * 100
    ks_daily = results["distributional"]["ks_test"]["n_pass"]
    ks_level = results["distributional"]["ks_level_test"]["n_pass"]
    eff_rank = results["cross_cell_correlation"]["gen_eff_rank"]
    suites_passed = sum(1 for s in results.values()
                       if isinstance(s, dict) and s.get("overall_pass", False))
    suites_total = 9

    print(f"\nbeta={beta}: {suites_passed}/{suites_total} suites")
    print(f"  CI: {ci_90:.1f}% (h=1: {ci_h1:.1f}%)")
    print(f"  KS daily: {ks_daily}/25, KS level: {ks_level}/25")
    print(f"  Eff rank: {eff_rank:.2f}")

    return {
        "beta": beta,
        "suites_passed": suites_passed,
        "ci_90": ci_90,
        "ci_h1": ci_h1,
        "ks_daily": ks_daily,
        "ks_level": ks_level,
        "eff_rank": eff_rank,
        "suite_details": {k: v.get("overall_pass", False) for k, v in results.items()
                         if isinstance(v, dict)},
    }


def main():
    parser = argparse.ArgumentParser(description="RC11-H0: Noise scaling probe")
    parser.add_argument("--model_path", type=str,
                       default="models/backfill/afcrps_144b/best_model.pt")
    parser.add_argument("--betas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_base", type=str, default="results/block_ar/148_probe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_results = {}
    for beta in args.betas:
        output_dir = f"{args.output_base}_beta{beta}"
        result = run_probe(
            args.model_path, beta, output_dir, args.device,
            args.max_batches, args.n_samples, args.seed,
        )
        all_results[f"beta_{beta}"] = result

    # Save comparison
    comparison_path = f"{args.output_base}_comparison.json"
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*80}")
    print(f"RC11-H0 NOISE SCALING PROBE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Beta':>6} | {'Suites':>8} | {'CI%':>8} | {'CI h=1%':>8} | {'KS_daily':>10} | {'KS_level':>10} | {'Eff_rank':>10}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for beta in args.betas:
        r = all_results[f"beta_{beta}"]
        print(f"{beta:>6.1f} | {r['suites_passed']:>2}/9     | {r['ci_90']:>7.1f} | {r['ci_h1']:>7.1f} | {r['ks_daily']:>5}/25    | {r['ks_level']:>5}/25    | {r['eff_rank']:>9.2f}")

    # Interpretation
    ci_at_3 = all_results[f"beta_3.0"]["ci_90"]
    if ci_at_3 >= 80:
        print(f"\n✓ AMPLITUDE BOTTLENECK: beta=3.0 CI={ci_at_3:.1f}% ≥ 80%")
        print("  → Spread encouragement (skip bypass, heteroscedastic) should help")
    else:
        print(f"\n✗ STRUCTURE BOTTLENECK: beta=3.0 CI={ci_at_3:.1f}% < 80%")
        print("  → Rank-1 prevents coverage. Fix structure before amplitude.")


if __name__ == "__main__":
    main()
