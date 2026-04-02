"""Evaluate all baselines (classical + deep) through the v2 test suite (9 suites).

v2 improvements over v1:
- Suite 8: distributional fidelity with KS-levels fix
- Suite 9: cross-cell correlation structure
- Fixed ACF computation, multi-sample kurtosis, random seed, EWMA warmup, tenor weights

Usage:
    # All 10 baselines (6 classical + 4 deep):
    PYTHONPATH=. python experiments/backfill/baselines/evaluate_baselines_v2.py \
        --n_samples 50 --max_batches 20 --device cuda

    # Specific baselines:
    PYTHONPATH=. python experiments/backfill/baselines/evaluate_baselines_v2.py \
        --baselines random_walk garch_ccc csdi --n_samples 50 --device cuda

    # Quick test:
    PYTHONPATH=. python experiments/backfill/baselines/evaluate_baselines_v2.py \
        --baselines random_walk --n_samples 10 --max_batches 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.backfill.baselines.classical_baselines import (
    RandomWalkBaseline,
    HistoricalSimulation,
    UnconditionalBootstrap,
    PCAVARBaseline,
    GARCHCCCBaseline,
    FilteredHistoricalSimulation,
)
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
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# All baseline names in evaluation order
CLASSICAL_BASELINES = [
    "random_walk", "historical_sim", "bootstrap",
    "pca_var", "garch_ccc", "filtered_hs",
]
DEEP_BASELINES = ["csdi", "timegrad", "deepvar", "vae"]
ALL_BASELINES = CLASSICAL_BASELINES + DEEP_BASELINES


def compute_crps(samples: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute CRPS (Continuous Ranked Probability Score) per horizon.

    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are iid samples and y is the observation.

    Lower is better. Proper scoring rule.

    Args:
        samples: (N, K, T, 5, 5) generated samples in [0, 1]
        ground_truth: (N, T, 5, 5) ground truth in [0, 1]
    Returns:
        dict with per-horizon and overall CRPS
    """
    N, K, T = samples.shape[:3]
    horizons = [1, 7, 14, 30]

    crps_per_horizon = {}
    for h in horizons:
        if h > T:
            continue
        t = h - 1
        s = samples[:, :, t]  # (N, K, 5, 5)
        y = ground_truth[:, t]  # (N, 5, 5)

        # E|X - y|: mean over samples of |sample - gt|
        mae_term = np.abs(s - y[:, None]).mean(axis=1)  # (N, 5, 5)

        # E|X - X'|: mean over pairs of |sample_i - sample_j|
        spread_term = np.zeros((N, 5, 5), dtype=np.float64)
        for i in range(K):
            for j in range(i + 1, K):
                spread_term += np.abs(s[:, i] - s[:, j])
        spread_term = spread_term * 2.0 / (K * (K - 1))

        crps = mae_term - 0.5 * spread_term  # (N, 5, 5)
        crps_per_horizon[str(h)] = float(crps.mean())

    overall_crps = float(np.mean(list(crps_per_horizon.values())))

    return {"per_horizon": crps_per_horizon, "overall": overall_crps}


def compute_energy_score(samples: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute Energy Score (multivariate CRPS) per horizon.

    ES = E||X - y||_2 - 0.5 * E||X - X'||_2
    where norms are over the spatial dimensions (25-dim vector).

    Lower is better. Captures cross-cell correlation quality.

    Args:
        samples: (N, K, T, 5, 5)
        ground_truth: (N, T, 5, 5)
    """
    N, K, T = samples.shape[:3]
    horizons = [1, 7, 14, 30]

    es_per_horizon = {}
    for h in horizons:
        if h > T:
            continue
        t = h - 1
        s = samples[:, :, t].reshape(N, K, 25)  # (N, K, 25)
        y = ground_truth[:, t].reshape(N, 25)  # (N, 25)

        # E||X - y||_2
        mae_term = np.linalg.norm(s - y[:, None], axis=2).mean(axis=1)  # (N,)

        # E||X - X'||_2 via subsampling
        n_pairs = min(500, K * (K - 1) // 2)
        rng = np.random.RandomState(42)
        spread_sum = np.zeros(N)
        for _ in range(n_pairs):
            i, j = rng.choice(K, size=2, replace=False)
            spread_sum += np.linalg.norm(s[:, i] - s[:, j], axis=1)
        spread_term = spread_sum / n_pairs

        es = mae_term - 0.5 * spread_term  # (N,)
        es_per_horizon[str(h)] = float(es.mean())

    overall_es = float(np.mean(list(es_per_horizon.values())))
    return {"per_horizon": es_per_horizon, "overall": overall_es}


def generate_baseline_samples(model, test_loader, n_samples, max_batches, device):
    """Generate samples from a baseline model.

    Returns:
        cond_samples: (N, n_samples, T, 5, 5) in [0, 1]
        ground_truth: (N, T, 5, 5) in [0, 1]
        history_arr: (N, T_hist, 5, 5) in [0, 1]
    """
    all_samples, all_gt, all_history = [], [], []

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break

        history = batch["history"].to(device)
        future_gt = denormalize_iv(batch["future"].to(device))
        samples = model.sample(history, n_samples=n_samples)
        history_denorm = denormalize_iv(history)

        all_samples.append(samples.cpu().numpy())
        all_gt.append(future_gt.cpu().numpy())
        all_history.append(history_denorm.cpu().numpy())

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{max_batches}")

    return (
        np.concatenate(all_samples, axis=0),
        np.concatenate(all_gt, axis=0),
        np.concatenate(all_history, axis=0),
    )


def run_baseline_evaluation(
    baseline_name, model, test_loader, test_dataset, returns,
    n_samples, max_batches, device, output_dir, config,
):
    """Run all 9 v2 test suites for a baseline model."""
    print("\n" + "=" * 70)
    print(f"  EVALUATING: {baseline_name}")
    print("=" * 70)

    t0 = time.time()
    print(f"\nGenerating {n_samples} samples per window...")
    cond_samples, ground_truth, history_arr = generate_baseline_samples(
        model, test_loader, n_samples, max_batches, device,
    )
    gen_time = time.time() - t0
    print(f"Generated {cond_samples.shape[0]} windows x {n_samples} samples "
          f"in {gen_time:.1f}s")

    results = {
        "baseline_name": baseline_name,
        "generation_time_s": gen_time,
        "n_windows": cond_samples.shape[0],
        "eval_config": {
            "n_samples": n_samples,
            "max_batches": max_batches,
            "batch_size": config["batch_size"],
            "seed": config.get("seed", 42),
            "train_end": config.get("train_end", 4040),
            "test_start": config["test_start"],
            "history_len": config["history_len"],
            "future_len": config["future_len"],
        },
    }

    # Scoring rules (CRPS, Energy Score)
    print("\n  Computing CRPS and Energy Score...")
    results["crps"] = compute_crps(cond_samples, ground_truth)
    results["energy_score"] = compute_energy_score(cond_samples, ground_truth)
    print(f"  CRPS: {results['crps']['overall']:.5f}  "
          f"ES: {results['energy_score']['overall']:.5f}")

    # Suite 1: Surface Validity
    results["surface"] = run_surface_validity_tests(cond_samples, ground_truth)

    # Suite 2: CI Coverage
    results["coverage"] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Suite 3: Conditionality
    # The v2 conditionality test takes a model and generates its own samples
    # via model.sample_batched(). All baselines implement sample_batched().
    cond_test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=0,
    )
    results["conditionality"] = run_conditionality_tests(
        model, cond_test_loader,
        n_samples=n_samples,
        max_batches=min(max_batches, 15),
        max_residual=20, device=device,
    )

    # Suite 4: Time Series Properties
    results["time_series"] = run_time_series_tests(cond_samples, ground_truth)

    # Suite 5: Block-AR Specific
    # Note: boundary smoothness is architecture-specific (trivial for non-AR baselines)
    # but growing uncertainty monotonicity is universally meaningful
    results["block_ar"] = run_block_ar_tests(cond_samples)

    # Suite 6: Cointegration
    if returns is not None:
        results["cointegration"] = run_cointegration_tests(
            cond_samples, ground_truth,
            returns=returns, test_start=config["test_start"],
            history_len=config["history_len"], future_len=config["future_len"],
        )

    # Suite 7: Regime Coverage
    results["regime_coverage"] = run_regime_coverage_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Suite 8: Distributional Fidelity (NEW in v2)
    results["distributional"] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Suite 9: Cross-Cell Correlation (NEW in v2)
    results["cross_cell_correlation"] = run_cross_cell_correlation_tests(
        cond_samples, ground_truth,
    )

    # Summary
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {baseline_name}")
    print("=" * 70)

    # Suites with annotations for paper footnotes
    # - S5 (block_ar): trivial for non-AR baselines (boundary smoothness is architecture-specific)
    # - S6 (cointegration): informational in the v2 suite's own logic
    suite_names = [
        ("surface", "Suite 1: Surface Validity", True),
        ("coverage", "Suite 2: CI Coverage", True),
        ("conditionality", "Suite 3: Conditionality", True),
        ("time_series", "Suite 4: Time Series", True),
        ("block_ar", "Suite 5: Block-AR Specific*", False),  # trivial for non-AR
        ("cointegration", "Suite 6: Cointegration†", False),  # informational
        ("regime_coverage", "Suite 7: Regime Coverage", True),
        ("distributional", "Suite 8: Distributional", True),
        ("cross_cell_correlation", "Suite 9: Cross-Cell Corr", True),
    ]

    n_pass = 0
    n_total = 0
    n_binding_pass = 0
    n_binding_total = 0
    for key, name, is_binding in suite_names:
        if key in results:
            passed = results[key].get("overall_pass") or results[key].get("pass", False)
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            n_total += 1
            if passed:
                n_pass += 1
            if is_binding:
                n_binding_total += 1
                if passed:
                    n_binding_pass += 1

    print(f"\n  Total: {n_pass}/{n_total} suites PASS "
          f"(binding: {n_binding_pass}/{n_binding_total})")
    print(f"  * Suite 5 is trivial for non-AR baselines (boundary smoothness)")
    print(f"  † Suite 6 is informational in the v2 test suite")
    results["suites_pass"] = n_pass
    results["suites_total"] = n_total
    results["suites_binding_pass"] = n_binding_pass
    results["suites_binding_total"] = n_binding_total

    # Save
    baseline_output_dir = Path(output_dir) / baseline_name
    baseline_output_dir.mkdir(parents=True, exist_ok=True)

    results_json = _convert_for_json(results)
    with open(baseline_output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"  Results saved to {baseline_output_dir}/results.json")
    return results


def _convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    return obj


def print_comparison_table(all_results: dict):
    """Print a comprehensive comparison table for all 9 suites."""
    print("\n" + "=" * 140)
    print("  COMPARISON TABLE: All Baselines (v2 — 9 Suites)")
    print("=" * 140)

    # Header
    header = (
        f"{'Method':<22} {'Pass':>5} "
        f"{'CRPS':>7} {'ES':>7} "
        f"{'CI@h1':>7} {'CI@h30':>7} {'CalErr':>7} "
        f"{'Kurt':>6} {'CalArb':>7} {'BflyArb':>7} "
        f"{'T/C':>5} "
        f"{'KS(d)':>6} {'KS(l)':>6} "
        f"{'CorrR':>6} {'RankR':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, res in all_results.items():
        suites = f"{res['suites_pass']}/{res['suites_total']}"

        crps = res.get("crps", {}).get("overall", 0)
        es = res.get("energy_score", {}).get("overall", 0)

        ph = res.get("coverage", {}).get("per_horizon", {})
        ci_h1 = (ph.get(1, {}) or ph.get("1", {}))
        ci_h1 = ci_h1.get(0.9, ci_h1.get("0.9", 0)) if isinstance(ci_h1, dict) else 0
        ci_h30 = (ph.get(30, {}) or ph.get("30", {}))
        ci_h30 = ci_h30.get(0.9, ci_h30.get("0.9", 0)) if isinstance(ci_h30, dict) else 0
        cal_err = res.get("coverage", {}).get("calibration_error", 0)

        kurt_data = res.get("time_series", {}).get("kurtosis", {})
        kurt = kurt_data.get("kurtosis_ratio", 0) if isinstance(kurt_data, dict) else 0

        cal = res.get("surface", {}).get("calendar", {}).get("calendar_avg_violation_rate", 0)
        but = res.get("surface", {}).get("butterfly", {}).get("butterfly_avg_violation_rate", 0)

        tc = res.get("conditionality", {}).get("turb_calm_ratio", 0)

        # Suite 8: distributional KS
        ks_daily = res.get("distributional", {}).get("ks_test", {}).get("n_pass", 0)
        ks_level = res.get("distributional", {}).get("ks_level_test", {}).get("n_pass", 0)

        # Suite 9: cross-cell correlation
        corr_r = res.get("cross_cell_correlation", {}).get("corr_ratio", 0)
        rank_r = res.get("cross_cell_correlation", {}).get("rank_ratio", 0)

        print(
            f"{name:<22} {suites:>5} "
            f"{crps:>7.4f} {es:>7.4f} "
            f"{ci_h1:>6.1%} {ci_h30:>6.1%} {cal_err:>7.3f} "
            f"{kurt:>6.3f} {cal:>6.1%} {but:>7.1%} "
            f"{tc:>5.2f} "
            f"{ks_daily:>4}/25 {ks_level:>4}/25 "
            f"{corr_r:>6.3f} {rank_r:>6.3f}"
        )

    # Reference: our best model
    print("-" * len(header))
    print(
        f"{'afCRPS 97a+qmap*':<22} {'5/9':>5} "
        f"{'~best':>7} {'~best':>7} "
        f"{'~87%':>7} {'~93%':>7} {'~0.05':>7} "
        f"{'0.919':>6} {'~9%':>7} {'~16%':>7} "
        f"{'1.64':>5} "
        f"{'24/25':>6} {'~?/25':>6} "
        f"{'~1.6':>6} {'~0.9':>6}"
    )
    print("\n  * afCRPS 97a+qmap is the proposed method (not run here "
          "-- see test_block_ar_requirements_v2.py)")

    # Per-suite pass/fail matrix
    print("\n  --- Per-Suite Pass/Fail ---")
    suite_keys = [
        ("surface", "S1:Surf"),
        ("coverage", "S2:CI"),
        ("conditionality", "S3:Cond"),
        ("time_series", "S4:TS"),
        ("block_ar", "S5:AR"),
        ("cointegration", "S6:Coint"),
        ("regime_coverage", "S7:Reg"),
        ("distributional", "S8:Dist"),
        ("cross_cell_correlation", "S9:Corr"),
    ]
    suite_header = f"{'Method':<22}" + "".join(f"{sn:>9}" for _, sn in suite_keys)
    print(suite_header)
    for name, res in all_results.items():
        row = f"{name:<22}"
        for key, _ in suite_keys:
            if key in res:
                passed = res[key].get("overall_pass") or res[key].get("pass", False)
                row += f"{'PASS':>9}" if passed else f"{'FAIL':>9}"
            else:
                row += f"{'N/A':>9}"
        print(row)

    # Per-horizon CRPS breakdown
    print("\n  --- CRPS by Horizon (lower = better) ---")
    h_header = f"{'Method':<22}" + "".join(f"{'h=' + h:>9}" for h in ["1", "7", "14", "30"])
    print(h_header)
    for name, res in all_results.items():
        crps_h = res.get("crps", {}).get("per_horizon", {})
        vals = "".join(
            f"{crps_h.get(h, crps_h.get(str(h), 0)):>9.5f}" for h in [1, 7, 14, 30]
        )
        print(f"{name:<22}{vals}")


def load_deep_baseline(name, device="cuda"):
    """Load a trained deep baseline model."""
    checkpoints = {
        "csdi": "models/backfill/baselines/csdi/best_model.pt",
        "timegrad": "models/backfill/baselines/timegrad/best_model.pt",
        "deepvar": "models/backfill/baselines/deepvar/best_model.pt",
        "vae": "models/backfill/context20_production/backfill_16yr.pt",
    }

    path = checkpoints[name]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {name}: {path}\n"
            f"Train it first or skip with --baselines <other baselines>"
        )

    if name == "csdi":
        from experiments.backfill.baselines.csdi_adapter import load_csdi_model
        return load_csdi_model(path, device)
    elif name == "timegrad":
        from experiments.backfill.baselines.timegrad_standalone import load_timegrad_model
        return load_timegrad_model(path, device)
    elif name == "deepvar":
        from experiments.backfill.baselines.deepvar_standalone import load_deepvar_model
        return load_deepvar_model(path, device)
    elif name == "vae":
        from experiments.backfill.baselines.vae_adapter import load_vae_model
        return load_vae_model(path, device)
    else:
        raise ValueError(f"Unknown deep baseline: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baselines through the v2 test suite (9 suites)"
    )
    parser.add_argument(
        "--baselines", nargs="*", default=None,
        help=f"Which baselines to evaluate (default: all 10). "
             f"Choices: {', '.join(ALL_BASELINES)}",
    )
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of ensemble members per window")
    parser.add_argument("--max_batches", type=int, default=20,
                        help="Maximum test batches to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (v2 feature)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="results/baselines_v2")
    parser.add_argument("--pca_components", type=int, default=5,
                        help="PCA components for PCA-VAR baseline")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    returns = data["ret"]

    print(f"  Surfaces: {surfaces.shape}, range [{surfaces.min():.3f}, {surfaces.max():.3f}]")
    print(f"  Returns:  {returns.shape}")

    # Standard Block-AR split (matches config_block_ar.py)
    history_len = 30
    future_len = 30
    train_end = 4040
    test_start = 4540

    config = {
        "history_len": history_len, "future_len": future_len,
        "batch_size": args.batch_size, "test_start": test_start,
        "train_end": train_end, "seed": args.seed,
    }

    train_surfaces = surfaces[:train_end]
    print(f"  Train: {train_surfaces.shape[0]} surfaces")
    print(f"  Test:  {surfaces.shape[0] - test_start} surfaces (from idx {test_start})")

    test_dataset = VolSurfaceDataset(
        surfaces=surfaces, history_len=history_len,
        future_len=future_len, start_idx=test_start,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    print(f"  Test dataset: {len(test_dataset)} windows")
    print(f"  Random seed: {args.seed}")

    # Determine which baselines to run
    if args.baselines is None:
        selected = list(ALL_BASELINES)
    else:
        selected = args.baselines
        # Validate
        for name in selected:
            if name not in ALL_BASELINES:
                parser.error(
                    f"Unknown baseline '{name}'. "
                    f"Choices: {', '.join(ALL_BASELINES)}"
                )

    # Separate into classical and deep
    classical_selected = [b for b in selected if b in CLASSICAL_BASELINES]
    deep_selected = [b for b in selected if b in DEEP_BASELINES]

    # Build classical baselines (all fitted on training data only)
    builders = {
        "random_walk": lambda: RandomWalkBaseline(train_surfaces, future_len=future_len),
        "historical_sim": lambda: HistoricalSimulation(
            train_surfaces, history_len=history_len, future_len=future_len),
        "bootstrap": lambda: UnconditionalBootstrap(train_surfaces, future_len=future_len),
        "pca_var": lambda: PCAVARBaseline(
            train_surfaces, future_len=future_len, n_components=args.pca_components),
        "garch_ccc": lambda: GARCHCCCBaseline(train_surfaces, future_len=future_len),
        "filtered_hs": lambda: FilteredHistoricalSimulation(
            train_surfaces, future_len=future_len),
    }

    baseline_models = {}
    for name in classical_selected:
        print(f"\nBuilding {name}...")
        baseline_models[name] = builders[name]()

    # Load deep baselines
    for name in deep_selected:
        print(f"\nLoading deep baseline: {name}...")
        try:
            baseline_models[name] = load_deep_baseline(name, device)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print(f"  Skipping {name}.")
            continue
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            print(f"  Skipping {name}.")
            continue

    if not baseline_models:
        print("\nNo baselines to evaluate. Exiting.")
        return

    print(f"\nWill evaluate {len(baseline_models)} baselines: "
          f"{', '.join(baseline_models.keys())}")

    # Evaluate
    all_results = {}
    for name, model in baseline_models.items():
        all_results[name] = run_baseline_evaluation(
            baseline_name=name, model=model,
            test_loader=test_loader, test_dataset=test_dataset,
            returns=returns, n_samples=args.n_samples,
            max_batches=args.max_batches, device=device,
            output_dir=args.output_dir, config=config,
        )

    # Comparison table
    print_comparison_table(all_results)

    # Save combined results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = _convert_for_json(all_results)
    with open(output_dir / "all_baselines_v2_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nAll results saved to {output_dir}/all_baselines_v2_results.json")


if __name__ == "__main__":
    main()
