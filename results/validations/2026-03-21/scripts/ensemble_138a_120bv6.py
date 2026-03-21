#!/usr/bin/env python
"""Ensemble evaluation: 138a best_model + 120b_v6 best_coverage_model.

50/50 ensemble (25 samples each, 50 total) evaluated on v2 test suites.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from diffusion.block_ar.block_ar_ddpm import denormalize_iv
from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

# Import v2 test functions
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_surface_validity_tests,
    run_ci_coverage_tests,
    run_time_series_tests,
    run_block_ar_tests,
    run_cointegration_tests,
    run_regime_coverage_tests,
    run_distributional_fidelity_tests,
    print_summary,
    convert_to_serializable,
)

# Import ensemble conditionality from the ensemble test script
from experiments.backfill.block_ar.test_ensemble_full import (
    run_ensemble_conditionality,
)


def load_model(path: str, device: str) -> SinglePassBlockAR:
    """Load a SinglePassBlockAR model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    sp_cfg = {k: v for k, v in raw_config.items()
              if k in SinglePassConfig.__dataclass_fields__}
    sp_config = SinglePassConfig(**sp_cfg)
    model = SinglePassBlockAR(sp_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters())
    is_joint = getattr(sp_config, "joint_decoder", False)
    is_ar = getattr(sp_config, "ar_frame", False)
    arch = "joint_transformer" if is_joint else ("AR_MLP" if is_ar else "block")
    print(f"  Loaded: {path}")
    print(f"    arch={arch}, noise={sp_config.noise_dist}, epoch={epoch}, params={n_params:,}")
    return model


def generate_ensemble_samples(models, samples_per_model, test_loader, max_batches, device):
    """Generate samples from all models and concatenate along sample dim."""
    cached_batches = []
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
        cached_batches.append(batch)

    n_batches = len(cached_batches)
    total_samples = sum(samples_per_model)
    print(f"\n  Ensemble: {len(models)} models, samples={samples_per_model}, total={total_samples}")
    print(f"  Test batches: {n_batches}")

    per_model_batch_samples = []
    for m_idx, (model, n_s) in enumerate(zip(models, samples_per_model)):
        print(f"\n  Model {m_idx}: generating {n_s} samples...")
        batch_samples = []
        with torch.no_grad():
            for b_idx in tqdm(range(n_batches), desc=f"  Model {m_idx}", leave=False):
                history = cached_batches[b_idx]["history"].to(device)
                extra = cached_batches[b_idx].get("history_returns")
                if extra is not None:
                    extra = extra.to(device)
                samples = model.sample_batched(
                    history, n_samples=n_s, extra_hist=extra,
                )
                batch_samples.append(samples.cpu().numpy())
        per_model_batch_samples.append(batch_samples)
        torch.cuda.empty_cache()

    all_cond = []
    all_gt = []
    all_hist = []
    for b_idx in range(n_batches):
        parts = [per_model_batch_samples[m][b_idx] for m in range(len(models))]
        combined = np.concatenate(parts, axis=1)
        all_cond.append(combined)
        gt = denormalize_iv(cached_batches[b_idx]["future"].to(device)).cpu().numpy()
        hist = denormalize_iv(cached_batches[b_idx]["history"].to(device)).cpu().numpy()
        all_gt.append(gt)
        all_hist.append(hist)

    cond_samples = np.concatenate(all_cond, axis=0)
    ground_truth = np.concatenate(all_gt, axis=0)
    history_arr = np.concatenate(all_hist, axis=0)

    return cond_samples, ground_truth, history_arr


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_batches = 20
    samples_per_model = [25, 25]

    model_paths = [
        str(ROOT / "models/backfill/afcrps_138a/best_model.pt"),
        str(ROOT / "models/backfill/afcrps_120b_v6/best_coverage_model.pt"),
    ]
    model_names = ["138a", "120b_v6"]

    output_dir = ROOT / "results/block_ar/ensemble_138a_120bv6_30d"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ENSEMBLE EVALUATION: 138a + 120b_v6 (v2 test suites)")
    print("=" * 60)
    for i, (name, path, ns) in enumerate(zip(model_names, model_paths, samples_per_model)):
        print(f"  [{i}] {name}: {ns} samples from {path}")
    print(f"  Total samples: {sum(samples_per_model)}")
    print(f"  Max batches: {max_batches}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    start_time = time.time()

    # Load models
    print("\nLoading models...")
    models = []
    for path in model_paths:
        model = load_model(path, device)
        models.append(model)

    # Load test data
    print("\nLoading test data...")
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]
    returns = data["ret"] if "ret" in data else None

    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0,
    )
    print(f"  Test windows: {len(test_dataset)}, batch_size: {config.batch_size}")

    # Generate ensemble samples
    print("\n" + "=" * 60)
    print("GENERATING ENSEMBLE SAMPLES")
    print("=" * 60)
    cond_samples, ground_truth, history_arr = generate_ensemble_samples(
        models, samples_per_model, test_loader, max_batches, device,
    )
    print(f"\n  Ensemble samples: {cond_samples.shape}")
    print(f"  Ground truth:     {ground_truth.shape}")
    print(f"  History:          {history_arr.shape}")

    # Run all 8 test suites (v2)
    results = {}

    # Suite 1: Surface Validity (v2)
    print("\n" + "=" * 60)
    print("Running v2 test suites...")
    results["surface"] = run_surface_validity_tests(cond_samples, ground_truth)

    # Suite 2: CI Coverage (v2)
    results["coverage"] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Suite 3: Conditionality (ensemble-specific — needs fresh inference)
    cond_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0,
    )
    results["conditionality"] = run_ensemble_conditionality(
        models, samples_per_model, cond_loader,
        max_batches=min(max_batches, 15), device=device,
    )

    # Suite 4: Time Series (v2)
    results["time_series"] = run_time_series_tests(cond_samples, ground_truth)

    # Suite 5: Block-AR (v2)
    results["block_ar"] = run_block_ar_tests(cond_samples, block_size=10)

    # Suite 6: Cointegration (v2)
    if returns is not None:
        results["cointegration"] = run_cointegration_tests(
            cond_samples, ground_truth,
            returns=returns,
            test_start=config.test_start,
            history_len=config.history_len,
            future_len=config.future_len,
        )

    # Suite 7: Regime Coverage (v2)
    results["regime_coverage"] = run_regime_coverage_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Suite 8: Distributional Fidelity (v2)
    results["distributional"] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Summary
    all_pass = print_summary(results)

    elapsed = time.time() - start_time

    # Eval provenance
    results["eval_config"] = {
        "ensemble_type": "50/50 ensemble",
        "models": model_names,
        "model_paths": model_paths,
        "samples_per_model": samples_per_model,
        "total_samples": sum(samples_per_model),
        "max_batches": max_batches,
        "test_version": "v2",
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save results
    results_serializable = convert_to_serializable(results)
    json_path = str(output_dir / "summary.json")
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Count suites
    suite_names = [
        ("surface", "overall_pass"),
        ("coverage", "pass"),
        ("conditionality", "pass"),
        ("time_series", "overall_pass"),
        ("block_ar", "overall_pass"),
        ("cointegration", "pass"),
        ("regime_coverage", "overall_pass"),
        ("distributional", "overall_pass"),
    ]
    n_pass = 0
    passing_suites = []
    failing_suites = []
    for suite_key, pass_key in suite_names:
        if suite_key in results:
            if results[suite_key].get(pass_key, False):
                n_pass += 1
                passing_suites.append(suite_key)
            else:
                failing_suites.append(suite_key)
    n_total = len([k for k, _ in suite_names if k in results])

    print(f"\nSuites: {n_pass}/{n_total} PASS")
    print(f"  Passing: {passing_suites}")
    print(f"  Failing: {failing_suites}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Compute composite score if available
    try:
        from autoresearch_session.compute_score import compute_score
        score = compute_score(json_path)
        print(f"\nCOMPOSITE SCORE: {score['total_score']} / {score['max_possible']} "
              f"(suites: {score['suites_passed']}/8)")

        score_path = str(output_dir / "composite_score.json")
        with open(score_path, "w") as f:
            json.dump(score, f, indent=2)
    except Exception as e:
        print(f"  Composite score error: {e}")


if __name__ == "__main__":
    main()
