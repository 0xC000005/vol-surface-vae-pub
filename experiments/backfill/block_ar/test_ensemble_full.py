#!/usr/bin/env python
"""Full 8-suite ensemble evaluation for multi-architecture models.

Loads multiple SinglePassBlockAR models (potentially different architectures),
generates samples from each, concatenates into a combined ensemble, and runs
all 8 test suites + composite score.

Supports per-model noise_dist overrides for inference.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/test_ensemble_full.py \
        --models 133f 108a 111b 120b \
        --samples_per_model 13 13 13 13 \
        --noise_overrides none gaussian none gaussian \
        --output_dir results/block_ar/E_best4_30d \
        --max_batches 20 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    denormalize_iv,
)
from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig
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
)


def load_single_pass_model(model_path: str, device: str,
                           noise_override: str = None) -> SinglePassBlockAR:
    """Load a SinglePassBlockAR model with optional noise_dist override."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]

    sp_cfg = {k: v for k, v in raw_config.items()
              if k in SinglePassConfig.__dataclass_fields__}

    if noise_override and noise_override != "none":
        original = sp_cfg.get("noise_dist", "gaussian")
        sp_cfg["noise_dist"] = noise_override
        print(f"    noise_dist: {original} -> {noise_override}")

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
    noise = sp_config.noise_dist
    print(f"    arch={arch}, noise={noise}, epoch={epoch}, params={n_params:,}")

    return model


def generate_ensemble_samples(
    models: List[SinglePassBlockAR],
    samples_per_model: List[int],
    test_loader: DataLoader,
    max_batches: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate samples from all models and concatenate along sample dim.

    Returns:
        cond_samples: (N, total_samples, T, 5, 5) in [0, 1]
        ground_truth: (N, T, 5, 5) in [0, 1]
        history: (N, H, 5, 5) in [0, 1]
    """
    # Cache test batches
    cached_batches = []
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
        cached_batches.append(batch)

    n_batches = len(cached_batches)
    total_samples = sum(samples_per_model)
    print(f"\n  Ensemble config: {len(models)} models, "
          f"samples={samples_per_model}, total={total_samples}")
    print(f"  Test batches: {n_batches}")

    # Generate per-model samples for each batch
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

    # Concatenate: for each batch, concat model samples along dim=1
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


def run_ensemble_conditionality(
    models: List[SinglePassBlockAR],
    samples_per_model: List[int],
    test_loader: DataLoader,
    max_batches: int,
    device: str,
) -> Dict:
    """Run conditionality test (Suite 3) for the ensemble.

    Generates conditional + unconditional (zero-history) samples from each model,
    then combines for turb/calm analysis.
    """
    MAX_UNCOND_BATCHES = 5

    cond_widths = []
    uncond_widths = []
    cond_maes = []
    uncond_maes = []
    per_window_cond_width_list = []
    all_batch_vov = []
    cond_width_sum = np.zeros((5, 5))
    uncond_width_sum = np.zeros((5, 5))
    cond_mae_sum = np.zeros((5, 5))
    uncond_mae_sum = np.zeros((5, 5))
    n_cell_samples = 0
    n_uncond_batches = 0

    cond_horizons = [1, 7, 14, 30]
    per_h_cond_width = {h: [] for h in cond_horizons}
    per_h_uncond_width = {h: [] for h in cond_horizons}

    for batch_idx, batch in enumerate(
        tqdm(test_loader, desc="Conditionality (ensemble)", total=max_batches)
    ):
        if batch_idx >= max_batches:
            break

        history = batch["history"].to(device)
        future_gt = denormalize_iv(batch["future"].to(device))
        B = history.shape[0]
        if B < 2:
            continue

        extra = batch.get("history_returns")
        if extra is not None:
            extra = extra.to(device)

        # Vol-of-vol for regime classification
        hist_np = denormalize_iv(history).cpu().numpy()
        mean_iv_hist = hist_np.mean(axis=(2, 3))
        daily_ch = np.diff(mean_iv_hist, axis=1)
        batch_vov = daily_ch.std(axis=1)
        all_batch_vov.append(batch_vov)

        # Conditional samples from all models
        cond_parts = []
        for model, n_s in zip(models, samples_per_model):
            with torch.no_grad():
                s = model.sample_batched(history, n_samples=n_s, extra_hist=extra)
            cond_parts.append(s.cpu().numpy())
        cond_np = np.concatenate(cond_parts, axis=1)

        gt_np = future_gt.cpu().numpy()
        eval_T = min(cond_np.shape[2], gt_np.shape[1])
        cond_np = cond_np[:, :, :eval_T]
        gt_np = gt_np[:, :eval_T]

        # CI width
        lo = np.percentile(cond_np, 5, axis=1)
        hi = np.percentile(cond_np, 95, axis=1)
        width = (hi - lo).mean(axis=(1, 2, 3))
        cond_widths.extend(width.tolist())

        cell_width = (hi - lo).mean(axis=1)  # (B, 5, 5)
        per_window_cond_width_list.append(cell_width)

        median_pred = np.median(cond_np, axis=1)
        cell_mae = np.abs(median_pred - gt_np).mean(axis=1)
        cond_width_sum += cell_width.sum(axis=0)
        cond_mae_sum += cell_mae.sum(axis=0)
        n_cell_samples += B

        mae = np.abs(median_pred - gt_np).mean(axis=(1, 2, 3))
        cond_maes.extend(mae.tolist())

        # Per-horizon
        for h in cond_horizons:
            h_idx = min(h - 1, eval_T - 1)
            lo_h = np.percentile(cond_np[:, :, h_idx], 5, axis=1)
            hi_h = np.percentile(cond_np[:, :, h_idx], 95, axis=1)
            wh = (hi_h - lo_h).mean(axis=(1, 2))
            per_h_cond_width[h].extend(wh.tolist())

        # Unconditional (first few batches only)
        if batch_idx < MAX_UNCOND_BATCHES:
            zero_history = torch.zeros_like(history)
            uncond_parts = []
            for model, n_s in zip(models, samples_per_model):
                with torch.no_grad():
                    s = model.sample_batched(zero_history, n_samples=n_s)
                uncond_parts.append(s.cpu().numpy())
            uncond_np = np.concatenate(uncond_parts, axis=1)[:, :, :eval_T]

            lo_u = np.percentile(uncond_np, 5, axis=1)
            hi_u = np.percentile(uncond_np, 95, axis=1)
            u_width = (hi_u - lo_u).mean(axis=(1, 2, 3))
            uncond_widths.extend(u_width.tolist())

            u_cell_width = (hi_u - lo_u).mean(axis=1)
            u_median = np.median(uncond_np, axis=1)
            u_cell_mae = np.abs(u_median - gt_np).mean(axis=1)
            uncond_width_sum += u_cell_width.sum(axis=0)
            uncond_mae_sum += u_cell_mae.sum(axis=0)
            n_uncond_batches += B

            u_mae = np.abs(u_median - gt_np).mean(axis=(1, 2, 3))
            uncond_maes.extend(u_mae.tolist())

            for h in cond_horizons:
                h_idx = min(h - 1, eval_T - 1)
                lo_uh = np.percentile(uncond_np[:, :, h_idx], 5, axis=1)
                hi_uh = np.percentile(uncond_np[:, :, h_idx], 95, axis=1)
                wuh = (hi_uh - lo_uh).mean(axis=(1, 2))
                per_h_uncond_width[h].extend(wuh.tolist())

        torch.cuda.empty_cache()

    # Compute results
    mean_cond_width = np.mean(cond_widths) if cond_widths else 0.01
    mean_uncond_width = np.mean(uncond_widths) if uncond_widths else 0.01
    width_ratio = mean_cond_width / max(mean_uncond_width, 1e-8)

    mean_cond_mae = np.mean(cond_maes) if cond_maes else 0.01
    mean_uncond_mae = np.mean(uncond_maes) if uncond_maes else 0.01
    mae_reduction_pct = max(0, (1 - mean_cond_mae / max(mean_uncond_mae, 1e-8))) * 100

    # Per-cell MAE reduction
    if n_cell_samples > 0 and n_uncond_batches > 0:
        avg_cond_mae = cond_mae_sum / n_cell_samples
        avg_uncond_mae = uncond_mae_sum / n_uncond_batches
        cell_mae_reduction = (1 - avg_cond_mae / np.clip(avg_uncond_mae, 1e-8, None)) * 100
        worst_cell_mae_reduction = float(cell_mae_reduction.min())
    else:
        worst_cell_mae_reduction = 0.0

    # Turb/calm split
    all_vov = np.concatenate(all_batch_vov)
    vov_median = np.median(all_vov)
    per_window_width = np.concatenate(per_window_cond_width_list, axis=0)

    turb_mask = all_vov > vov_median
    calm_mask = ~turb_mask

    if turb_mask.sum() > 0 and calm_mask.sum() > 0:
        turb_avg = per_window_width[turb_mask].mean()
        calm_avg = per_window_width[calm_mask].mean()
        turb_calm_ratio = float(turb_avg / max(calm_avg, 1e-8))
    else:
        turb_calm_ratio = 1.0

    # Growing uncertainty
    h_means = [np.mean(per_h_cond_width[h]) for h in cond_horizons
               if per_h_cond_width[h]]
    growing = (all(h_means[i] <= h_means[i+1] * 1.1
                   for i in range(len(h_means)-1))
               if len(h_means) > 1 else True)

    # Per-regime conditionality (informational)
    per_regime_cond = {}
    if n_uncond_batches > 0:
        for regime, mask in [("calm", calm_mask), ("turb", turb_mask)]:
            if mask.sum() > 0:
                r_width = per_window_width[mask].mean()
                per_regime_cond[regime] = {
                    "avg_width_ratio": float(r_width / max(mean_uncond_width, 1e-8)),
                    "worst_cell_width_ratio": float(r_width / max(mean_uncond_width, 1e-8)),
                }

    results = {
        "width_ratio": round(width_ratio, 4),
        "mae_reduction_pct": round(mae_reduction_pct, 1),
        "mae_pass": mae_reduction_pct > 5.0,
        "worst_cell_mae_reduction": round(worst_cell_mae_reduction, 1),
        "worst_cell_mae_pass": worst_cell_mae_reduction > -10.0,
        "turb_calm_ratio": round(turb_calm_ratio, 4),
        "turb_calm_pass": turb_calm_ratio > 1.15,
        "growing_uncertainty_monotonic": growing,
        "per_regime_conditionality": per_regime_cond,
        "pass": ((turb_calm_ratio > 1.15)
                 and (mae_reduction_pct > 5.0)
                 and (worst_cell_mae_reduction > -10.0)),
    }

    print("\n" + "=" * 60)
    print("TEST SUITE 3: CONDITIONALITY (ENSEMBLE)")
    print("=" * 60)
    print(f"  Turb/Calm ratio:        {turb_calm_ratio:.3f} (target >1.15) "
          f"{'PASS' if turb_calm_ratio > 1.15 else 'FAIL'}")
    print(f"  Width ratio (c/u):      {width_ratio:.3f} (informational)")
    print(f"  MAE reduction:          {mae_reduction_pct:.1f}% "
          f"{'PASS' if mae_reduction_pct > 5.0 else 'FAIL'}")
    print(f"  Worst cell MAE red:     {worst_cell_mae_reduction:.1f}% "
          f"{'PASS' if worst_cell_mae_reduction > -10.0 else 'FAIL'}")
    print(f"  Growing uncertainty:    {'PASS' if growing else 'FAIL'}")
    print(f"  Overall:                {'PASS' if results['pass'] else 'FAIL'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Full 8-suite ensemble evaluation for multi-architecture models"
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model IDs (e.g., 133f 108a 111b 120b)")
    parser.add_argument("--samples_per_model", nargs="+", type=int, required=True,
                        help="Samples per model (e.g., 17 17 17 or 13 13 13 13)")
    parser.add_argument("--noise_overrides", nargs="+", default=None,
                        help="Per-model noise_dist override (none/gaussian/student_t)")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_base", type=str,
                        default="models/backfill/afcrps_{}/best_model.pt",
                        help="Path template with {} for model ID")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    assert len(args.models) == len(args.samples_per_model), \
        f"models ({len(args.models)}) and samples_per_model ({len(args.samples_per_model)}) must match"

    if args.noise_overrides:
        assert len(args.noise_overrides) == len(args.models), \
            f"noise_overrides ({len(args.noise_overrides)}) must match models ({len(args.models)})"
    else:
        args.noise_overrides = ["none"] * len(args.models)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = sum(args.samples_per_model)

    print("=" * 60)
    print("MULTI-ARCHITECTURE ENSEMBLE EVALUATION")
    print("=" * 60)
    for i, (mid, ns, no) in enumerate(zip(args.models, args.samples_per_model,
                                           args.noise_overrides)):
        print(f"  [{i}] {mid}: {ns} samples, noise={no}")
    print(f"  Total: {total_samples} samples")
    print(f"  Max batches: {args.max_batches}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    device = args.device

    # Load models
    print("\nLoading models...")
    models = []
    for i, model_id in enumerate(args.models):
        path = args.model_base.format(model_id)
        print(f"  [{i}] {model_id}: {path}")
        model = load_single_pass_model(path, device, args.noise_overrides[i])
        models.append(model)

    # Load test data
    print("\nLoading test data...")
    from experiments.backfill.block_ar.config_block_ar import get_default_config
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]
    returns = data["ret"] if "ret" in data else None

    test_dataset = VolSurfaceDataset(
        surfaces,
        config.history_len,
        config.future_len,
        start_idx=config.test_start,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"  Test windows: {len(test_dataset)}, batch_size: {config.batch_size}")

    # =====================================================================
    # Generate ensemble samples
    # =====================================================================
    print("\n" + "=" * 60)
    print("GENERATING ENSEMBLE SAMPLES")
    print("=" * 60)
    cond_samples, ground_truth, history_arr = generate_ensemble_samples(
        models, args.samples_per_model, test_loader, args.max_batches, device,
    )
    print(f"\n  Ensemble samples: {cond_samples.shape}")
    print(f"  Ground truth:     {ground_truth.shape}")
    print(f"  History:          {history_arr.shape}")

    # =====================================================================
    # Run all 8 test suites
    # =====================================================================
    results = {}

    # Suite 1: Surface Validity
    results["surface"] = run_surface_validity_tests(cond_samples, ground_truth)

    # Suite 2: CI Coverage
    results["coverage"] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Suite 3: Conditionality (needs fresh model inference)
    cond_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    results["conditionality"] = run_ensemble_conditionality(
        models, args.samples_per_model, cond_loader,
        max_batches=min(args.max_batches, 15), device=device,
    )

    # Suite 4: Time Series
    results["time_series"] = run_time_series_tests(cond_samples, ground_truth)

    # Suite 5: Block-AR (growing uncertainty)
    results["block_ar"] = run_block_ar_tests(cond_samples, block_size=10)

    # Suite 6: Cointegration
    if returns is not None:
        results["cointegration"] = run_cointegration_tests(
            cond_samples, ground_truth,
            returns=returns,
            test_start=config.test_start,
            history_len=config.history_len,
            future_len=config.future_len,
        )

    # Suite 7: Regime Coverage
    results["regime_coverage"] = run_regime_coverage_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Suite 8: Distributional Fidelity
    results["distributional"] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history_arr,
    )

    # Summary
    all_pass = print_summary(results)

    # Eval provenance
    results["eval_config"] = {
        "ensemble_type": "multi_architecture",
        "models": args.models,
        "model_paths": [args.model_base.format(m) for m in args.models],
        "samples_per_model": args.samples_per_model,
        "noise_overrides": args.noise_overrides,
        "total_samples": total_samples,
        "max_batches": args.max_batches,
    }

    # Save results JSON
    results_serializable = convert_to_serializable(results)
    json_path = str(output_dir / "summary.json")
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Compute composite score
    try:
        from autoresearch_session.compute_score import compute_score
        score = compute_score(json_path)
        print(f"\n{'=' * 60}")
        print(f"COMPOSITE SCORE: {score['total_score']} / {score['max_possible']} "
              f"(suites: {score['suites_passed']}/8)")
        print(f"{'=' * 60}")
        print(json.dumps(score["components"], indent=2))

        score_path = str(output_dir / "composite_score.json")
        with open(score_path, "w") as f:
            json.dump(score, f, indent=2)
    except Exception as e:
        print(f"  Composite score error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
