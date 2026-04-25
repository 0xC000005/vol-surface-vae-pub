#!/usr/bin/env python
"""426a: validation-oracle marginal quantile feasibility diagnostic for 392a.

This is not a deployable calibrated model. It deliberately uses validation futures to
measure whether a monotone marginal level correction is sufficient to cross the 8/11
frontier without changing the learned 392a generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    HistoryFutureDictDataset,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (  # noqa: E402
    suite_summary,
)
from experiments.backfill.block_ar.evaluate_403a_calibrated_risk_system import (  # noqa: E402
    CalibratedSampler,
    fit_calibration_tables,
    sample_native,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (  # noqa: E402
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--oracle_samples", type=int, default=48)
    parser.add_argument("--n_quantiles", type=int, default=201)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--seed", type=int, default=426)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    base_model.eval()

    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )

    print("Fitting validation-oracle marginal quantile map")
    oracle_samples = sample_native(
        model=base_model,
        history_norm=batch.history_norm,
        n_samples=args.oracle_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    tables = fit_calibration_tables(
        calib_samples=oracle_samples,
        calib_future=batch.future_01.detach().cpu().numpy(),
        calib_history=batch.history_01.detach().cpu().numpy(),
        n_quantiles=args.n_quantiles,
        alpha=args.alpha,
        regime_bins=args.regime_bins,
        min_bin_windows=args.min_bin_windows,
    )

    model = CalibratedSampler(base_model, tables, device).eval()

    outputs: list[np.ndarray] = []
    for start in range(0, batch.history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, batch.history_01.shape[0])
        with torch.no_grad():
            samples = model.sample_batched(
                batch.history_norm[start:end],
                n_samples=args.samples,
                n_steps=args.future_len,
                chunk_size=args.chunk_size,
                history_is_normalized=True,
            )
        outputs.append(samples.detach().cpu().numpy())
    cond_samples = np.concatenate(outputs, axis=0)
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    conditionality = run_conditionality_tests(
        model,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    cointegration = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    regime_coverage = run_regime_coverage_tests(cond_samples, ground_truth, history_01)
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "diagnostic": {
                "kind": "validation_oracle_marginal_quantile_map",
                "oracle_uses_validation_future": True,
                "claim": "feasibility_upper_bound_not_deployable_model",
                "oracle_samples": args.oracle_samples,
                "n_quantiles": args.n_quantiles,
                "alpha": args.alpha,
                "regime_bins": bool(args.regime_bins),
                "min_bin_windows": args.min_bin_windows,
                "seed": args.seed,
                "vov_q20": tables.vov_q20,
                "vov_q80": tables.vov_q80,
            },
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
            "rollout_start": int(rollout_start),
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "regime_coverage": regime_coverage,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")

    lines = [
        f"- base model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        "- diagnostic: `validation oracle marginal quantile map; not deployable`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        "",
        "**Fidelity / Structure**",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- median-bias fraction cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- median-bias magnitude cells: `{distributional['median_bias']['n_mag_pass']}/25`",
        f"- regime layer2: `{regime_coverage['layer2_n_passing']}/{regime_coverage['layer2_n_total']}`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "426a Validation-Oracle Quantile Diagnostic", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()

