#!/usr/bin/env python
"""
220b: Multi-horizon path-quality suite for recursive one-day kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    HistoryFutureDictDataset,
    OneDayKernelRolloutWrapper,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    rollout_samples_in_batches,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_ci_coverage_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_surface_validity_tests,
)


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    names = [
        ("surface", results["surface"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("conditionality", results["conditionality"]["overall_pass"]),
        ("distributional_fidelity", results["distributional_fidelity"]["overall_pass"]),
        ("cross_cell_correlation", results["cross_cell_correlation"]["overall_pass"]),
        ("mean_reversion", results["mean_reversion"]["overall_pass"]),
        ("pathwise_jump_realism", results["pathwise_jump_realism"]["overall_pass"]),
    ]
    failed = [name for name, passed in names if not passed]
    return sum(int(passed) for _name, passed in names), failed


def main() -> None:
    parser = argparse.ArgumentParser(description="220b recursive multi-horizon path suite")
    parser.add_argument("--model_type", type=str, default="212ai")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--force_native_anchor", action="store_true",
                        help="Force native-path rollout (model.sample_batched) with inference "
                             "anchor(0.50) for any 227a-family checkpoint. Use this to compare "
                             "229a honestly against 232 variants (which always use this regime).")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)

    # 232 variants train without anchor to preserve 229a's innovation-law calibration
    # (see 231 negative finding). They are *evaluated* with inference-time anchor(0.5),
    # which is the production recipe for the 229a family.
    if args.model_type in {'232a', '232b', '232c', '232d'}:
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 232 variant: use_scale_anchor=True, alpha=0.50 at inference")

    # 233a variants always use anchor (set in constructor). Keep it redundantly here
    # so --force_native_anchor still produces consistent settings regardless of how
    # the checkpoint was saved.
    if args.model_type.startswith('233a'):
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 233a {args.model_type}: use_scale_anchor=True, alpha=0.50 at inference")

    # --force_native_anchor: apply the same regime to 227a/229a baselines for fair comparison
    if args.force_native_anchor and args.model_type == '227a':
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] --force_native_anchor: use_scale_anchor=True alpha=0.50, native path")

    wrapper = OneDayKernelRolloutWrapper(model).eval()

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

    # Use model's native sample_batched for models that generate multi-day trajectories
    # (recurrent/adapter 221d+, smooth transport 183c, etc.)
    use_native = hasattr(model, 'sample_batched') and (
        hasattr(model, 'temporal_adapter')
        or args.model_type == '183c'
        or args.model_type in {'231a', '231b', '231c', '232a', '232b', '232c', '232d'}
        or args.model_type.startswith('233a')
        or args.force_native_anchor
    )
    if use_native:
        from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
        print("Using native sample_batched (recurrent + adapter)")
        outputs = []
        n_steps = batch.future_01.shape[1]
        for start in range(0, batch.history_01.shape[0], args.batch_size):
            end = min(start + args.batch_size, batch.history_01.shape[0])
            hist_batch = normalize_iv(batch.history_01[start:end])
            with torch.no_grad():
                samp = model.sample_batched(
                    hist_batch, n_samples=args.samples, n_steps=n_steps, chunk_size=args.chunk_size,
                )
            outputs.append(samp.cpu().numpy())
        cond_samples = np.concatenate(outputs, axis=0)
    else:
        cond_samples = rollout_samples_in_batches(
            wrapper=wrapper,
            history_norm=batch.history_norm,
            n_samples=args.samples,
            n_steps=batch.future_01.shape[1],
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)

    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    # For native multi-day models, pass model directly (has sample_batched)
    # For one-day kernels, pass wrapper (wraps sample_next_iv)
    cond_model = model if use_native else wrapper
    conditionality = run_conditionality_tests(
        cond_model,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {
        "n_pass": n_pass,
        "n_total": 7,
        "failed_suites": failed,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/7`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Coverage**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- worst cell h30: `{coverage['worst_cell_per_horizon'].get(30, float('nan')):.3f}`",
        f"- best cell h30: `{coverage['best_cell_per_horizon'].get(30, float('nan')):.3f}`",
        "",
        "**Conditionality**",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- MAE reduction vs shuffled: `{conditionality.get('mae_reduction_pct', float('nan')):.1f}%`",
        "",
        "**Mean Reversion**",
        f"- aggregate slope ratio: `{mean_reversion.get('mr_gt_ratio', float('nan')):.3f}`",
        f"- active pass count: `{mean_reversion.get('active_pass_count', 0)}/{mean_reversion.get('active_cell_count', 0)}`",
        f"- full-horizon overall: `{mean_reversion.get('full_horizon', {}).get('overall_pass', False)}`",
        "",
        "**Distributional Fidelity**",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- worst floor occupancy: `{distributional['explosion']['worst_cell_floor']:.3%}`",
        "",
        "**Cross-Cell / Pathwise**",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "220b Multi-Horizon Path Suite", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
