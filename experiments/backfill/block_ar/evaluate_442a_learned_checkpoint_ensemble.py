#!/usr/bin/env python
"""442a: equal-weight learned checkpoint ensemble.

This is a deployable learned-law ensemble, not a calibration layer. It combines samples
from frozen learned conditional generators with no validation-future information and no
validation-tuned weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_403a_calibrated_risk_system import sample_native  # noqa: E402
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)


DEFAULT_CHECKPOINTS = [
    "models/backfill/391a_recent_rollout_energy_w02_s42/best_model.pt",
    "models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    "models/backfill/393a_recent_rollout_energy_w01_s42/best_model.pt",
]


def sample_checkpoint_ensemble(
    model_type: str,
    checkpoints: list[str],
    samples_per_model: list[int],
    history_norm: torch.Tensor,
    future_len: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(checkpoints) != len(samples_per_model):
        raise ValueError("checkpoints and samples_per_model must have the same length")
    outputs: list[np.ndarray] = []
    members: list[dict[str, Any]] = []
    for idx, (checkpoint, n_samples) in enumerate(zip(checkpoints, samples_per_model, strict=True)):
        if int(n_samples) <= 0:
            continue
        print(f"Sampling ensemble member {idx + 1}/{len(checkpoints)}: {checkpoint} ({n_samples} samples)")
        model, payload = load_one_day_kernel(model_type, checkpoint, device)
        model.eval()
        samples = sample_native(
            model=model,
            history_norm=history_norm,
            n_samples=int(n_samples),
            n_steps=future_len,
            batch_size=batch_size,
            chunk_size=chunk_size,
            device=device,
        )
        outputs.append(samples)
        members.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_epoch": int(payload.get("epoch", -1)),
                "samples": int(n_samples),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not outputs:
        raise ValueError("At least one ensemble member must have positive samples")
    return np.concatenate(outputs, axis=1).astype(np.float32), members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--samples_per_model", nargs="+", type=int, default=[16, 16, 16])
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=442)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
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
    cond_samples, members = sample_checkpoint_ensemble(
        model_type=args.model_type,
        checkpoints=list(args.checkpoints),
        samples_per_model=list(args.samples_per_model),
        history_norm=batch.history_norm,
        future_len=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(hist_norm_np.shape[0])}
    model = FixedDeployableSampler(samples_by_key).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=model,
        data_path=args.data_path,
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
        batch_size=args.batch_size,
        conditionality_samples=args.conditionality_samples,
        conditionality_max_batches=args.conditionality_max_batches,
        device=device,
    )

    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size
    results["config"] = {
        "model_type": args.model_type,
        "deployability": {
            "kind": "equal_weight_learned_checkpoint_ensemble",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_per_window_oracle_miss_placement": False,
            "uses_validation_tuned_weights": False,
            "not_post_hoc_calibration": True,
            "ensemble_members": members,
        },
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": int(cond_samples.shape[1]),
        "conditionality_samples": int(args.conditionality_samples),
        "rollout_start": int(rollout_start),
        "seed": int(args.seed),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")

    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    distributional = results["distributional_fidelity"]
    regime = results["regime_coverage"]
    cross_cell = results["cross_cell_correlation"]
    mean_rev = results["mean_reversion"]
    pathwise = results["pathwise_jump_realism"]
    member_lines = [
        f"- member: `{m['checkpoint']}`, samples: `{m['samples']}`, epoch: `{m['checkpoint_epoch']}`"
        for m in members
    ]
    lines = [
        "- policy: `equal-weight learned checkpoint ensemble`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- uses validation future: `{results['config']['deployability']['uses_validation_future']}`",
        f"- total samples per window: `{cond_samples.shape[1]}`",
        *member_lines,
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- regime layer2: `{regime['layer2_n_passing']}/{regime['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "442a Equal-Weight Learned Checkpoint Ensemble", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
