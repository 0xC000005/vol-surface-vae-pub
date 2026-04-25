#!/usr/bin/env python
"""496a: center-preserving learned stress ensemble.

The sampler combines 392a frontier samples with 494a interval-score stress samples,
then recenters the full ensemble around the 392a sample median. This is a deployable
policy because it uses only frozen learned generators and the current history.
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


def sample_member(
    model_type: str,
    checkpoint: str,
    history_norm: torch.Tensor,
    n_samples: int,
    future_len: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    meta = {
        "checkpoint": checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "samples": int(n_samples),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return samples.astype(np.float32), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--base_checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument(
        "--stress_checkpoint",
        default="models/backfill/494a_recent_interval_score_w002_s42/best_model.pt",
    )
    parser.add_argument("--base_samples", type=int, default=32)
    parser.add_argument("--stress_samples", type=int, default=16)
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
    parser.add_argument("--seed", type=int, default=496)
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
    print(f"Sampling base center/member: {args.base_checkpoint}")
    base_samples, base_meta = sample_member(
        args.model_type,
        args.base_checkpoint,
        batch.history_norm,
        args.base_samples,
        args.future_len,
        args.batch_size,
        args.chunk_size,
        device,
    )
    print(f"Sampling stress member: {args.stress_checkpoint}")
    stress_samples, stress_meta = sample_member(
        args.model_type,
        args.stress_checkpoint,
        batch.history_norm,
        args.stress_samples,
        args.future_len,
        args.batch_size,
        args.chunk_size,
        device,
    )
    raw_ensemble = np.concatenate([base_samples, stress_samples], axis=1).astype(np.float32)
    base_center = np.median(base_samples, axis=1)
    ensemble_center = np.median(raw_ensemble, axis=1)
    cond_samples = np.clip(
        base_center[:, None] + (raw_ensemble - ensemble_center[:, None]),
        0.0,
        1.0,
    ).astype(np.float32)

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
            "kind": "center_preserving_learned_stress_ensemble",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_validation_tuned_weights": False,
            "not_post_hoc_calibration": True,
            "center_source": base_meta,
            "ensemble_members": [base_meta, stress_meta],
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
    lines = [
        "- policy: `center-preserving learned stress ensemble`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- uses validation future: `{results['config']['deployability']['uses_validation_future']}`",
        f"- total samples per window: `{cond_samples.shape[1]}`",
        f"- base checkpoint: `{args.base_checkpoint}`, samples: `{args.base_samples}`",
        f"- stress checkpoint: `{args.stress_checkpoint}`, samples: `{args.stress_samples}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- regime layer2: `{regime['layer2_n_passing']}/{regime['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- median-bias cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "496a Center-Preserving Learned Stress Ensemble", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
