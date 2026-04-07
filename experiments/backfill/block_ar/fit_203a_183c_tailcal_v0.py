#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.visualize_management_report_183c_v1 import load_model
from experiments.backfill.block_ar.tailcal_mapper import (
    compute_vol_of_vol,
    encode_teacher_basis,
    fit_tailcal_map,
    regime_bucket_from_vov,
    repeat_forward_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit 203a tail calibrator on 183c validation split")
    parser.add_argument(
        "--model_path",
        default="models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt",
    )
    parser.add_argument(
        "--output_map",
        default="results/validations/2026-04-07/analysis/203_design/203a_183c_tailcal_v0/tailcal_map.npz",
    )
    parser.add_argument(
        "--output_summary",
        default="results/validations/2026-04-07/analysis/203_design/203a_183c_tailcal_v0/fit_summary.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--log_shift_clip", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, config = load_model(args.model_path, device)
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)

    history_len = int(config["history_len"])
    future_len = int(config["future_len"])
    test_start = 4511
    max_train_idx = test_start - history_len - future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)

    history_np = val_hist.detach().cpu().numpy()
    vov = compute_vol_of_vol(history_np)
    vov_edges = np.array([np.quantile(vov, 0.2), np.quantile(vov, 0.8)], dtype=np.float32)
    regime_bucket = regime_bucket_from_vov(vov, vov_edges)
    block_splits = np.array_split(np.arange(future_len), 3)
    block_bounds = np.array([int(block_splits[0][0]), int(block_splits[1][0]), int(block_splits[2][0]), future_len], dtype=np.int64)
    quantile_levels = np.linspace(0.01, 0.99, 99, dtype=np.float32)

    gen_log_r_all: list[np.ndarray] = []
    gt_log_r_all: list[np.ndarray] = []
    bucket_all: list[np.ndarray] = []
    block_all: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(val_indices), args.batch_size):
            end = min(start + args.batch_size, len(val_indices))
            hist = val_hist[start:end].to(device)
            fut = val_future[start:end].to(device)
            batch = hist.shape[0]
            outputs = model.forward_from_history(hist)
            gt_u = iv_to_unconstrained(
                fut,
                lo=model.support_lo,
                hi=model.support_hi,
                eps=model.support_eps,
            )
            gt_basis = encode_teacher_basis(model, gt_u, outputs)

            samples_u = model.sample_future_u(hist, n_samples=args.fit_samples)
            outputs_rep = repeat_forward_outputs(outputs, args.fit_samples)
            gen_basis = encode_teacher_basis(
                model,
                samples_u.reshape(batch * args.fit_samples, future_len, -1),
                outputs_rep,
            )

            gt_basis_np = gt_basis.detach().cpu().numpy()
            gen_basis_np = gen_basis.detach().cpu().numpy().reshape(batch, args.fit_samples, future_len, -1)
            for block_idx, block in enumerate(block_splits):
                lo = int(block[0])
                hi = int(block[-1]) + 1
                gt_block = gt_basis_np[:, lo:hi, :].reshape(batch, -1)
                gen_block = gen_basis_np[:, :, lo:hi, :].reshape(batch * args.fit_samples, -1)
                gt_log_r = np.log(np.clip(np.linalg.norm(gt_block, axis=1), 1e-8, None))
                gen_log_r = np.log(np.clip(np.linalg.norm(gen_block, axis=1), 1e-8, None))
                gen_log_r_all.append(gen_log_r)
                gt_log_r_all.append(np.repeat(gt_log_r, args.fit_samples))
                bucket_all.append(np.repeat(regime_bucket[start:end], args.fit_samples))
                block_all.append(np.full(batch * args.fit_samples, block_idx, dtype=np.int64))

    gen_log_r_all_np = np.concatenate(gen_log_r_all)
    gt_log_r_all_np = np.concatenate(gt_log_r_all)
    bucket_all_np = np.concatenate(bucket_all)
    block_all_np = np.concatenate(block_all)

    fit = fit_tailcal_map(
        gen_log_r=gen_log_r_all_np,
        gt_log_r=gt_log_r_all_np,
        regime_bucket=bucket_all_np,
        block_idx=block_all_np,
        quantile_levels=quantile_levels,
        vov_edges=vov_edges,
        block_bounds=block_bounds,
        future_len=future_len,
        support_lo=float(model.support_lo),
        support_hi=float(model.support_hi),
        support_eps=float(model.support_eps),
        log_shift_clip=float(args.log_shift_clip),
    )
    Path(args.output_map).parent.mkdir(parents=True, exist_ok=True)
    fit.save(args.output_map)

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "output_map": str(Path(args.output_map).resolve()),
        "n_val_windows": int(len(val_indices)),
        "fit_samples_per_window": int(args.fit_samples),
        "vov_edges": [float(vov_edges[0]), float(vov_edges[1])],
        "block_bounds": [int(x) for x in block_bounds.tolist()],
        "log_shift_clip": float(args.log_shift_clip),
        "fit_summary": fit.fit_summary,
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
