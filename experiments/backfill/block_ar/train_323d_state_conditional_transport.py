#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.conditional_marginal_copula_model import (
    StateConditionalEmpiricalTransportConfig,
    save_state_conditional_transport_checkpoint,
)
from diffusion.block_ar.future_scalar_ar_mixture_density_model import (
    load_model as load_scalar_ar_model,
)
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_320a_future_scalar_ar_mixture_density_model import (
    make_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="323d state-conditional empirical marginal transport"
    )
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/321c_v0_s42/best_model.pt")
    parser.add_argument("--n_quantiles", type=int, default=101)
    parser.add_argument("--n_level_bins", type=int, default=5)
    parser.add_argument("--n_vol_bins", type=int, default=3)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--q_min", type=float, default=0.005)
    parser.add_argument("--q_max", type=float, default=0.995)
    parser.add_argument("--calib_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--max_calib_windows", type=int, default=0)

    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--logit_eps", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    tensors, h, w, d = make_dataset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=str(device),
    )
    train_hist, train_future = tensors[0], tensors[1]
    if args.max_calib_windows > 0:
        train_hist = train_hist[: args.max_calib_windows]
        train_future = train_future[: args.max_calib_windows]

    hist_flat = train_hist.view(train_hist.shape[0], args.history_len, d)
    fut_flat = train_future.view(train_future.shape[0], args.future_len, d)
    hist_logits = iv_to_logit(hist_flat, args.logit_eps)
    last_logits = hist_logits[:, -1].cpu()
    edge_levels = torch.linspace(0, 1, args.n_level_bins + 1)[1:-1]
    level_edges = torch.quantile(last_logits, edge_levels, dim=0).transpose(0, 1).contiguous()

    daily = hist_flat[:, 1:] - hist_flat[:, :-1]
    realized_var = daily.pow(2).mean(dim=(1, 2)).cpu()
    vol_edge_levels = torch.linspace(0, 1, args.n_vol_bins + 1)[1:-1]
    vol_edges = torch.quantile(realized_var, vol_edge_levels).contiguous()

    level_bins = torch.empty_like(last_logits, dtype=torch.long)
    for cell in range(d):
        level_bins[:, cell] = torch.bucketize(last_logits[:, cell], level_edges[cell])
    vol_bins = torch.bucketize(realized_var, vol_edges)
    bin_idx = level_bins * args.n_vol_bins + vol_bins[:, None]
    n_bins = args.n_level_bins * args.n_vol_bins
    bin_counts = torch.bincount(bin_idx.reshape(-1), minlength=n_bins)

    loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    base_model, _ = load_scalar_ar_model(args.base_checkpoint, device)
    base_model.eval()

    source_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, (hist_01, fut_01) in enumerate(loader, start=1):
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                hist_01.shape[0],
                hist_01.shape[1],
                -1,
            )
            base_samples = base_model.sample_batched(
                hist_norm,
                n_samples=args.calib_samples,
                n_steps=args.future_len,
                chunk_size=args.chunk_size,
                history_is_normalized=True,
            )
            source_chunks.append(
                iv_to_logit(
                    base_samples.view(
                        base_samples.shape[0],
                        base_samples.shape[1],
                        args.future_len,
                        d,
                    ),
                    args.logit_eps,
                ).cpu()
            )
            target_chunks.append(
                iv_to_logit(
                    fut_01.to(device, non_blocking=True).view(
                        fut_01.shape[0],
                        args.future_len,
                        d,
                    ),
                    args.logit_eps,
                ).cpu()
            )
            if batch_idx % 10 == 0:
                print(f"calibration batches={batch_idx}")

    source = torch.cat(source_chunks, dim=0)
    target = torch.cat(target_chunks, dim=0)
    q_levels = torch.linspace(args.q_min, args.q_max, args.n_quantiles)
    source_q = torch.empty(d, n_bins, args.future_len, args.n_quantiles)
    target_q = torch.empty_like(source_q)

    global_source_q = torch.empty(d, args.future_len, args.n_quantiles)
    global_target_q = torch.empty_like(global_source_q)
    for cell in range(d):
        global_source_q[cell] = torch.quantile(
            source[:, :, :, cell].reshape(-1, args.future_len),
            q_levels,
            dim=0,
        ).transpose(0, 1)
        global_target_q[cell] = torch.quantile(
            target[:, :, cell],
            q_levels,
            dim=0,
        ).transpose(0, 1)

    fallback_bins = 0
    for cell in range(d):
        for bin_id in range(n_bins):
            mask = bin_idx[:, cell] == bin_id
            if int(mask.sum()) < args.min_bin_windows:
                source_q[cell, bin_id] = global_source_q[cell]
                target_q[cell, bin_id] = global_target_q[cell]
                fallback_bins += 1
                continue
            source_q[cell, bin_id] = torch.quantile(
                source[mask, :, :, cell].reshape(-1, args.future_len),
                q_levels,
                dim=0,
            ).transpose(0, 1)
            target_q[cell, bin_id] = torch.quantile(
                target[mask, :, cell],
                q_levels,
                dim=0,
            ).transpose(0, 1)

    cfg = StateConditionalEmpiricalTransportConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        logit_eps=args.logit_eps,
        max_sample_chunk=args.chunk_size,
        base_checkpoint=args.base_checkpoint,
        n_quantiles=args.n_quantiles,
        n_level_bins=args.n_level_bins,
        n_vol_bins=args.n_vol_bins,
    )
    for name in ["best_model.pt", "final_model.pt"]:
        save_state_conditional_transport_checkpoint(
            str(out_dir / name),
            cfg,
            source_quantiles=source_q,
            target_quantiles=target_q,
            quantile_levels=q_levels,
            level_edges=level_edges,
            vol_edges=vol_edges,
        )
    summary = {
        "config": asdict(cfg),
        "n_source_paths": int(source.shape[0] * source.shape[1]),
        "n_target_paths": int(target.shape[0]),
        "n_bins": int(n_bins),
        "fallback_cell_bins": int(fallback_bins),
        "bin_counts": bin_counts.tolist(),
        "grid": [int(h), int(w), int(d)],
        "q_min": args.q_min,
        "q_max": args.q_max,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
