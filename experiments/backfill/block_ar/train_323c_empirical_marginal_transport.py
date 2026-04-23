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
    EmpiricalMarginalTransportConfig,
    save_transport_checkpoint,
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
        description="323c empirical marginal transport over a neural copula sampler"
    )
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/321c_v0_s42/best_model.pt")
    parser.add_argument("--n_quantiles", type=int, default=101)
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
            base_flat = base_samples.view(
                base_samples.shape[0] * base_samples.shape[1],
                args.future_len,
                d,
            )
            source_chunks.append(iv_to_logit(base_flat, args.logit_eps).cpu())
            target_chunks.append(
                iv_to_logit(fut_01.to(device, non_blocking=True).view(fut_01.shape[0], args.future_len, d), args.logit_eps).cpu()
            )
            if batch_idx % 10 == 0:
                print(
                    f"calibration batches={batch_idx} "
                    f"source_paths={sum(x.shape[0] for x in source_chunks)}"
                )

    source = torch.cat(source_chunks, dim=0)
    target = torch.cat(target_chunks, dim=0)
    q_levels = torch.linspace(args.q_min, args.q_max, args.n_quantiles)
    source_q = torch.quantile(source, q_levels, dim=0).permute(1, 2, 0).contiguous()
    target_q = torch.quantile(target, q_levels, dim=0).permute(1, 2, 0).contiguous()

    cfg = EmpiricalMarginalTransportConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        logit_eps=args.logit_eps,
        max_sample_chunk=args.chunk_size,
        base_checkpoint=args.base_checkpoint,
        n_quantiles=args.n_quantiles,
    )
    save_transport_checkpoint(
        str(out_dir / "best_model.pt"),
        cfg,
        source_quantiles=source_q,
        target_quantiles=target_q,
        quantile_levels=q_levels,
    )
    save_transport_checkpoint(
        str(out_dir / "final_model.pt"),
        cfg,
        source_quantiles=source_q,
        target_quantiles=target_q,
        quantile_levels=q_levels,
    )
    summary = {
        "config": asdict(cfg),
        "n_source_paths": int(source.shape[0]),
        "n_target_paths": int(target.shape[0]),
        "grid": [int(h), int(w), int(d)],
        "q_min": args.q_min,
        "q_max": args.q_max,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
