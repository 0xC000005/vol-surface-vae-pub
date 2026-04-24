#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_path_flow_matching import (
    EmpiricalNormalScorePathFMConfig,
    EmpiricalNormalScorePathFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_330a_causal_future_memory_transition_flow import (
    make_dataset,
)


def _flatten_cells(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.reshape(-1, x.shape[-2] * x.shape[-1])
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"Expected 3D or 4D window tensor, got shape {tuple(x.shape)}")


def compute_empirical_quantiles(
    train_hist_01: torch.Tensor,
    train_future_01: torch.Tensor,
    n_quantiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels = (torch.arange(n_quantiles, device=train_hist_01.device, dtype=torch.float32) + 0.5)
    levels = levels / float(n_quantiles)
    hist_flat = _flatten_cells(train_hist_01)
    future_flat = _flatten_cells(train_future_01)
    history_source = torch.cat([hist_flat, future_flat], dim=0)
    history_q = torch.quantile(history_source.float(), levels, dim=0).transpose(0, 1)
    future_q = torch.quantile(future_flat.float(), levels, dim=0).transpose(0, 1)
    return history_q.contiguous(), future_q.contiguous(), levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="339a empirical normal-score full-path flow matching"
    )
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--mixer_type", choices=["axial", "transformer"], default="axial")
    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--history_hidden", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--global_mixer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow_time_dim", type=int, default=32)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--flow_steps", type=int, default=32)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--max_sample_chunk", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)

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
        device=device,
    )
    train_hist, train_future, val_hist, val_future = tensors
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    cfg = EmpiricalNormalScorePathFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_ff=args.token_ff,
        model_dropout=args.model_dropout,
        mixer_type=args.mixer_type,
        context_dim=args.context_dim,
        history_hidden=args.history_hidden,
        encoder_dropout=args.encoder_dropout,
        token_heads=args.token_heads,
        global_mixer=args.global_mixer,
        transition_features=args.transition_features,
        flow_time_dim=args.flow_time_dim,
        n_quantiles=args.n_quantiles,
        cdf_eps=args.cdf_eps,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
        max_sample_chunk=args.max_sample_chunk,
    )
    model = EmpiricalNormalScorePathFlowMatching(cfg).to(device)
    history_q, future_q, quantile_levels = compute_empirical_quantiles(
        train_hist,
        train_future,
        n_quantiles=args.n_quantiles,
    )
    model.set_empirical_quantiles(history_q, future_q, quantile_levels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_01, fut_01 in loader:
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                hist_01.shape[0], hist_01.shape[1], -1
            )
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                fut_01.shape[0], fut_01.shape[1], -1
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(hist_norm, fut_norm)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={h} W={w} D={d}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "val_total": val_avg["total"],
            "val_future_score_std": val_avg["future_score_std"],
            "val_future_score_abs": val_avg["future_score_abs"],
            "val_implied_transition_std": val_avg["implied_transition_std"],
            "val_implied_transition_abs": val_avg["implied_transition_abs"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"score_std={rec['val_future_score_std']:.3f} "
            f"score_abs={rec['val_future_score_abs']:.3f} "
            f"trans_std={rec['val_implied_transition_std']:.3f} "
            f"trans_abs={rec['val_implied_transition_abs']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2))
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "config": asdict(cfg),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
