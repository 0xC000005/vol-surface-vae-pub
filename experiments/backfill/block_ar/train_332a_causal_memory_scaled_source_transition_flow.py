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

from diffusion.block_ar.causal_memory_scaled_source_transition_flow_matching import (
    CausalMemoryScaledSourceTransitionFMConfig,
    CausalMemoryScaledSourceTransitionFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_330a_causal_future_memory_transition_flow import (
    compute_logit_stats,
    make_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="332a causal memory conditional-source transition FM")
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_ff", type=int, default=256)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=3)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--source_scale_clip", type=float, default=2.0)
    parser.add_argument("--logit_eps", type=float, default=1e-4)
    parser.add_argument("--logit_std_floor", type=float, default=1e-3)
    parser.add_argument("--flow_steps", type=int, default=32)
    parser.add_argument("--sample_temperature", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=36)
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

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
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

    cfg = CausalMemoryScaledSourceTransitionFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        memory_dim=args.memory_dim,
        memory_layers=args.memory_layers,
        memory_heads=args.memory_heads,
        memory_ff=args.memory_ff,
        model_dropout=args.model_dropout,
        time_dim=args.time_dim,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_heads=args.token_heads,
        token_ff=args.token_ff,
        source_scale_clip=args.source_scale_clip,
        logit_eps=args.logit_eps,
        logit_std_floor=args.logit_std_floor,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
    )
    model = CausalMemoryScaledSourceTransitionFlowMatching(cfg).to(device)
    mean, std = compute_logit_stats(
        train_hist,
        train_future,
        logit_eps=args.logit_eps,
        std_floor=args.logit_std_floor,
    )
    model.set_logit_stats(mean, std)

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
            "train_fm_loss": train_avg["fm_loss"],
            "train_scale_nll": train_avg["scale_nll"],
            "train_source_std_mean": train_avg["source_std_mean"],
            "train_source_std_std": train_avg["source_std_std"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_scale_nll": val_avg["scale_nll"],
            "val_source_std_mean": val_avg["source_std_mean"],
            "val_source_std_std": val_avg["source_std_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] "
            f"train={rec['train_total']:.5f} val={rec['val_total']:.5f} "
            f"fm={rec['val_fm_loss']:.5f} scale={rec['val_scale_nll']:.5f} "
            f"src_mean={rec['val_source_std_mean']:.3f} src_std={rec['val_source_std_std']:.3f} "
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
