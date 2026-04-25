#!/usr/bin/env python
"""470a: learned residual score-flow around a frozen 392a conditional center."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    load_model as load_base_model,
)
from diffusion.block_ar.frozen_center_residual_score_flow import (  # noqa: E402
    FrozenCenterResidualScenarioGenerator,
    FrozenCenterResidualScoreFMConfig,
    FrozenCenterResidualScoreFlow,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_391a_recent_rollout_energy_finetune import (  # noqa: E402
    build_recent_block,
)


@torch.no_grad()
def precompute_score_dataset(
    base_model: torch.nn.Module,
    hist_01: torch.Tensor,
    fut_01: torch.Tensor,
    center_samples: int,
    center_chunk_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_scores: list[torch.Tensor] = []
    center_scores: list[torch.Tensor] = []
    future_scores: list[torch.Tensor] = []
    n_steps = int(fut_01.shape[1])
    for start in range(0, hist_01.shape[0], batch_size):
        end = min(start + batch_size, hist_01.shape[0])
        hist_batch = hist_01[start:end].to(device, non_blocking=True)
        fut_batch = fut_01[start:end].to(device, non_blocking=True)
        hist_norm = normalize_iv(hist_batch).view(hist_batch.shape[0], hist_batch.shape[1], -1)
        fut_norm = normalize_iv(fut_batch).view(fut_batch.shape[0], fut_batch.shape[1], -1)
        sampled_centers = base_model.sample_batched(
            hist_norm,
            n_samples=center_samples,
            n_steps=n_steps,
            chunk_size=center_chunk_size,
            history_is_normalized=True,
        )
        center_01 = sampled_centers.median(dim=1).values
        center_norm = normalize_iv(center_01).view(center_01.shape[0], n_steps, -1)
        history_scores.append(base_model.history_scores(hist_norm).cpu())
        center_scores.append(base_model.target_future_scores(center_norm).cpu())
        future_scores.append(base_model.target_future_scores(fut_norm).cpu())
    return (
        torch.cat(history_scores, dim=0),
        torch.cat(center_scores, dim=0),
        torch.cat(future_scores, dim=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--center_samples", type=int, default=8)
    parser.add_argument("--center_chunk_size", type=int, default=4)
    parser.add_argument("--precompute_batch_size", type=int, default=16)

    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--history_hidden", type=int, default=128)
    parser.add_argument("--center_hidden", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=3)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--flow_time_dim", type=int, default=32)
    parser.add_argument("--flow_steps", type=int, default=24)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--max_sample_chunk", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, base_payload = load_base_model(args.base_checkpoint, device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)
    if base_model.cfg.history_len != args.history_len or base_model.cfg.future_len != args.future_len:
        raise ValueError("Base checkpoint horizon configuration does not match requested data")

    hist_01, fut_01, indices = build_recent_block(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        adaptation_windows=args.adaptation_windows,
        device=device,
    )
    n_total = int(hist_01.shape[0])
    n_val = max(1, int(round(n_total * float(args.holdout_frac))))
    n_train = n_total - n_val

    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Base epoch: {base_payload.get('epoch')}  base best_val: {base_payload.get('best_val')}")
    print(f"Recent windows: {n_total} index range: {indices[0]}..{indices[-1]}")
    print(f"Train/holdout: {n_train}/{n_val}")
    print("Precomputing frozen-center score tensors...")
    hist_scores, center_scores, fut_scores = precompute_score_dataset(
        base_model=base_model,
        hist_01=hist_01,
        fut_01=fut_01,
        center_samples=args.center_samples,
        center_chunk_size=args.center_chunk_size,
        batch_size=args.precompute_batch_size,
        device=device,
    )
    torch.save(
        {
            "history_scores": hist_scores,
            "center_scores": center_scores,
            "future_scores": fut_scores,
            "indices": torch.from_numpy(indices.copy()),
        },
        out_dir / "score_dataset.pt",
    )

    cfg = FrozenCenterResidualScoreFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=base_model.cfg.n_cells,
        context_dim=args.context_dim,
        history_hidden=args.history_hidden,
        center_hidden=args.center_hidden,
        encoder_dropout=args.encoder_dropout,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_heads=args.token_heads,
        token_ff=args.token_ff,
        model_dropout=args.model_dropout,
        flow_time_dim=args.flow_time_dim,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
        max_sample_chunk=args.max_sample_chunk,
        center_samples=args.center_samples,
        center_chunk_size=args.center_chunk_size,
    )
    model = FrozenCenterResidualScoreFlow(cfg).to(device)
    print(f"Residual params: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = TensorDataset(
        hist_scores[:n_train],
        center_scores[:n_train],
        fut_scores[:n_train],
    )
    val_ds = TensorDataset(
        hist_scores[n_train:],
        center_scores[n_train:],
        fut_scores[n_train:],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_epoch(loader: DataLoader, train_mode: bool, max_batches: int) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_b, center_b, fut_b in loader:
            if max_batches > 0 and n_batches >= max_batches:
                break
            hist_b = hist_b.to(device, non_blocking=True)
            center_b = center_b.to(device, non_blocking=True)
            fut_b = fut_b.to(device, non_blocking=True)
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(hist_b, center_b, fut_b)
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

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True, max_batches=args.max_train_batches)
        val_avg = run_epoch(val_loader, train_mode=False, max_batches=args.max_val_batches)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_residual_score_std": train_avg["residual_score_std"],
            "val_total": val_avg["total"],
            "val_residual_score_std": val_avg["residual_score_std"],
            "val_residual_score_abs": val_avg["residual_score_abs"],
            "val_center_score_std": val_avg["center_score_std"],
            "val_future_score_std": val_avg["future_score_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"resid_std={rec['val_residual_score_std']:.3f} "
            f"resid_abs={rec['val_residual_score_abs']:.3f} "
            f"center/future_std={rec['val_center_score_std']:.3f}/{rec['val_future_score_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(
                str(out_dir / "best_model.pt"),
                model,
                cfg,
                epoch,
                best_val,
                args.base_checkpoint,
            )

    save_checkpoint(
        str(out_dir / "final_model.pt"),
        model,
        cfg,
        args.epochs,
        best_val,
        args.base_checkpoint,
    )
    generator = FrozenCenterResidualScenarioGenerator(
        residual_flow=model,
        base_model=base_model,
        base_checkpoint_path=args.base_checkpoint,
    )
    generator.eval()
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "base_checkpoint": args.base_checkpoint,
        "base_epoch": base_payload.get("epoch"),
        "base_best_val": base_payload.get("best_val"),
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "center_samples": int(args.center_samples),
        "center_chunk_size": int(args.center_chunk_size),
        "residual_params": int(sum(p.numel() for p in model.parameters())),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
