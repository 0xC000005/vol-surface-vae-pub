#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_330a_causal_future_memory_transition_flow import (
    make_dataset,
)


def one_step_fm_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    memory_state: torch.Tensor,
    current_score: torch.Tensor,
    target_next_score: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x1 = target_next_score - current_score
    x0 = torch.randn_like(x1)
    bsz, n_cells = x1.shape
    t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target_velocity = x1 - x0
    pred_velocity = model.predict_velocity(x_t, current_score, memory_state, t)
    loss = F.mse_loss(pred_velocity, target_velocity)
    metrics = {
        "rollin_loss": loss.detach(),
        "rollin_transition_std": x1.std(unbiased=False).detach(),
        "rollin_transition_abs": x1.abs().mean().detach(),
        "rollin_target_velocity_std": target_velocity.std(unbiased=False).detach(),
    }
    return loss, metrics


@torch.no_grad()
def generate_prefix_scores(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_scores: torch.Tensor,
    n_generated_steps: int,
    flow_steps: int,
    temperature: float,
) -> torch.Tensor:
    prefix = history_scores.clone()
    if n_generated_steps <= 0:
        return prefix
    dt = 1.0 / float(flow_steps)
    for _step in range(n_generated_steps):
        memory_state = model._encode_prefix_scores(prefix)[:, -1]
        current_score = prefix[:, -1]
        x = temperature * torch.randn_like(current_score)
        for flow_step in range(flow_steps):
            t = torch.full(
                (history_scores.shape[0],),
                (flow_step + 0.5) * dt,
                device=history_scores.device,
                dtype=history_scores.dtype,
            )
            x = x + dt * model.predict_velocity(x, current_score, memory_state, t)
        prefix = torch.cat([prefix, (current_score + x)[:, None, :]], dim=1)
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="344a on-policy fine-tune for 340c empirical-normal-score AR-FM"
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="models/backfill/340c_v0_s42/best_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--rollin_flow_steps", type=int, default=8)
    parser.add_argument("--rollin_temperature", type=float, default=1.0)

    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
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
    model, payload = load_model(args.init_checkpoint, device)
    cfg = model.cfg
    tensors, h, w, d = make_dataset(
        data_path=args.data_path,
        history_len=cfg.history_len,
        future_len=cfg.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=device,
    )
    if d != cfg.n_cells:
        raise ValueError(f"Checkpoint n_cells={cfg.n_cells} but data has {d}")
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
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
                teacher_loss, teacher_metrics = model.training_loss(hist_norm, fut_norm)
                history_scores = model.history_scores(hist_norm)
                target_future_scores = model.target_future_scores(fut_norm)
                horizon_idx = int(
                    torch.randint(0, cfg.future_len, (1,), device=device).item()
                )
                generated_prefix = generate_prefix_scores(
                    model,
                    history_scores,
                    n_generated_steps=horizon_idx,
                    flow_steps=args.rollin_flow_steps,
                    temperature=args.rollin_temperature,
                )
                memory_state = model._encode_prefix_scores(generated_prefix.detach())[
                    :, -1
                ]
                current_score = generated_prefix[:, -1].detach()
                target_next = target_future_scores[:, horizon_idx]
                rollin_loss, rollin_metrics = one_step_fm_loss(
                    model,
                    memory_state,
                    current_score,
                    target_next,
                )
                total = 0.5 * (teacher_loss + rollin_loss)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.clip_grad
                        )
                    optimizer.step()

            metrics = {
                "total": total.detach(),
                "teacher_loss": teacher_metrics["total"],
                "rollin_loss": rollin_metrics["rollin_loss"],
                "teacher_transition_std": teacher_metrics["transition_std"],
                "rollin_transition_std": rollin_metrics["rollin_transition_std"],
                "rollin_transition_abs": rollin_metrics["rollin_transition_abs"],
                "rollin_horizon": torch.tensor(float(horizon_idx + 1), device=device),
                "memory_abs": teacher_metrics["memory_abs"],
            }
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Init checkpoint: {args.init_checkpoint} epoch={payload.get('epoch')}")
    print(
        f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}"
    )
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
            "train_teacher_loss": train_avg["teacher_loss"],
            "train_rollin_loss": train_avg["rollin_loss"],
            "val_total": val_avg["total"],
            "val_teacher_loss": val_avg["teacher_loss"],
            "val_rollin_loss": val_avg["rollin_loss"],
            "val_rollin_transition_std": val_avg["rollin_transition_std"],
            "val_rollin_transition_abs": val_avg["rollin_transition_abs"],
            "val_rollin_horizon": val_avg["rollin_horizon"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"teacher={rec['val_teacher_loss']:.5f} "
            f"rollin={rec['val_rollin_loss']:.5f} "
            f"roll_std={rec['val_rollin_transition_std']:.3f} "
            f"h={rec['val_rollin_horizon']:.1f} "
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
        "init_checkpoint": args.init_checkpoint,
        "config": asdict(cfg),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
