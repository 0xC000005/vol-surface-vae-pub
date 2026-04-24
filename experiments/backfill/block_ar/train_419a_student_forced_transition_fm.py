#!/usr/bin/env python
"""419a: student-forced transition-FM fine-tune for the 392a AR frontier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import build_multistep_windows  # noqa: E402
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


def build_recent_block(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    adaptation_windows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    val_start = max_train_idx - val_size
    n = min(int(adaptation_windows), int(val_start))
    indices = np.arange(val_start - n, val_start)
    history_01, future_01 = build_multistep_windows(indices, surf_tensor, history_len, future_len)
    future_01 = future_01.view(history_01.shape[0], future_len, 5, 5)
    return history_01, future_01, indices


@torch.no_grad()
def sample_generated_prefix_scores(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    n_steps: int,
    flow_steps: int,
) -> torch.Tensor:
    """Return history plus one free-running score path under the current model."""
    was_training = model.training
    model.eval()
    history_scores = model.history_scores(history_norm)
    prefix = history_scores.clone()
    n_flow = max(1, int(flow_steps))
    dt = 1.0 / float(n_flow)

    for _ in range(n_steps):
        memory_state = model._encode_prefix_scores(prefix)[:, -1]
        current_score = prefix[:, -1]
        x = model.cfg.sample_temperature * torch.randn_like(current_score)
        noise_scale = model._conditional_noise_scale(memory_state)
        if noise_scale is not None:
            x = x * noise_scale
        for flow_step in range(n_flow):
            t = torch.full(
                (history_scores.shape[0],),
                (flow_step + 0.5) * dt,
                device=history_scores.device,
                dtype=history_scores.dtype,
            )
            x = x + dt * model.predict_velocity(x, current_score, memory_state, t)
        prefix = torch.cat([prefix, (current_score + x)[:, None, :]], dim=1)

    model.train(was_training)
    return prefix


def transition_fm_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    current_score: torch.Tensor,
    memory_state: torch.Tensor,
    target_next_score: torch.Tensor,
) -> torch.Tensor:
    x1 = target_next_score - current_score
    x0 = torch.randn_like(x1)
    noise_scale = model._conditional_noise_scale(memory_state)
    if noise_scale is not None:
        x0 = x0 * noise_scale
    bsz, n_cells = x1.shape
    t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target_velocity = x1 - x0
    pred_velocity = model.predict_velocity(x_t, current_score, memory_state, t)
    return F.mse_loss(pred_velocity, target_velocity)


def selected_horizons(horizon: int, count: int, device: torch.device) -> torch.Tensor:
    if count <= 0 or count >= horizon:
        return torch.arange(horizon, device=device)
    grid = torch.linspace(0, horizon - 1, steps=count, device=device)
    return grid.round().long().unique()


def student_forced_fm_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    rollout_flow_steps: int,
    student_horizon_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_scores = model.target_future_scores(future_norm)
    generated_prefix = sample_generated_prefix_scores(
        model=model,
        history_norm=history_norm,
        n_steps=target_scores.shape[1],
        flow_steps=rollout_flow_steps,
    )
    h_indices = selected_horizons(
        horizon=target_scores.shape[1],
        count=student_horizon_count,
        device=target_scores.device,
    )

    losses: list[torch.Tensor] = []
    off_policy_abs: list[torch.Tensor] = []
    for h_idx in h_indices.tolist():
        prefix_len = model.cfg.history_len + h_idx
        prefix = generated_prefix[:, :prefix_len].detach()
        memory_state = model._encode_prefix_scores(prefix)[:, -1]
        current_score = prefix[:, -1]
        target_next = target_scores[:, h_idx]
        losses.append(transition_fm_loss(model, current_score, memory_state, target_next))
        off_policy_abs.append((target_next - current_score).abs().mean().detach())

    loss = torch.stack(losses).mean()
    metrics = {
        "student_fm_loss": loss.detach(),
        "student_off_policy_abs": torch.stack(off_policy_abs).mean(),
        "student_horizon_count": torch.tensor(
            float(len(h_indices)), device=target_scores.device
        ),
    }
    return loss, metrics


def combined_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    fm_anchor_weight: float,
    student_weight: float,
    rollout_flow_steps: int,
    student_horizon_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(history_norm, future_norm)
    student_loss, student_metrics = student_forced_fm_loss(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        rollout_flow_steps=rollout_flow_steps,
        student_horizon_count=student_horizon_count,
    )
    total = float(fm_anchor_weight) * fm_loss + float(student_weight) * student_loss
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        **student_metrics,
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--student_weight", type=float, default=0.1)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--student_horizon_count", type=int, default=6)
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
    model, payload = load_model(args.checkpoint, device)
    model.train()
    if model.cfg.history_len != args.history_len or model.cfg.future_len != args.future_len:
        raise ValueError("Checkpoint horizon configuration does not match requested data")

    hist_01, fut_01, indices = build_recent_block(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        adaptation_windows=args.adaptation_windows,
        device=device,
    )
    n_total = hist_01.shape[0]
    n_val = max(1, int(round(n_total * float(args.holdout_frac))))
    n_train = n_total - n_val
    train_hist, val_hist = hist_01[:n_train], hist_01[n_train:]
    train_fut, val_fut = fut_01[:n_train], fut_01[n_train:]

    train_loader = DataLoader(
        TensorDataset(train_hist, train_fut),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_fut),
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
        for hist_batch, fut_batch in loader:
            if max_batches > 0 and n_batches >= max_batches:
                break
            hist_norm = normalize_iv(hist_batch.to(device, non_blocking=True)).view(
                hist_batch.shape[0], hist_batch.shape[1], -1
            )
            fut_norm = normalize_iv(fut_batch.to(device, non_blocking=True)).view(
                fut_batch.shape[0], fut_batch.shape[1], -1
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = combined_loss(
                    model=model,
                    history_norm=hist_norm,
                    future_norm=fut_norm,
                    fm_anchor_weight=args.fm_anchor_weight,
                    student_weight=args.student_weight,
                    rollout_flow_steps=args.rollout_flow_steps,
                    student_horizon_count=args.student_horizon_count,
                )
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

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')}  source best_val: {payload.get('best_val')}")
    print(f"Recent windows: {n_total} index range: {indices[0]}..{indices[-1]}")
    print(f"Train/holdout: {n_train}/{n_val}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

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
            "train_fm_loss": train_avg["fm_loss"],
            "train_student_fm_loss": train_avg["student_fm_loss"],
            "train_student_off_policy_abs": train_avg["student_off_policy_abs"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_student_fm_loss": val_avg["student_fm_loss"],
            "val_student_off_policy_abs": val_avg["student_off_policy_abs"],
            "val_transition_std": val_avg["transition_std"],
            "student_horizon_count": val_avg["student_horizon_count"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"student={rec['val_student_fm_loss']:.5f} "
            f"off={rec['val_student_off_policy_abs']:.3f} "
            f"trans_std={rec['val_transition_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, model.cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, model.cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "objective": {
            "fm_anchor_weight": args.fm_anchor_weight,
            "student_weight": args.student_weight,
            "rollout_flow_steps": args.rollout_flow_steps,
            "student_horizon_count": args.student_horizon_count,
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
