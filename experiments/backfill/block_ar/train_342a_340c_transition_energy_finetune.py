#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
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


def energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multivariate CRPS/energy score in normalized score coordinates."""
    bsz, n_samples, n_cells = samples.shape
    scale = math.sqrt(float(n_cells))
    target_dist = torch.sqrt(
        (samples - target[:, None, :]).pow(2).sum(dim=-1) + eps
    ).mean(dim=1) / scale
    pair_dist = torch.cdist(samples, samples, p=2).mean(dim=(1, 2)) / scale
    score = target_dist - 0.5 * pair_dist
    return score.mean(), target_dist.mean(), pair_dist.mean()


def sample_transition_with_grad(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    current_score: torch.Tensor,
    memory_state: torch.Tensor,
    n_samples: int,
    flow_steps: int,
) -> torch.Tensor:
    bsz, n_cells = current_score.shape
    k = int(n_samples)
    x = model.cfg.sample_temperature * torch.randn(
        bsz * k,
        n_cells,
        device=current_score.device,
        dtype=current_score.dtype,
    )
    current_rep = current_score.repeat_interleave(k, dim=0)
    memory_rep = memory_state.repeat_interleave(k, dim=0)
    dt = 1.0 / float(flow_steps)
    for step in range(flow_steps):
        t = torch.full(
            (bsz * k,),
            (step + 0.5) * dt,
            device=current_score.device,
            dtype=current_score.dtype,
        )
        x = x + dt * model.predict_velocity(x, current_rep, memory_rep, t)
    return x.view(bsz, k, n_cells)


def combined_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    train_sample_count: int,
    energy_flow_steps: int,
    energy_eps: float,
    energy_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    memory_states, current_scores, future_scores = model.teacher_forced_memory(
        history_norm,
        future_norm,
    )
    x1 = future_scores - current_scores
    x0 = torch.randn_like(x1)
    bsz, horizon, n_cells = x1.shape
    t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
    x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
    target_velocity = x1 - x0
    pred_velocity = model.predict_velocity(
        x_t.reshape(bsz * horizon, n_cells),
        current_scores.reshape(bsz * horizon, n_cells),
        memory_states.reshape(bsz * horizon, model.cfg.memory_dim),
        t.reshape(bsz * horizon),
    ).view_as(x1)
    fm_loss = F.mse_loss(pred_velocity, target_velocity)

    h_idx = int(torch.randint(0, horizon, (1,), device=x1.device).item())
    sampled_transition = sample_transition_with_grad(
        model=model,
        current_score=current_scores[:, h_idx],
        memory_state=memory_states[:, h_idx],
        n_samples=train_sample_count,
        flow_steps=energy_flow_steps,
    )
    sampled_next = current_scores[:, h_idx, None, :] + sampled_transition
    target_next = future_scores[:, h_idx]
    es_loss, target_dist, pair_dist = energy_score(
        sampled_next,
        target_next,
        eps=energy_eps,
    )
    total = fm_loss + float(energy_weight) * es_loss
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "energy": es_loss.detach(),
        "energy_target_dist": target_dist.detach(),
        "energy_pair_dist": pair_dist.detach(),
        "energy_horizon": torch.tensor(float(h_idx + 1), device=x1.device),
        "transition_std": x1.std(unbiased=False).detach(),
        "sample_next_std": sampled_next.std(unbiased=False).detach(),
        "target_next_std": target_next.std(unbiased=False).detach(),
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="342a: 340c transition-law energy-score fine-tune"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/backfill/340c_v0_s42/best_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--energy_flow_steps", type=int, default=8)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)

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
    model, payload = load_model(args.checkpoint, device)
    if model.cfg.history_len != args.history_len or model.cfg.future_len != args.future_len:
        raise ValueError("Checkpoint horizon configuration does not match the requested data")
    if model.cfg.n_cells != d:
        raise ValueError("Checkpoint cell count does not match the requested data")

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
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []

    def run_epoch(
        loader: DataLoader,
        train_mode: bool,
        max_batches: int,
    ) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_01, fut_01 in loader:
            if max_batches > 0 and n_batches >= max_batches:
                break
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                hist_01.shape[0],
                hist_01.shape[1],
                -1,
            )
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                fut_01.shape[0],
                fut_01.shape[1],
                -1,
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = combined_loss(
                    model=model,
                    history_norm=hist_norm,
                    future_norm=fut_norm,
                    train_sample_count=args.train_sample_count,
                    energy_flow_steps=args.energy_flow_steps,
                    energy_eps=args.energy_eps,
                    energy_weight=args.energy_weight,
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
    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={h} W={w} D={d}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(
            train_loader,
            train_mode=True,
            max_batches=args.max_train_batches,
        )
        val_avg = run_epoch(
            val_loader,
            train_mode=False,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_fm_loss": train_avg["fm_loss"],
            "train_energy": train_avg["energy"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_energy": val_avg["energy"],
            "val_energy_target_dist": val_avg["energy_target_dist"],
            "val_energy_pair_dist": val_avg["energy_pair_dist"],
            "val_transition_std": val_avg["transition_std"],
            "val_sample_next_std": val_avg["sample_next_std"],
            "val_target_next_std": val_avg["target_next_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"es={rec['val_energy']:.5f} target={rec['val_energy_target_dist']:.3f} "
            f"pair={rec['val_energy_pair_dist']:.3f} "
            f"sample_std={rec['val_sample_next_std']:.3f} "
            f"target_std={rec['val_target_next_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, model.cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, model.cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2))
    summary = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "config": payload["config"],
        "objective": {
            "fm_anchor_weight": 1.0,
            "energy_weight": args.energy_weight,
            "train_sample_count": args.train_sample_count,
            "energy_flow_steps": args.energy_flow_steps,
            "energy_eps": args.energy_eps,
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
