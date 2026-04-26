#!/usr/bin/env python
"""546a: raw-IV path energy fine-tune for the 544a local-shift AR flow."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.local_shift_normalized_empirical_score_transition_flow_matching import (
    LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_330a_causal_future_memory_transition_flow import (
    make_dataset,
)


def raw_iv_path_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy score over full raw-IV paths, normalized by target path dispersion."""
    bsz, n_samples, horizon, n_cells = samples.shape
    dim = horizon * n_cells
    samples_flat = samples.reshape(bsz, n_samples, dim)
    target_flat = target.reshape(bsz, dim)
    target_std = target_flat.std(dim=1, unbiased=False).mean().detach().clamp_min(1e-4)
    scale = math.sqrt(float(dim)) * target_std
    target_dist = torch.sqrt(
        (samples_flat - target_flat[:, None, :]).pow(2).sum(dim=-1) + eps
    ).mean(dim=1) / scale
    pair_dist = torch.cdist(samples_flat, samples_flat, p=2).mean(dim=(1, 2)) / scale
    score = target_dist - 0.5 * pair_dist
    return score.mean(), target_dist.mean(), pair_dist.mean()


def sample_rollout_iv_with_grad(
    model: LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
) -> torch.Tensor:
    history_01 = denormalize_iv(model._flatten(history_norm))
    center, scale = model.causal_local_stats(history_01)
    history_scores = model._values_to_scores(model.to_local_values(history_01, center, scale))
    bsz = history_scores.shape[0]
    k = int(n_samples)
    n_flow = max(1, int(flow_steps))
    dt = 1.0 / float(n_flow)
    prefix = (
        history_scores.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    chunk_center = (
        center.unsqueeze(1)
        .expand(bsz, k, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.n_cells)
    )
    chunk_scale = (
        scale.unsqueeze(1)
        .expand(bsz, k, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.n_cells)
    )
    frames: list[torch.Tensor] = []
    for _step in range(n_steps):
        memory_state = model._encode_prefix_scores(
            prefix,
            center=chunk_center,
            scale=chunk_scale,
        )[:, -1]
        current_score = prefix[:, -1]
        x = model.cfg.sample_temperature * torch.randn_like(current_score)
        for flow_step in range(n_flow):
            t = torch.full(
                (bsz * k,),
                (flow_step + 0.5) * dt,
                device=history_scores.device,
                dtype=history_scores.dtype,
            )
            x = x + dt * model.predict_velocity(x, current_score, memory_state, t)
        next_score = current_score + x
        next_local = model._scores_to_values(next_score)
        next_iv = (next_local * chunk_scale + chunk_center).clamp(0.0, 1.0)
        frames.append(next_iv.view(bsz, k, model.cfg.n_cells))
        prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
    return torch.stack(frames, dim=2)


def combined_loss(
    model: LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_eps: float,
    raw_energy_weight: float,
    fm_anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(history_norm, future_norm)
    target_iv = denormalize_iv(model._flatten(future_norm))
    sampled_iv = sample_rollout_iv_with_grad(
        model=model,
        history_norm=history_norm,
        n_samples=train_sample_count,
        n_steps=target_iv.shape[1],
        flow_steps=rollout_flow_steps,
    )
    raw_energy, target_dist, pair_dist = raw_iv_path_energy_score(
        sampled_iv,
        target_iv,
        eps=energy_eps,
    )
    total = float(fm_anchor_weight) * fm_loss + float(raw_energy_weight) * raw_energy
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "raw_iv_energy": raw_energy.detach(),
        "raw_target_dist": target_dist.detach(),
        "raw_pair_dist": pair_dist.detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        "sample_iv_std": sampled_iv.std(unbiased=False).detach(),
        "target_iv_std": target_iv.std(unbiased=False).detach(),
        "sample_h1_std": sampled_iv[:, :, 0].std(unbiased=False).detach(),
        "sample_h30_std": sampled_iv[:, :, -1].std(unbiased=False).detach(),
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/544a_local_shift_normalized_ar_flow_s544/best_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=2)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--raw_energy_weight", type=float, default=1.0)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)

    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=546)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

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
                hist_01.shape[0], hist_01.shape[1], -1
            )
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                fut_01.shape[0], fut_01.shape[1], -1
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = combined_loss(
                    model=model,
                    history_norm=hist_norm,
                    future_norm=fut_norm,
                    train_sample_count=args.train_sample_count,
                    rollout_flow_steps=args.rollout_flow_steps,
                    energy_eps=args.energy_eps,
                    raw_energy_weight=args.raw_energy_weight,
                    fm_anchor_weight=args.fm_anchor_weight,
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

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
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
            "train_raw_iv_energy": train_avg["raw_iv_energy"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_raw_iv_energy": val_avg["raw_iv_energy"],
            "val_raw_target_dist": val_avg["raw_target_dist"],
            "val_raw_pair_dist": val_avg["raw_pair_dist"],
            "val_transition_std": val_avg["transition_std"],
            "val_sample_iv_std": val_avg["sample_iv_std"],
            "val_target_iv_std": val_avg["target_iv_std"],
            "val_sample_h1_std": val_avg["sample_h1_std"],
            "val_sample_h30_std": val_avg["sample_h30_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"raw_es={rec['val_raw_iv_energy']:.5f} "
            f"target={rec['val_raw_target_dist']:.3f} pair={rec['val_raw_pair_dist']:.3f} "
            f"iv_std={rec['val_sample_iv_std']:.4f}/{rec['val_target_iv_std']:.4f} "
            f"h1/h30={rec['val_sample_h1_std']:.4f}/{rec['val_sample_h30_std']:.4f} "
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
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "config": payload["config"],
        "objective": {
            "fm_anchor_weight": args.fm_anchor_weight,
            "raw_energy_weight": args.raw_energy_weight,
            "train_sample_count": args.train_sample_count,
            "rollout_flow_steps": args.rollout_flow_steps,
            "energy_eps": args.energy_eps,
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
