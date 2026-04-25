#!/usr/bin/env python
"""455a: free-running joint (history, future) MMD fine-tune for 392a."""

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
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (  # noqa: E402
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_353a_340c_full_rollout_energy_finetune import (  # noqa: E402
    sample_rollout_scores_with_grad,
)
from experiments.backfill.block_ar.train_453a_recent_rollout_iv_marginal_crps_finetune import (  # noqa: E402
    build_recent_block,
)


def _median_bandwidth_sq(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x.new_tensor(1.0)
    with torch.no_grad():
        dist_sq = torch.cdist(x.detach(), x.detach(), p=2).square()
        mask = ~torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        vals = dist_sq[mask]
        vals = vals[vals > 1e-12]
        if vals.numel() == 0:
            return x.new_tensor(1.0)
        return vals.median().clamp_min(1e-6)


def _rbf_from_bandwidth(x: torch.Tensor, y: torch.Tensor, bandwidth_sq: torch.Tensor) -> torch.Tensor:
    dist_sq = torch.cdist(x, y, p=2).square()
    return torch.exp(-0.5 * dist_sq / bandwidth_sq.clamp_min(1e-6))


def joint_mmd_loss(
    history_iv: torch.Tensor,
    generated_iv: torch.Tensor,
    target_iv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, n_samples, horizon, n_cells = generated_iv.shape
    hist_real = history_iv.reshape(bsz, -1)
    fut_real = target_iv.reshape(bsz, horizon * n_cells)
    hist_gen = hist_real[:, None].expand(bsz, n_samples, hist_real.shape[-1]).reshape(
        bsz * n_samples,
        hist_real.shape[-1],
    )
    fut_gen = generated_iv.reshape(bsz * n_samples, horizon * n_cells)

    hist_all = torch.cat([hist_real, hist_gen], dim=0)
    fut_all = torch.cat([fut_real, fut_gen], dim=0)
    hist_bw = _median_bandwidth_sq(hist_all)
    fut_bw = _median_bandwidth_sq(fut_all)

    k_h_rr = _rbf_from_bandwidth(hist_real, hist_real, hist_bw)
    k_y_rr = _rbf_from_bandwidth(fut_real, fut_real, fut_bw)
    k_h_gg = _rbf_from_bandwidth(hist_gen, hist_gen, hist_bw)
    k_y_gg = _rbf_from_bandwidth(fut_gen, fut_gen, fut_bw)
    k_h_rg = _rbf_from_bandwidth(hist_real, hist_gen, hist_bw)
    k_y_rg = _rbf_from_bandwidth(fut_real, fut_gen, fut_bw)

    rr = (k_h_rr * k_y_rr).mean()
    gg = (k_h_gg * k_y_gg).mean()
    rg = (k_h_rg * k_y_rg).mean()
    mmd = (rr + gg - 2.0 * rg).clamp_min(0.0)
    return mmd, rr.detach(), gg.detach(), rg.detach()


def combined_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    train_sample_count: int,
    rollout_flow_steps: int,
    mmd_weight: float,
    fm_anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(history_norm, future_norm)
    sampled_scores = sample_rollout_scores_with_grad(
        model=model,
        history_norm=history_norm,
        n_samples=train_sample_count,
        n_steps=future_norm.shape[1],
        flow_steps=rollout_flow_steps,
    )
    sampled_iv = model._scores_to_values(sampled_scores)
    history_iv = denormalize_iv(history_norm).view(history_norm.shape[0], history_norm.shape[1], -1)
    target_iv = denormalize_iv(future_norm).view(
        future_norm.shape[0],
        future_norm.shape[1],
        -1,
    )
    mmd, rr, gg, rg = joint_mmd_loss(history_iv, sampled_iv, target_iv)
    total = float(fm_anchor_weight) * fm_loss + float(mmd_weight) * mmd
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "joint_mmd": mmd.detach(),
        "mmd_rr": rr,
        "mmd_gg": gg,
        "mmd_rg": rg,
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
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--mmd_weight", type=float, default=0.5)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
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
                hist_batch.shape[0],
                hist_batch.shape[1],
                -1,
            )
            fut_norm = normalize_iv(fut_batch.to(device, non_blocking=True)).view(
                fut_batch.shape[0],
                fut_batch.shape[1],
                -1,
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = combined_loss(
                    model=model,
                    history_norm=hist_norm,
                    future_norm=fut_norm,
                    train_sample_count=args.train_sample_count,
                    rollout_flow_steps=args.rollout_flow_steps,
                    mmd_weight=args.mmd_weight,
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
            "train_joint_mmd": train_avg["joint_mmd"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_joint_mmd": val_avg["joint_mmd"],
            "val_mmd_rr": val_avg["mmd_rr"],
            "val_mmd_gg": val_avg["mmd_gg"],
            "val_mmd_rg": val_avg["mmd_rg"],
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
            f"mmd={rec['val_joint_mmd']:.5f} "
            f"rr/gg/rg={rec['val_mmd_rr']:.3f}/{rec['val_mmd_gg']:.3f}/{rec['val_mmd_rg']:.3f} "
            f"std={rec['val_sample_iv_std']:.4f}/{rec['val_target_iv_std']:.4f} "
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
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "objective": {
            "fm_anchor_weight": args.fm_anchor_weight,
            "mmd_weight": args.mmd_weight,
            "train_sample_count": args.train_sample_count,
            "rollout_flow_steps": args.rollout_flow_steps,
            "kernel": "joint_history_future_rbf_mmd",
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
