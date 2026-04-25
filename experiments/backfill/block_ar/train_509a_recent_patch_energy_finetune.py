#!/usr/bin/env python
"""509a: MMPD-inspired patch-energy fine-tune from the 392a AR frontier."""

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

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_353a_340c_full_rollout_energy_finetune import (  # noqa: E402
    sample_rollout_scores_with_grad,
)
from experiments.backfill.block_ar.train_391a_recent_rollout_energy_finetune import (  # noqa: E402
    build_recent_block,
)


def patch_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    patch_len: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy score averaged over overlapping future patches."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    bsz, n_samples, horizon, n_cells = samples.shape
    length = int(patch_len)
    if length < 1 or length > horizon:
        raise ValueError(f"patch_len must be in [1,{horizon}], got {patch_len}")
    scale = math.sqrt(float(length * n_cells))
    scores: list[torch.Tensor] = []
    target_terms: list[torch.Tensor] = []
    pair_terms: list[torch.Tensor] = []
    for start in range(0, horizon - length + 1):
        sample_patch = samples[:, :, start : start + length].reshape(
            bsz,
            n_samples,
            length * n_cells,
        )
        target_patch = target[:, start : start + length].reshape(bsz, length * n_cells)
        target_dist = torch.sqrt(
            (sample_patch - target_patch[:, None, :]).pow(2).sum(dim=-1) + eps
        ).mean(dim=1) / scale
        pair_dist = torch.cdist(sample_patch, sample_patch, p=2).mean(dim=(1, 2)) / scale
        scores.append(target_dist - 0.5 * pair_dist)
        target_terms.append(target_dist)
        pair_terms.append(pair_dist)
    score = torch.stack(scores, dim=0).mean()
    target_term = torch.stack(target_terms, dim=0).mean()
    pair_term = torch.stack(pair_terms, dim=0).mean()
    return score, target_term, pair_term


def combined_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    train_sample_count: int,
    rollout_flow_steps: int,
    patch_len: int,
    patch_energy_weight: float,
    fm_anchor_weight: float,
    energy_eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(history_norm, future_norm)
    target_scores = model.target_future_scores(future_norm)
    sampled_scores = sample_rollout_scores_with_grad(
        model=model,
        history_norm=history_norm,
        n_samples=train_sample_count,
        n_steps=target_scores.shape[1],
        flow_steps=rollout_flow_steps,
    )
    patch_loss, target_dist, pair_dist = patch_energy_score(
        samples=sampled_scores,
        target=target_scores,
        patch_len=patch_len,
        eps=energy_eps,
    )
    total = float(fm_anchor_weight) * fm_loss + float(patch_energy_weight) * patch_loss
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "patch_energy": patch_loss.detach(),
        "patch_target_dist": target_dist.detach(),
        "patch_pair_dist": pair_dist.detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        "sample_score_std": sampled_scores.std(unbiased=False).detach(),
        "target_score_std": target_scores.std(unbiased=False).detach(),
        "sample_h1_std": sampled_scores[:, :, 0].std(unbiased=False).detach(),
        "sample_h30_std": sampled_scores[:, :, -1].std(unbiased=False).detach(),
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
    parser.add_argument("--patch_len", type=int, default=5)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--patch_energy_weight", type=float, default=0.05)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                    patch_len=args.patch_len,
                    patch_energy_weight=args.patch_energy_weight,
                    fm_anchor_weight=args.fm_anchor_weight,
                    energy_eps=args.energy_eps,
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
            "train_patch_energy": train_avg["patch_energy"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_patch_energy": val_avg["patch_energy"],
            "val_patch_target_dist": val_avg["patch_target_dist"],
            "val_patch_pair_dist": val_avg["patch_pair_dist"],
            "val_sample_score_std": val_avg["sample_score_std"],
            "val_target_score_std": val_avg["target_score_std"],
            "val_sample_h1_std": val_avg["sample_h1_std"],
            "val_sample_h30_std": val_avg["sample_h30_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"patch={rec['val_patch_energy']:.5f} "
            f"target={rec['val_patch_target_dist']:.3f} pair={rec['val_patch_pair_dist']:.3f} "
            f"std={rec['val_sample_score_std']:.3f}/{rec['val_target_score_std']:.3f} "
            f"h1/h30={rec['val_sample_h1_std']:.3f}/{rec['val_sample_h30_std']:.3f} "
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
            "patch_energy_weight": args.patch_energy_weight,
            "patch_len": args.patch_len,
            "train_sample_count": args.train_sample_count,
            "rollout_flow_steps": args.rollout_flow_steps,
            "loss": "overlapping_patch_energy",
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
