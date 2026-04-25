#!/usr/bin/env python
"""483a: frozen 392a proposal with learned density-ratio resampling."""

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
from diffusion.block_ar.frozen_proposal_density_ratio_resampler import (  # noqa: E402
    FrozenProposalDensityRatioConfig,
    FrozenProposalDensityRatioScorer,
    density_ratio_nce_loss,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_391a_recent_rollout_energy_finetune import (  # noqa: E402
    build_recent_block,
)


@torch.no_grad()
def precompute_ratio_dataset(
    base_model: torch.nn.Module,
    hist_01: torch.Tensor,
    fut_01: torch.Tensor,
    train_negatives: int,
    source_chunk_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_scores: list[torch.Tensor] = []
    positive_scores: list[torch.Tensor] = []
    negative_scores: list[torch.Tensor] = []
    n_steps = int(fut_01.shape[1])
    for start in range(0, hist_01.shape[0], batch_size):
        end = min(start + batch_size, hist_01.shape[0])
        hist_batch = hist_01[start:end].to(device, non_blocking=True)
        fut_batch = fut_01[start:end].to(device, non_blocking=True)
        hist_norm = normalize_iv(hist_batch).view(hist_batch.shape[0], hist_batch.shape[1], -1)
        fut_norm = normalize_iv(fut_batch).view(fut_batch.shape[0], fut_batch.shape[1], -1)
        proposal_01 = base_model.sample_batched(
            hist_norm,
            n_samples=train_negatives,
            n_steps=n_steps,
            chunk_size=source_chunk_size,
            history_is_normalized=True,
        )
        flat_proposal = proposal_01.reshape(
            proposal_01.shape[0] * proposal_01.shape[1],
            n_steps,
            5,
            5,
        )
        proposal_norm = normalize_iv(flat_proposal).view(
            flat_proposal.shape[0],
            n_steps,
            -1,
        )
        proposal_scores = base_model.target_future_scores(proposal_norm).view(
            proposal_01.shape[0],
            proposal_01.shape[1],
            n_steps,
            base_model.cfg.n_cells,
        )
        history_scores.append(base_model.history_scores(hist_norm).cpu())
        positive_scores.append(base_model.target_future_scores(fut_norm).cpu())
        negative_scores.append(proposal_scores.cpu())
    return (
        torch.cat(history_scores, dim=0),
        torch.cat(positive_scores, dim=0),
        torch.cat(negative_scores, dim=0),
    )


def epoch_pass(
    model: FrozenProposalDensityRatioScorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    logit_l2: float,
    max_batches: int,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: dict[str, float] = {}
    n_obs = 0
    for batch_idx, (history, positive, negative) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        history = history.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = density_ratio_nce_loss(
            model,
            history_scores=history,
            positive_future_scores=positive,
            negative_future_scores=negative,
            logit_l2=logit_l2,
        )
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bsz = int(history.shape[0])
        n_obs += bsz
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * bsz
    if n_obs == 0:
        raise RuntimeError("No batches processed")
    return {key: value / n_obs for key, value in totals.items()}


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
    parser.add_argument("--train_negatives", type=int, default=32)
    parser.add_argument("--source_chunk_size", type=int, default=4)
    parser.add_argument("--precompute_batch_size", type=int, default=8)

    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--stats_hidden", type=int, default=192)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--candidate_chunk_size", type=int, default=8)
    parser.add_argument("--max_score_chunk", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--logit_l2", type=float, default=1e-4)
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
    print("Precomputing frozen 392a proposal futures...")
    history_scores, positive_scores, negative_scores = precompute_ratio_dataset(
        base_model=base_model,
        hist_01=hist_01,
        fut_01=fut_01,
        train_negatives=args.train_negatives,
        source_chunk_size=args.source_chunk_size,
        batch_size=args.precompute_batch_size,
        device=device,
    )
    torch.save(
        {
            "history_scores": history_scores,
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
            "indices": torch.from_numpy(indices.copy()),
        },
        out_dir / "ratio_dataset.pt",
    )

    cfg = FrozenProposalDensityRatioConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=base_model.cfg.n_cells,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        stats_hidden=args.stats_hidden,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        candidate_multiplier=args.candidate_multiplier,
        candidate_chunk_size=args.candidate_chunk_size,
        max_score_chunk=args.max_score_chunk,
    )
    model = FrozenProposalDensityRatioScorer(cfg).to(device)
    print(f"Scorer params: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = TensorDataset(
        history_scores[:n_train],
        positive_scores[:n_train],
        negative_scores[:n_train],
    )
    val_ds = TensorDataset(
        history_scores[n_train:],
        positive_scores[n_train:],
        negative_scores[n_train:],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = epoch_pass(
            model,
            train_loader,
            optimizer,
            device,
            logit_l2=args.logit_l2,
            max_batches=args.max_train_batches,
        )
        val_metrics = epoch_pass(
            model,
            val_loader,
            None,
            device,
            logit_l2=args.logit_l2,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        if val_metrics["ce"] < best_val:
            best_val = val_metrics["ce"]
            best_epoch = epoch
            save_checkpoint(
                str(out_dir / "best_model.pt"),
                model,
                cfg,
                epoch,
                best_val,
                args.base_checkpoint,
            )
        print(
            f"[ep {epoch:03d}] "
            f"train_ce={train_metrics['ce']:.4f} val_ce={val_metrics['ce']:.4f} "
            f"train_acc={train_metrics['acc']:.3f} val_acc={val_metrics['acc']:.3f} "
            f"pos={val_metrics['pos_logit']:.3f} neg={val_metrics['neg_logit']:.3f} "
            f"std={val_metrics['logit_std']:.3f} ent={val_metrics['entropy']:.3f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} time={time.time() - t0:.1f}s",
            flush=True,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_ce": best_val,
        "n_train": n_train,
        "n_val": n_val,
        "config": vars(args),
    }
    print(json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
