#!/usr/bin/env python
"""558a: hard-state replay patch-energy fine-tune from the 510a frontier."""

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


def hard_state_scores_from_intervals(
    samples: np.ndarray,
    future_01: np.ndarray,
    target_coverage: float = 0.90,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute calibration-only hard-state scores from generated intervals."""
    samples = np.asarray(samples, dtype=np.float64)
    future = np.asarray(future_01, dtype=np.float64)
    if samples.ndim != 5:
        raise ValueError("samples must have shape (windows, samples, horizon, rows, cols)")
    if future.ndim != 4:
        raise ValueError("future_01 must have shape (windows, horizon, rows, cols)")
    if samples.shape[0] != future.shape[0] or samples.shape[2:] != future.shape[1:]:
        raise ValueError("samples and future_01 have incompatible shapes")

    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    covered = (future >= q05) & (future <= q95)
    upper = future > q95
    lower = future < q05
    coverage = covered.mean(axis=(1, 2, 3))
    upper_rate = upper.mean(axis=(1, 2, 3))
    lower_rate = lower.mean(axis=(1, 2, 3))
    under_target = np.maximum(0.0, float(target_coverage) - coverage)
    scores = under_target + upper_rate + lower_rate
    return scores.astype(np.float64), {
        "coverage90": coverage,
        "coverage_under_target": under_target,
        "upper_miss_rate": upper_rate,
        "lower_miss_rate": lower_rate,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def rank_replay_weights(scores: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Convert hard-state scores to smooth mean-one replay weights."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if values.shape[0] == 0:
        raise ValueError("scores must be non-empty")
    if values.shape[0] == 1 or np.ptp(values) <= 0.0:
        return np.ones(values.shape[0], dtype=np.float32)
    rank = _rankdata(values) / float(values.shape[0] - 1)
    weights = 1.0 + float(strength) * rank * rank
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def patch_energy_score_per_window(
    samples: torch.Tensor,
    target: torch.Tensor,
    patch_len: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy score per window, averaged over overlapping future patches."""
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
        target_terms.append(target_dist)
        pair_terms.append(pair_dist)
        scores.append(target_dist - 0.5 * pair_dist)
    return (
        torch.stack(scores, dim=1).mean(dim=1),
        torch.stack(target_terms, dim=1).mean(dim=1),
        torch.stack(pair_terms, dim=1).mean(dim=1),
    )


def weighted_combined_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    replay_weight: torch.Tensor,
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
    per_score, per_target, per_pair = patch_energy_score_per_window(
        samples=sampled_scores,
        target=target_scores,
        patch_len=patch_len,
        eps=energy_eps,
    )
    weight = replay_weight.to(device=per_score.device, dtype=per_score.dtype)
    weight = weight / weight.mean().clamp_min(1e-8)
    patch_loss = (weight * per_score).mean()
    target_dist = (weight * per_target).mean()
    pair_dist = (weight * per_pair).mean()
    total = float(fm_anchor_weight) * fm_loss + float(patch_energy_weight) * patch_loss
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "patch_energy": patch_loss.detach(),
        "patch_target_dist": target_dist.detach(),
        "patch_pair_dist": pair_dist.detach(),
        "replay_weight_mean": weight.mean().detach(),
        "replay_weight_max": weight.max().detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        "sample_score_std": sampled_scores.std(unbiased=False).detach(),
        "target_score_std": target_scores.std(unbiased=False).detach(),
    }
    return total, metrics


@torch.no_grad()
def sample_model_values(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_01: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    outs: list[np.ndarray] = []
    model.eval()
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist_norm = normalize_iv(history_01[start:end].to(device)).view(
            end - start,
            history_01.shape[1],
            5,
            5,
        )
        samples = model.sample_batched(
            hist_norm,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        )
        outs.append(samples.detach().cpu().numpy())
        print(f"  hard-weight sampling windows {end}/{history_01.shape[0]}", flush=True)
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt",
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
    parser.add_argument("--weight_samples", type=int, default=32)
    parser.add_argument("--weight_chunk_size", type=int, default=8)
    parser.add_argument("--replay_strength", type=float, default=2.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=558)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
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
    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')}  source best_val: {payload.get('best_val')}")
    print(f"Recent windows: {hist_01.shape[0]} index range: {indices[0]}..{indices[-1]}")
    print("Computing calibration hard-state replay weights from frozen source...")
    weight_samples = sample_model_values(
        model=model,
        history_01=hist_01,
        n_samples=args.weight_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.weight_chunk_size,
        device=device,
    )
    scores, hard_metrics = hard_state_scores_from_intervals(
        weight_samples,
        fut_01.detach().cpu().numpy(),
    )
    weights_np = rank_replay_weights(scores, strength=args.replay_strength)
    weights = torch.from_numpy(weights_np).to(device)

    n_total = hist_01.shape[0]
    n_val = max(1, int(round(n_total * float(args.holdout_frac))))
    n_train = n_total - n_val
    train_hist, val_hist = hist_01[:n_train], hist_01[n_train:]
    train_fut, val_fut = fut_01[:n_train], fut_01[n_train:]
    train_w, val_w = weights[:n_train], weights[n_train:]
    train_loader = DataLoader(
        TensorDataset(train_hist, train_fut, train_w),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_fut, val_w),
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
        for hist_batch, fut_batch, weight_batch in loader:
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
                loss, metrics = weighted_combined_loss(
                    model=model,
                    history_norm=hist_norm,
                    future_norm=fut_norm,
                    replay_weight=weight_batch.to(device, non_blocking=True),
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

    print(f"Train/holdout: {n_train}/{n_val}")
    print(
        "Replay weights: "
        f"score_mean={scores.mean():.4f} score_p90={np.quantile(scores, 0.90):.4f} "
        f"w_min={weights_np.min():.3f} w_max={weights_np.max():.3f}"
    )
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
            "train_patch_energy": train_avg["patch_energy"],
            "train_replay_weight_max": train_avg["replay_weight_max"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_patch_energy": val_avg["patch_energy"],
            "val_patch_target_dist": val_avg["patch_target_dist"],
            "val_patch_pair_dist": val_avg["patch_pair_dist"],
            "val_replay_weight_max": val_avg["replay_weight_max"],
            "val_sample_score_std": val_avg["sample_score_std"],
            "val_target_score_std": val_avg["target_score_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"patch={rec['val_patch_energy']:.5f} wmax={rec['val_replay_weight_max']:.2f} "
            f"std={rec['val_sample_score_std']:.3f}/{rec['val_target_score_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s",
            flush=True,
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, model.cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, model.cfg, args.epochs, best_val)
    hard_summary = {
        "score_mean": float(scores.mean()),
        "score_p50": float(np.quantile(scores, 0.50)),
        "score_p90": float(np.quantile(scores, 0.90)),
        "score_max": float(scores.max()),
        "coverage90_mean": float(hard_metrics["coverage90"].mean()),
        "upper_miss_rate_mean": float(hard_metrics["upper_miss_rate"].mean()),
        "lower_miss_rate_mean": float(hard_metrics["lower_miss_rate"].mean()),
        "weight_min": float(weights_np.min()),
        "weight_mean": float(weights_np.mean()),
        "weight_max": float(weights_np.max()),
    }
    (out_dir / "hard_state_weights.json").write_text(
        json.dumps(
            {
                "indices": [int(x) for x in indices],
                "scores": scores.tolist(),
                "weights": weights_np.tolist(),
                "summary": hard_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "hard_state_summary": hard_summary,
        "objective": {
            "fm_anchor_weight": args.fm_anchor_weight,
            "patch_energy_weight": args.patch_energy_weight,
            "patch_len": args.patch_len,
            "train_sample_count": args.train_sample_count,
            "rollout_flow_steps": args.rollout_flow_steps,
            "weight_samples": args.weight_samples,
            "replay_strength": args.replay_strength,
            "loss": "hard_state_replay_patch_energy",
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
