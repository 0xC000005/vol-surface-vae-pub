#!/usr/bin/env python
"""467a: recent-window conditional critic fine-tune for the 392a generator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
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


class ConditionalPathCritic(nn.Module):
    def __init__(
        self,
        history_len: int,
        future_len: int,
        n_cells: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = (history_len + future_len) * n_cells
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, history_scores: torch.Tensor, future_scores: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history_scores, future_scores], dim=1)
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=1)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--adv_weight", type=float, default=0.01)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--critic_hidden", type=int, default=512)
    parser.add_argument("--critic_dropout", type=float, default=0.1)
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
    if model.cfg.history_len != args.history_len or model.cfg.future_len != args.future_len:
        raise ValueError("Checkpoint horizon configuration does not match requested data")

    critic = ConditionalPathCritic(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=model.cfg.n_cells,
        hidden_dim=args.critic_hidden,
        dropout=args.critic_dropout,
    ).to(device)

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

    gen_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.weight_decay,
    )
    gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=args.epochs)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=args.epochs)

    def run_epoch(loader: DataLoader, train_mode: bool, max_batches: int) -> dict[str, float]:
        model.train(train_mode)
        critic.train(train_mode)
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
            history_scores = model.history_scores(hist_norm)
            target_scores = model.target_future_scores(fut_norm)

            critic_loss = torch.tensor(0.0, device=device)
            real_logit = torch.tensor(0.0, device=device)
            fake_logit = torch.tensor(0.0, device=device)
            if train_mode:
                with torch.no_grad():
                    fake_scores_detached = sample_rollout_scores_with_grad(
                        model=model,
                        history_norm=hist_norm,
                        n_samples=args.train_sample_count,
                        n_steps=target_scores.shape[1],
                        flow_steps=args.rollout_flow_steps,
                    ).detach()
                hist_rep = history_scores.detach().repeat_interleave(
                    args.train_sample_count,
                    dim=0,
                )
                fake_flat = fake_scores_detached.reshape(
                    hist_rep.shape[0],
                    target_scores.shape[1],
                    target_scores.shape[2],
                )
                real_logits = critic(history_scores.detach(), target_scores.detach())
                fake_logits = critic(hist_rep, fake_flat)
                critic_loss = (
                    F.softplus(-real_logits).mean()
                    + F.softplus(fake_logits).mean()
                )
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.clip_grad)
                critic_optimizer.step()
                real_logit = real_logits.mean().detach()
                fake_logit = fake_logits.mean().detach()

            with torch.set_grad_enabled(train_mode):
                fm_loss, fm_metrics = model.training_loss(hist_norm, fut_norm)
                fake_scores = sample_rollout_scores_with_grad(
                    model=model,
                    history_norm=hist_norm,
                    n_samples=args.train_sample_count,
                    n_steps=target_scores.shape[1],
                    flow_steps=args.rollout_flow_steps,
                )
                hist_rep = history_scores.repeat_interleave(args.train_sample_count, dim=0)
                fake_flat = fake_scores.reshape(
                    hist_rep.shape[0],
                    target_scores.shape[1],
                    target_scores.shape[2],
                )
                set_requires_grad(critic, False)
                gen_logits = critic(hist_rep, fake_flat)
                set_requires_grad(critic, True)
                gen_adv = F.softplus(-gen_logits).mean()
                total = float(args.fm_anchor_weight) * fm_loss + float(args.adv_weight) * gen_adv
                if train_mode:
                    gen_optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    gen_optimizer.step()

            metrics = {
                "total": total.detach(),
                "fm_loss": fm_loss.detach(),
                "gen_adv": gen_adv.detach(),
                "critic_loss": critic_loss.detach(),
                "critic_real_logit": real_logit.detach(),
                "critic_fake_logit": fake_logit.detach(),
                "transition_std": fm_metrics["transition_std"].detach(),
                "sample_score_std": fake_scores.detach().std(unbiased=False),
                "target_score_std": target_scores.detach().std(unbiased=False),
                "sample_h1_std": fake_scores.detach()[:, :, 0].std(unbiased=False),
                "sample_h30_std": fake_scores.detach()[:, :, -1].std(unbiased=False),
            }
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')}  source best_val: {payload.get('best_val')}")
    print(f"Recent windows: {n_total} index range: {indices[0]}..{indices[-1]}")
    print(f"Train/holdout: {n_train}/{n_val}")
    print(f"Generator params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True, max_batches=args.max_train_batches)
        val_avg = run_epoch(val_loader, train_mode=False, max_batches=args.max_val_batches)
        gen_scheduler.step()
        critic_scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_fm_loss": train_avg["fm_loss"],
            "train_gen_adv": train_avg["gen_adv"],
            "train_critic_loss": train_avg["critic_loss"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_gen_adv": val_avg["gen_adv"],
            "val_sample_score_std": val_avg["sample_score_std"],
            "val_target_score_std": val_avg["target_score_std"],
            "val_sample_h1_std": val_avg["sample_h1_std"],
            "val_sample_h30_std": val_avg["sample_h30_std"],
            "gen_lr": gen_optimizer.param_groups[0]["lr"],
            "critic_lr": critic_optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"adv={rec['val_gen_adv']:.5f} "
            f"std={rec['val_sample_score_std']:.3f}/{rec['val_target_score_std']:.3f} "
            f"h1/h30={rec['val_sample_h1_std']:.3f}/{rec['val_sample_h30_std']:.3f} "
            f"lr={rec['gen_lr']:.2e}/{rec['critic_lr']:.2e} time={rec['sec']:.1f}s"
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, model.cfg, epoch, best_val)
            torch.save(
                {
                    "epoch": epoch,
                    "critic_state_dict": critic.state_dict(),
                    "config": vars(args),
                },
                out_dir / "best_critic.pt",
            )

    save_checkpoint(str(out_dir / "final_model.pt"), model, model.cfg, args.epochs, best_val)
    torch.save(
        {
            "epoch": args.epochs,
            "critic_state_dict": critic.state_dict(),
            "config": vars(args),
        },
        out_dir / "final_critic.pt",
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
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "objective": {
            "fm_anchor_weight": args.fm_anchor_weight,
            "adv_weight": args.adv_weight,
            "train_sample_count": args.train_sample_count,
            "rollout_flow_steps": args.rollout_flow_steps,
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
