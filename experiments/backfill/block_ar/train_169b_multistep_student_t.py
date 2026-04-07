#!/usr/bin/env python
"""
169b: Multi-Step Teacher-Forced Student-t Density Model

Safe extension of 169a:
  - same support-aware transform
  - same low-rank + diagonal Student-t head
  - same fixed-nu option (recommended: nu=8)
  - same GRU encoder and spatial decoder family

Only the training objective changes:
  - train on the full 30-step future with teacher-forced multistep NLL
  - evaluate with free rollout using sample_batched() on a small validation subset

This is the minimal rollout-aware density experiment intended to answer whether
the one-step fixed-nu recipe survives when trained against the actual horizon.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_169b_multistep_student_t.py \
        --epochs 20 --batch_size 16 --rank 5 --fixed_nu 8.0 \
        --output_dir models/backfill/student_t_169b --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys; sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    OneStepStudentTARModel,
    effective_rank,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)


def build_multistep_windows(
    indices: np.ndarray,
    surf: torch.Tensor,
    hist_len: int,
    future_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build history and future tensors directly on device."""
    idx = torch.from_numpy(indices).long().to(surf.device)
    offsets_h = torch.arange(hist_len, device=surf.device).unsqueeze(0)
    offsets_f = torch.arange(future_len, device=surf.device).unsqueeze(0)
    hist_idx = idx.unsqueeze(1) + offsets_h
    fut_idx = idx.unsqueeze(1) + hist_len + offsets_f
    hist = surf[hist_idx]  # (N, H, 5, 5)
    future = surf[fut_idx].reshape(len(indices), future_len, -1)  # (N, T, 25)
    return hist, future


def teacher_forced_multistep_loss(
    model: OneStepStudentTARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Teacher-forced multistep NLL over the full future horizon.

    Uses the existing GRU encoder state and appends true future frames step by step,
    matching the autoregressive rollout structure without changing the model class.
    """
    batch_size, hist_len = history_01.shape[:2]
    future_len = future_01.shape[1]
    n_cells = future_01.shape[-1]

    hist_norm = normalize_iv(history_01).reshape(batch_size, hist_len, n_cells)
    gru_outputs, gru_state = model.encoder.gru(hist_norm)
    prev_01 = history_01[:, -1].reshape(batch_size, n_cells)

    total_nll = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_diag = 0.0
    total_nu = 0.0

    for step in range(future_len):
        attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
        cond = model.encoder.bottleneck(pooled)

        prev_u = iv_to_unconstrained(
            prev_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mu, factor, diag, nu = model.decoder(cond, prev_u)

        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        nll_t = model.student_t_nll(target_u, mu, factor, diag, nu)
        cov_t = model.covariance(factor, diag)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        total_nll = total_nll + nll_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(cov_t).mean()
        total_diag = total_diag + diag.mean()
        total_nu = total_nu + nu.mean()

        next_norm = normalize_iv(target_t).unsqueeze(1)
        next_out, gru_state = model.encoder.gru(next_norm, gru_state)
        gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
        prev_01 = target_t

    scale = 1.0 / future_len
    metrics = {
        "multistep_nll": total_nll * scale,
        "multistep_mae": total_mae * scale,
        "pred_eff_rank": total_rank * scale,
        "diag_mean": total_diag * scale,
        "nu_mean": total_nu * scale,
    }
    return total_nll * scale, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: OneStepStudentTARModel,
    val_loader: DataLoader,
) -> dict:
    model.eval()
    total_nll = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_diag = 0.0
    total_nu = 0.0
    total_count = 0

    for history_01, future_01 in val_loader:
        loss, metrics = teacher_forced_multistep_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        total_nll += loss.item() * batch_size
        total_mae += metrics["multistep_mae"].item() * batch_size
        total_rank += metrics["pred_eff_rank"].item() * batch_size
        total_diag += metrics["diag_mean"].item() * batch_size
        total_nu += metrics["nu_mean"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {
            "val_multistep_nll": float("nan"),
            "val_multistep_mae": float("nan"),
            "val_pred_eff_rank": float("nan"),
            "val_diag_mean": float("nan"),
            "val_nu_mean": float("nan"),
        }

    return {
        "val_multistep_nll": total_nll / total_count,
        "val_multistep_mae": total_mae / total_count,
        "val_pred_eff_rank": total_rank / total_count,
        "val_diag_mean": total_diag / total_count,
        "val_nu_mean": total_nu / total_count,
    }


@torch.no_grad()
def evaluate_rollout_subset(
    model: OneStepStudentTARModel,
    val_loader: DataLoader,
    rollout_val_samples: int,
    rollout_eval_limit: int,
) -> dict:
    """Small free-rollout probe to monitor exposure bias during training."""
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []

    for history_01, future_01 in val_loader:
        if total_count >= rollout_eval_limit:
            break
        if total_count + history_01.shape[0] > rollout_eval_limit:
            keep = rollout_eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples = model.sample_batched(
            history_norm,
            n_samples=rollout_val_samples,
            n_steps=future_01.shape[1],
        )  # (B, K, T, 5, 5) in [0.01, 1.0]

        future_grid = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        median = samples.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_grid).abs().mean()
        support_viol = (
            (samples < model.support_lo) | (samples > model.support_hi)
        ).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_count += history_01.shape[0]

        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {
            "rollout_cov90": float("nan"),
            "rollout_width90": float("nan"),
            "rollout_mae": float("nan"),
            "rollout_support_violation_rate": float("nan"),
            "rollout_turb_calm_ratio": float("nan"),
        }

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    if calm_mask.any() and turb_mask.any():
        turb_calm_ratio = (widths[turb_mask].mean() / widths[calm_mask].mean()).item()
    else:
        turb_calm_ratio = float("nan")

    return {
        "rollout_cov90": total_cov / total_count,
        "rollout_width90": total_width / total_count,
        "rollout_mae": total_mae / total_count,
        "rollout_support_violation_rate": total_support_viol / total_count,
        "rollout_turb_calm_ratio": turb_calm_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="169b: multistep transformed Student-t prototype")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=8.0,
                        help="Recommended safe setting: fixed scalar nu=8.0")
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--rollout_eval_every", type=int, default=1)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    hist_len = args.history_len
    future_len = args.future_len

    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[:args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[:args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, hist_len, future_len
    )
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, hist_len, future_len
    )

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
    )

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        rank=args.rank,
        diag_floor=args.diag_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
    )
    model = OneStepStudentTARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n{'=' * 64}")
    print("169b: Multi-Step Teacher-Forced Student-t Density Model")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Rank={args.rank} | d_model={args.d_model}")
    print(f"  Support transform: logit(({args.support_lo}, {args.support_hi}))")
    print("  Objective: teacher-forced multistep Student-t NLL")
    print(f"  Tail parameter: {'learned' if args.fixed_nu is None else f'fixed nu={args.fixed_nu}'}")

    optimizer = torch.optim.AdamW([
        {
            "params": model.encoder.parameters(),
            "lr": args.lr_encoder,
            "weight_decay": args.weight_decay_encoder,
        },
        {
            "params": model.decoder.parameters(),
            "lr": args.lr_decoder,
            "weight_decay": args.weight_decay_decoder,
        },
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        ep_mae = 0.0
        ep_rank = 0.0
        ep_diag = 0.0
        ep_nu = 0.0
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = teacher_forced_multistep_loss(model, history_01, future_01)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_mae += metrics["multistep_mae"].item()
            ep_rank += metrics["pred_eff_rank"].item()
            ep_diag += metrics["diag_mean"].item()
            ep_nu += metrics["nu_mean"].item()
            nb += 1

        scheduler.step()

        train_metrics = {
            "train_multistep_nll": ep_loss / max(nb, 1),
            "train_multistep_mae": ep_mae / max(nb, 1),
            "train_pred_eff_rank": ep_rank / max(nb, 1),
            "train_diag_mean": ep_diag / max(nb, 1),
            "train_nu_mean": ep_nu / max(nb, 1),
        }
        val_metrics = evaluate_teacher_forced(model, val_loader)

        rollout_metrics = {}
        if args.rollout_eval_every > 0 and epoch % args.rollout_eval_every == 0:
            rollout_metrics = evaluate_rollout_subset(
                model,
                val_loader,
                rollout_val_samples=args.rollout_val_samples,
                rollout_eval_limit=args.rollout_eval_limit,
            )
        else:
            rollout_metrics = {
                "rollout_cov90": float("nan"),
                "rollout_width90": float("nan"),
                "rollout_mae": float("nan"),
                "rollout_support_violation_rate": float("nan"),
                "rollout_turb_calm_ratio": float("nan"),
            }

        elapsed = time.time() - t0
        is_best = val_metrics["val_multistep_nll"] < best_val
        if is_best:
            best_val = val_metrics["val_multistep_nll"]
            best_metrics = {**val_metrics, **rollout_metrics}
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_multistep_nll": best_val,
                "config": {
                    "type": "multi_step_student_t_169b",
                    "encoder": vars(encoder_config),
                    "decoder": decoder_config,
                    "support_lo": args.support_lo,
                    "support_hi": args.support_hi,
                    "support_eps": args.support_eps,
                    "fixed_nu": args.fixed_nu,
                    "history_len": hist_len,
                    "future_len": future_len,
                    "train_windows": len(train_indices),
                    "val_windows": len(val_indices),
                },
                "best_metrics": best_metrics,
            }, f"{args.output_dir}/best_model.pt")

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            **rollout_metrics,
        }
        history.append(row)

        print(
            f"Ep {epoch:3d}  "
            f"train_nll={train_metrics['train_multistep_nll']:.4f}  "
            f"val_nll={val_metrics['val_multistep_nll']:.4f}  "
            f"roll_cov90={rollout_metrics['rollout_cov90']:.4f}  "
            f"roll_tc={rollout_metrics['rollout_turb_calm_ratio']:.3f}  "
            f"rank={val_metrics['val_pred_eff_rank']:.2f}  "
            f"nu={val_metrics['val_nu_mean']:.2f}  "
            f"viol={rollout_metrics['rollout_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_multistep_nll": history[-1]["val_multistep_nll"] if history else float("nan"),
        "config": {
            "type": "multi_step_student_t_169b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "fixed_nu": args.fixed_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_val_multistep_nll": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest val multistep NLL: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"rollout_cov90={best_metrics['rollout_cov90']:.4f}, "
            f"rollout_turb_calm_ratio={best_metrics['rollout_turb_calm_ratio']:.3f}, "
            f"rollout_support_violation_rate={best_metrics['rollout_support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
