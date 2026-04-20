#!/usr/bin/env python
"""
241a: 183c + paper-exact Contrastive Flow Matching (Stojanovski et al., arXiv 2506.05350).

L_total = fm_loss + local/band/smooth/budget regs (same as 183c)
        - λ_contrastive * ||pred_v - neg_target_v||^2

neg_target_v is the TARGET velocity of a randomly permuted batch element (no gradient on
neg_target_v). This is Algorithm 1 of the paper: random negatives, no class labels required.

Warm-start from 183c best. Single training stage (everything unfrozen). bf16 autocast.
Tuned DataLoader. Grad accumulation for effective batch 128.

Bitter Lesson: zero hand features. Regime labels NOT used — the paper does not need them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import make_serializable
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    checkpoint_key,
    evaluate_frontier_subset,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel,
    evaluate_teacher_forced,
)


def warm_start_with_gates(model: StateMetricTransportModel, ckpt_path: str, device):
    """Warm-start that includes local_state_gate / band_state_gate (183c's skips them)."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = dict(payload["model_state_dict"])
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  Warm start (with gates) loaded from {ckpt_path}")
    print(f"  Warm start missing: {len(missing)} | unexpected: {len(unexpected)} | shape-skipped: {len(skipped)}")


def contrastive_fm_loss(
    model: StateMetricTransportModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    lambda_contrastive: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """183c loss + paper-exact contrastive term with random batch negatives."""
    target_u = iv_to_unconstrained(
        future_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps,
    )
    with torch.no_grad():
        (
            mu, time_factor, time_diag, cell_factor, cell_diag, scale,
            flow_context, base_local_delta, block_logits,
        ) = model.forward_from_history(history_01)
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u, mu=mu, time_factor=time_factor, time_diag=time_diag,
            cell_factor=cell_factor, cell_diag=cell_diag, scale=scale,
            base_local_delta=base_local_delta, block_logits=block_logits,
        ).view(history_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, target_band_log = compute_control_targets(model, target_basis)

    target_basis_flat = target_basis.reshape(target_basis.shape[0], -1)
    z0, prior_stats = model.prior.sample(
        target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype,
    )
    t = torch.rand(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis_flat
    target_v = target_basis_flat - z0

    pred_v, control_metrics = model.transport_velocity(z_t, t, path_context)
    fm_loss = F.mse_loss(pred_v, target_v)

    # Paper-exact contrastive term: random-permutation negative target velocity.
    if lambda_contrastive > 0:
        B = target_basis.shape[0]
        perm = torch.randperm(B, device=target_basis.device)
        # Reject identity permutations (probability ~ 1/B!): rotate if fixed-point-heavy.
        if (perm == torch.arange(B, device=perm.device)).all():
            perm = torch.roll(perm, 1)
        with torch.no_grad():
            neg_target_v = target_basis_flat[perm] - z0[perm]
        contrastive_neg = F.mse_loss(pred_v, neg_target_v)
        cfm_loss = fm_loss - lambda_contrastive * contrastive_neg
    else:
        contrastive_neg = torch.zeros((), device=target_basis.device)
        cfm_loss = fm_loss

    # Rest of 183c's loss (control targets, smoothness, budget) — unchanged.
    raw_local, raw_band, metric_local, metric_band, _local_gate, _band_gate = model.build_state_metric_controls(
        z_t, t, path_context,
    )
    local_loss = F.smooth_l1_loss(metric_local, target_local_log)
    band_loss = F.smooth_l1_loss(metric_band, target_band_log)
    surf = metric_local.view(metric_local.shape[0], metric_local.shape[1], 5, 5)
    time_smooth = (metric_local[:, 1:] - metric_local[:, :-1]).abs().mean()
    row_smooth = (surf[:, :, 1:] - surf[:, :, :-1]).abs().mean()
    col_smooth = (surf[:, :, :, 1:] - surf[:, :, :, :-1]).abs().mean()
    smooth_reg = time_smooth + row_smooth + col_smooth
    budget_reg = (
        (model.local_metric_budget() - model.metric_config["init_local_metric_budget"]).pow(2)
        + (model.band_metric_budget() - model.metric_config["init_band_metric_budget"]).pow(2)
    )

    total = (
        cfm_loss
        + model.integrated_config["local_loss_weight"] * local_loss
        + model.integrated_config["band_loss_weight"] * band_loss
        + model.integrated_config["smooth_reg_weight"] * smooth_reg
        + model.metric_config["budget_reg_weight"] * budget_reg
    )

    high_mask = model.path_geometry.high_band_mask().to(target_basis.device, dtype=target_basis.dtype)
    target_jump_like = (target_basis_flat.abs() > 2.5).float().mean(dim=-1)

    metrics = {
        "flow_match_loss": fm_loss.detach(),
        "contrastive_neg": contrastive_neg.detach(),
        "contrastive_ratio": (contrastive_neg / fm_loss.clamp_min(1e-8)).detach(),
        "target_basis_std_mean": target_basis_flat.std(dim=-1).mean(),
        "prior_std_mean": prior_stats["prior_std_mean"],
        "prior_jump_prob_mean": prior_stats["prior_jump_prob_mean"],
        "prior_jump_scale_mean": prior_stats["prior_jump_scale_mean"],
        "prior_active_rate": prior_stats["prior_active_rate"],
        "target_jump_like_rate": target_jump_like.mean(),
        "target_high_band_abs_mean": (
            (target_basis_flat.abs() * high_mask.unsqueeze(0)).sum(dim=-1) / high_mask.sum().clamp_min(1.0)
        ).mean(),
        "sdr_total_loss": total.detach(),
        "sdr_local_loss": local_loss.detach(),
        "sdr_band_loss": band_loss.detach(),
        "sdr_smooth_reg": smooth_reg.detach(),
        "sdr_budget_reg": budget_reg.detach(),
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        **control_metrics,
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="241a: 183c + Contrastive Flow Matching")
    # Core training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr_ctrl", type=float, default=2.5e-4)
    parser.add_argument("--lr_path", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    # Contrastive FM
    parser.add_argument("--lambda_contrastive", type=float, default=0.05,
                        help="Paper's λ; >= 0.1 risks velocity collapse (paper Table 5).")
    # Model hyperparameters (must match 183c checkpoint config)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_low_scale_clip", type=float, default=1.2)
    parser.add_argument("--flow_mid_scale_clip", type=float, default=0.7)
    parser.add_argument("--flow_high_scale_clip", type=float, default=0.35)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--n_blocks", type=int, default=5)
    parser.add_argument("--n_templates", type=int, default=3)
    parser.add_argument("--mix_chunk_size", type=int, default=27)
    parser.add_argument("--template_diag_clip", type=float, default=0.30)
    parser.add_argument("--template_offdiag_clip", type=float, default=0.18)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
    parser.add_argument("--path_context_dim", type=int, default=256)
    parser.add_argument("--path_context_hidden_dim", type=int, default=256)
    parser.add_argument("--path_d_model", type=int, default=192)
    parser.add_argument("--path_heads", type=int, default=4)
    parser.add_argument("--path_layers", type=int, default=4)
    parser.add_argument("--path_ff_mult", type=int, default=4)
    parser.add_argument("--path_time_embed_dim", type=int, default=64)
    parser.add_argument("--path_ode_steps", type=int, default=8)
    parser.add_argument("--prior_low_std", type=float, default=1.00)
    parser.add_argument("--prior_mid_std", type=float, default=0.75)
    parser.add_argument("--prior_high_std", type=float, default=0.35)
    parser.add_argument("--prior_low_jump_prob", type=float, default=0.005)
    parser.add_argument("--prior_mid_jump_prob", type=float, default=0.025)
    parser.add_argument("--prior_high_jump_prob", type=float, default=0.070)
    parser.add_argument("--prior_low_jump_scale", type=float, default=0.10)
    parser.add_argument("--prior_mid_jump_scale", type=float, default=0.30)
    parser.add_argument("--prior_high_jump_scale", type=float, default=0.75)
    parser.add_argument("--width_rank", type=int, default=4)
    parser.add_argument("--width_hidden_dim", type=int, default=256)
    parser.add_argument("--width_clip", type=float, default=0.80)
    parser.add_argument("--band_hidden_dim", type=int, default=128)
    parser.add_argument("--band_clip", type=float, default=0.45)
    parser.add_argument("--local_loss_weight", type=float, default=0.75)
    parser.add_argument("--band_loss_weight", type=float, default=0.50)
    parser.add_argument("--smooth_reg_weight", type=float, default=0.04)
    parser.add_argument("--state_gate_hidden_dim", type=int, default=128)
    parser.add_argument("--init_local_state_gate", type=float, default=0.30)
    parser.add_argument("--init_band_state_gate", type=float, default=0.25)
    parser.add_argument("--local_metric_min", type=float, default=0.12)
    parser.add_argument("--local_metric_max", type=float, default=0.55)
    parser.add_argument("--band_metric_min", type=float, default=0.08)
    parser.add_argument("--band_metric_max", type=float, default=0.35)
    parser.add_argument("--init_local_metric_budget", type=float, default=0.24)
    parser.add_argument("--init_band_metric_budget", type=float, default=0.14)
    parser.add_argument("--local_metric_clip", type=float, default=0.60)
    parser.add_argument("--band_metric_clip", type=float, default=0.35)
    parser.add_argument("--budget_reg_weight", type=float, default=0.02)
    # Infra
    parser.add_argument("--warm_start_path", type=str, default=None)
    parser.add_argument("--warmstart_include_gates", action="store_true", default=True)
    parser.add_argument("--no_warmstart_include_gates", dest="warmstart_include_gates", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--dataloader_workers", type=int, default=4)
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", dest="dataloader_pin_memory", action="store_false")
    parser.add_argument("--dataloader_persistent_workers", action="store_true", default=True)
    parser.add_argument("--no_persistent_workers", dest="dataloader_persistent_workers", action="store_false")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    # Match 183c pattern: put surf_tensor on GPU so window tensors are on-device.
    # Downstream evaluation functions assume batched tensors are already on-device.
    surf_tensor = torch.from_numpy(surfaces).to(args.device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, args.history_len, args.future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, args.history_len, args.future_len)
    # num_workers=0 required because GPU tensors can't cross worker-process boundary.
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    encoder_config = EncoderConfig(input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0)
    decoder_config = dict(
        n_frames=args.future_len, n_cells=25, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        cond_dim=128, time_rank=args.time_rank, cell_rank=args.cell_rank,
        diag_floor=args.diag_floor, scale_floor=args.scale_floor, init_diag=args.init_diag, init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim, local_delta_clip=args.local_delta_clip,
        n_blocks=args.n_blocks, n_templates=args.n_templates,
        template_diag_clip=args.template_diag_clip, template_offdiag_clip=args.template_offdiag_clip,
        drift_strength_max=args.drift_strength_max, equilibrium_offset_clip=args.equilibrium_offset_clip,
        init_drift_strength=args.init_drift_strength,
    )
    flow_config = dict(
        dim=args.future_len * 25, context_dim=args.flow_context_dim, n_frames=args.future_len,
        grid_h=5, grid_w=5, hidden_dim=args.flow_hidden_dim, n_layers=args.flow_layers,
        low_scale_clip=args.flow_low_scale_clip, mid_scale_clip=args.flow_mid_scale_clip,
        high_scale_clip=args.flow_high_scale_clip,
    )
    path_config = dict(
        context_dim=args.path_context_dim, context_hidden_dim=args.path_context_hidden_dim,
        d_model=args.path_d_model, n_heads=args.path_heads, n_layers=args.path_layers,
        ff_mult=args.path_ff_mult, time_embed_dim=args.path_time_embed_dim, n_ode_steps=args.path_ode_steps,
    )
    prior_config = dict(
        low_std=args.prior_low_std, mid_std=args.prior_mid_std, high_std=args.prior_high_std,
        low_jump_prob=args.prior_low_jump_prob, mid_jump_prob=args.prior_mid_jump_prob, high_jump_prob=args.prior_high_jump_prob,
        low_jump_scale=args.prior_low_jump_scale, mid_jump_scale=args.prior_mid_jump_scale, high_jump_scale=args.prior_high_jump_scale,
    )
    integrated_config = dict(
        width_rank=args.width_rank, width_hidden_dim=args.width_hidden_dim, width_clip=args.width_clip,
        band_hidden_dim=args.band_hidden_dim, band_clip=args.band_clip,
        local_loss_weight=args.local_loss_weight, band_loss_weight=args.band_loss_weight,
        smooth_reg_weight=args.smooth_reg_weight,
        init_local_strength=0.05, init_band_strength=0.05,
    )
    state_config = dict(
        gate_hidden_dim=args.state_gate_hidden_dim,
        init_local_state_gate=args.init_local_state_gate,
        init_band_state_gate=args.init_band_state_gate,
    )
    metric_config = dict(
        local_metric_min=args.local_metric_min, local_metric_max=args.local_metric_max,
        band_metric_min=args.band_metric_min, band_metric_max=args.band_metric_max,
        init_local_metric_budget=args.init_local_metric_budget, init_band_metric_budget=args.init_band_metric_budget,
        local_metric_clip=args.local_metric_clip, band_metric_clip=args.band_metric_clip,
        budget_reg_weight=args.budget_reg_weight,
    )

    model = StateMetricTransportModel(
        encoder_config=encoder_config, decoder_config=decoder_config, flow_config=flow_config,
        path_config=path_config, prior_config=prior_config, integrated_config=integrated_config,
        state_config=state_config, metric_config=metric_config,
        support_lo=args.support_lo, support_hi=args.support_hi, support_eps=args.support_eps,
        cov_jitter=args.cov_jitter, base_nu=args.base_nu, mix_chunk_size=args.mix_chunk_size,
    ).to(args.device)

    if args.warm_start_path:
        if args.warmstart_include_gates:
            warm_start_with_gates(model, args.warm_start_path, args.device)
        else:
            model.maybe_load_warm_start(args.warm_start_path, args.device)

    # Single-stage fine-tune: everything unfrozen (inherits 183c's Stage-2 training config).
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)
    model.path_transport.requires_grad_(True)
    model.path_context_adapter.requires_grad_(True)
    model.width_allocator.requires_grad_(True)
    model.band_tail.requires_grad_(True)
    model.local_state_gate.requires_grad_(True)
    model.band_state_gate.requires_grad_(True)
    ctrl_params = (
        list(model.width_allocator.parameters())
        + list(model.band_tail.parameters())
        + list(model.local_state_gate.parameters())
        + list(model.band_state_gate.parameters())
        + [model.local_metric_budget_logit, model.band_metric_budget_logit]
    )
    path_params = list(model.path_transport.parameters()) + list(model.path_context_adapter.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": ctrl_params, "lr": args.lr_ctrl},
            {"params": path_params, "lr": args.lr_path},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print(f"\n{'=' * 72}\n241a: 183c + Contrastive Flow Matching (paper-exact)\n{'=' * 72}")
    print(f"  Epochs: {args.epochs} | B={args.batch_size} ga={args.grad_accum} (effective {args.batch_size*args.grad_accum})")
    print(f"  λ_contrastive = {args.lambda_contrastive}")
    print(f"  bf16 autocast: {args.bf16} | Warm start (with gates): {args.warm_start_path}")

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_totals: dict[str, float] = {}
        nb = 0
        optimizer.zero_grad()
        for batch_idx, (history_01, future_01) in enumerate(train_loader):
            history_01 = history_01.to(args.device, non_blocking=True)
            future_01 = future_01.to(args.device, non_blocking=True)
            with autocast_ctx:
                loss, metrics = contrastive_fm_loss(model, history_01, future_01, args.lambda_contrastive)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {batch_idx}")
            (loss / args.grad_accum).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            for key, value in metrics.items():
                v = value.item() if torch.is_tensor(value) else float(value)
                ep_totals[key] = ep_totals.get(key, 0.0) + v
            nb += 1
            global_step += 1

            if global_step % args.log_every_n_steps == 0:
                fm = metrics["flow_match_loss"].item()
                cn = metrics["contrastive_neg"].item()
                ratio = metrics["contrastive_ratio"].item()
                print(
                    f"  [ep {epoch} step {global_step}] fm={fm:.4f} cneg={cn:.4f} "
                    f"ratio={ratio:.3f} total={loss.item():.4f}"
                )

        scheduler.step()
        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in ep_totals.items()}
        val_metrics = evaluate_teacher_forced(model, val_loader)
        joint_metrics = evaluate_joint_subset(
            model, val_loader,
            joint_val_samples=args.joint_val_samples, eval_limit=args.joint_eval_limit,
        )
        frontier_metrics = evaluate_frontier_subset(
            model, val_loader,
            joint_val_samples=args.joint_val_samples, eval_limit=args.joint_eval_limit,
        )
        current_key = checkpoint_key(val_metrics, frontier_metrics, joint_metrics)
        is_best = best_key is None or current_key < best_key
        if is_best:
            best_key = current_key
            best_metrics = {**val_metrics, **joint_metrics, **frontier_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": {
                        "type": "241a_contrastive_fm_state_metric_transport",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "cov_jitter": args.cov_jitter,
                        "base_nu": args.base_nu,
                        "mix_chunk_size": args.mix_chunk_size,
                        "lambda_contrastive": args.lambda_contrastive,
                    },
                    "args": vars(args),
                },
                Path(args.output_dir) / "best_model.pt",
            )

        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "time_sec": epoch_time,
            **train_metrics,
            **val_metrics,
            **joint_metrics,
            **frontier_metrics,
            "is_best": is_best,
        }
        history.append(make_serializable(row))
        history_path.write_text(json.dumps(history, indent=2))

        fm = train_metrics.get("train_flow_match_loss", 0.0)
        cn = train_metrics.get("train_contrastive_neg", 0.0)
        print(
            f"  [ep {epoch}/{args.epochs}] fm={fm:.4f} cneg={cn:.4f} "
            f"val_fm={val_metrics.get('val_flow_match_loss', 0.0):.4f} "
            f"best={is_best} time={epoch_time:.1f}s"
        )

    # Always save final.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "config": {
                "type": "241a_contrastive_fm_state_metric_transport",
                "encoder": vars(encoder_config),
                "decoder": decoder_config, "flow": flow_config, "path": path_config,
                "prior": prior_config, "integrated": integrated_config,
                "state": state_config, "metric": metric_config,
                "support_lo": args.support_lo, "support_hi": args.support_hi,
                "support_eps": args.support_eps, "cov_jitter": args.cov_jitter,
                "base_nu": args.base_nu, "mix_chunk_size": args.mix_chunk_size,
                "lambda_contrastive": args.lambda_contrastive,
            },
            "args": vars(args),
        },
        Path(args.output_dir) / "final_model.pt",
    )
    print(f"\n{'=' * 72}\nDone. Best selection key: {best_key}\n{'=' * 72}")


if __name__ == "__main__":
    main()
