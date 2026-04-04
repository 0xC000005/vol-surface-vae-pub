#!/usr/bin/env python
"""
167b Gradient Health Analysis — Follow-up to 167a gradient investigation.

Verifies that 167b's 4 fixes (CLN frozen, wd=0, FiLM (1+gamma), simple Linear)
restored gradient health to load_head.

Compares to 167a ep20 reference numbers:
  base_head total grad_norm = 0.2676
  load_head total grad_norm = 0.0391
  cond_resid_film total grad_norm = 0.0018
  load_head[-1].weight WD/grad ratio = 27.27 (WD dominated)

Outputs:
  - Gradient norms for all head groups
  - Weight decay counterfactual (what WD=0.01 would have done)
  - FiLM gamma/beta analysis
  - Attribution: which fix contributed most
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys; sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig

# Import model classes from training script
from experiments.backfill.block_ar.train_167b_clean_isolation import (
    ARFactorizedCleanModel,
    FactorizedDecoderClean,
    SpatialTransformerDecoder,
    ConditionalNorm,
    afcrps_per_frame,
    variogram_score_per_frame,
    interval_score,
    reflecting_boundary,
    normalize_iv,
    denormalize_iv,
    compute_cond_ref,
    make_serializable,
)


def compute_grad_stats(name, param):
    """Compute gradient statistics for a single parameter."""
    result = {
        "name": name,
        "param_norm": param.data.norm().item(),
        "param_numel": param.numel(),
        "has_grad": param.grad is not None,
    }
    if param.grad is not None:
        g = param.grad
        result.update({
            "grad_norm": g.norm().item(),
            "grad_mean": g.mean().item(),
            "grad_std": g.std().item() if g.numel() > 1 else float("nan"),
            "grad_abs_mean": g.abs().mean().item(),
            "grad_max": g.abs().max().item(),
            "grad_to_param_ratio": (g.norm() / (param.data.norm() + 1e-12)).item(),
        })
    else:
        result.update({
            "grad_norm": None, "grad_mean": None, "grad_std": None,
            "grad_abs_mean": None, "grad_max": None, "grad_to_param_ratio": None,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_members", type=int, default=16)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--n_factors", type=int, default=5)
    parser.add_argument("--bptt_steps", type=int, default=5)
    args = parser.parse_args()

    device = args.device
    CKPT_PATH = "models/backfill/afcrps_167b/checkpoint_epoch_20.pt"
    OUTPUT_DIR = Path("results/validations/2026-04-04/analysis/167b_followup")
    VERIFY_DIR = Path("results/validations/2026-04-04/verification_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    # Loss config (same as training, epoch 20 = past warmup)
    LAMBDA_VS = 1.0
    LAMBDA_IS = 0.005
    LAMBDA_FLOOR = 2.5
    FLOOR_TAU = 0.005
    ALPHA = 0.95
    SPREAD_WEIGHT = 0.5
    H, T, C = 30, 30, 25

    print("=" * 70)
    print("167b Gradient Health Analysis")
    print("=" * 70)

    # --- Load checkpoint ---
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, weights_only=False, map_location=device)
    config = checkpoint["config"]
    print(f"  Epoch: {checkpoint['epoch']}, Val loss: {checkpoint['val_loss']:.6f}")

    encoder_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0,
    )
    decoder_config = config["decoder"]
    model = ARFactorizedCleanModel(encoder_config, decoder_config, n_factors=args.n_factors).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze CLN (same as training)
    cln_frozen = 0
    for name, param in model.decoder.named_parameters():
        if 'cln' in name or 'ff_cln' in name:
            param.requires_grad = False
            cln_frozen += param.numel()
    print(f"  CLN frozen: {cln_frozen:,} params")

    # --- Load data ---
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    TEST_START = 4511
    max_train_idx = TEST_START - H - T
    VAL_SIZE = 441
    train_indices = np.arange(0, max_train_idx - VAL_SIZE)
    surf_tensor = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    # Build windows
    idx = torch.from_numpy(train_indices).long()
    offsets_h = torch.arange(H, device=device).unsqueeze(0)
    offsets_f = torch.arange(T, device=device).unsqueeze(0)
    hist_idx = idx.to(device).unsqueeze(1) + offsets_h
    fut_idx = idx.to(device).unsqueeze(1) + H + offsets_f
    train_hist = surf_tensor[hist_idx]
    train_future = surf_tensor[fut_idx].reshape(len(train_indices), T, C)
    train_last = surf_tensor[idx.to(device) + H - 1].reshape(len(train_indices), C)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future, train_last),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # Compute cond_ref (same as training)
    cond_ref = compute_cond_ref(model, surf_tensor, train_indices, device, H, C)
    model.decoder.cond_ref.copy_(cond_ref)

    # ====================================================================
    # PART 1: Gradient magnitude analysis (one forward+backward pass)
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 1: Gradient Magnitude Analysis")
    print("=" * 70)

    model.train()
    K = args.n_members

    # Get one batch
    batch_iter = iter(train_loader)
    hist, gt_frames, last_frame = next(batch_iter)
    B = hist.shape[0]

    # Forward pass (full BPTT, same as training)
    hist_norm = normalize_iv(hist)
    hist_flat = hist_norm.reshape(B, H, C)
    gru_outputs, h_last = model.encoder.gru(hist_flat)
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    cond = model.encoder.bottleneck(h_pooled)

    cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    prev = last_frame.unsqueeze(1).expand(B, K, C).reshape(B * K, C)
    h_gru = h_last.detach().unsqueeze(2).expand(1, B, K, -1).reshape(1, B * K, -1).contiguous()
    gru_outs = gru_outputs.detach().unsqueeze(1).expand(B, K, H, -1).reshape(B * K, H, -1)

    model.zero_grad()
    N = args.bptt_steps
    window_loss = 0

    for t in range(T):
        z_t = torch.randn(B * K, args.noise_dim, device=device)
        if t > 0:
            al_t = model.encoder.attn_proj(gru_outs).squeeze(-1)
            aw_t = F.softmax(al_t, dim=1)
            hp_t = (aw_t.unsqueeze(-1) * gru_outs).sum(dim=1)
            cond_K = model.encoder.bottleneck(hp_t)

        delta_base, L = model.decoder(cond_K, prev, z_t)
        eps = torch.randn(B * K, args.n_factors, device=device)
        delta = delta_base + torch.einsum("bcr,br->bc", L, eps)
        frame_t = prev + torch.tanh(delta)

        frame_BK = frame_t.reshape(B, K, C)
        gt_t = gt_frames[:, t, :]
        loss_t, mae_t, spread_t = afcrps_per_frame(frame_BK, gt_t, alpha=ALPHA, spread_weight=SPREAD_WEIGHT)
        is_t = interval_score(frame_BK, gt_t)
        vs_t = variogram_score_per_frame(frame_BK, gt_t)
        floor_barrier = FLOOR_TAU * F.softplus(-frame_t / FLOOR_TAU).mean()

        step_loss = (loss_t + LAMBDA_IS * is_t + LAMBDA_VS * vs_t + LAMBDA_FLOOR * floor_barrier) / T
        window_loss = window_loss + step_loss

        is_window_end = ((t + 1) % N == 0) or (t == T - 1)
        if is_window_end:
            window_loss.backward()
            window_loss = 0
            prev = frame_t.detach()
            h_gru = h_gru.detach()
            gru_outs = gru_outs.detach()
        else:
            prev = frame_t

        frame_norm = normalize_iv(prev.detach()).unsqueeze(1)
        gru_out, h_gru = model.encoder.gru(frame_norm, h_gru)
        gru_outs = torch.cat([gru_outs, gru_out], dim=1)

    # Collect gradients
    gradients = {}

    # base_head
    for suffix in ["weight", "bias"]:
        p = getattr(model.decoder.base_head, suffix)
        gradients[f"base_head.{suffix}"] = compute_grad_stats(f"base_head.{suffix}", p)

    # load_head (167b: single Linear, no hidden layer)
    for suffix in ["weight", "bias"]:
        p = getattr(model.decoder.load_head, suffix)
        gradients[f"load_head.{suffix}"] = compute_grad_stats(f"load_head.{suffix}", p)

    # cond_resid_film (Sequential: [Linear, SiLU, Linear])
    film_layers = [(0, "cond_resid_film[0]"), (2, "cond_resid_film[-1]")]
    for layer_idx, label in film_layers:
        layer = model.decoder.cond_resid_film[layer_idx]
        for suffix in ["weight", "bias"]:
            p = getattr(layer, suffix)
            gradients[f"{label}.{suffix}"] = compute_grad_stats(f"{label}.{suffix}", p)

    # output_proj (should be unused, zero-init)
    for suffix in ["weight", "bias"]:
        p = getattr(model.decoder.output_proj, suffix)
        gradients[f"output_proj.{suffix}"] = compute_grad_stats(f"output_proj.{suffix}", p)

    # Trunk layers (sample: layer 0 and 3 CLN, input_proj, cond_proj, noise_proj)
    for li in [0, 3]:
        layer = model.decoder.layers[li]
        p = layer['cln'].scale_proj.weight
        gradients[f"layers[{li}].cln.scale_proj.weight"] = compute_grad_stats(
            f"layers[{li}].cln.scale_proj.weight", p)
        p = layer['cln'].scale_proj.bias
        gradients[f"layers[{li}].cln.scale_proj.bias"] = compute_grad_stats(
            f"layers[{li}].cln.scale_proj.bias", p)

    gradients["input_proj.weight"] = compute_grad_stats(
        "input_proj.weight", model.decoder.input_proj.weight)
    gradients["cond_proj[0].weight"] = compute_grad_stats(
        "cond_proj[0].weight", model.decoder.cond_proj[0].weight)
    gradients["noise_proj[0].weight"] = compute_grad_stats(
        "noise_proj[0].weight", model.decoder.noise_proj[0].weight)

    # Group norms
    def group_grad_norm(prefix, module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    group_norms = {
        "base_head_total": group_grad_norm("base_head", model.decoder.base_head),
        "load_head_total": group_grad_norm("load_head", model.decoder.load_head),
        "cond_resid_film_total": group_grad_norm("cond_resid_film", model.decoder.cond_resid_film),
    }

    # Trunk groups (CLN frozen, so should be ~0)
    cln_grad = 0.0
    attn_grad = 0.0
    ff_grad = 0.0
    for layer in model.decoder.layers:
        for p in layer['cln'].parameters():
            if p.grad is not None:
                cln_grad += p.grad.norm().item() ** 2
        for p in layer['attn'].parameters():
            if p.grad is not None:
                attn_grad += p.grad.norm().item() ** 2
        for p in layer['ff'].parameters():
            if p.grad is not None:
                ff_grad += p.grad.norm().item() ** 2
        for p in layer['ff_cln'].parameters():
            if p.grad is not None:
                cln_grad += p.grad.norm().item() ** 2
    group_norms["trunk_cln_total"] = cln_grad ** 0.5
    group_norms["trunk_attn_total"] = attn_grad ** 0.5
    group_norms["trunk_ff_total"] = ff_grad ** 0.5

    print("\nGradient norms (group totals):")
    for k, v in group_norms.items():
        print(f"  {k}: {v:.6f}")

    print("\nKey individual gradients:")
    for key in ["base_head.weight", "base_head.bias", "load_head.weight", "load_head.bias",
                 "cond_resid_film[-1].weight", "cond_resid_film[-1].bias"]:
        g = gradients[key]
        print(f"  {key}: grad_norm={g['grad_norm']:.6f}, param_norm={g['param_norm']:.6f}, "
              f"ratio={g['grad_to_param_ratio']:.4f}" if g['grad_norm'] is not None else f"  {key}: NO GRAD")

    # ====================================================================
    # PART 2: Weight decay counterfactual
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 2: Weight Decay Counterfactual (what if wd=0.01 were still active?)")
    print("=" * 70)

    wd_analysis = {}
    # In 167b, load_head is a single Linear (weight + bias)
    for suffix in ["weight", "bias"]:
        p = getattr(model.decoder.load_head, suffix)
        grad_info = gradients[f"load_head.{suffix}"]

        wd_removal = 0.01 * p.data.norm().item()  # what WD would subtract
        grad_addition = grad_info["grad_norm"] if grad_info["grad_norm"] is not None else 0.0

        # Per-element: WD subtracts wd*param from each element, grad adds grad to each element
        # The proper comparison is per-element average
        param_abs_mean = p.data.abs().mean().item()
        wd_per_element = 0.01 * param_abs_mean
        grad_per_element = grad_info["grad_abs_mean"] if grad_info["grad_abs_mean"] is not None else 0.0

        wd_dominates = wd_per_element > grad_per_element
        wd_to_grad_ratio = wd_per_element / (grad_per_element + 1e-15)

        entry = {
            "param_norm": p.data.norm().item(),
            "param_abs_mean": param_abs_mean,
            "grad_norm": grad_addition,
            "grad_abs_mean": grad_per_element,
            "hypothetical_wd_removal_norm": wd_removal,
            "hypothetical_wd_per_element": wd_per_element,
            "wd_would_dominate": wd_dominates,
            "wd_to_grad_ratio": wd_to_grad_ratio,
            "actual_wd": 0.0,
            "fix_effective": not wd_dominates,
        }
        wd_analysis[f"load_head.{suffix}"] = entry
        status = "WD WOULD DOMINATE" if wd_dominates else "GRADIENT WINS"
        print(f"\n  load_head.{suffix}:")
        print(f"    param_norm={p.data.norm().item():.6f}")
        print(f"    grad_norm={grad_addition:.6f}")
        print(f"    hypothetical WD removal (norm): {wd_removal:.6f}")
        print(f"    WD/grad ratio (per-element): {wd_to_grad_ratio:.2f}x")
        print(f"    Status: {status}")

    # Also check cond_resid_film
    for layer_idx, label in [(0, "cond_resid_film[0]"), (2, "cond_resid_film[-1]")]:
        layer = model.decoder.cond_resid_film[layer_idx]
        p = layer.weight
        grad_info = gradients[f"{label}.weight"]
        param_abs_mean = p.data.abs().mean().item()
        wd_per_element = 0.01 * param_abs_mean
        grad_per_element = grad_info["grad_abs_mean"] if grad_info["grad_abs_mean"] is not None else 0.0
        wd_to_grad_ratio = wd_per_element / (grad_per_element + 1e-15)
        wd_analysis[f"{label}.weight"] = {
            "param_norm": p.data.norm().item(),
            "grad_norm": grad_info["grad_norm"],
            "hypothetical_wd_per_element": wd_per_element,
            "grad_abs_mean": grad_per_element,
            "wd_to_grad_ratio": wd_to_grad_ratio,
            "wd_would_dominate": wd_per_element > grad_per_element,
        }
        print(f"\n  {label}.weight: WD/grad={wd_to_grad_ratio:.2f}x")

    # ====================================================================
    # PART 3: FiLM pathway health
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 3: FiLM Pathway Health")
    print("=" * 70)

    model.eval()
    film_analysis = {}

    # Gather gamma/beta statistics from a larger sample
    gammas = []
    betas = []
    with torch.no_grad():
        for i, (h_batch, _, l_batch) in enumerate(train_loader):
            if i >= 10:  # 10 batches ~160 samples
                break
            B_i = h_batch.shape[0]
            h_norm = normalize_iv(h_batch).reshape(B_i, H, C)
            go, _ = model.encoder.gru(h_norm)
            al = model.encoder.attn_proj(go).squeeze(-1)
            aw = F.softmax(al, dim=1)
            hp = (aw.unsqueeze(-1) * go).sum(dim=1)
            cond_i = model.encoder.bottleneck(hp)

            cond_resid = cond_i - model.decoder.cond_ref
            film_params = model.decoder.cond_resid_film(cond_resid)
            gamma, beta = film_params.chunk(2, dim=-1)
            gammas.append(gamma.cpu())
            betas.append(beta.cpu())

    gammas = torch.cat(gammas, dim=0)  # (N, d_model)
    betas = torch.cat(betas, dim=0)

    film_analysis = {
        "gamma_mean": gammas.mean().item(),
        "gamma_std": gammas.std().item(),
        "gamma_abs_mean": gammas.abs().mean().item(),
        "gamma_max": gammas.abs().max().item(),
        "gamma_near_zero_frac": (gammas.abs() < 0.01).float().mean().item(),
        "gamma_per_dim_std": gammas.std(dim=0).mean().item(),
        "beta_mean": betas.mean().item(),
        "beta_std": betas.std().item(),
        "beta_abs_mean": betas.abs().mean().item(),
        "beta_max": betas.abs().max().item(),
        "beta_near_zero_frac": (betas.abs() < 0.01).float().mean().item(),
        "beta_per_dim_std": betas.std(dim=0).mean().item(),
        "effective_scale_mean": (1 + gammas).mean().item(),
        "effective_scale_std": (1 + gammas).std().item(),
        "effective_scale_min": (1 + gammas).min().item(),
        "effective_scale_max": (1 + gammas).max().item(),
        "n_samples": gammas.shape[0],
    }

    # FiLM gradient from Part 1
    film_grad_total = group_norms["cond_resid_film_total"]
    film_analysis["gradient_norm_total"] = film_grad_total
    film_analysis["gradient_is_meaningful"] = film_grad_total > 0.001

    print(f"\n  gamma: mean={film_analysis['gamma_mean']:.6f}, std={film_analysis['gamma_std']:.6f}, "
          f"abs_mean={film_analysis['gamma_abs_mean']:.6f}")
    print(f"  gamma near zero (<0.01): {film_analysis['gamma_near_zero_frac']:.1%}")
    print(f"  effective scale (1+gamma): mean={film_analysis['effective_scale_mean']:.6f}, "
          f"range=[{film_analysis['effective_scale_min']:.4f}, {film_analysis['effective_scale_max']:.4f}]")
    print(f"\n  beta: mean={film_analysis['beta_mean']:.6f}, std={film_analysis['beta_std']:.6f}, "
          f"abs_mean={film_analysis['beta_abs_mean']:.6f}")
    print(f"  beta near zero (<0.01): {film_analysis['beta_near_zero_frac']:.1%}")
    print(f"\n  FiLM gradient total: {film_grad_total:.6f} "
          f"({'MEANINGFUL' if film_analysis['gradient_is_meaningful'] else 'INERT'})")

    # ====================================================================
    # PART 4: Attribution — which fix contributed most?
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 4: Attribution Analysis")
    print("=" * 70)

    # 167a ep20 reference numbers
    REF_167A = {
        "base_head_total": 0.2676,
        "load_head_total": 0.0391,
        "cond_resid_film_total": 0.0018,
        "base_to_load_ratio": 6.84,
        "load_head_weight_grad_norm": 0.00257,  # load_head[-1].weight (output layer)
        "load_head_weight_wd_ratio": 27.27,
        "cln_total": 0.00420,
        "L_norm": 0.0255,  # L_norm at ep20
    }

    # Current 167b numbers
    curr = {
        "base_head_total": group_norms["base_head_total"],
        "load_head_total": group_norms["load_head_total"],
        "cond_resid_film_total": group_norms["cond_resid_film_total"],
        "cln_total": group_norms["trunk_cln_total"],
    }
    curr["base_to_load_ratio"] = curr["base_head_total"] / (curr["load_head_total"] + 1e-12)

    # L_norm from checkpoint diagnostics (replicate from training)
    with torch.no_grad():
        sh = normalize_iv(train_hist[:4]).reshape(4, H, C)
        sg, _ = model.encoder.gru(sh)
        sa = F.softmax(model.encoder.attn_proj(sg).squeeze(-1), dim=1)
        sc = model.encoder.bottleneck((sa.unsqueeze(-1) * sg).sum(dim=1))
        sz = torch.randn(4, args.noise_dim, device=device)
        _, L_diag = model.decoder(sc, train_last[:4], sz)
        curr_L_norm = L_diag.norm().item() / 4
        curr_L_std = L_diag.std().item()
        L_flat = L_diag.reshape(-1, args.n_factors)
        _, sv, _ = torch.linalg.svd(L_flat)
        sv_norm = sv / sv.sum()
        curr_L_eff_rank = torch.exp(-(sv_norm * torch.log(sv_norm + 1e-10)).sum()).item()

    print(f"\n  167a ep20 reference:")
    print(f"    base_head grad: {REF_167A['base_head_total']:.4f}")
    print(f"    load_head grad: {REF_167A['load_head_total']:.4f}")
    print(f"    film grad:      {REF_167A['cond_resid_film_total']:.4f}")
    print(f"    CLN grad:       {REF_167A['cln_total']:.5f}")
    print(f"    base/load ratio: {REF_167A['base_to_load_ratio']:.2f}")
    print(f"    L_norm:         {REF_167A['L_norm']:.4f}")

    print(f"\n  167b ep20 current:")
    print(f"    base_head grad: {curr['base_head_total']:.4f}")
    print(f"    load_head grad: {curr['load_head_total']:.4f}")
    print(f"    film grad:      {curr['cond_resid_film_total']:.4f}")
    print(f"    CLN grad:       {curr['cln_total']:.5f}")
    print(f"    base/load ratio: {curr['base_to_load_ratio']:.2f}")
    print(f"    L_norm:         {curr_L_norm:.4f}")
    print(f"    L_std:          {curr_L_std:.4f}")
    print(f"    L_eff_rank:     {curr_L_eff_rank:.2f}")

    # Ratios
    load_grad_improvement = curr["load_head_total"] / (REF_167A["load_head_total"] + 1e-12)
    base_grad_change = curr["base_head_total"] / (REF_167A["base_head_total"] + 1e-12)
    film_grad_improvement = curr["cond_resid_film_total"] / (REF_167A["cond_resid_film_total"] + 1e-12)
    ratio_improvement = REF_167A["base_to_load_ratio"] / (curr["base_to_load_ratio"] + 1e-12)
    L_norm_improvement = curr_L_norm / (REF_167A["L_norm"] + 1e-12)

    print(f"\n  Changes 167a→167b:")
    print(f"    load_head grad: {load_grad_improvement:.2f}x ({REF_167A['load_head_total']:.4f} → {curr['load_head_total']:.4f})")
    print(f"    base_head grad: {base_grad_change:.2f}x")
    print(f"    film grad:      {film_grad_improvement:.2f}x")
    print(f"    base/load ratio: {REF_167A['base_to_load_ratio']:.2f} → {curr['base_to_load_ratio']:.2f} ({ratio_improvement:.2f}x improvement)")
    print(f"    L_norm:         {REF_167A['L_norm']:.4f} → {curr_L_norm:.4f} ({L_norm_improvement:.2f}x)")

    # Attribution decomposition
    # Fix 1: CLN frozen → gradient that used to flow to CLN is now redistributed
    # Fix 2: wd=0 → gradient no longer fighting weight decay
    # Fix 3: FiLM (1+gamma) → better conditioning path
    # Fix 4: Simple Linear → shorter gradient path

    # CLN gradient in 167a was small (0.0042) relative to load_head (0.039)
    # So CLN competition was NOT the dominant issue
    cln_fraction = REF_167A["cln_total"] / (REF_167A["load_head_total"] + REF_167A["cln_total"])

    # WD counterfactual: compute how much gradient load_head would lose
    load_weight_grad = gradients["load_head.weight"]["grad_abs_mean"] or 0
    load_weight_param = gradients["load_head.weight"]["param_norm"] or 0
    hypothetical_wd_per_elem = 0.01 * (getattr(model.decoder.load_head, "weight").data.abs().mean().item())
    wd_fraction_of_grad = hypothetical_wd_per_elem / (load_weight_grad + 1e-15)

    attribution = {
        "fix1_cln_frozen": {
            "description": "CLN frozen → no competing diversity path",
            "cln_grad_167a": REF_167A["cln_total"],
            "cln_grad_167b": curr["cln_total"],
            "cln_fraction_of_load_167a": cln_fraction,
            "impact": "MINOR" if cln_fraction < 0.2 else "MODERATE",
            "reasoning": f"CLN gradient was only {cln_fraction:.1%} of load_head gradient in 167a"
        },
        "fix2_wd_removed": {
            "description": "wd=0 for load_head → gradient not fighting decay",
            "167a_wd_to_grad_ratio": REF_167A["load_head_weight_wd_ratio"],
            "167b_hypothetical_wd_to_grad_ratio": wd_analysis["load_head.weight"]["wd_to_grad_ratio"],
            "167b_would_wd_dominate": wd_analysis["load_head.weight"]["wd_would_dominate"],
            "impact": "CRITICAL" if wd_analysis["load_head.weight"]["wd_would_dominate"] else "MODERATE",
            "reasoning": (
                f"In 167a, WD was 27.3x the gradient. In 167b (counterfactual), WD would be "
                f"{wd_analysis['load_head.weight']['wd_to_grad_ratio']:.1f}x the gradient. "
                f"{'WD still dominates even with other fixes.' if wd_analysis['load_head.weight']['wd_would_dominate'] else 'Other fixes made gradient large enough to overcome WD.'}"
            ),
        },
        "fix3_film_identity": {
            "description": "FiLM with (1+gamma)*h+beta → identity at init, not suppressive",
            "film_grad_167a": REF_167A["cond_resid_film_total"],
            "film_grad_167b": curr["cond_resid_film_total"],
            "film_grad_improvement": film_grad_improvement,
            "gamma_diverged": film_analysis["gamma_abs_mean"] > 0.01,
            "gamma_abs_mean": film_analysis["gamma_abs_mean"],
            "impact": "MODERATE" if film_grad_improvement > 2.0 else "MINOR",
            "reasoning": (
                f"FiLM gradient went {film_grad_improvement:.2f}x. "
                f"gamma abs mean = {film_analysis['gamma_abs_mean']:.4f} "
                f"({'diverged from identity' if film_analysis['gamma_abs_mean'] > 0.01 else 'near identity'})"
            ),
        },
        "fix4_simple_linear": {
            "description": "Single Linear head → no hidden layer, direct gradient path",
            "167a_had_hidden_layer": True,
            "167a_hidden_grad_norm": 0.000618,  # load_head[0].weight in 167a
            "167a_output_grad_norm": 0.00257,    # load_head[-1].weight in 167a
            "167b_direct_grad_norm": gradients["load_head.weight"]["grad_norm"],
            "gradient_path_improvement": (gradients["load_head.weight"]["grad_norm"] or 0) / 0.00257,
            "impact": "MODERATE",
            "reasoning": (
                f"167a had 2-layer MLP with vanishing grad at hidden layer (0.000618). "
                f"167b single Linear has grad {gradients['load_head.weight']['grad_norm']:.6f}. "
                f"Direct path eliminates bottleneck."
            ),
        },
    }

    # Determine primary driver
    impacts = {k: v["impact"] for k, v in attribution.items()}
    critical_fixes = [k for k, v in impacts.items() if v == "CRITICAL"]
    if not critical_fixes:
        critical_fixes = [k for k, v in impacts.items() if v == "MODERATE"]

    print(f"\n  Attribution:")
    for fix_name, fix_data in attribution.items():
        print(f"\n    {fix_name}: {fix_data['impact']}")
        print(f"      {fix_data['description']}")
        print(f"      {fix_data['reasoning']}")

    # ====================================================================
    # PART 5: L_norm trajectory comparison
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 5: L_norm Health Summary")
    print("=" * 70)

    # Load training history for trajectory
    with open("models/backfill/afcrps_167b/training_history.json") as f:
        training_hist = json.load(f)

    L_norm_trajectory = [(h["epoch"], h["L_norm"]) for h in training_hist]
    print(f"\n  167b L_norm trajectory:")
    for ep, ln in L_norm_trajectory:
        direction = "GROWING" if ep > 1 and ln > L_norm_trajectory[0][1] else "STABLE" if ep == 1 else ""
        print(f"    ep{ep:2d}: L_norm={ln:.4f}  {direction}")

    print(f"\n  167a comparison:")
    print(f"    167a ep1→ep20: L_norm 0.08 → 0.009 (DYING, -89%)")
    print(f"    167b ep1→ep20: L_norm {L_norm_trajectory[0][1]:.4f} → {L_norm_trajectory[-1][1]:.4f} "
          f"({'GROWING' if L_norm_trajectory[-1][1] > L_norm_trajectory[0][1] else 'STABLE/DYING'}, "
          f"{((L_norm_trajectory[-1][1] / L_norm_trajectory[0][1]) - 1) * 100:+.0f}%)")

    # ====================================================================
    # Compile results
    # ====================================================================
    results = {
        "metadata": {
            "script": "167b_gradient_health.py",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": CKPT_PATH,
            "device": device,
            "batch_size": args.batch_size,
            "n_members": K,
            "noise_dim": args.noise_dim,
            "n_factors": args.n_factors,
            "loss_config": {
                "lambda_vs": LAMBDA_VS,
                "lambda_is": LAMBDA_IS,
                "lambda_floor": LAMBDA_FLOOR,
                "floor_tau": FLOOR_TAU,
                "bptt_steps": N,
            },
        },
        "part1_gradients": {
            "individual": gradients,
            "group_norms": group_norms,
        },
        "part2_weight_decay_counterfactual": wd_analysis,
        "part3_film_health": film_analysis,
        "part4_attribution": attribution,
        "part5_L_norm_health": {
            "L_norm_ep20": curr_L_norm,
            "L_std_ep20": curr_L_std,
            "L_eff_rank_ep20": curr_L_eff_rank,
            "L_norm_trajectory": L_norm_trajectory,
            "167a_L_norm_ep20": REF_167A["L_norm"],
            "167b_vs_167a_ratio": L_norm_improvement,
        },
        "comparison_167a_vs_167b": {
            "base_head_grad": {
                "167a_ep20": REF_167A["base_head_total"],
                "167b_ep20": curr["base_head_total"],
                "ratio": base_grad_change,
            },
            "load_head_grad": {
                "167a_ep20": REF_167A["load_head_total"],
                "167b_ep20": curr["load_head_total"],
                "ratio": load_grad_improvement,
            },
            "film_grad": {
                "167a_ep20": REF_167A["cond_resid_film_total"],
                "167b_ep20": curr["cond_resid_film_total"],
                "ratio": film_grad_improvement,
            },
            "base_to_load_ratio": {
                "167a_ep20": REF_167A["base_to_load_ratio"],
                "167b_ep20": curr["base_to_load_ratio"],
                "improvement": ratio_improvement,
            },
            "L_norm": {
                "167a_ep20": REF_167A["L_norm"],
                "167b_ep20": curr_L_norm,
                "ratio": L_norm_improvement,
            },
        },
        "verdict": {
            "load_head_alive": curr_L_norm > REF_167A["L_norm"],
            "load_head_growing": L_norm_trajectory[-1][1] > L_norm_trajectory[0][1],
            "gradient_balanced": curr["base_to_load_ratio"] < 5.0,
            "film_active": film_analysis["gamma_abs_mean"] > 0.001,
            "primary_fix": critical_fixes[0] if critical_fixes else "combined",
            "all_fix_impacts": impacts,
            "summary": "",
        },
    }

    # Generate summary
    summary_lines = []
    if results["verdict"]["load_head_alive"]:
        summary_lines.append(
            f"load_head is ALIVE: L_norm {curr_L_norm:.4f} vs 167a's {REF_167A['L_norm']:.4f} "
            f"({L_norm_improvement:.1f}x)")
    else:
        summary_lines.append(
            f"load_head is WEAKER: L_norm {curr_L_norm:.4f} vs 167a's {REF_167A['L_norm']:.4f}")

    if results["verdict"]["load_head_growing"]:
        summary_lines.append(
            f"L_norm GROWING over training: {L_norm_trajectory[0][1]:.4f} → {L_norm_trajectory[-1][1]:.4f}")
    else:
        summary_lines.append(
            f"L_norm trajectory: {L_norm_trajectory[0][1]:.4f} → {L_norm_trajectory[-1][1]:.4f}")

    summary_lines.append(
        f"Gradient balance: base/load = {curr['base_to_load_ratio']:.2f} "
        f"(167a was {REF_167A['base_to_load_ratio']:.2f})")

    summary_lines.append(
        f"Primary driver: {results['verdict']['primary_fix']}")

    results["verdict"]["summary"] = " | ".join(summary_lines)

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {results['verdict']['summary']}")
    print(f"{'=' * 70}")

    # Save results
    grad_path = OUTPUT_DIR / "gradient_health.json"
    with open(grad_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nSaved: {grad_path}")

    # Save verification copy
    verify_path = VERIFY_DIR / "167b_gradient_health.json"
    verification = {
        "experiment": "167b_clean_isolation",
        "analysis": "gradient_health_followup",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "key_metrics": {
            "load_head_grad_norm_167b": curr["load_head_total"],
            "load_head_grad_norm_167a": REF_167A["load_head_total"],
            "load_head_grad_improvement": load_grad_improvement,
            "base_to_load_ratio_167b": curr["base_to_load_ratio"],
            "base_to_load_ratio_167a": REF_167A["base_to_load_ratio"],
            "L_norm_167b_ep20": curr_L_norm,
            "L_norm_167a_ep20": REF_167A["L_norm"],
            "L_norm_improvement": L_norm_improvement,
            "L_eff_rank": curr_L_eff_rank,
            "film_gamma_abs_mean": film_analysis["gamma_abs_mean"],
            "film_gradient_meaningful": film_analysis["gradient_is_meaningful"],
            "hypothetical_wd_would_dominate": wd_analysis["load_head.weight"]["wd_would_dominate"],
        },
        "verdict": results["verdict"],
    }
    with open(verify_path, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"Saved: {verify_path}")


if __name__ == "__main__":
    main()
