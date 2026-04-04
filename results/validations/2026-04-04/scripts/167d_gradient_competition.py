#!/usr/bin/env python
"""
167d Gradient Competition Analysis: CLN vs L@eps

Measures whether CLN (noise modulation inside attention trunk) and L@eps
(post-attention factor loading) DIRECTLY compete for gradient signal.

Tests:
1. Gradient magnitude ratio per cell: CLN scale_proj vs load_head
2. Ablation gradient test: L detached -> does CLN gradient increase?
3. Reverse ablation: CLN detached -> does L gradient increase?
4. Per-cell analysis for worst S3 cells

Usage:
    PYTHONPATH=. python results/validations/2026-04-04/scripts/167d_gradient_competition.py \
        --model_path models/backfill/afcrps_167d/best_model.pt \
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig

# Import model classes from 167d training script
from experiments.backfill.block_ar.train_167d_e2e_factorized import (
    ARFactorizedCleanModel,
    ConditionalNorm,
    SpatialTransformerDecoder,
    FactorizedDecoderClean,
    normalize_iv,
    denormalize_iv,
    afcrps_per_frame,
    variogram_score_per_frame,
    interval_score,
    reflecting_boundary,
    compute_cond_ref,
    make_serializable,
)


def load_model(model_path, device):
    """Load 167d model from checkpoint."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    encoder_config = EncoderConfig(**cfg["encoder"])
    model = ARFactorizedCleanModel(encoder_config, cfg["decoder"], n_factors=cfg.get("n_factors", 5))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, cfg


def prepare_data(device, batch_size=16, seed=42):
    """Load training data and prepare one batch."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, T, C = 30, 30, 25

    # Use training indices
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
    hist = surf_tensor[hist_idx]
    future = surf_tensor[fut_idx].reshape(len(train_indices), T, C)
    last_frame = surf_tensor[idx.to(device) + H - 1].reshape(len(train_indices), C)

    # Create loader and get first batch
    loader = DataLoader(
        TensorDataset(hist, future, last_frame),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    batch = next(iter(loader))
    return batch, surf_tensor, train_indices


def run_forward_backward_step0(model, batch, K, device, noise_dim,
                                n_factors, alpha=0.95, spread_weight=0.5,
                                lambda_vs=1.0, lambda_is=0.005,
                                lambda_floor=2.5, floor_tau=0.005,
                                detach_L=False, zero_noise_for_cln=False,
                                seed=123):
    """
    Run one forward+backward pass for AR step t=0 only.

    Args:
        detach_L: If True, detach L before combining with delta_base (ablation)
        zero_noise_for_cln: If True, pass zeros as noise to trunk (CLN ablation)
    """
    torch.manual_seed(seed)
    model.zero_grad()
    model.train()

    hist, gt_frames, last_frame = batch
    B = hist.shape[0]
    H, C = 30, 25

    # Encoder
    hist_norm = normalize_iv(hist)
    hist_flat = hist_norm.reshape(B, H, C)
    gru_outputs, h_last = model.encoder.gru(hist_flat)
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    cond = model.encoder.bottleneck(h_pooled)

    # Expand for K members
    cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    prev = last_frame.unsqueeze(1).expand(B, K, C).reshape(B * K, C)

    # Step t=0
    z_t = torch.randn(B * K, noise_dim, device=device)

    if zero_noise_for_cln:
        # For CLN ablation: pass zero noise so CLN has no diversifying effect
        # CLN with zero noise: scale_proj(0)=bias, bias_proj(0)=bias
        # But since CLN is zero-init, at init scale=0, bias=0 -> identity
        # After training, bias terms may be nonzero but noise-dependent modulation is killed
        z_trunk = torch.zeros_like(z_t)
    else:
        z_trunk = z_t

    # Forward through decoder trunk manually to intercept
    h = model.decoder.input_proj(prev.unsqueeze(-1))
    h = h + model.decoder.cond_proj(cond_K).unsqueeze(1)
    h = h + model.decoder.spatial_pos
    z_proj = model.decoder.noise_proj(z_trunk)

    for li, layer in enumerate(model.decoder.layers):
        ls_a = model.decoder.ls_params[li * 2]
        ls_f = model.decoder.ls_params[li * 2 + 1]
        h_norm = layer['cln'](h, z_proj)
        attn_out, _ = layer['attn'](h_norm, h_norm, h_norm)
        h = h + ls_a * attn_out
        h = h + ls_f * layer['ff'](layer['ff_cln'](h, z_proj))

    # Base innovation
    delta_base = model.decoder.base_head(h).squeeze(-1)  # (B*K, 25)

    # FiLM + load_head
    cond_resid = cond_K - model.decoder.cond_ref
    film_params = model.decoder.cond_resid_film(cond_resid)
    gamma, beta = film_params.chunk(2, dim=-1)
    h_modulated = (1 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
    L = model.decoder.load_head(h_modulated)  # (B*K, 25, n_factors)

    if detach_L:
        L_use = L.detach()
    else:
        L_use = L

    eps = torch.randn(B * K, n_factors, device=device)
    delta = delta_base + torch.einsum("bcr,br->bc", L_use, eps)
    frame_t = prev + torch.tanh(delta)

    frame_BK = frame_t.reshape(B, K, C)
    gt_t = gt_frames[:, 0, :]

    # Loss computation matching training
    loss_t, mae_t, spread_t = afcrps_per_frame(frame_BK, gt_t, alpha=alpha, spread_weight=spread_weight)
    is_t = interval_score(frame_BK, gt_t)
    vs_t = variogram_score_per_frame(frame_BK, gt_t)
    floor_barrier = floor_tau * F.softplus(-frame_t / floor_tau).mean()

    total_loss = loss_t + lambda_is * is_t + lambda_vs * vs_t + lambda_floor * floor_barrier
    total_loss.backward()

    return {
        "loss": total_loss.item(),
        "mae": mae_t.item(),
        "spread": spread_t.item(),
        "vs": vs_t.item(),
        "is": is_t.item(),
        "floor": floor_barrier.item(),
    }


def extract_gradients(model, device):
    """Extract per-component gradient vectors."""
    results = {}

    # CLN gradients: all parameters named 'cln' or 'ff_cln'
    cln_grads = []
    cln_param_names = []
    for name, param in model.decoder.named_parameters():
        if ('cln' in name or 'ff_cln' in name) and 'load_head' not in name and 'cond_resid_film' not in name:
            if param.grad is not None:
                cln_grads.append(param.grad.detach().flatten())
                cln_param_names.append(name)
    results["cln_grad"] = torch.cat(cln_grads) if cln_grads else torch.zeros(1, device=device)
    results["cln_param_names"] = cln_param_names

    # load_head gradients
    load_grads = []
    load_param_names = []
    for name, param in model.decoder.load_head.named_parameters():
        if param.grad is not None:
            load_grads.append(param.grad.detach().flatten())
            load_param_names.append(f"load_head.{name}")
    results["load_grad"] = torch.cat(load_grads) if load_grads else torch.zeros(1, device=device)
    results["load_param_names"] = load_param_names

    # base_head gradients
    base_grads = []
    for name, param in model.decoder.base_head.named_parameters():
        if param.grad is not None:
            base_grads.append(param.grad.detach().flatten())
    results["base_grad"] = torch.cat(base_grads) if base_grads else torch.zeros(1, device=device)

    # FiLM gradients
    film_grads = []
    for name, param in model.decoder.cond_resid_film.named_parameters():
        if param.grad is not None:
            film_grads.append(param.grad.detach().flatten())
    results["film_grad"] = torch.cat(film_grads) if film_grads else torch.zeros(1, device=device)

    return results


def per_cell_gradient_magnitudes(model, device):
    """
    Extract per-cell gradient magnitudes for CLN scale_proj and load_head.

    CLN scale_proj: Linear(noise_dim=32, n_cells*d_model=25*128=3200)
        Weight shape: (3200, 32) -> reshaped to (25, 128, 32)
        For cell c: weight[c*128:(c+1)*128, :] and bias[c*128:(c+1)*128]

    load_head: Linear(d_model=128, n_factors=5)
        Weight shape: (5, 128), bias shape: (5,)
        This is shared across cells -- the per-cell output depends on the input h_modulated
        which varies per cell. But the parameter gradients aggregate across cells.
        To get per-cell gradient contribution, we need to check the Jacobian.
    """
    results = {}
    n_cells = 25
    d_model = 128

    # CLN scale_proj per cell (layers 0-3, both cln and ff_cln)
    cln_per_cell = torch.zeros(n_cells, device=device)
    for li, layer in enumerate(model.decoder.layers):
        for cln_key in ['cln', 'ff_cln']:
            cln = layer[cln_key]
            # scale_proj: weight (n_cells*d_model, noise_dim), bias (n_cells*d_model)
            if cln.scale_proj.weight.grad is not None:
                w_grad = cln.scale_proj.weight.grad  # (3200, 32)
                b_grad = cln.scale_proj.bias.grad    # (3200,)
                for c in range(n_cells):
                    start = c * d_model
                    end = (c + 1) * d_model
                    cell_w_grad = w_grad[start:end, :].norm()
                    cell_b_grad = b_grad[start:end].norm()
                    cln_per_cell[c] += (cell_w_grad + cell_b_grad).item()
            # Also add bias_proj contribution
            if cln.bias_proj.weight.grad is not None:
                w_grad = cln.bias_proj.weight.grad
                b_grad = cln.bias_proj.bias.grad
                for c in range(n_cells):
                    start = c * d_model
                    end = (c + 1) * d_model
                    cell_w_grad = w_grad[start:end, :].norm()
                    cell_b_grad = b_grad[start:end].norm()
                    cln_per_cell[c] += (cell_w_grad + cell_b_grad).item()

    results["cln_per_cell"] = cln_per_cell.cpu().numpy()

    # load_head is shared across cells, so we measure using output_proj approach
    # load_head.weight: (n_factors, d_model), load_head.bias: (n_factors,)
    # The gradient aggregates across all cells. To get per-cell contribution
    # we'd need hooks. Instead, report the total.
    if model.decoder.load_head.weight.grad is not None:
        lh_w_norm = model.decoder.load_head.weight.grad.norm().item()
        lh_b_norm = model.decoder.load_head.bias.grad.norm().item()
        results["load_head_total"] = lh_w_norm + lh_b_norm
    else:
        results["load_head_total"] = 0.0

    return results


def per_cell_load_head_gradient_via_hooks(model, batch, K, device, noise_dim, n_factors, seed=123):
    """
    Use forward hooks to measure per-cell gradient contribution to load_head.

    Register a hook on h_modulated to capture the per-cell hidden states,
    then compute per-cell gradient magnitude based on the chain rule.
    """
    torch.manual_seed(seed)
    model.zero_grad()
    model.train()

    hist, gt_frames, last_frame = batch
    B = hist.shape[0]
    H, C = 30, 25

    # Encoder
    hist_norm = normalize_iv(hist)
    hist_flat = hist_norm.reshape(B, H, C)
    gru_outputs, h_last = model.encoder.gru(hist_flat)
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    cond = model.encoder.bottleneck(h_pooled)

    cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    prev = last_frame.unsqueeze(1).expand(B, K, C).reshape(B * K, C)

    z_t = torch.randn(B * K, noise_dim, device=device)

    # Full forward through decoder
    delta_base, L = model.decoder(cond_K, prev, z_t)
    eps = torch.randn(B * K, n_factors, device=device)
    delta = delta_base + torch.einsum("bcr,br->bc", L, eps)
    frame_t = prev + torch.tanh(delta)

    frame_BK = frame_t.reshape(B, K, C)
    gt_t = gt_frames[:, 0, :]

    loss_t, _, _ = afcrps_per_frame(frame_BK, gt_t, alpha=0.95, spread_weight=0.5)
    is_t = interval_score(frame_BK, gt_t)
    vs_t = variogram_score_per_frame(frame_BK, gt_t)
    floor_barrier = 0.005 * F.softplus(-frame_t / 0.005).mean()
    total_loss = loss_t + 0.005 * is_t + 1.0 * vs_t + 2.5 * floor_barrier

    # Per-cell: compute gradient of loss w.r.t. L for each cell
    # L shape: (B*K, 25, n_factors)
    # We want dLoss/dL[b,c,:] for each cell c
    L.retain_grad()
    total_loss.backward()

    per_cell_L_grad = torch.zeros(C, device=device)
    if L.grad is not None:
        # L.grad shape: (B*K, 25, n_factors)
        # Per-cell: average gradient magnitude across batch and factors
        for c in range(C):
            per_cell_L_grad[c] = L.grad[:, c, :].norm(dim=-1).mean().item()

    return per_cell_L_grad.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/backfill/afcrps_167d/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    t0 = time.time()

    print("=" * 70)
    print("167d GRADIENT COMPETITION ANALYSIS: CLN vs L@eps")
    print("=" * 70)

    # 1. Load model
    print("\n[1] Loading model...")
    model, cfg = load_model(args.model_path, device)
    noise_dim = cfg["decoder"]["noise_dim"]
    n_factors = cfg.get("n_factors", 5)
    K = args.K

    n_cln = sum(p.numel() for name, p in model.decoder.named_parameters()
                if 'cln' in name and 'load_head' not in name and 'cond_resid_film' not in name)
    n_load = sum(p.numel() for p in model.decoder.load_head.parameters())
    n_base = sum(p.numel() for p in model.decoder.base_head.parameters())
    n_film = sum(p.numel() for p in model.decoder.cond_resid_film.parameters())
    print(f"  CLN params: {n_cln:,}")
    print(f"  load_head params: {n_load:,}")
    print(f"  base_head params: {n_base:,}")
    print(f"  FiLM params: {n_film:,}")

    # 2. Prepare data
    print("\n[2] Preparing data...")
    batch, surf_tensor, train_indices = prepare_data(device, args.batch_size, args.seed)

    # Update cond_ref
    cond_ref = compute_cond_ref(model, surf_tensor, train_indices, device, 30, 25)
    model.decoder.cond_ref.copy_(cond_ref)

    results = OrderedDict()

    # ============================================================
    # TEST A: Normal forward+backward (baseline)
    # ============================================================
    print("\n[3] TEST A: Normal forward+backward (baseline)...")
    loss_info_normal = run_forward_backward_step0(
        model, batch, K, device, noise_dim, n_factors,
        detach_L=False, zero_noise_for_cln=False, seed=123
    )
    grads_normal = extract_gradients(model, device)
    per_cell_normal = per_cell_gradient_magnitudes(model, device)
    per_cell_L_normal = per_cell_load_head_gradient_via_hooks(
        model, batch, K, device, noise_dim, n_factors, seed=123
    )

    print(f"  Loss: {loss_info_normal['loss']:.6f}")
    print(f"  CLN grad norm: {grads_normal['cln_grad'].norm():.6f}")
    print(f"  load_head grad norm: {grads_normal['load_grad'].norm():.6f}")
    print(f"  base_head grad norm: {grads_normal['base_grad'].norm():.6f}")
    print(f"  FiLM grad norm: {grads_normal['film_grad'].norm():.6f}")

    results["normal"] = {
        "loss_info": loss_info_normal,
        "cln_grad_norm": grads_normal["cln_grad"].norm().item(),
        "load_grad_norm": grads_normal["load_grad"].norm().item(),
        "base_grad_norm": grads_normal["base_grad"].norm().item(),
        "film_grad_norm": grads_normal["film_grad"].norm().item(),
        "cln_per_cell": per_cell_normal["cln_per_cell"].tolist(),
        "load_head_total": per_cell_normal["load_head_total"],
        "L_per_cell_grad": per_cell_L_normal.tolist(),
    }

    # ============================================================
    # TEST B: L detached (ablation - does CLN gradient increase?)
    # ============================================================
    print("\n[4] TEST B: L detached ablation...")
    loss_info_L_detach = run_forward_backward_step0(
        model, batch, K, device, noise_dim, n_factors,
        detach_L=True, zero_noise_for_cln=False, seed=123
    )
    grads_L_detach = extract_gradients(model, device)
    per_cell_L_detach = per_cell_gradient_magnitudes(model, device)

    cln_normal = grads_normal["cln_grad"].norm().item()
    cln_L_detach = grads_L_detach["cln_grad"].norm().item()
    cln_ratio = cln_L_detach / max(cln_normal, 1e-12)

    print(f"  Loss: {loss_info_L_detach['loss']:.6f}")
    print(f"  CLN grad norm (L detached): {cln_L_detach:.6f}")
    print(f"  CLN grad norm (normal):     {cln_normal:.6f}")
    print(f"  Ratio (detached/normal):    {cln_ratio:.4f}")
    if cln_ratio > 1.05:
        print(f"  >>> COMPETITION DETECTED: CLN grad increases {(cln_ratio-1)*100:.1f}% when L detached")
    elif cln_ratio < 0.95:
        print(f"  >>> COOPERATION DETECTED: CLN grad decreases {(1-cln_ratio)*100:.1f}% when L detached")
    else:
        print(f"  >>> INDEPENDENT: CLN grad barely changes (within 5%)")

    # Per-cell: does CLN gradient change per cell when L is detached?
    cln_per_cell_ratio = np.array(per_cell_L_detach["cln_per_cell"]) / np.maximum(
        np.array(per_cell_normal["cln_per_cell"]), 1e-12
    )

    results["L_detached"] = {
        "loss_info": loss_info_L_detach,
        "cln_grad_norm": cln_L_detach,
        "cln_ratio_vs_normal": cln_ratio,
        "competition_detected": bool(cln_ratio > 1.05),
        "base_grad_norm": grads_L_detach["base_grad"].norm().item(),
        "cln_per_cell": per_cell_L_detach["cln_per_cell"].tolist(),
        "cln_per_cell_ratio": cln_per_cell_ratio.tolist(),
    }

    # ============================================================
    # TEST C: CLN noise zeroed (reverse ablation - does L gradient increase?)
    # ============================================================
    print("\n[5] TEST C: CLN noise zeroed (reverse ablation)...")
    loss_info_cln_zero = run_forward_backward_step0(
        model, batch, K, device, noise_dim, n_factors,
        detach_L=False, zero_noise_for_cln=True, seed=123
    )
    grads_cln_zero = extract_gradients(model, device)

    load_normal = grads_normal["load_grad"].norm().item()
    load_cln_zero = grads_cln_zero["load_grad"].norm().item()
    load_ratio = load_cln_zero / max(load_normal, 1e-12)

    # Per-cell L gradient when CLN is zeroed
    per_cell_L_cln_zero = per_cell_load_head_gradient_via_hooks(
        model, batch, K, device, noise_dim, n_factors, seed=123
    )
    # Ah wait -- we need to actually pass zero noise. Let's do a manual version.
    # The hook-based approach calls model.decoder() which uses full noise.
    # We need a modified approach. Let's compute per-cell L grad ratio from
    # the parameter gradient ratio instead.

    print(f"  Loss: {loss_info_cln_zero['loss']:.6f}")
    print(f"  load_head grad norm (CLN zeroed): {load_cln_zero:.6f}")
    print(f"  load_head grad norm (normal):     {load_normal:.6f}")
    print(f"  Ratio (zeroed/normal):            {load_ratio:.4f}")
    if load_ratio > 1.05:
        print(f"  >>> COMPETITION DETECTED: L grad increases {(load_ratio-1)*100:.1f}% when CLN zeroed")
    elif load_ratio < 0.95:
        print(f"  >>> COOPERATION DETECTED: L grad decreases {(1-load_ratio)*100:.1f}% when CLN zeroed")
    else:
        print(f"  >>> INDEPENDENT: L grad barely changes (within 5%)")

    results["CLN_zeroed"] = {
        "loss_info": loss_info_cln_zero,
        "load_grad_norm": load_cln_zero,
        "load_ratio_vs_normal": load_ratio,
        "competition_detected": bool(load_ratio > 1.05),
        "base_grad_norm": grads_cln_zero["base_grad"].norm().item(),
        "cln_grad_norm": grads_cln_zero["cln_grad"].norm().item(),
    }

    # ============================================================
    # TEST D: Per-cell analysis for worst S3 cells
    # ============================================================
    print("\n[6] TEST D: Per-cell analysis...")
    cln_cells = np.array(per_cell_normal["cln_per_cell"])
    L_cells = per_cell_L_normal

    # Cell grid: 5x5, cell (r,c) = r*5 + c
    # Worst S3 cells: (1,4) = cell 9, (1,0) = cell 5
    worst_cells = {"(1,4)": 9, "(1,0)": 5}
    center_cells = {"(2,2)": 12, "(2,1)": 11, "(1,2)": 7}

    print(f"\n  {'Cell':<10} {'CLN grad':>12} {'L grad':>12} {'CLN/L ratio':>14} {'Dominance':>12}")
    print(f"  {'-'*60}")

    cell_analysis = {}
    all_cells = {**worst_cells, **center_cells}
    for label, idx in all_cells.items():
        cln_g = cln_cells[idx]
        l_g = L_cells[idx]
        ratio = cln_g / max(l_g, 1e-12)
        dom = "CLN" if ratio > 2.0 else ("L" if ratio < 0.5 else "balanced")
        print(f"  {label:<10} {cln_g:>12.6f} {l_g:>12.6f} {ratio:>14.4f} {dom:>12}")
        cell_analysis[label] = {
            "cell_idx": idx,
            "cln_grad": float(cln_g),
            "L_grad": float(l_g),
            "cln_over_L_ratio": float(ratio),
            "dominance": dom,
        }

    # Full grid
    print(f"\n  Full grid CLN/L ratio (5x5, moneyness x tenor):")
    cln_over_L = cln_cells / np.maximum(L_cells, 1e-12)
    for r in range(5):
        row = "  "
        for c in range(5):
            idx = r * 5 + c
            row += f"{cln_over_L[idx]:8.2f}"
        print(row)

    # CLN per-cell change when L detached
    print(f"\n  CLN grad ratio (L_detached / normal) per cell (5x5):")
    for r in range(5):
        row = "  "
        for c in range(5):
            idx = r * 5 + c
            row += f"{cln_per_cell_ratio[idx]:8.4f}"
        print(row)

    results["per_cell"] = {
        "worst_s3_cells": cell_analysis,
        "cln_per_cell": cln_cells.tolist(),
        "L_per_cell": L_cells.tolist(),
        "cln_over_L_ratio_grid": cln_over_L.tolist(),
        "cln_ratio_when_L_detached_grid": cln_per_cell_ratio.tolist(),
    }

    # ============================================================
    # TEST E: Gradient cosine similarity between components
    # ============================================================
    print("\n[7] TEST E: Gradient cosine similarity between components...")
    # Use the normal gradients
    # CLN and load_head have different dimensionality, so cosine doesn't apply directly.
    # Instead measure: correlation of per-cell gradient magnitudes
    cln_mag = cln_cells
    L_mag = L_cells
    correlation = np.corrcoef(cln_mag, L_mag)[0, 1]
    print(f"  Correlation(CLN_per_cell, L_per_cell): {correlation:.4f}")
    print(f"  CLN gradient std across cells: {cln_mag.std():.6f}")
    print(f"  L gradient std across cells:   {L_mag.std():.6f}")

    # Edge vs center analysis
    edge_cells = [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]
    center_cell_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]

    cln_edge_mean = cln_mag[edge_cells].mean()
    cln_center_mean = cln_mag[center_cell_indices].mean()
    L_edge_mean = L_mag[edge_cells].mean()
    L_center_mean = L_mag[center_cell_indices].mean()

    print(f"\n  CLN gradient: edge={cln_edge_mean:.6f}, center={cln_center_mean:.6f}, ratio={cln_edge_mean/max(cln_center_mean, 1e-12):.4f}")
    print(f"  L gradient:   edge={L_edge_mean:.6f}, center={L_center_mean:.6f}, ratio={L_edge_mean/max(L_center_mean, 1e-12):.4f}")

    results["gradient_correlation"] = {
        "cln_L_per_cell_correlation": float(correlation),
        "cln_edge_mean": float(cln_edge_mean),
        "cln_center_mean": float(cln_center_mean),
        "cln_edge_over_center": float(cln_edge_mean / max(cln_center_mean, 1e-12)),
        "L_edge_mean": float(L_edge_mean),
        "L_center_mean": float(L_center_mean),
        "L_edge_over_center": float(L_edge_mean / max(L_center_mean, 1e-12)),
    }

    # ============================================================
    # TEST F: Multi-batch stability (5 different batches)
    # ============================================================
    print("\n[8] TEST F: Multi-batch stability (5 batches)...")
    cln_norms_across = []
    load_norms_across = []
    cln_L_detach_ratios = []
    load_cln_zero_ratios = []

    for b_seed in range(5):
        batch_b, _, _ = prepare_data(device, args.batch_size, seed=42 + b_seed)
        cond_ref_b = compute_cond_ref(model, surf_tensor, train_indices, device, 30, 25)
        model.decoder.cond_ref.copy_(cond_ref_b)

        # Normal
        _ = run_forward_backward_step0(
            model, batch_b, K, device, noise_dim, n_factors,
            detach_L=False, zero_noise_for_cln=False, seed=123 + b_seed
        )
        g_n = extract_gradients(model, device)
        cln_n = g_n["cln_grad"].norm().item()
        load_n = g_n["load_grad"].norm().item()

        # L detached
        _ = run_forward_backward_step0(
            model, batch_b, K, device, noise_dim, n_factors,
            detach_L=True, zero_noise_for_cln=False, seed=123 + b_seed
        )
        g_ld = extract_gradients(model, device)
        cln_ld = g_ld["cln_grad"].norm().item()

        # CLN zeroed
        _ = run_forward_backward_step0(
            model, batch_b, K, device, noise_dim, n_factors,
            detach_L=False, zero_noise_for_cln=True, seed=123 + b_seed
        )
        g_cz = extract_gradients(model, device)
        load_cz = g_cz["load_grad"].norm().item()

        cln_norms_across.append(cln_n)
        load_norms_across.append(load_n)
        cln_L_detach_ratios.append(cln_ld / max(cln_n, 1e-12))
        load_cln_zero_ratios.append(load_cz / max(load_n, 1e-12))

    print(f"  CLN grad norms across 5 batches: mean={np.mean(cln_norms_across):.6f}, std={np.std(cln_norms_across):.6f}")
    print(f"  load_head grad norms:            mean={np.mean(load_norms_across):.6f}, std={np.std(load_norms_across):.6f}")
    print(f"  CLN ratio (L detach/normal):     mean={np.mean(cln_L_detach_ratios):.4f}, std={np.std(cln_L_detach_ratios):.4f}")
    print(f"  L ratio (CLN zero/normal):       mean={np.mean(load_cln_zero_ratios):.4f}, std={np.std(load_cln_zero_ratios):.4f}")

    results["multi_batch"] = {
        "cln_norms": cln_norms_across,
        "load_norms": load_norms_across,
        "cln_L_detach_ratios": cln_L_detach_ratios,
        "load_cln_zero_ratios": load_cln_zero_ratios,
        "cln_L_detach_ratio_mean": float(np.mean(cln_L_detach_ratios)),
        "cln_L_detach_ratio_std": float(np.std(cln_L_detach_ratios)),
        "load_cln_zero_ratio_mean": float(np.mean(load_cln_zero_ratios)),
        "load_cln_zero_ratio_std": float(np.std(load_cln_zero_ratios)),
    }

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Gradient Competition Analysis")
    print("=" * 70)

    # Competition verdict
    cln_competition = cln_ratio > 1.05
    L_competition = load_ratio > 1.05
    both_compete = cln_competition and L_competition

    print(f"\n  [A] CLN grad increases when L detached: {cln_competition} (ratio={cln_ratio:.4f})")
    print(f"  [B] L grad increases when CLN zeroed:   {L_competition} (ratio={load_ratio:.4f})")
    print(f"  [C] Bidirectional competition:          {both_compete}")
    print(f"  [D] Per-cell CLN/L correlation:         {correlation:.4f}")

    # Multi-batch stability
    stable_cln = np.std(cln_L_detach_ratios) < 0.1
    stable_L = np.std(load_cln_zero_ratios) < 0.1
    print(f"  [E] Multi-batch stable (CLN):           {stable_cln} (std={np.std(cln_L_detach_ratios):.4f})")
    print(f"  [F] Multi-batch stable (L):             {stable_L} (std={np.std(load_cln_zero_ratios):.4f})")

    verdict = "UNKNOWN"
    if both_compete:
        verdict = "BIDIRECTIONAL_COMPETITION"
    elif cln_competition:
        verdict = "CLN_SUPPRESSED_BY_L"
    elif L_competition:
        verdict = "L_SUPPRESSED_BY_CLN"
    elif cln_ratio < 0.95 and load_ratio < 0.95:
        verdict = "COOPERATION"
    else:
        verdict = "INDEPENDENT"

    print(f"\n  VERDICT: {verdict}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    results["summary"] = {
        "verdict": verdict,
        "cln_competition": bool(cln_competition),
        "L_competition": bool(L_competition),
        "cln_ratio_L_detached": float(cln_ratio),
        "L_ratio_CLN_zeroed": float(load_ratio),
        "per_cell_correlation": float(correlation),
        "multi_batch_stable": bool(stable_cln and stable_L),
        "cln_L_detach_ratio_mean": float(np.mean(cln_L_detach_ratios)),
        "load_cln_zero_ratio_mean": float(np.mean(load_cln_zero_ratios)),
        "elapsed_seconds": time.time() - t0,
    }

    # Save results
    out_dir = Path("results/validations/2026-04-04/analysis/167d_followup")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradient_competition.json"
    with open(out_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Verification results
    verify_dir = Path("results/validations/2026-04-04/verification_results")
    verify_dir.mkdir(parents=True, exist_ok=True)
    verify_path = verify_dir / "167d_gradient_competition.json"
    verify_data = {
        "test": "167d_gradient_competition",
        "model": args.model_path,
        "verdict": verdict,
        "cln_competition": bool(cln_competition),
        "L_competition": bool(L_competition),
        "cln_ratio_L_detached": float(cln_ratio),
        "L_ratio_CLN_zeroed": float(load_ratio),
        "per_cell_correlation": float(correlation),
        "multi_batch_cln_ratio_mean": float(np.mean(cln_L_detach_ratios)),
        "multi_batch_load_ratio_mean": float(np.mean(load_cln_zero_ratios)),
        "multi_batch_cln_ratio_std": float(np.std(cln_L_detach_ratios)),
        "multi_batch_load_ratio_std": float(np.std(load_cln_zero_ratios)),
        "worst_s3_cell_1_4": cell_analysis.get("(1,4)", {}),
        "worst_s3_cell_1_0": cell_analysis.get("(1,0)", {}),
        "cln_edge_over_center": float(cln_edge_mean / max(cln_center_mean, 1e-12)),
        "L_edge_over_center": float(L_edge_mean / max(L_center_mean, 1e-12)),
    }
    with open(verify_path, "w") as f:
        json.dump(make_serializable(verify_data), f, indent=2)
    print(f"  Verification saved to: {verify_path}")


if __name__ == "__main__":
    main()
