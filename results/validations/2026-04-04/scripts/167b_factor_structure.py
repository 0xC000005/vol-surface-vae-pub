#!/usr/bin/env python
"""
167b Factor Structure Analysis: Compare model vs GT eigenvalue spectra,
L matrix structure, per-cell spread, and condition sensitivity.

Uses FINAL model checkpoint (final_model.pt).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig

# --- Import model classes from 167b training script ---
# We inline the necessary classes to avoid import issues

def normalize_iv(surfaces):
    return surfaces * 2.0 - 1.0

def denormalize_iv(surfaces):
    return (surfaces + 1.0) / 2.0

def reflecting_boundary(x, lo=0.01, hi=1.0):
    below = x < lo
    above = x > hi
    x = torch.where(below, 2 * lo - x, x)
    x = torch.where(above, 2 * hi - x, x)
    return x.clamp(lo, hi)


class ConditionalNorm(nn.Module):
    def __init__(self, d_model, noise_dim, n_cells=25):
        super().__init__()
        self.n_cells = n_cells
        self.d_model = d_model
        self.scale_proj = nn.Linear(noise_dim, n_cells * d_model)
        self.bias_proj = nn.Linear(noise_dim, n_cells * d_model)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, x, z):
        B = z.shape[0]
        scale = self.scale_proj(z).view(B, self.n_cells, self.d_model)
        bias = self.bias_proj(z).view(B, self.n_cells, self.d_model)
        return (scale + 1.0) * x + bias


class SpatialTransformerDecoder(nn.Module):
    def __init__(self, n_cells=25, d_model=128, n_heads=4, n_layers=4,
                 cond_dim=128, noise_dim=32):
        super().__init__()
        self.n_cells = n_cells
        self.d_model = d_model
        self.noise_dim = noise_dim
        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.SiLU(), nn.Linear(d_model, noise_dim),
        )
        self.layers = nn.ModuleList()
        self.ls_params = nn.ParameterList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'cln': ConditionalNorm(d_model, noise_dim, n_cells),
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'ff_cln': ConditionalNorm(d_model, noise_dim, n_cells),
            }))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
        self.output_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward_trunk(self, cond, prev_frame, noise):
        h = self.input_proj(prev_frame.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos
        z = self.noise_proj(noise)
        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[li * 2]
            ls_f = self.ls_params[li * 2 + 1]
            h_norm = layer['cln'](h, z)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out
            h = h + ls_f * layer['ff'](layer['ff_cln'](h, z))
        return h


class FactorizedDecoderClean(SpatialTransformerDecoder):
    def __init__(self, n_cells=25, d_model=128, n_heads=4, n_layers=4,
                 cond_dim=128, noise_dim=32, n_factors=5):
        super().__init__(n_cells=n_cells, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, cond_dim=cond_dim, noise_dim=noise_dim)
        self.n_factors = n_factors
        self.base_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)
        self.load_head = nn.Linear(d_model, n_factors)
        nn.init.normal_(self.load_head.weight, std=0.01)
        nn.init.zeros_(self.load_head.bias)
        self.cond_resid_film = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        nn.init.zeros_(self.cond_resid_film[-1].weight)
        nn.init.zeros_(self.cond_resid_film[-1].bias)
        self.register_buffer('cond_ref', torch.zeros(cond_dim))

    def forward(self, cond, prev_frame, noise):
        h = self.forward_trunk(cond, prev_frame, noise)
        delta_base = self.base_head(h).squeeze(-1)
        cond_resid = cond - self.cond_ref
        film_params = self.cond_resid_film(cond_resid)
        gamma, beta = film_params.chunk(2, dim=-1)
        h_modulated = (1 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
        L = self.load_head(h_modulated)
        return delta_base, L


class ARFactorizedCleanModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, n_factors=5):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = FactorizedDecoderClean(**decoder_config, n_factors=n_factors)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_factors = n_factors

    def sample_batched(self, history, n_samples=50, return_L=False):
        """Generate samples. If return_L=True, also return L matrices per step."""
        B = history.shape[0]
        T = 30
        device = history.device
        CHUNK = 10

        with torch.no_grad():
            last_frame = denormalize_iv(history[:, -1]).reshape(B, 25)
            hist_flat = history.reshape(B, history.shape[1], -1)
            gru_outputs_base, h_last_base = self.encoder.gru(hist_flat)

            all_samples = []
            all_L = [] if return_L else None

            for start in range(0, n_samples, CHUNK):
                k = min(CHUNK, n_samples - start)
                last_k = last_frame.unsqueeze(1).expand(B, k, -1).reshape(B * k, 25)
                gru_out_k = gru_outputs_base.unsqueeze(1).expand(
                    B, k, -1, -1).reshape(B * k, -1, self.encoder_config.gru_hidden_dim)
                h_last_k = h_last_base.unsqueeze(2).expand(
                    1, B, k, -1).reshape(1, B * k, -1)

                attn_logits = self.encoder.attn_proj(gru_out_k).squeeze(-1)
                attn_weights = F.softmax(attn_logits, dim=1)
                h_pooled = (attn_weights.unsqueeze(-1) * gru_out_k).sum(dim=1)
                cond_init = self.encoder.bottleneck(h_pooled)

                frames = []
                L_per_step = [] if return_L else None
                prev = last_k
                gru_state = h_last_k.contiguous()
                gru_outs = gru_out_k

                for t in range(T):
                    z_t = torch.randn(B * k, self.decoder.noise_dim, device=device)
                    if t > 0:
                        al = self.encoder.attn_proj(gru_outs).squeeze(-1)
                        aw = F.softmax(al, dim=1)
                        hp = (aw.unsqueeze(-1) * gru_outs).sum(dim=1)
                        cond_t = self.encoder.bottleneck(hp)
                    else:
                        cond_t = cond_init

                    delta_base, L = self.decoder(cond_t, prev, z_t)
                    eps = torch.randn(B * k, self.n_factors, device=device)
                    delta = delta_base + torch.einsum("bcr,br->bc", L, eps)
                    frame_t = prev + torch.tanh(delta)
                    frame_t = reflecting_boundary(frame_t)
                    frames.append(frame_t)

                    if return_L:
                        # Store L reshaped to (B, k, 25, n_factors)
                        L_per_step.append(L.reshape(B, k, 25, self.n_factors))

                    fn = normalize_iv(frame_t).unsqueeze(1)
                    go, gru_state = self.encoder.gru(fn, gru_state)
                    gru_outs = torch.cat([gru_outs, go], dim=1)
                    prev = frame_t

                frames = torch.stack(frames, dim=1).reshape(B, k, T, 5, 5)
                all_samples.append(frames)

                if return_L:
                    # Stack: (T, B, k, 25, n_factors) -> use first chunk only for L
                    L_per_step = torch.stack(L_per_step, dim=0)  # (T, B, k, 25, nf)
                    all_L.append(L_per_step)

            samples = torch.cat(all_samples, dim=1)
            if return_L:
                # Return L from first chunk only (representative)
                return samples, all_L[0]  # (T, B, k, 25, nf)
            return samples


def load_model(model_path, device):
    """Load 167b model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    encoder_config = EncoderConfig(**cfg["encoder"])
    decoder_config = cfg["decoder"]
    n_factors = cfg.get("n_factors", 5)
    model = ARFactorizedCleanModel(encoder_config, decoder_config, n_factors=n_factors)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def compute_eff_rank(eigenvalues):
    """Effective rank from eigenvalue spectrum via entropy."""
    ev = np.abs(eigenvalues)
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return 1.0
    p = ev / ev.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return np.exp(entropy)


def cosine_similarity_matrix(A, B):
    """Cosine similarity between columns of A and B."""
    # A: (n, k1), B: (n, k2) -> (k1, k2)
    A_norm = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return A_norm.T @ B_norm


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="167b Factor Structure Analysis")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/afcrps_167b/final_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_windows", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--output_dir", type=str,
                        default="results/validations/2026-04-04/analysis/167b_followup")
    args = parser.parse_args()

    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Load data ---
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    returns = data["ret"]
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25
    surf_tensor = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    # --- Load model ---
    model = load_model(args.model_path, device)
    n_factors = model.n_factors
    print(f"n_factors = {n_factors}")

    # --- Build test windows ---
    test_indices = np.arange(args.test_start, min(args.test_start + args.n_windows, N_total - H - T))
    n_windows = len(test_indices)
    print(f"Using {n_windows} test windows starting at index {args.test_start}")

    # Build history and future for all windows
    hist_list = []
    future_list = []
    last_frame_list = []
    for idx in test_indices:
        hist = surf_tensor[idx:idx + H].unsqueeze(0)  # (1, 30, 5, 5)
        fut = surfaces[idx + H:idx + H + T].reshape(T, C)  # (30, 25)
        last = surfaces[idx + H - 1].reshape(C)  # (25,)
        hist_list.append(hist)
        future_list.append(fut)
        last_frame_list.append(last)

    hist_batch = torch.cat(hist_list, dim=0)  # (n_windows, 30, 5, 5)
    future_np = np.stack(future_list, axis=0)  # (n_windows, 30, 25)
    last_frame_np = np.stack(last_frame_list, axis=0)  # (n_windows, 25)

    # --- Compute VoV for regime classification ---
    vov_list = []
    for idx in test_indices:
        hist_ret = returns[idx:idx + H]
        vov_list.append(np.std(hist_ret))
    vov = np.array(vov_list)
    vov_median = np.median(vov)
    calm_mask = vov <= vov_median
    turb_mask = vov > vov_median
    print(f"VoV median: {vov_median:.6f}, Calm: {calm_mask.sum()}, Turbulent: {turb_mask.sum()}")

    # ============================================================
    # PART 1: Factor Structure Comparison (PCA on deltas)
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: Factor Structure Comparison (PCA on generated vs GT deltas)")
    print("=" * 60)

    # Generate samples
    hist_norm = normalize_iv(hist_batch)
    with torch.no_grad():
        samples, L_all = model.sample_batched(hist_norm, n_samples=args.n_samples, return_L=True)
    # samples: (n_windows, n_samples, 30, 5, 5)
    samples_flat = samples.cpu().numpy().reshape(n_windows, args.n_samples, T, C)
    print(f"Generated samples shape: {samples_flat.shape}")

    # L_all: (T, n_windows, chunk_k, 25, n_factors) - from first chunk of 10
    L_all_np = L_all.cpu().numpy()  # (T=30, B=n_windows, k=10, 25, n_factors)
    print(f"L matrices shape: {L_all_np.shape}")

    # Compute deltas for generated and GT
    # For generated: delta = frame_t - frame_{t-1}, for t=1..29
    # For GT: delta = future_t - future_{t-1}
    gen_deltas_all = []
    gt_deltas_all = []
    for w in range(n_windows):
        # GT deltas: from future surfaces
        gt_frames = future_np[w]  # (30, 25)
        gt_prev = np.concatenate([last_frame_np[w:w+1], gt_frames[:-1]], axis=0)  # (30, 25)
        gt_delta = gt_frames - gt_prev  # (30, 25)
        gt_deltas_all.append(gt_delta)

        # Generated deltas: from sampled frames
        for s in range(args.n_samples):
            gen_frames = samples_flat[w, s]  # (30, 25)
            gen_prev = np.concatenate([last_frame_np[w:w+1], gen_frames[:-1]], axis=0)
            gen_delta = gen_frames - gen_prev  # (30, 25)
            gen_deltas_all.append(gen_delta)

    gt_deltas = np.concatenate(gt_deltas_all, axis=0)  # (n_windows * 30, 25)
    gen_deltas = np.concatenate(gen_deltas_all, axis=0)  # (n_windows * n_samples * 30, 25)
    print(f"GT deltas: {gt_deltas.shape}, Gen deltas: {gen_deltas.shape}")

    # PCA on GT deltas
    gt_mean = gt_deltas.mean(axis=0)
    gt_centered = gt_deltas - gt_mean
    gt_cov = np.cov(gt_centered.T)
    gt_eigenvalues, gt_eigenvectors = np.linalg.eigh(gt_cov)
    gt_eigenvalues = gt_eigenvalues[::-1]
    gt_eigenvectors = gt_eigenvectors[:, ::-1]

    # PCA on generated deltas
    gen_mean = gen_deltas.mean(axis=0)
    gen_centered = gen_deltas - gen_mean
    gen_cov = np.cov(gen_centered.T)
    gen_eigenvalues, gen_eigenvectors = np.linalg.eigh(gen_cov)
    gen_eigenvalues = gen_eigenvalues[::-1]
    gen_eigenvectors = gen_eigenvectors[:, ::-1]

    # Metrics
    gt_eff_rank = compute_eff_rank(gt_eigenvalues)
    gen_eff_rank = compute_eff_rank(gen_eigenvalues)
    gt_pc1_share = gt_eigenvalues[0] / gt_eigenvalues.sum()
    gen_pc1_share = gen_eigenvalues[0] / gen_eigenvalues.sum()
    gt_var_explained = np.cumsum(gt_eigenvalues) / gt_eigenvalues.sum()
    gen_var_explained = np.cumsum(gen_eigenvalues) / gen_eigenvalues.sum()

    # Cosine similarity between loadings
    n_compare = min(5, len(gt_eigenvalues))
    cos_sim = cosine_similarity_matrix(gt_eigenvectors[:, :n_compare],
                                        gen_eigenvectors[:, :n_compare])
    # Best alignment (max abs cosine sim for each GT PC)
    best_alignment = np.max(np.abs(cos_sim), axis=1)

    print(f"\n--- Eigenvalue Spectrum ---")
    print(f"GT  eff_rank: {gt_eff_rank:.3f}, PC1 share: {gt_pc1_share:.3f}")
    print(f"Gen eff_rank: {gen_eff_rank:.3f}, PC1 share: {gen_pc1_share:.3f}")
    print(f"\nGT  eigenvalues (top 10): {gt_eigenvalues[:10]}")
    print(f"Gen eigenvalues (top 10): {gen_eigenvalues[:10]}")
    print(f"\nGT  cumvar: {gt_var_explained[:10]}")
    print(f"Gen cumvar: {gen_var_explained[:10]}")
    print(f"\nCosine similarity (GT PCs vs Gen PCs):")
    print(f"  Matrix:\n{cos_sim}")
    print(f"  Best alignment per GT PC: {best_alignment}")

    factor_comparison = {
        "gt_eff_rank": float(gt_eff_rank),
        "gen_eff_rank": float(gen_eff_rank),
        "gt_pc1_share": float(gt_pc1_share),
        "gen_pc1_share": float(gen_pc1_share),
        "gt_eigenvalues_top10": gt_eigenvalues[:10].tolist(),
        "gen_eigenvalues_top10": gen_eigenvalues[:10].tolist(),
        "gt_cumvar_top5": gt_var_explained[:5].tolist(),
        "gen_cumvar_top5": gen_var_explained[:5].tolist(),
        "cosine_sim_matrix": cos_sim.tolist(),
        "best_alignment_per_gt_pc": best_alignment.tolist(),
        "eigenvalue_ratio_gen_over_gt": (gen_eigenvalues[:10] / (gt_eigenvalues[:10] + 1e-12)).tolist(),
    }

    # ============================================================
    # PART 2: L Matrix Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: L Matrix Analysis")
    print("=" * 60)

    # Extract L for each window at t=0 (initial step)
    # L_all: (T=30, B=n_windows, k=10, 25, n_factors)
    # Take mean across k (samples) and look at t=0
    L_t0 = L_all_np[0]  # (n_windows, 10, 25, n_factors)
    L_t0_mean = L_t0.mean(axis=1)  # (n_windows, 25, n_factors)

    # Also look at L across all timesteps
    L_all_mean = L_all_np.mean(axis=2)  # (T, n_windows, 25, n_factors)

    # SVD of L pooled across windows (t=0)
    L_pooled = L_t0_mean.reshape(-1, n_factors)  # (n_windows*25, n_factors)
    U, s, Vt = np.linalg.svd(L_pooled, full_matrices=False)
    L_eff_rank_svd = compute_eff_rank(s ** 2)
    L_sv_share = s / s.sum()

    print(f"\n--- L SVD (pooled across windows, t=0) ---")
    print(f"Singular values: {s}")
    print(f"SV shares: {L_sv_share}")
    print(f"L eff_rank (from SVD): {L_eff_rank_svd:.3f}")
    print(f"Right singular vectors (Vt):\n{Vt}")

    # Compare L's left singular vectors with GT PCA directions
    # Reshape L to (25, n_factors) per window, then check if columns align with GT PCs
    L_per_window_alignment = []
    for w in range(n_windows):
        L_w = L_t0_mean[w]  # (25, n_factors)
        # SVD of this L
        U_w, s_w, Vt_w = np.linalg.svd(L_w, full_matrices=False)
        # U_w columns are cell loading directions
        # Compare with GT eigenvectors
        cos_L_gt = cosine_similarity_matrix(U_w[:, :min(5, n_factors)],
                                             gt_eigenvectors[:, :5])
        best_per_L = np.max(np.abs(cos_L_gt), axis=1)
        L_per_window_alignment.append({
            "singular_values": s_w.tolist(),
            "eff_rank": float(compute_eff_rank(s_w ** 2)),
            "best_alignment_with_gt_pcs": best_per_L.tolist(),
        })

    L_eff_ranks_per_window = [a["eff_rank"] for a in L_per_window_alignment]
    L_gt_align_mean = np.mean([a["best_alignment_with_gt_pcs"] for a in L_per_window_alignment], axis=0)

    print(f"\n--- Per-window L analysis ---")
    print(f"L eff_rank per window: mean={np.mean(L_eff_ranks_per_window):.3f}, "
          f"std={np.std(L_eff_ranks_per_window):.3f}, "
          f"range=[{np.min(L_eff_ranks_per_window):.3f}, {np.max(L_eff_ranks_per_window):.3f}]")
    print(f"Mean alignment of L singular vectors with GT PCs: {L_gt_align_mean}")

    # L evolution across timesteps
    L_eff_rank_by_t = []
    L_norm_by_t = []
    for t in range(T):
        L_t = L_all_mean[t]  # (n_windows, 25, n_factors)
        L_flat = L_t.reshape(-1, n_factors)
        _, sv_t, _ = np.linalg.svd(L_flat, full_matrices=False)
        L_eff_rank_by_t.append(float(compute_eff_rank(sv_t ** 2)))
        L_norm_by_t.append(float(np.linalg.norm(L_flat)))

    print(f"\n--- L evolution across timesteps ---")
    print(f"L eff_rank: t=0={L_eff_rank_by_t[0]:.3f}, t=14={L_eff_rank_by_t[14]:.3f}, "
          f"t=29={L_eff_rank_by_t[29]:.3f}")
    print(f"L norm: t=0={L_norm_by_t[0]:.4f}, t=14={L_norm_by_t[14]:.4f}, "
          f"t=29={L_norm_by_t[29]:.4f}")

    # Compare output delta eff_rank vs L eff_rank
    # Output delta eff_rank: compute per-window
    gen_eff_ranks_per_window = []
    for w in range(n_windows):
        gen_w = []
        for s_idx in range(args.n_samples):
            gen_frames = samples_flat[w, s_idx]  # (30, 25)
            gen_prev = np.concatenate([last_frame_np[w:w+1], gen_frames[:-1]], axis=0)
            gen_delta = gen_frames - gen_prev
            gen_w.append(gen_delta)
        gen_w = np.concatenate(gen_w, axis=0)  # (n_samples*30, 25)
        gen_cov_w = np.cov(gen_w.T)
        ev_w = np.linalg.eigvalsh(gen_cov_w)[::-1]
        gen_eff_ranks_per_window.append(float(compute_eff_rank(ev_w)))

    print(f"\nOutput delta eff_rank per window: mean={np.mean(gen_eff_ranks_per_window):.3f}, "
          f"std={np.std(gen_eff_ranks_per_window):.3f}")
    print(f"L eff_rank per window: mean={np.mean(L_eff_ranks_per_window):.3f}")
    print(f"Ratio output_eff_rank/L_eff_rank: "
          f"{np.mean(gen_eff_ranks_per_window) / np.mean(L_eff_ranks_per_window):.3f}")

    l_matrix_analysis = {
        "pooled_svd": {
            "singular_values": s.tolist(),
            "sv_shares": L_sv_share.tolist(),
            "eff_rank": float(L_eff_rank_svd),
            "right_singular_vectors": Vt.tolist(),
        },
        "per_window": {
            "eff_rank_mean": float(np.mean(L_eff_ranks_per_window)),
            "eff_rank_std": float(np.std(L_eff_ranks_per_window)),
            "eff_rank_min": float(np.min(L_eff_ranks_per_window)),
            "eff_rank_max": float(np.max(L_eff_ranks_per_window)),
            "gt_alignment_mean": L_gt_align_mean.tolist(),
        },
        "temporal_evolution": {
            "eff_rank_by_t": L_eff_rank_by_t,
            "norm_by_t": L_norm_by_t,
        },
        "output_vs_L_eff_rank": {
            "output_delta_eff_rank_mean": float(np.mean(gen_eff_ranks_per_window)),
            "L_eff_rank_mean": float(np.mean(L_eff_ranks_per_window)),
            "ratio": float(np.mean(gen_eff_ranks_per_window) / np.mean(L_eff_ranks_per_window)),
        },
    }

    # ============================================================
    # PART 3: Per-cell Spread Decomposition
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3: Per-cell Spread Decomposition")
    print("=" * 60)

    # Spread from L@eps: var(L@eps) for each cell = sum_j L_{c,j}^2
    # Per-cell spread from L (t=0, averaged across windows and samples)
    L_spread_per_cell = np.zeros(C)
    for w in range(n_windows):
        L_w = L_t0_mean[w]  # (25, n_factors)
        # Variance contribution: sum of squared loadings per cell
        cell_var = (L_w ** 2).sum(axis=1)  # (25,)
        L_spread_per_cell += cell_var
    L_spread_per_cell /= n_windows

    # GT spread per cell: variance of GT deltas per cell
    gt_cell_var = np.var(gt_deltas, axis=0)  # (25,)

    # Generated spread per cell
    gen_cell_var = np.var(gen_deltas, axis=0)  # (25,)

    # Normalize for comparison
    L_spread_norm = L_spread_per_cell / (L_spread_per_cell.sum() + 1e-12)
    gt_spread_norm = gt_cell_var / (gt_cell_var.sum() + 1e-12)
    gen_spread_norm = gen_cell_var / (gen_cell_var.sum() + 1e-12)

    # Correlation between spread distributions
    from scipy.stats import pearsonr, spearmanr
    L_gt_corr, _ = pearsonr(L_spread_norm, gt_spread_norm)
    gen_gt_corr, _ = pearsonr(gen_spread_norm, gt_spread_norm)
    L_gt_spearman, _ = spearmanr(L_spread_norm, gt_spread_norm)
    gen_gt_spearman, _ = spearmanr(gen_spread_norm, gt_spread_norm)

    print(f"\n--- Spread per cell (normalized) ---")
    print(f"L spread vs GT: Pearson={L_gt_corr:.3f}, Spearman={L_gt_spearman:.3f}")
    print(f"Gen spread vs GT: Pearson={gen_gt_corr:.3f}, Spearman={gen_gt_spearman:.3f}")
    print(f"\nL spread (normalized): {L_spread_norm}")
    print(f"GT spread (normalized): {gt_spread_norm}")
    print(f"Gen spread (normalized): {gen_spread_norm}")

    # Condition-dependent spread: does L change per window?
    L_spread_per_window = np.zeros((n_windows, C))
    for w in range(n_windows):
        L_w = L_t0_mean[w]
        L_spread_per_window[w] = (L_w ** 2).sum(axis=1)

    # Coefficient of variation across windows per cell
    L_spread_cv = np.std(L_spread_per_window, axis=0) / (np.mean(L_spread_per_window, axis=0) + 1e-12)
    print(f"\nL spread CV across windows per cell: mean={L_spread_cv.mean():.3f}, "
          f"range=[{L_spread_cv.min():.3f}, {L_spread_cv.max():.3f}]")

    # Reshape spread for 5x5 grid display
    cell_labels = ["M" + str(i) + "T" + str(j) for i in range(5) for j in range(5)]

    spread_analysis = {
        "L_spread_vs_gt": {
            "pearson": float(L_gt_corr),
            "spearman": float(L_gt_spearman),
        },
        "gen_spread_vs_gt": {
            "pearson": float(gen_gt_corr),
            "spearman": float(gen_gt_spearman),
        },
        "L_spread_normalized": L_spread_norm.tolist(),
        "gt_spread_normalized": gt_spread_norm.tolist(),
        "gen_spread_normalized": gen_spread_norm.tolist(),
        "L_spread_raw": L_spread_per_cell.tolist(),
        "gt_spread_raw": gt_cell_var.tolist(),
        "gen_spread_raw": gen_cell_var.tolist(),
        "L_spread_cv_per_cell": L_spread_cv.tolist(),
        "L_spread_cv_mean": float(L_spread_cv.mean()),
        "condition_dependent": bool(L_spread_cv.mean() > 0.05),
    }

    # ============================================================
    # PART 4: Condition Sensitivity (Calm vs Turbulent)
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 4: Condition Sensitivity (Calm vs Turbulent)")
    print("=" * 60)

    calm_idx = np.where(calm_mask)[0]
    turb_idx = np.where(turb_mask)[0]

    # L matrices for calm vs turbulent
    L_calm = L_t0_mean[calm_idx]  # (n_calm, 25, n_factors)
    L_turb = L_t0_mean[turb_idx]  # (n_turb, 25, n_factors)

    # Average L norms
    L_calm_norm = np.mean([np.linalg.norm(L_calm[i]) for i in range(len(calm_idx))])
    L_turb_norm = np.mean([np.linalg.norm(L_turb[i]) for i in range(len(turb_idx))])

    # Per-cell spread comparison
    calm_spread = np.mean([(L_calm[i] ** 2).sum(axis=1) for i in range(len(calm_idx))], axis=0)
    turb_spread = np.mean([(L_turb[i] ** 2).sum(axis=1) for i in range(len(turb_idx))], axis=0)
    turb_calm_ratio = turb_spread / (calm_spread + 1e-12)

    # GT calm vs turbulent spread
    gt_calm_deltas = np.concatenate([gt_deltas_all[i] for i in calm_idx], axis=0)
    gt_turb_deltas = np.concatenate([gt_deltas_all[i] for i in turb_idx], axis=0)
    gt_calm_var = np.var(gt_calm_deltas, axis=0)
    gt_turb_var = np.var(gt_turb_deltas, axis=0)
    gt_turb_calm_ratio = gt_turb_var / (gt_calm_var + 1e-12)

    # Compare generated calm vs turb
    gen_calm_deltas = []
    gen_turb_deltas = []
    for w_idx in calm_idx:
        for s_idx in range(args.n_samples):
            gen_frames = samples_flat[w_idx, s_idx]
            gen_prev = np.concatenate([last_frame_np[w_idx:w_idx+1], gen_frames[:-1]], axis=0)
            gen_calm_deltas.append(gen_frames - gen_prev)
    for w_idx in turb_idx:
        for s_idx in range(args.n_samples):
            gen_frames = samples_flat[w_idx, s_idx]
            gen_prev = np.concatenate([last_frame_np[w_idx:w_idx+1], gen_frames[:-1]], axis=0)
            gen_turb_deltas.append(gen_frames - gen_prev)

    gen_calm_deltas = np.concatenate(gen_calm_deltas, axis=0)
    gen_turb_deltas = np.concatenate(gen_turb_deltas, axis=0)
    gen_calm_var = np.var(gen_calm_deltas, axis=0)
    gen_turb_var = np.var(gen_turb_deltas, axis=0)
    gen_turb_calm_ratio = gen_turb_var / (gen_calm_var + 1e-12)

    # L eff_rank by regime
    L_calm_pooled = L_calm.reshape(-1, n_factors)
    L_turb_pooled = L_turb.reshape(-1, n_factors)
    _, sv_calm, _ = np.linalg.svd(L_calm_pooled, full_matrices=False)
    _, sv_turb, _ = np.linalg.svd(L_turb_pooled, full_matrices=False)
    L_calm_eff_rank = compute_eff_rank(sv_calm ** 2)
    L_turb_eff_rank = compute_eff_rank(sv_turb ** 2)

    # FiLM effect: cosine similarity between mean L in calm vs turb
    L_calm_mean = L_calm.mean(axis=0)  # (25, n_factors)
    L_turb_mean = L_turb.mean(axis=0)
    cos_calm_turb = cosine_similarity_matrix(L_calm_mean, L_turb_mean)
    diag_cos = np.diag(cos_calm_turb)

    print(f"\n--- L norms by regime ---")
    print(f"Calm L norm (mean): {L_calm_norm:.4f}")
    print(f"Turb L norm (mean): {L_turb_norm:.4f}")
    print(f"Turb/Calm ratio: {L_turb_norm / (L_calm_norm + 1e-12):.3f}")

    print(f"\n--- Per-cell turb/calm spread ratio ---")
    print(f"L-based turb/calm ratio per cell: mean={turb_calm_ratio.mean():.3f}, "
          f"range=[{turb_calm_ratio.min():.3f}, {turb_calm_ratio.max():.3f}]")
    print(f"GT turb/calm ratio per cell: mean={gt_turb_calm_ratio.mean():.3f}, "
          f"range=[{gt_turb_calm_ratio.min():.3f}, {gt_turb_calm_ratio.max():.3f}]")
    print(f"Gen turb/calm ratio per cell: mean={gen_turb_calm_ratio.mean():.3f}, "
          f"range=[{gen_turb_calm_ratio.min():.3f}, {gen_turb_calm_ratio.max():.3f}]")

    print(f"\n--- L structure by regime ---")
    print(f"L calm eff_rank: {L_calm_eff_rank:.3f}")
    print(f"L turb eff_rank: {L_turb_eff_rank:.3f}")
    print(f"Calm-turb L factor cosine similarity (diag): {diag_cos}")

    print(f"\n--- FiLM sensitivity ---")
    print(f"Mean |cos_sim| between calm and turb L factors: {np.mean(np.abs(diag_cos)):.3f}")
    film_changes_structure = bool(np.mean(np.abs(diag_cos)) < 0.95)
    print(f"FiLM changes factor structure: {film_changes_structure}")

    # Correlation of turb/calm ratios
    ratio_corr, _ = pearsonr(turb_calm_ratio, gt_turb_calm_ratio)
    gen_ratio_corr, _ = pearsonr(gen_turb_calm_ratio, gt_turb_calm_ratio)

    condition_analysis = {
        "L_norm": {
            "calm": float(L_calm_norm),
            "turb": float(L_turb_norm),
            "turb_calm_ratio": float(L_turb_norm / (L_calm_norm + 1e-12)),
        },
        "turb_calm_spread_ratio": {
            "L_based": {
                "mean": float(turb_calm_ratio.mean()),
                "per_cell": turb_calm_ratio.tolist(),
            },
            "gt": {
                "mean": float(gt_turb_calm_ratio.mean()),
                "per_cell": gt_turb_calm_ratio.tolist(),
            },
            "gen": {
                "mean": float(gen_turb_calm_ratio.mean()),
                "per_cell": gen_turb_calm_ratio.tolist(),
            },
            "L_vs_gt_corr": float(ratio_corr),
            "gen_vs_gt_corr": float(gen_ratio_corr),
        },
        "L_eff_rank_by_regime": {
            "calm": float(L_calm_eff_rank),
            "turb": float(L_turb_eff_rank),
        },
        "film_effect": {
            "factor_cos_sim_calm_vs_turb": diag_cos.tolist(),
            "mean_abs_cos_sim": float(np.mean(np.abs(diag_cos))),
            "changes_structure": film_changes_structure,
        },
        "vov_stats": {
            "median": float(vov_median),
            "n_calm": int(calm_mask.sum()),
            "n_turb": int(turb_mask.sum()),
        },
    }

    # ============================================================
    # Summary and Verdict
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Key metrics for verdict
    eff_rank_ratio = gen_eff_rank / gt_eff_rank
    pc1_ratio = gen_pc1_share / gt_pc1_share
    mean_pc_alignment = np.mean(best_alignment[:3])  # Top 3 PCs
    L_output_ratio = np.mean(gen_eff_ranks_per_window) / np.mean(L_eff_ranks_per_window)

    print(f"\n1. Factor structure:")
    print(f"   GT eff_rank={gt_eff_rank:.3f}, Gen eff_rank={gen_eff_rank:.3f}, ratio={eff_rank_ratio:.3f}")
    print(f"   GT PC1={gt_pc1_share:.3f}, Gen PC1={gen_pc1_share:.3f}")
    print(f"   Mean alignment (top 3 PCs): {mean_pc_alignment:.3f}")

    print(f"\n2. L matrix:")
    print(f"   L eff_rank (per-window mean): {np.mean(L_eff_ranks_per_window):.3f}")
    print(f"   Output eff_rank / L eff_rank: {L_output_ratio:.3f}")
    print(f"   L-GT PC alignment: {L_gt_align_mean[:3]}")

    print(f"\n3. Spread distribution:")
    print(f"   L vs GT spread: Pearson={L_gt_corr:.3f}, Spearman={L_gt_spearman:.3f}")
    print(f"   Condition-dependent: {spread_analysis['condition_dependent']} (CV={L_spread_cv.mean():.3f})")

    print(f"\n4. Condition sensitivity:")
    print(f"   L turb/calm norm ratio: {L_turb_norm / (L_calm_norm + 1e-12):.3f}")
    print(f"   Gen turb/calm spread: {gen_turb_calm_ratio.mean():.3f} (GT: {gt_turb_calm_ratio.mean():.3f})")
    print(f"   FiLM changes structure: {film_changes_structure}")

    # Principled improvement assessment
    principled = True
    issues = []

    if eff_rank_ratio < 0.5:
        issues.append(f"Gen eff_rank too low ({gen_eff_rank:.2f} vs GT {gt_eff_rank:.2f})")
        principled = False
    if eff_rank_ratio > 2.0:
        issues.append(f"Gen eff_rank too high ({gen_eff_rank:.2f} vs GT {gt_eff_rank:.2f})")
        principled = False
    if gen_pc1_share > 0.8:
        issues.append(f"Gen PC1 dominance too high ({gen_pc1_share:.3f})")
        principled = False
    if mean_pc_alignment < 0.5:
        issues.append(f"Poor PC alignment ({mean_pc_alignment:.3f})")
    if L_gt_corr < 0.3:
        issues.append(f"Poor L-GT spread correlation ({L_gt_corr:.3f})")

    verdict = "PRINCIPLED IMPROVEMENT" if principled and len(issues) == 0 else \
              "PARTIALLY PRINCIPLED" if principled else "NOT PRINCIPLED"
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    if issues:
        print(f"Issues: {issues}")
    print(f"{'='*60}")

    summary = {
        "verdict": verdict,
        "issues": issues,
        "key_metrics": {
            "gt_eff_rank": float(gt_eff_rank),
            "gen_eff_rank": float(gen_eff_rank),
            "eff_rank_ratio": float(eff_rank_ratio),
            "gt_pc1_share": float(gt_pc1_share),
            "gen_pc1_share": float(gen_pc1_share),
            "mean_pc_alignment_top3": float(mean_pc_alignment),
            "L_eff_rank_per_window_mean": float(np.mean(L_eff_ranks_per_window)),
            "output_L_eff_rank_ratio": float(L_output_ratio),
            "L_gt_spread_pearson": float(L_gt_corr),
            "L_gt_spread_spearman": float(L_gt_spearman),
            "L_turb_calm_norm_ratio": float(L_turb_norm / (L_calm_norm + 1e-12)),
            "film_changes_structure": film_changes_structure,
        },
    }

    # Save all results
    results = {
        "model_path": args.model_path,
        "n_windows": n_windows,
        "n_samples": args.n_samples,
        "test_start": args.test_start,
        "factor_comparison": factor_comparison,
        "l_matrix_analysis": l_matrix_analysis,
        "spread_analysis": spread_analysis,
        "condition_analysis": condition_analysis,
        "summary": summary,
    }

    out_path = Path(args.output_dir) / "factor_structure.json"
    with open(out_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Also save verification copy
    verif_path = Path("results/validations/2026-04-04/verification_results/167b_factor_structure.json")
    verif_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verif_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"Verification copy saved to {verif_path}")


if __name__ == "__main__":
    main()
