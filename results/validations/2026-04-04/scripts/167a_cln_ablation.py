#!/usr/bin/env python
"""
167a CLN vs Factor Path Ablation Study

Ablation on epoch 20 checkpoint (when L was still somewhat active, L_norm~0.08).
For 30 test windows with 100 noise samples each, measure 4 conditions:
1. Both pathways (normal forward)
2. CLN only (factor eps=0)
3. Factor only (CLN z=0)
4. Neither (both=0)

Key question: Does CLN compete with and suppress the factor path's rank contribution?
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_167a_factorized import (
    ARFactorizedTransformerModel,
    FactorizedSpatialTransformerDecoder,
    normalize_iv,
    denormalize_iv,
    reflecting_boundary,
)


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def compute_effective_rank(X):
    """Effective rank from singular values (Vershynin definition).
    X: (n_samples, n_features)
    """
    _, s, _ = torch.svd(X - X.mean(dim=0, keepdim=True))
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0, 1.0
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    eff_rank = entropy.exp().item()
    pc1_dominance = (s[0] ** 2 / (s ** 2).sum()).item()
    return eff_rank, pc1_dominance


def compute_cross_cell_correlation(members):
    """Average pairwise correlation across ensemble members.
    members: (K, 25) -- K members, 25 cells
    """
    K = members.shape[0]
    if K < 2:
        return 0.0
    # Flatten to (K, 25), compute correlation matrix
    members_centered = members - members.mean(dim=0, keepdim=True)
    norms = members_centered.norm(dim=1, keepdim=True).clamp(min=1e-10)
    members_normed = members_centered / norms
    corr_matrix = members_normed @ members_normed.t()  # (K, K)
    # Average upper triangle
    mask = torch.triu(torch.ones(K, K, device=members.device), diagonal=1).bool()
    return corr_matrix[mask].mean().item()


def ar_generate_ablated(model, condition, last_frame, n_steps, gru_state, gru_outputs,
                        zero_cln=False, zero_factor=False):
    """AR generation with ablation options for CLN and factor noise."""
    B = condition.shape[0]
    device = condition.device
    noise_dim = model.decoder.noise_dim
    n_factors = model.n_factors
    frames = []
    prev = last_frame

    for t in range(n_steps):
        # CLN noise: if zero_cln, pass zeros instead of random
        if zero_cln:
            z_t = torch.zeros(B, noise_dim, device=device)
        else:
            z_t = torch.randn(B, noise_dim, device=device)

        if gru_state is not None and gru_outputs is not None:
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond_t = model.encoder.bottleneck(h_pooled)
        else:
            cond_t = condition

        # Factorized output
        delta_base, L = model.decoder(cond_t, prev, z_t)

        # Factor noise: if zero_factor, pass zeros instead of random
        if zero_factor:
            eps = torch.zeros(B, n_factors, device=device)
        else:
            eps = torch.randn(B, n_factors, device=device)

        delta = delta_base + torch.einsum("bcr,br->bc", L, eps)
        frame_t = prev + torch.tanh(delta)
        frame_t = reflecting_boundary(frame_t)
        frames.append(frame_t)

        if gru_state is not None:
            frame_norm = normalize_iv(frame_t).unsqueeze(1)
            gru_out, gru_state = model.encoder.gru(frame_norm, gru_state)
            gru_outputs = torch.cat([gru_outputs, gru_out], dim=1)

        prev = frame_t

    return torch.stack(frames, dim=1)


def sample_ablated(model, history, n_samples, zero_cln=False, zero_factor=False):
    """Generate samples with ablation options. Returns (B, n_samples, T, 25)."""
    B = history.shape[0]
    T = 30
    CHUNK = 10

    with torch.no_grad():
        last_frame = denormalize_iv(history[:, -1]).reshape(B, 25)
        hist_flat = history.reshape(B, history.shape[1], -1)
        gru_outputs_base, h_last_base = model.encoder.gru(hist_flat)

        all_samples = []
        for start in range(0, n_samples, CHUNK):
            k = min(CHUNK, n_samples - start)
            last_k = last_frame.unsqueeze(1).expand(B, k, -1).reshape(B * k, 25)
            gru_out_k = gru_outputs_base.unsqueeze(1).expand(
                B, k, -1, -1).reshape(B * k, -1, model.encoder_config.gru_hidden_dim)
            h_last_k = h_last_base.unsqueeze(2).expand(
                1, B, k, -1).reshape(1, B * k, -1)

            attn_logits = model.encoder.attn_proj(gru_out_k).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_out_k).sum(dim=1)
            cond_init = model.encoder.bottleneck(h_pooled)

            frames = ar_generate_ablated(
                model, cond_init, last_k, n_steps=T,
                gru_state=h_last_k.contiguous(),
                gru_outputs=gru_out_k,
                zero_cln=zero_cln,
                zero_factor=zero_factor,
            )
            frames = frames.reshape(B, k, T, 25)
            all_samples.append(frames)

        samples = torch.cat(all_samples, dim=1)
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)
    surf_tensor = torch.from_numpy(surfaces).float().to(device)
    N = len(surfaces)
    H = 30  # history length
    T = 30  # forecast horizon
    C = 25

    # Test split starts at index 4540
    test_start = 4540
    test_end = N - H - T
    test_indices = np.arange(test_start, test_end)
    n_windows = min(30, len(test_indices))
    print(f"Test windows: {n_windows} (from {len(test_indices)} available)")

    # Select evenly spaced windows
    window_indices = test_indices[np.linspace(0, len(test_indices) - 1, n_windows, dtype=int)]

    # Load model
    ckpt_path = "models/backfill/afcrps_167a/checkpoint_epoch_20.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    config = ckpt["config"]

    encoder_config = EncoderConfig(**config["encoder"])
    decoder_config = config["decoder"]
    n_factors = config["n_factors"]

    model = ARFactorizedTransformerModel(encoder_config, decoder_config, n_factors=n_factors)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")

    # Check L norm at this checkpoint
    L_weight = model.decoder.load_head[-1].weight
    print(f"load_head final layer weight norm: {L_weight.norm().item():.4f}")
    print(f"load_head final layer weight mean abs: {L_weight.abs().mean().item():.6f}")

    n_samples = 100
    ablation_modes = {
        "both": {"zero_cln": False, "zero_factor": False},
        "cln_only": {"zero_cln": False, "zero_factor": True},
        "factor_only": {"zero_cln": True, "zero_factor": False},
        "neither": {"zero_cln": True, "zero_factor": True},
    }

    # Results storage
    results = {mode: {
        "per_cell_spread": [],        # (n_windows, 25) mean std across members per cell
        "cross_cell_corr": [],         # (n_windows,) avg pairwise corr
        "effective_rank": [],          # (n_windows,)
        "pc1_dominance": [],           # (n_windows,)
        "total_spread": [],            # (n_windows,)
        "per_horizon_spread": [],      # (n_windows, T) spread by horizon
    } for mode in ablation_modes}

    # For PC direction comparison, store top-3 PC directions at horizon 15 (midpoint)
    pc_directions = {mode: [] for mode in ablation_modes}

    torch.manual_seed(42)
    t0 = time.time()

    for wi, idx in enumerate(window_indices):
        if wi % 5 == 0:
            print(f"Window {wi+1}/{n_windows} (idx={idx})...")

        # Build history
        offsets = torch.arange(H, device=device)
        hist_idx = idx + offsets
        hist = surf_tensor[hist_idx].unsqueeze(0)  # (1, H, 5, 5)
        hist_norm = normalize_iv(hist)

        for mode_name, mode_kwargs in ablation_modes.items():
            # Use same seed per window for fair comparison
            torch.manual_seed(42 + wi * 1000)

            # Generate samples: (1, n_samples, T, 25)
            samples = sample_ablated(model, hist_norm, n_samples, **mode_kwargs)
            samples = samples.squeeze(0)  # (n_samples, T, 25)

            # Per-cell spread: std across members, mean over horizons
            per_cell_std = samples.std(dim=0)  # (T, 25)
            mean_per_cell_spread = per_cell_std.mean(dim=0)  # (25,)
            results[mode_name]["per_cell_spread"].append(mean_per_cell_spread.cpu().numpy())

            # Per-horizon spread
            per_horizon_spread = per_cell_std.mean(dim=1)  # (T,)
            results[mode_name]["per_horizon_spread"].append(per_horizon_spread.cpu().numpy())

            # Total spread
            total_spread = per_cell_std.mean().item()
            results[mode_name]["total_spread"].append(total_spread)

            # Cross-cell correlation and effective rank at each horizon, then average
            eff_ranks = []
            pc1_doms = []
            corrs = []
            mid_horizon = T // 2  # horizon 15

            for t in range(T):
                members_t = samples[:, t, :]  # (n_samples, 25)
                er, pc1 = compute_effective_rank(members_t)
                eff_ranks.append(er)
                pc1_doms.append(pc1)
                corrs.append(compute_cross_cell_correlation(members_t))

                # Save PC directions at mid-horizon for subspace comparison
                if t == mid_horizon:
                    centered = members_t - members_t.mean(dim=0, keepdim=True)
                    try:
                        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                        # Top 3 right singular vectors (directions in cell-space)
                        pc_dirs = Vh[:3, :].cpu().numpy()  # (3, 25)
                    except Exception:
                        pc_dirs = np.zeros((3, 25))
                    pc_directions[mode_name].append(pc_dirs)

            results[mode_name]["effective_rank"].append(np.mean(eff_ranks))
            results[mode_name]["pc1_dominance"].append(np.mean(pc1_doms))
            results[mode_name]["cross_cell_corr"].append(np.mean(corrs))

    elapsed = time.time() - t0
    print(f"\nAll ablations complete in {elapsed:.1f}s")

    # --- Aggregate results ---
    summary = {}
    for mode_name in ablation_modes:
        r = results[mode_name]
        summary[mode_name] = {
            "mean_total_spread": float(np.mean(r["total_spread"])),
            "std_total_spread": float(np.std(r["total_spread"])),
            "mean_effective_rank": float(np.mean(r["effective_rank"])),
            "std_effective_rank": float(np.std(r["effective_rank"])),
            "mean_pc1_dominance": float(np.mean(r["pc1_dominance"])),
            "std_pc1_dominance": float(np.std(r["pc1_dominance"])),
            "mean_cross_cell_corr": float(np.mean(r["cross_cell_corr"])),
            "std_cross_cell_corr": float(np.std(r["cross_cell_corr"])),
            "per_cell_spread_mean": np.mean(r["per_cell_spread"], axis=0).tolist(),
            "per_horizon_spread_mean": np.mean(r["per_horizon_spread"], axis=0).tolist(),
        }

    # --- Subspace comparison: CLN-only vs Factor-only ---
    # Measure alignment of top PC directions between CLN-only and factor-only
    subspace_alignment = []
    for wi in range(n_windows):
        pc_cln = pc_directions["cln_only"][wi]    # (3, 25)
        pc_fac = pc_directions["factor_only"][wi]  # (3, 25)

        # Compute cosine similarity matrix between top-3 PCs
        # Normalize rows
        pc_cln_n = pc_cln / (np.linalg.norm(pc_cln, axis=1, keepdims=True) + 1e-10)
        pc_fac_n = pc_fac / (np.linalg.norm(pc_fac, axis=1, keepdims=True) + 1e-10)
        cos_sim = np.abs(pc_cln_n @ pc_fac_n.T)  # (3, 3)

        # Principal angle: max alignment of PC1_cln with any factor PC
        pc1_max_align = float(cos_sim[0, :].max())
        # Average of max alignments for top-3
        avg_max_align = float(np.mean([cos_sim[i, :].max() for i in range(3)]))
        # Subspace overlap: Frobenius norm of cross-correlation
        subspace_overlap = float(np.linalg.norm(cos_sim, 'fro') / 3.0)

        subspace_alignment.append({
            "pc1_max_align": pc1_max_align,
            "avg_max_align": avg_max_align,
            "subspace_overlap": subspace_overlap,
        })

    summary["subspace_comparison"] = {
        "description": "Alignment of top-3 PC directions between CLN-only and factor-only at horizon 15",
        "mean_pc1_max_align": float(np.mean([s["pc1_max_align"] for s in subspace_alignment])),
        "mean_avg_max_align": float(np.mean([s["avg_max_align"] for s in subspace_alignment])),
        "mean_subspace_overlap": float(np.mean([s["subspace_overlap"] for s in subspace_alignment])),
        "per_window": subspace_alignment,
    }

    # --- Spread decomposition ---
    both_spread = summary["both"]["mean_total_spread"]
    cln_spread = summary["cln_only"]["mean_total_spread"]
    fac_spread = summary["factor_only"]["mean_total_spread"]
    neither_spread = summary["neither"]["mean_total_spread"]

    summary["spread_decomposition"] = {
        "both_spread": both_spread,
        "cln_only_spread": cln_spread,
        "factor_only_spread": fac_spread,
        "neither_spread": neither_spread,
        "cln_contribution_pct": float((cln_spread - neither_spread) / (both_spread - neither_spread + 1e-10) * 100),
        "factor_contribution_pct": float((fac_spread - neither_spread) / (both_spread - neither_spread + 1e-10) * 100),
        "interaction_pct": float(
            100 - (cln_spread - neither_spread + fac_spread - neither_spread) /
            (both_spread - neither_spread + 1e-10) * 100
        ),
    }

    # --- Key diagnostic: does factor path produce HIGHER rank when CLN is off? ---
    factor_only_rank = summary["factor_only"]["mean_effective_rank"]
    both_rank = summary["both"]["mean_effective_rank"]
    cln_only_rank = summary["cln_only"]["mean_effective_rank"]

    summary["key_question"] = {
        "question": "When CLN is removed (z=0), does factor path alone produce HIGHER rank diversity?",
        "factor_only_rank": factor_only_rank,
        "both_rank": both_rank,
        "cln_only_rank": cln_only_rank,
        "factor_rank_minus_both": float(factor_only_rank - both_rank),
        "answer": (
            "YES: CLN competes with factor path" if factor_only_rank > both_rank + 0.5
            else "NO: Factor path itself produces low-rank output (problem is in L itself)"
            if factor_only_rank < both_rank - 0.5
            else "MARGINAL: Ranks are similar (within 0.5)"
        ),
    }

    # Also compare PC1 dominance
    summary["key_question"]["factor_only_pc1"] = summary["factor_only"]["mean_pc1_dominance"]
    summary["key_question"]["both_pc1"] = summary["both"]["mean_pc1_dominance"]
    summary["key_question"]["cln_only_pc1"] = summary["cln_only"]["mean_pc1_dominance"]

    # --- Print report ---
    print("\n" + "="*80)
    print("167a CLN vs Factor Path Ablation — Summary")
    print("="*80)

    for mode in ["both", "cln_only", "factor_only", "neither"]:
        s = summary[mode]
        print(f"\n{mode.upper():15s}: spread={s['mean_total_spread']:.6f}  "
              f"eff_rank={s['mean_effective_rank']:.2f}  "
              f"pc1={s['mean_pc1_dominance']:.3f}  "
              f"corr={s['mean_cross_cell_corr']:.3f}")

    print(f"\n--- Spread Decomposition ---")
    sd = summary["spread_decomposition"]
    print(f"  CLN contribution:    {sd['cln_contribution_pct']:.1f}%")
    print(f"  Factor contribution: {sd['factor_contribution_pct']:.1f}%")
    print(f"  Interaction:         {sd['interaction_pct']:.1f}%")

    print(f"\n--- Key Question ---")
    kq = summary["key_question"]
    print(f"  Factor-only rank:    {kq['factor_only_rank']:.2f}")
    print(f"  Both rank:           {kq['both_rank']:.2f}")
    print(f"  CLN-only rank:       {kq['cln_only_rank']:.2f}")
    print(f"  Delta (fac - both):  {kq['factor_rank_minus_both']:.2f}")
    print(f"  ANSWER: {kq['answer']}")

    print(f"\n--- Subspace Alignment (CLN vs Factor PCs) ---")
    sc = summary["subspace_comparison"]
    print(f"  PC1 max alignment:   {sc['mean_pc1_max_align']:.3f}")
    print(f"  Avg max alignment:   {sc['mean_avg_max_align']:.3f}")
    print(f"  Subspace overlap:    {sc['mean_subspace_overlap']:.3f}")
    print(f"  (1.0 = identical subspaces, 0.0 = orthogonal)")

    # Per-cell spread heatmap (text)
    print(f"\n--- Per-cell Spread (5x5 grid, both pathway) ---")
    both_pcs = np.array(summary["both"]["per_cell_spread_mean"]).reshape(5, 5)
    for row in range(5):
        print("  " + "  ".join(f"{both_pcs[row, c]:.4f}" for c in range(5)))

    print(f"\n--- Per-cell Spread (5x5 grid, factor-only) ---")
    fac_pcs = np.array(summary["factor_only"]["per_cell_spread_mean"]).reshape(5, 5)
    for row in range(5):
        print("  " + "  ".join(f"{fac_pcs[row, c]:.4f}" for c in range(5)))

    # --- Save results ---
    out_dir = Path("results/validations/2026-04-04")

    # Full results
    results_path = out_dir / "analysis" / "167a_gradient" / "cln_ablation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Verification summary
    verification = {
        "experiment": "167a_cln_ablation",
        "checkpoint": ckpt_path,
        "epoch": ckpt["epoch"],
        "n_windows": n_windows,
        "n_samples": n_samples,
        "test_split_start": test_start,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "key_finding": kq["answer"],
        "spread_decomposition": sd,
        "effective_ranks": {
            "both": both_rank,
            "cln_only": cln_only_rank,
            "factor_only": factor_only_rank,
            "neither": summary["neither"]["mean_effective_rank"],
        },
        "pc1_dominance": {
            "both": kq["both_pc1"],
            "cln_only": kq["cln_only_pc1"],
            "factor_only": kq["factor_only_pc1"],
            "neither": summary["neither"]["mean_pc1_dominance"],
        },
        "subspace_alignment": {
            "pc1_max_align": sc["mean_pc1_max_align"],
            "avg_max_align": sc["mean_avg_max_align"],
            "overlap": sc["mean_subspace_overlap"],
        },
    }
    verif_path = out_dir / "verification_results" / "167a_cln_ablation.json"
    with open(verif_path, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"Verification saved to: {verif_path}")


if __name__ == "__main__":
    main()
