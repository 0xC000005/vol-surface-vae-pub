#!/usr/bin/env python
"""
164a Post-training diagnostics (RC20 Stage 3).

7 diagnostics as specified in RC20 compass:
1. Per-cell spread ratio grid (5x5)
2. Per-cell kurtosis grid (5x5)
3. Within-window effective rank
4. tanh hit rate
5. Reflecting boundary hit rate
6. turb/calm from full 1252 windows
7. Per-cell cointegration

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_164a.py \
        --model_path models/backfill/afcrps_164a/best_model.pt \
        --device cuda
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import kurtosis

import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_164a_ar_spatial import (
    ARSpatialTransformerModel, normalize_iv, denormalize_iv, reflecting_boundary,
)
from diffusion.block_ar.gru_encoder import EncoderConfig


def load_model(path, device="cuda"):
    cp = torch.load(path, map_location=device, weights_only=False)
    cfg = cp["config"]
    enc_cfg = EncoderConfig(**cfg["encoder"])
    model = ARSpatialTransformerModel(enc_cfg, cfg["decoder"]).to(device)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, cfg


def generate_samples(model, surfaces, indices, n_samples=50, device="cuda"):
    """Generate samples for given indices. Returns (N, K, T, C) in [0,1]."""
    H, T, C = 30, 30, 25
    all_samples = []

    with torch.no_grad():
        for idx in indices:
            hist = torch.from_numpy(
                surfaces[idx:idx+H][None].astype(np.float32)
            ).to(device)
            hist_norm = normalize_iv(hist)

            samples = model.sample_batched(hist_norm, n_samples=n_samples)
            # (1, K, T, 5, 5) → (K, T, C)
            all_samples.append(samples[0].reshape(n_samples, T, C).cpu().numpy())

    return np.array(all_samples)  # (N, K, T, C)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    model, cfg = load_model(args.model_path, device)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    ret = data["ret"]
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25

    TEST_START = 4511
    test_indices = np.arange(TEST_START, N_total - H - T + 1)
    print(f"Test windows: {len(test_indices)}")

    # ── Generate samples for diagnostics ──
    # Use 200 windows for diagnostics (fast), full 1252 for turb/calm
    diag_indices = test_indices[:200]
    print(f"\nGenerating samples for {len(diag_indices)} diagnostic windows...")
    samples = generate_samples(model, surfaces, diag_indices,
                               n_samples=args.n_samples, device=device)
    gt = np.array([surfaces[i+H:i+H+T].reshape(T, C) for i in diag_indices])
    print(f"  Samples shape: {samples.shape}")

    # ═══ Diagnostic 1: Per-cell spread ratio grid ═══
    print("\n" + "="*60)
    print("D1: Per-cell spread ratio (ensemble_std / GT_std)")
    print("="*60)
    gen_std = samples.std(axis=1).mean(axis=(0, 1))  # (C,) average over windows and time
    # GT spread: std of daily changes across windows
    gt_changes = np.diff(gt, axis=1)  # (N, T-1, C)
    gt_std = gt_changes.std(axis=(0, 1))  # (C,)
    spread_ratio = gen_std / (gt_std + 1e-8)
    spread_grid = spread_ratio.reshape(5, 5)
    print("  Spread ratio grid (5x5):")
    for row in range(5):
        cells = "  ".join(f"{spread_grid[row, c]:.2f}" for c in range(5))
        print(f"    [{cells}]")
    print(f"  Range: [{spread_ratio.min():.2f}, {spread_ratio.max():.2f}]")
    print(f"  Target: range < 3x")

    # ═══ Diagnostic 2: Per-cell kurtosis grid ═══
    print("\n" + "="*60)
    print("D2: Per-cell kurtosis ratio (gen / GT)")
    print("="*60)
    gen_changes = np.diff(samples[:, 0], axis=1)  # (N, T-1, C) — first member
    gen_kurt = np.array([kurtosis(gen_changes[:, :, c].flatten()) for c in range(C)])
    gt_kurt = np.array([kurtosis(gt_changes[:, :, c].flatten()) for c in range(C)])
    kurt_ratio = gen_kurt / (gt_kurt + 1e-8)
    kurt_grid = kurt_ratio.reshape(5, 5)
    print("  Kurtosis ratio grid (5x5):")
    for row in range(5):
        cells = "  ".join(f"{kurt_grid[row, c]:.2f}" for c in range(5))
        print(f"    [{cells}]")
    in_range = ((kurt_ratio >= 0.5) & (kurt_ratio <= 2.0)).sum()
    print(f"  In [0.5, 2.0]: {in_range}/25")
    print(f"  Aggregate: {gen_kurt.mean():.3f} / {gt_kurt.mean():.3f} = {gen_kurt.mean() / (gt_kurt.mean() + 1e-8):.3f}")

    # ═══ Diagnostic 3: Within-window effective rank ═══
    print("\n" + "="*60)
    print("D3: Within-window effective rank")
    print("="*60)
    eff_ranks = []
    for i in range(min(200, len(diag_indices))):
        # Each window: (K, T*C) reshaped
        member_vecs = samples[i].reshape(args.n_samples, -1)  # (K, T*C)
        member_vecs -= member_vecs.mean(axis=0, keepdims=True)
        try:
            svd = np.linalg.svd(member_vecs, compute_uv=False)
            p = svd / svd.sum()
            p = p[p > 1e-10]
            eff_rank = np.exp(-np.sum(p * np.log(p)))
            eff_ranks.append(eff_rank)
        except:
            pass
    print(f"  Mean eff_rank: {np.mean(eff_ranks):.2f}")
    print(f"  Std: {np.std(eff_ranks):.2f}")
    print(f"  Target: > 2.5 (GT ~ 5.0, 161a = 1.78, 163a = 2.68)")

    # ═══ Diagnostic 4: tanh hit rate ═══
    print("\n" + "="*60)
    print("D4: tanh hit rate")
    print("="*60)
    # Estimate: run a few forward passes and check delta magnitudes
    n_check = 50
    check_idx = test_indices[:n_check]
    tanh_hits = 0; total_deltas = 0
    with torch.no_grad():
        for idx in check_idx:
            hist = torch.from_numpy(
                surfaces[idx:idx+H][None].astype(np.float32)
            ).to(device)
            hist_norm = normalize_iv(hist)

            # Manual AR to inspect deltas
            cond = model.encode(hist_norm)
            last_frame = hist[0, -1].reshape(1, C)
            prev = last_frame

            for t in range(T):
                z = torch.randn(1, model.decoder.noise_dim, device=device)
                delta = model.decoder(cond, prev, z)
                # tanh saturation: |delta| > 2.0 means tanh is > 0.96
                hits = (delta.abs() > 2.0).sum().item()
                tanh_hits += hits
                total_deltas += delta.numel()
                frame = prev + torch.tanh(delta)
                frame = reflecting_boundary(frame, 0.01, 1.0)
                prev = frame

    hit_rate = tanh_hits / max(total_deltas, 1)
    print(f"  tanh hit rate (|delta| > 2.0): {hit_rate:.4%}")
    print(f"  ({tanh_hits}/{total_deltas})")
    print(f"  Target: < 0.1% (model learned bounds internally)")

    # ═══ Diagnostic 5: Reflecting boundary hit rate ═══
    print("\n" + "="*60)
    print("D5: Reflecting boundary hit rate")
    print("="*60)
    boundary_hits = 0; total_frames = 0
    with torch.no_grad():
        for idx in check_idx:
            hist = torch.from_numpy(
                surfaces[idx:idx+H][None].astype(np.float32)
            ).to(device)
            hist_norm = normalize_iv(hist)
            cond = model.encode(hist_norm)
            last_frame = hist[0, -1].reshape(1, C)
            prev = last_frame
            for t in range(T):
                z = torch.randn(1, model.decoder.noise_dim, device=device)
                delta = model.decoder(cond, prev, z)
                frame_raw = prev + torch.tanh(delta)
                out_of_bounds = ((frame_raw < 0.01) | (frame_raw > 1.0)).sum().item()
                boundary_hits += out_of_bounds
                total_frames += frame_raw.numel()
                prev = reflecting_boundary(frame_raw, 0.01, 1.0)

    bound_rate = boundary_hits / max(total_frames, 1)
    print(f"  Boundary hit rate: {bound_rate:.4%}")
    print(f"  ({boundary_hits}/{total_frames})")
    print(f"  Target: < 0.1%")

    # ═══ Diagnostic 6: turb/calm from full 1252 windows ═══
    print("\n" + "="*60)
    print("D6: turb/calm ratio (full 1252 windows)")
    print("="*60)
    print(f"  Generating samples for all {len(test_indices)} test windows...")
    full_samples = generate_samples(model, surfaces, test_indices,
                                    n_samples=args.n_samples, device=device)
    # Spread per window: std over members, averaged over time and cells
    spreads = full_samples.std(axis=1).mean(axis=(1, 2))  # (N,)

    # 30-day realized vol from returns
    vol_30d = np.array([
        np.std(ret[max(0, i-30):i]) if i >= 30 else np.std(ret[:i+1])
        for i in test_indices
    ])
    q75 = np.percentile(vol_30d, 75)
    q25 = np.percentile(vol_30d, 25)
    turb_mask = vol_30d >= q75
    calm_mask = vol_30d <= q25

    if turb_mask.sum() > 5 and calm_mask.sum() > 5:
        turb_spread = spreads[turb_mask].mean()
        calm_spread = spreads[calm_mask].mean()
        turb_calm = turb_spread / (calm_spread + 1e-8)
        print(f"  Turbulent spread: {turb_spread:.4f} ({turb_mask.sum()} windows)")
        print(f"  Calm spread: {calm_spread:.4f} ({calm_mask.sum()} windows)")
        print(f"  turb/calm ratio: {turb_calm:.3f}")
        print(f"  Target: > 1.15 (99m_v2 = 1.49, 161a = 1.209)")
    else:
        turb_calm = None
        print("  Not enough turb/calm windows")

    # ═══ Diagnostic 7: Per-cell cointegration ═══
    print("\n" + "="*60)
    print("D7: Per-cell cointegration")
    print("="*60)
    from statsmodels.tsa.stattools import adfuller
    coint_pass = 0
    for c in range(C):
        # Test: gen mean trajectory - GT should be stationary
        gen_mean = full_samples[:, :, :, c].mean(axis=1)  # (N, T) — mean over members
        gt_full = np.array([surfaces[i+H:i+H+T].reshape(T, C)[:, c] for i in test_indices])
        residuals = gen_mean - gt_full  # (N, T)
        # Flatten and test stationarity
        flat_resid = residuals.flatten()
        try:
            result = adfuller(flat_resid, maxlag=5)
            if result[1] < 0.05:  # p < 0.05 → stationary → cointegrated
                coint_pass += 1
        except:
            pass

    coint_ratio = coint_pass / C
    print(f"  Cointegration pass: {coint_pass}/25 cells")
    print(f"  Ratio: {coint_ratio:.3f}")
    print(f"  Target: > 0.50 (99m_v2 = PASS, 161a = 0.431)")

    # ═══ Summary ═══
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"  D1 Spread ratio range: [{spread_ratio.min():.2f}, {spread_ratio.max():.2f}] (target: < 3x)")
    print(f"  D2 Kurtosis in range: {in_range}/25 (target: > 15)")
    print(f"  D3 Eff rank: {np.mean(eff_ranks):.2f} (target: > 2.5)")
    print(f"  D4 tanh hit rate: {hit_rate:.4%} (target: < 0.1%)")
    print(f"  D5 Boundary hit rate: {bound_rate:.4%} (target: < 0.1%)")
    print(f"  D6 turb/calm: {turb_calm:.3f}" if turb_calm else "  D6 turb/calm: N/A")
    print(f"  D7 Cointegration: {coint_pass}/25 = {coint_ratio:.3f} (target: > 0.50)")

    # Save results
    output_dir = Path(args.model_path).parent
    results = {
        "spread_ratio_range": [float(spread_ratio.min()), float(spread_ratio.max())],
        "spread_ratio_grid": spread_grid.tolist(),
        "kurtosis_in_range": int(in_range),
        "kurtosis_grid": kurt_grid.tolist(),
        "eff_rank_mean": float(np.mean(eff_ranks)),
        "tanh_hit_rate": float(hit_rate),
        "boundary_hit_rate": float(bound_rate),
        "turb_calm": float(turb_calm) if turb_calm else None,
        "cointegration_pass": int(coint_pass),
        "cointegration_ratio": float(coint_ratio),
    }
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
