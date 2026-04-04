#!/usr/bin/env python
"""
Consistent Mean Reversion Measurement Across RC22 Models

Methodology (applied identically to every model):
1. Load 200 test windows (start index 4540)
2. For each window, run encoder to get condition
3. Run decoder for FIRST STEP ONLY: prev_frame -> delta
4. For factorized models: delta = delta_base (no L@eps, to isolate centering)
5. For baseline: delta = decoder(cond, prev, noise=zeros) (deterministic, no noise)
6. Regress delta vs prev_frame across all 200*25=5000 data points -> slope
7. Also compute GT slope: (future_frame_1 - last_frame) vs last_frame

Additional: Measure WITH L@eps (avg over 50 noise samples) for 167b and 167d.

Output:
- results/validations/2026-04-04/analysis/167d_followup/consistent_mean_reversion.json
- results/validations/2026-04-04/verification_results/consistent_mean_reversion.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")


def normalize_iv(surfaces):
    return surfaces * 2.0 - 1.0


def denormalize_iv(surfaces):
    return (surfaces + 1.0) / 2.0


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def regress(x, y):
    """OLS regression y = a + b*x. Returns slope, intercept, R^2."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0
    x_bar = x.mean()
    y_bar = y.mean()
    ss_xx = ((x - x_bar) ** 2).sum()
    ss_xy = ((x - x_bar) * (y - y_bar)).sum()
    if ss_xx < 1e-15:
        return 0.0, y_bar, 0.0
    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar
    y_hat = slope * x + intercept
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_bar) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return float(slope), float(intercept), float(r2)


def regress_per_cell(prev_flat, delta_flat, n_cells=25):
    """Per-cell regression. prev_flat: (N, 25), delta_flat: (N, 25)."""
    slopes = np.zeros(n_cells)
    r2s = np.zeros(n_cells)
    for c in range(n_cells):
        s, _, r = regress(prev_flat[:, c], delta_flat[:, c])
        slopes[c] = s
        r2s[c] = r
    return slopes, r2s


def load_data(data_path, start_idx=4540, n_windows=200):
    """Load test windows: history (30 frames) + future (30 frames)."""
    data = np.load(data_path)
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    N = len(surfaces)

    histories = []
    futures = []
    for i in range(start_idx, min(start_idx + n_windows, N - 60)):
        hist = surfaces[i:i + 30]      # (30, 5, 5) in [0, 1]
        fut = surfaces[i + 30:i + 60]  # (30, 5, 5) in [0, 1]
        histories.append(hist)
        futures.append(fut)

    histories = np.stack(histories)  # (W, 30, 5, 5)
    futures = np.stack(futures)      # (W, 30, 5, 5)
    return histories, futures


# ============================================================================
# Model loading — each model uses its own class from its training script
# ============================================================================

def load_baseline(model_path, device):
    """Load 164a baseline (ARSpatialTransformerModel)."""
    from experiments.backfill.block_ar.train_164a_v3_percell_bptt_softplus import (
        ARSpatialTransformerModel,
    )
    from diffusion.block_ar.gru_encoder import EncoderConfig

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", ckpt.get("model_config", {}))

    enc_cfg = EncoderConfig(
        input_dim=cfg.get("input_dim", 25),
        gru_hidden_dim=cfg.get("gru_hidden_dim", 64),
        bottleneck_dim=cfg.get("bottleneck_dim", 128),
    )
    dec_cfg = {
        "n_cells": 25,
        "d_model": cfg.get("d_model", 128),
        "n_heads": cfg.get("n_heads", 4),
        "n_layers": cfg.get("n_layers", 4),
        "cond_dim": cfg.get("bottleneck_dim", 128),
        "noise_dim": cfg.get("noise_dim", 32),
    }
    model = ARSpatialTransformerModel(enc_cfg, dec_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_factorized(model_path, train_script_module, device):
    """Load 167b or 167d (ARFactorizedCleanModel)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", ckpt.get("model_config", {}))

    from diffusion.block_ar.gru_encoder import EncoderConfig

    enc_cfg = EncoderConfig(
        input_dim=cfg.get("input_dim", 25),
        gru_hidden_dim=cfg.get("gru_hidden_dim", 64),
        bottleneck_dim=cfg.get("bottleneck_dim", 128),
    )
    dec_cfg = {
        "n_cells": 25,
        "d_model": cfg.get("d_model", 128),
        "n_heads": cfg.get("n_heads", 4),
        "n_layers": cfg.get("n_layers", 4),
        "cond_dim": cfg.get("bottleneck_dim", 128),
        "noise_dim": cfg.get("noise_dim", 32),
    }
    n_factors = cfg.get("n_factors", 5)
    model = train_script_module.ARFactorizedCleanModel(enc_cfg, dec_cfg, n_factors=n_factors)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


# ============================================================================
# First-step delta extraction
# ============================================================================

def encode_condition(model, history_norm, device):
    """Run encoder to get condition vector. history_norm: (B, 30, 5, 5) in [-1,1]."""
    B = history_norm.shape[0]
    hist_flat = history_norm.reshape(B, 30, 25)
    gru_outputs, h_last = model.encoder.gru(hist_flat)
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    cond = model.encoder.bottleneck(h_pooled)
    return cond


@torch.no_grad()
def get_delta_baseline(model, histories, device, batch_size=50):
    """Get first-step delta for baseline (noise=zeros, deterministic)."""
    W = len(histories)
    all_delta = []
    all_prev = []
    for start in range(0, W, batch_size):
        end = min(start + batch_size, W)
        hist_raw = torch.tensor(histories[start:end], dtype=torch.float32, device=device)
        hist_norm = normalize_iv(hist_raw)  # [-1, 1]
        last_frame = hist_raw[:, -1].reshape(-1, 25)  # [0, 1]

        cond = encode_condition(model, hist_norm, device)
        noise = torch.zeros(end - start, model.decoder.noise_dim, device=device)
        delta = model.decoder(cond, last_frame, noise)  # (B, 25) raw delta
        delta = torch.tanh(delta)  # apply tanh as in ar_generate

        all_delta.append(delta.cpu().numpy())
        all_prev.append(last_frame.cpu().numpy())

    return np.concatenate(all_delta), np.concatenate(all_prev)


@torch.no_grad()
def get_delta_factorized_base_only(model, histories, device, batch_size=50):
    """Get first-step delta_base for factorized model (no L@eps)."""
    W = len(histories)
    all_delta = []
    all_prev = []
    for start in range(0, W, batch_size):
        end = min(start + batch_size, W)
        hist_raw = torch.tensor(histories[start:end], dtype=torch.float32, device=device)
        hist_norm = normalize_iv(hist_raw)
        last_frame = hist_raw[:, -1].reshape(-1, 25)

        cond = encode_condition(model, hist_norm, device)
        noise = torch.zeros(end - start, model.decoder.noise_dim, device=device)
        delta_base, L = model.decoder(cond, last_frame, noise)
        delta_base = torch.tanh(delta_base)  # apply tanh as in ar_generate

        all_delta.append(delta_base.cpu().numpy())
        all_prev.append(last_frame.cpu().numpy())

    return np.concatenate(all_delta), np.concatenate(all_prev)


@torch.no_grad()
def get_delta_factorized_with_L(model, histories, device, n_noise=50, batch_size=50):
    """Get first-step delta WITH L@eps, averaged over n_noise samples."""
    W = len(histories)
    all_delta = []
    all_prev = []
    for start in range(0, W, batch_size):
        end = min(start + batch_size, W)
        B = end - start
        hist_raw = torch.tensor(histories[start:end], dtype=torch.float32, device=device)
        hist_norm = normalize_iv(hist_raw)
        last_frame = hist_raw[:, -1].reshape(B, 25)

        cond = encode_condition(model, hist_norm, device)

        # Average over n_noise noise samples
        delta_sum = torch.zeros(B, 25, device=device)
        for _ in range(n_noise):
            noise = torch.randn(B, model.decoder.noise_dim, device=device)
            delta_base, L = model.decoder(cond, last_frame, noise)
            eps = torch.randn(B, model.n_factors, device=device)
            delta_full = delta_base + torch.einsum("bcr,br->bc", L, eps)
            delta_full = torch.tanh(delta_full)
            delta_sum += delta_full
        delta_avg = delta_sum / n_noise

        all_delta.append(delta_avg.cpu().numpy())
        all_prev.append(last_frame.cpu().numpy())

    return np.concatenate(all_delta), np.concatenate(all_prev)


def compute_gt_delta(histories, futures):
    """GT first-step: future[0] - history[-1]."""
    last_frame = histories[:, -1].reshape(-1, 25)
    first_future = futures[:, 0].reshape(-1, 25)
    delta = first_future - last_frame
    return delta, last_frame


def format_grid(values, shape=(5, 5)):
    """Format 25 values as 5x5 grid."""
    return np.array(values).reshape(shape).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_windows", type=int, default=200)
    parser.add_argument("--start_idx", type=int, default=4540)
    parser.add_argument("--n_noise", type=int, default=50,
                        help="Noise samples for L@eps averaging")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = "data/vol_surface_with_ret.npz"

    print(f"Loading data from {data_path}, start={args.start_idx}, n_windows={args.n_windows}")
    histories, futures = load_data(data_path, args.start_idx, args.n_windows)
    W = len(histories)
    print(f"Loaded {W} windows, {W * 25} data points")

    results = {
        "methodology": {
            "description": "First-step-only mean reversion measurement",
            "n_windows": W,
            "start_idx": args.start_idx,
            "n_data_points": W * 25,
            "n_noise_for_L": args.n_noise,
            "steps": [
                "1. Load test windows from data",
                "2. Encode history -> condition vector",
                "3. First step only: prev_frame -> delta (NO teacher forcing beyond step 0)",
                "4. Baseline: noise=zeros (deterministic). Factorized: delta_base only.",
                "5. Regress delta vs prev_frame -> slope (mean reversion coefficient)",
                "6. GT: (future[0] - history[-1]) vs history[-1]",
            ],
        },
        "models": {},
    }

    # ── GT measurement ────────────────────────────────────────────────────
    print("\n=== GT Mean Reversion ===")
    gt_delta, gt_prev = compute_gt_delta(histories, futures)
    gt_agg_slope, gt_agg_intercept, gt_agg_r2 = regress(gt_prev, gt_delta)
    gt_cell_slopes, gt_cell_r2s = regress_per_cell(gt_prev, gt_delta)

    results["models"]["GT"] = {
        "aggregate_slope": gt_agg_slope,
        "aggregate_intercept": gt_agg_intercept,
        "aggregate_r2": gt_agg_r2,
        "cell_slopes_5x5": format_grid(gt_cell_slopes),
        "cell_r2_5x5": format_grid(gt_cell_r2s),
        "cell_slope_range": [float(gt_cell_slopes.min()), float(gt_cell_slopes.max())],
        "cell_slope_mean": float(gt_cell_slopes.mean()),
    }
    print(f"  Aggregate slope: {gt_agg_slope:.4f}, R2: {gt_agg_r2:.4f}")
    print(f"  Cell slope range: [{gt_cell_slopes.min():.4f}, {gt_cell_slopes.max():.4f}]")

    # ── Baseline (164a) ───────────────────────────────────────────────────
    print("\n=== Baseline (164a_v3_percell_bptt_softplus) ===")
    baseline_path = "models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt"
    baseline_model = load_baseline(baseline_path, device)
    bl_delta, bl_prev = get_delta_baseline(baseline_model, histories, device)
    bl_agg_slope, bl_agg_intercept, bl_agg_r2 = regress(bl_prev, bl_delta)
    bl_cell_slopes, bl_cell_r2s = regress_per_cell(bl_prev, bl_delta)

    results["models"]["baseline_164a"] = {
        "model_path": baseline_path,
        "method": "noise=zeros, deterministic delta = tanh(decoder(cond, prev, zeros))",
        "aggregate_slope": bl_agg_slope,
        "aggregate_intercept": bl_agg_intercept,
        "aggregate_r2": bl_agg_r2,
        "mr_gt_ratio": bl_agg_slope / gt_agg_slope if abs(gt_agg_slope) > 1e-8 else None,
        "cell_slopes_5x5": format_grid(bl_cell_slopes),
        "cell_r2_5x5": format_grid(bl_cell_r2s),
        "cell_slope_range": [float(bl_cell_slopes.min()), float(bl_cell_slopes.max())],
        "cell_slope_mean": float(bl_cell_slopes.mean()),
    }
    print(f"  Aggregate slope: {bl_agg_slope:.4f}, R2: {bl_agg_r2:.4f}")
    print(f"  MR/GT ratio: {bl_agg_slope / gt_agg_slope:.3f}")
    print(f"  Cell slope range: [{bl_cell_slopes.min():.4f}, {bl_cell_slopes.max():.4f}]")
    del baseline_model
    torch.cuda.empty_cache()

    # ── 167b (CLN frozen) ─────────────────────────────────────────────────
    print("\n=== 167b (CLN frozen, factorized) ===")
    import experiments.backfill.block_ar.train_167b_clean_isolation as mod_167b

    model_167b_path = "models/backfill/afcrps_167b/best_model.pt"
    model_167b = load_factorized(model_167b_path, mod_167b, device)

    # Base only
    b_delta, b_prev = get_delta_factorized_base_only(model_167b, histories, device)
    b_agg_slope, b_agg_intercept, b_agg_r2 = regress(b_prev, b_delta)
    b_cell_slopes, b_cell_r2s = regress_per_cell(b_prev, b_delta)

    results["models"]["167b_base_only"] = {
        "model_path": model_167b_path,
        "method": "delta_base only (no L@eps), noise=zeros, tanh applied",
        "aggregate_slope": b_agg_slope,
        "aggregate_intercept": b_agg_intercept,
        "aggregate_r2": b_agg_r2,
        "mr_gt_ratio": b_agg_slope / gt_agg_slope if abs(gt_agg_slope) > 1e-8 else None,
        "cell_slopes_5x5": format_grid(b_cell_slopes),
        "cell_r2_5x5": format_grid(b_cell_r2s),
        "cell_slope_range": [float(b_cell_slopes.min()), float(b_cell_slopes.max())],
        "cell_slope_mean": float(b_cell_slopes.mean()),
    }
    print(f"  Base-only aggregate slope: {b_agg_slope:.4f}, R2: {b_agg_r2:.4f}")
    print(f"  MR/GT ratio: {b_agg_slope / gt_agg_slope:.3f}")

    # With L@eps
    print(f"  Computing with L@eps ({args.n_noise} noise samples)...")
    bl_delta, bl_prev = get_delta_factorized_with_L(model_167b, histories, device, args.n_noise)
    bl_agg_slope, bl_agg_intercept, bl_agg_r2 = regress(bl_prev, bl_delta)
    bl_cell_slopes, bl_cell_r2s = regress_per_cell(bl_prev, bl_delta)

    results["models"]["167b_with_L"] = {
        "model_path": model_167b_path,
        "method": f"delta_base + L@eps, averaged over {args.n_noise} noise samples, tanh applied",
        "aggregate_slope": bl_agg_slope,
        "aggregate_intercept": bl_agg_intercept,
        "aggregate_r2": bl_agg_r2,
        "mr_gt_ratio": bl_agg_slope / gt_agg_slope if abs(gt_agg_slope) > 1e-8 else None,
        "cell_slopes_5x5": format_grid(bl_cell_slopes),
        "cell_r2_5x5": format_grid(bl_cell_r2s),
        "cell_slope_range": [float(bl_cell_slopes.min()), float(bl_cell_slopes.max())],
        "cell_slope_mean": float(bl_cell_slopes.mean()),
        "L_effect_on_slope": bl_agg_slope - b_agg_slope,
    }
    print(f"  With-L aggregate slope: {bl_agg_slope:.4f}, R2: {bl_agg_r2:.4f}")
    print(f"  MR/GT ratio: {bl_agg_slope / gt_agg_slope:.3f}")
    print(f"  L effect on slope: {bl_agg_slope - b_agg_slope:+.4f}")
    del model_167b
    torch.cuda.empty_cache()

    # ── 167d (CLN active) ─────────────────────────────────────────────────
    print("\n=== 167d (CLN active, factorized) ===")
    import experiments.backfill.block_ar.train_167d_e2e_factorized as mod_167d

    model_167d_path = "models/backfill/afcrps_167d/best_model.pt"
    model_167d = load_factorized(model_167d_path, mod_167d, device)

    # Base only
    d_delta, d_prev = get_delta_factorized_base_only(model_167d, histories, device)
    d_agg_slope, d_agg_intercept, d_agg_r2 = regress(d_prev, d_delta)
    d_cell_slopes, d_cell_r2s = regress_per_cell(d_prev, d_delta)

    results["models"]["167d_base_only"] = {
        "model_path": model_167d_path,
        "method": "delta_base only (no L@eps), noise=zeros, tanh applied",
        "aggregate_slope": d_agg_slope,
        "aggregate_intercept": d_agg_intercept,
        "aggregate_r2": d_agg_r2,
        "mr_gt_ratio": d_agg_slope / gt_agg_slope if abs(gt_agg_slope) > 1e-8 else None,
        "cell_slopes_5x5": format_grid(d_cell_slopes),
        "cell_r2_5x5": format_grid(d_cell_r2s),
        "cell_slope_range": [float(d_cell_slopes.min()), float(d_cell_slopes.max())],
        "cell_slope_mean": float(d_cell_slopes.mean()),
    }
    print(f"  Base-only aggregate slope: {d_agg_slope:.4f}, R2: {d_agg_r2:.4f}")
    print(f"  MR/GT ratio: {d_agg_slope / gt_agg_slope:.3f}")

    # With L@eps
    print(f"  Computing with L@eps ({args.n_noise} noise samples)...")
    dl_delta, dl_prev = get_delta_factorized_with_L(model_167d, histories, device, args.n_noise)
    dl_agg_slope, dl_agg_intercept, dl_agg_r2 = regress(dl_prev, dl_delta)
    dl_cell_slopes, dl_cell_r2s = regress_per_cell(dl_prev, dl_delta)

    results["models"]["167d_with_L"] = {
        "model_path": model_167d_path,
        "method": f"delta_base + L@eps, averaged over {args.n_noise} noise samples, tanh applied",
        "aggregate_slope": dl_agg_slope,
        "aggregate_intercept": dl_agg_intercept,
        "aggregate_r2": dl_agg_r2,
        "mr_gt_ratio": dl_agg_slope / gt_agg_slope if abs(gt_agg_slope) > 1e-8 else None,
        "cell_slopes_5x5": format_grid(dl_cell_slopes),
        "cell_r2_5x5": format_grid(dl_cell_r2s),
        "cell_slope_range": [float(dl_cell_slopes.min()), float(dl_cell_slopes.max())],
        "cell_slope_mean": float(dl_cell_slopes.mean()),
        "L_effect_on_slope": dl_agg_slope - d_agg_slope,
    }
    print(f"  With-L aggregate slope: {dl_agg_slope:.4f}, R2: {dl_agg_r2:.4f}")
    print(f"  MR/GT ratio: {dl_agg_slope / gt_agg_slope:.3f}")
    print(f"  L effect on slope: {dl_agg_slope - d_agg_slope:+.4f}")
    del model_167d
    torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CONSISTENT MEAN REVERSION COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25} {'Agg Slope':>10} {'MR/GT':>8} {'Cell Range':>20} {'R2':>8}")
    print("-" * 80)

    rows = [
        ("GT", gt_agg_slope, 1.000, gt_cell_slopes, gt_agg_r2),
        ("Baseline (164a)", bl_agg_slope_val := results["models"]["baseline_164a"]["aggregate_slope"],
         results["models"]["baseline_164a"]["mr_gt_ratio"],
         bl_cell_slopes, results["models"]["baseline_164a"]["aggregate_r2"]),
        ("167b base-only", results["models"]["167b_base_only"]["aggregate_slope"],
         results["models"]["167b_base_only"]["mr_gt_ratio"],
         b_cell_slopes, results["models"]["167b_base_only"]["aggregate_r2"]),
        ("167b +L@eps", results["models"]["167b_with_L"]["aggregate_slope"],
         results["models"]["167b_with_L"]["mr_gt_ratio"],
         bl_cell_slopes, results["models"]["167b_with_L"]["aggregate_r2"]),
        ("167d base-only", results["models"]["167d_base_only"]["aggregate_slope"],
         results["models"]["167d_base_only"]["mr_gt_ratio"],
         d_cell_slopes, results["models"]["167d_base_only"]["aggregate_r2"]),
        ("167d +L@eps", results["models"]["167d_with_L"]["aggregate_slope"],
         results["models"]["167d_with_L"]["mr_gt_ratio"],
         dl_cell_slopes, results["models"]["167d_with_L"]["aggregate_r2"]),
    ]

    for name, slope, ratio, cell_s, r2 in rows:
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        cell_range = f"[{cell_s.min():.4f}, {cell_s.max():.4f}]"
        print(f"{name:<25} {slope:>10.4f} {ratio_str:>8} {cell_range:>20} {r2:>8.4f}")

    print("\n--- L@eps Effect ---")
    print(f"167b: L changes slope by {results['models']['167b_with_L']['L_effect_on_slope']:+.4f}")
    print(f"167d: L changes slope by {results['models']['167d_with_L']['L_effect_on_slope']:+.4f}")

    # ── Per-cell spatial patterns ─────────────────────────────────────────
    print("\n--- Per-Cell Slope Grids (5x5: moneyness x tenor) ---")
    for name, slopes in [
        ("GT", gt_cell_slopes),
        ("Baseline (164a)", bl_cell_slopes if 'bl_cell_slopes' in dir() else
         np.array(results["models"]["baseline_164a"]["cell_slopes_5x5"]).ravel()),
        ("167b base-only", b_cell_slopes),
        ("167d base-only", d_cell_slopes),
    ]:
        print(f"\n  {name}:")
        grid = slopes.reshape(5, 5)
        for row in grid:
            print("    " + "  ".join(f"{v:+.4f}" for v in row))

    # ── Save results ──────────────────────────────────────────────────────
    results_serializable = make_serializable(results)

    analysis_path = "results/validations/2026-04-04/analysis/167d_followup/consistent_mean_reversion.json"
    verify_path = "results/validations/2026-04-04/verification_results/consistent_mean_reversion.json"

    for path in [analysis_path, verify_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nSaved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
