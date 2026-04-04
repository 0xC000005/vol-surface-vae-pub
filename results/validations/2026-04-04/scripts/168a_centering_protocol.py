#!/usr/bin/env python
"""
Centering Protocol: Compare deterministic-path slope vs sampled-mean slope.

Measures for 168a (K=4) and baseline (K=16):
1. Deterministic-path slope: decoder with noise=zeros, regress delta vs prev_frame
2. Sampled first-step mean slope: average delta over 50 MC draws, regress vs prev_frame
3. (168a only) Spread: std of delta across MC draws
4. (168a only) Centering/spread ratio: |mean(delta)| / std(delta)

Both models use ARSpatialTransformerModel from train_164a_v3_percell_bptt_softplus.py
Test split: start 4540, 200 windows.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_164a_v3_percell_bptt_softplus import (
    ARSpatialTransformerModel,
    normalize_iv,
    denormalize_iv,
)
from diffusion.block_ar.gru_encoder import EncoderConfig


def load_model(path, device):
    """Load ARSpatialTransformerModel from checkpoint."""
    ckpt = torch.load(path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    enc_cfg = EncoderConfig(**cfg["encoder"])
    dec_cfg = cfg["decoder"]
    model = ARSpatialTransformerModel(enc_cfg, dec_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def get_test_windows(start=4540, n_windows=200):
    """Load data and extract test windows."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    H = 30

    histories = []
    last_frames = []
    gt_first_frames = []

    for i in range(start, start + n_windows):
        hist = surfaces[i : i + H]  # (30, 5, 5) in [0,1]
        future_first = surfaces[i + H]  # (5, 5) ground truth next frame
        histories.append(hist)
        last_frames.append(hist[-1].flatten())  # (25,) in [0,1]
        gt_first_frames.append(future_first.flatten())  # (25,) in [0,1]

    histories = np.stack(histories)  # (N, 30, 5, 5)
    last_frames = np.stack(last_frames)  # (N, 25)
    gt_first_frames = np.stack(gt_first_frames)  # (N, 25)

    return histories, last_frames, gt_first_frames


def encode_histories(model, histories, device, batch_size=50):
    """Encode all histories to get conditions and GRU states."""
    N = histories.shape[0]
    conditions = []
    gru_outputs_list = []
    gru_states_list = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            hist_t = torch.tensor(histories[start:end], dtype=torch.float32, device=device)
            hist_norm = normalize_iv(hist_t)  # [-1,1]
            B = hist_t.shape[0]

            # Run GRU on history
            hist_flat = hist_norm.reshape(B, 30, -1)  # (B, 30, 25)
            gru_out, h_last = model.encoder.gru(hist_flat)

            # Attention pool for condition
            attn_logits = model.encoder.attn_proj(gru_out).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_out).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)

            conditions.append(cond.cpu())
            gru_outputs_list.append(gru_out.cpu())
            gru_states_list.append(h_last.cpu())

    conditions = torch.cat(conditions, dim=0)
    gru_outputs = torch.cat(gru_outputs_list, dim=0)
    gru_states = torch.cat(gru_states_list, dim=1)

    return conditions, gru_outputs, gru_states


def measure_deterministic_slope(model, conditions, last_frames, device, batch_size=50):
    """Measure first-step delta with noise=zeros (deterministic path)."""
    N = conditions.shape[0]
    noise_dim = model.decoder.noise_dim
    all_deltas = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            cond = conditions[start:end].to(device)
            prev = last_frames[start:end].to(device)
            z = torch.zeros(end - start, noise_dim, device=device)

            delta = model.decoder(cond, prev, z)
            all_deltas.append(delta.cpu())

    all_deltas = torch.cat(all_deltas, dim=0)  # (N, 25)
    return all_deltas.numpy()


def measure_sampled_slope(model, conditions, last_frames, device,
                          n_mc=50, batch_size=50):
    """Measure first-step delta averaged over MC noise draws."""
    N = conditions.shape[0]
    noise_dim = model.decoder.noise_dim

    # Collect all MC draws: (N, n_mc, 25)
    all_deltas = np.zeros((N, n_mc, 25))

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start
            cond = conditions[start:end].to(device)
            prev = last_frames[start:end].to(device)

            for m in range(n_mc):
                z = torch.randn(B, noise_dim, device=device)
                delta = model.decoder(cond, prev, z)
                all_deltas[start:end, m, :] = delta.cpu().numpy()

    # Mean delta across MC draws: (N, 25)
    mean_deltas = all_deltas.mean(axis=1)
    # Std across MC draws: (N, 25)
    std_deltas = all_deltas.std(axis=1)

    return mean_deltas, std_deltas, all_deltas


def regress_delta_vs_prevframe(deltas, last_frames):
    """Regress delta vs prev_frame per cell. Returns slopes and r-squared.

    For each cell: delta_i = slope * prev_frame_i + intercept
    Negative slope = mean-reversion (good centering).
    """
    N, C = deltas.shape
    slopes = np.zeros(C)
    intercepts = np.zeros(C)
    r_squared = np.zeros(C)

    for c in range(C):
        res = stats.linregress(last_frames[:, c], deltas[:, c])
        slopes[c] = res.slope
        intercepts[c] = res.intercept
        r_squared[c] = res.rvalue ** 2

    return slopes, intercepts, r_squared


def make_serializable(obj):
    """Convert numpy types to native Python types."""
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
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def run_centering_protocol(model, model_name, histories, last_frames_np,
                           gt_first_frames_np, device, compute_spread=False):
    """Run full centering protocol for one model."""
    print(f"\n{'='*60}")
    print(f"  Centering Protocol: {model_name}")
    print(f"{'='*60}")

    # Encode
    print("  Encoding histories...")
    conditions, gru_outputs, gru_states = encode_histories(model, histories, device)

    last_frames_t = torch.tensor(last_frames_np, dtype=torch.float32)

    # 1. Deterministic-path slope (noise=zeros)
    print("  Measuring deterministic-path slope (noise=zeros)...")
    det_deltas = measure_deterministic_slope(model, conditions, last_frames_t, device)
    det_slopes, det_intercepts, det_r2 = regress_delta_vs_prevframe(
        det_deltas, last_frames_np
    )

    print(f"    Deterministic slope: mean={det_slopes.mean():.4f}, "
          f"median={np.median(det_slopes):.4f}")
    print(f"    R2: mean={det_r2.mean():.4f}")

    # 2. Sampled first-step mean slope (50 MC draws)
    print("  Measuring sampled-mean slope (50 MC draws)...")
    mean_deltas, std_deltas, all_deltas = measure_sampled_slope(
        model, conditions, last_frames_t, device, n_mc=50
    )
    mc_slopes, mc_intercepts, mc_r2 = regress_delta_vs_prevframe(
        mean_deltas, last_frames_np
    )

    print(f"    MC-mean slope: mean={mc_slopes.mean():.4f}, "
          f"median={np.median(mc_slopes):.4f}")
    print(f"    R2: mean={mc_r2.mean():.4f}")

    # GT slope: regress (gt_first - prev) vs prev
    gt_deltas = gt_first_frames_np - last_frames_np
    gt_slopes, gt_intercepts, gt_r2 = regress_delta_vs_prevframe(
        gt_deltas, last_frames_np
    )
    print(f"    GT slope: mean={gt_slopes.mean():.4f}, median={np.median(gt_slopes):.4f}")
    print(f"    GT R2: mean={gt_r2.mean():.4f}")

    # Slope ratio vs GT
    det_ratio = det_slopes.mean() / gt_slopes.mean() if gt_slopes.mean() != 0 else float('inf')
    mc_ratio = mc_slopes.mean() / gt_slopes.mean() if gt_slopes.mean() != 0 else float('inf')
    print(f"\n    Slope ratio (model/GT):")
    print(f"      Deterministic: {det_ratio:.3f}")
    print(f"      MC-mean:       {mc_ratio:.3f}")

    results = {
        "model_name": model_name,
        "n_windows": len(last_frames_np),
        "n_mc_draws": 50,
        "deterministic": {
            "slopes_mean": float(det_slopes.mean()),
            "slopes_median": float(np.median(det_slopes)),
            "slopes_per_cell": det_slopes.tolist(),
            "intercepts_per_cell": det_intercepts.tolist(),
            "r2_mean": float(det_r2.mean()),
            "r2_per_cell": det_r2.tolist(),
            "delta_mean_per_cell": det_deltas.mean(axis=0).tolist(),
            "delta_std_per_cell": det_deltas.std(axis=0).tolist(),
        },
        "mc_mean": {
            "slopes_mean": float(mc_slopes.mean()),
            "slopes_median": float(np.median(mc_slopes)),
            "slopes_per_cell": mc_slopes.tolist(),
            "intercepts_per_cell": mc_intercepts.tolist(),
            "r2_mean": float(mc_r2.mean()),
            "r2_per_cell": mc_r2.tolist(),
            "delta_mean_per_cell": mean_deltas.mean(axis=0).tolist(),
            "delta_std_per_cell": mean_deltas.std(axis=0).tolist(),
        },
        "gt": {
            "slopes_mean": float(gt_slopes.mean()),
            "slopes_median": float(np.median(gt_slopes)),
            "slopes_per_cell": gt_slopes.tolist(),
            "r2_mean": float(gt_r2.mean()),
            "r2_per_cell": gt_r2.tolist(),
        },
        "slope_ratio_det_vs_gt": float(det_ratio),
        "slope_ratio_mc_vs_gt": float(mc_ratio),
    }

    # 3 & 4: Spread and centering/spread ratio (168a only)
    if compute_spread:
        print("\n  Computing spread and centering/spread ratio...")

        # Spread: std across MC draws per window per cell, then average
        # std_deltas is (N, 25) already
        spread_mean = std_deltas.mean()
        spread_per_cell = std_deltas.mean(axis=0)

        # Centering magnitude: |mean(delta)| per window per cell
        centering_mag = np.abs(mean_deltas)  # (N, 25)

        # Centering/spread ratio per window per cell
        ratio = centering_mag / (std_deltas + 1e-10)  # (N, 25)
        ratio_mean = ratio.mean()
        ratio_per_cell = ratio.mean(axis=0)

        print(f"    Spread (mean std across MC): {spread_mean:.6f}")
        print(f"    |mean(delta)| (centering mag): {centering_mag.mean():.6f}")
        print(f"    Centering/spread ratio: {ratio_mean:.3f}")
        print(f"    Ratio per cell (mean): min={ratio_per_cell.min():.3f}, "
              f"max={ratio_per_cell.max():.3f}")

        results["spread"] = {
            "mean_std_across_mc": float(spread_mean),
            "std_per_cell": spread_per_cell.tolist(),
        }
        results["centering_spread_ratio"] = {
            "mean_ratio": float(ratio_mean),
            "ratio_per_cell": ratio_per_cell.tolist(),
            "centering_mag_mean": float(centering_mag.mean()),
            "centering_mag_per_cell": centering_mag.mean(axis=0).tolist(),
        }

        # Bonus: per-window ratio distribution
        ratio_per_window = ratio.mean(axis=1)  # (N,)
        results["centering_spread_ratio"]["per_window_stats"] = {
            "mean": float(ratio_per_window.mean()),
            "std": float(ratio_per_window.std()),
            "median": float(np.median(ratio_per_window)),
            "p25": float(np.percentile(ratio_per_window, 25)),
            "p75": float(np.percentile(ratio_per_window, 75)),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device

    t0 = time.time()

    # Load data
    print("Loading test windows (start=4540, n=200)...")
    histories, last_frames, gt_first_frames = get_test_windows(start=4540, n_windows=200)
    print(f"  Loaded {len(last_frames)} windows")

    # Model paths
    model_168a_path = "models/backfill/afcrps_168a/best_model.pt"
    baseline_path = "models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt"

    # Load models
    print("\nLoading 168a (K=4)...")
    model_168a, cfg_168a = load_model(model_168a_path, device)
    print(f"  K={cfg_168a['n_members']}, lambda_vs={cfg_168a.get('lambda_vs')}, "
          f"lambda_is={cfg_168a.get('lambda_is')}")

    print("Loading baseline (K=16)...")
    model_baseline, cfg_baseline = load_model(baseline_path, device)
    print(f"  K={cfg_baseline['n_members']}, lambda_vs={cfg_baseline.get('lambda_vs')}, "
          f"lambda_is={cfg_baseline.get('lambda_is')}")

    # Run protocol for both
    results_168a = run_centering_protocol(
        model_168a, "168a (K=4)", histories, last_frames, gt_first_frames,
        device, compute_spread=True
    )
    results_baseline = run_centering_protocol(
        model_baseline, "baseline (K=16)", histories, last_frames, gt_first_frames,
        device, compute_spread=True  # also measure spread for baseline to compare
    )

    # ── Comparison ──
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")

    gt_slope = results_168a["gt"]["slopes_mean"]
    print(f"\n  GT mean-reversion slope: {gt_slope:.4f}")
    print(f"\n  {'Metric':<35} {'168a (K=4)':>12} {'Baseline (K=16)':>15}")
    print(f"  {'-'*62}")

    metrics = [
        ("Det slope (noise=0)", "deterministic", "slopes_mean"),
        ("MC-mean slope (50 draws)", "mc_mean", "slopes_mean"),
        ("Det R2", "deterministic", "r2_mean"),
        ("MC-mean R2", "mc_mean", "r2_mean"),
    ]
    for label, key, subkey in metrics:
        v168 = results_168a[key][subkey]
        vbase = results_baseline[key][subkey]
        print(f"  {label:<35} {v168:>12.4f} {vbase:>15.4f}")

    print(f"\n  {'Slope ratio vs GT':<35} {'168a':>12} {'Baseline':>15}")
    print(f"  {'-'*62}")
    print(f"  {'Deterministic':<35} {results_168a['slope_ratio_det_vs_gt']:>12.3f} "
          f"{results_baseline['slope_ratio_det_vs_gt']:>15.3f}")
    print(f"  {'MC-mean':<35} {results_168a['slope_ratio_mc_vs_gt']:>12.3f} "
          f"{results_baseline['slope_ratio_mc_vs_gt']:>15.3f}")

    # Spread comparison
    if "spread" in results_168a and "spread" in results_baseline:
        print(f"\n  {'Spread / Diversity':<35} {'168a':>12} {'Baseline':>15}")
        print(f"  {'-'*62}")
        s168 = results_168a["spread"]["mean_std_across_mc"]
        sbase = results_baseline["spread"]["mean_std_across_mc"]
        print(f"  {'Mean spread (std over MC)':<35} {s168:>12.6f} {sbase:>15.6f}")

        c168 = results_168a["centering_spread_ratio"]["centering_mag_mean"]
        cbase = results_baseline["centering_spread_ratio"]["centering_mag_mean"]
        print(f"  {'|mean(delta)| (centering mag)':<35} {c168:>12.6f} {cbase:>15.6f}")

        r168 = results_168a["centering_spread_ratio"]["mean_ratio"]
        rbase = results_baseline["centering_spread_ratio"]["mean_ratio"]
        print(f"  {'Centering/spread ratio':<35} {r168:>12.3f} {rbase:>15.3f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Save results ──
    combined = {
        "protocol": "centering_protocol_v1",
        "date": "2026-04-04",
        "test_windows": {"start": 4540, "n_windows": 200},
        "n_mc_draws": 50,
        "models": {
            "168a": {
                "path": model_168a_path,
                "K": cfg_168a["n_members"],
                "lambda_vs": cfg_168a.get("lambda_vs"),
                "lambda_is": cfg_168a.get("lambda_is"),
            },
            "baseline": {
                "path": baseline_path,
                "K": cfg_baseline["n_members"],
                "lambda_vs": cfg_baseline.get("lambda_vs"),
                "lambda_is": cfg_baseline.get("lambda_is"),
            },
        },
        "results": {
            "168a": results_168a,
            "baseline": results_baseline,
        },
        "comparison": {
            "gt_slope_mean": gt_slope,
            "det_slope_168a": results_168a["deterministic"]["slopes_mean"],
            "det_slope_baseline": results_baseline["deterministic"]["slopes_mean"],
            "mc_slope_168a": results_168a["mc_mean"]["slopes_mean"],
            "mc_slope_baseline": results_baseline["mc_mean"]["slopes_mean"],
            "slope_ratio_det_168a_vs_gt": results_168a["slope_ratio_det_vs_gt"],
            "slope_ratio_det_baseline_vs_gt": results_baseline["slope_ratio_det_vs_gt"],
            "slope_ratio_mc_168a_vs_gt": results_168a["slope_ratio_mc_vs_gt"],
            "slope_ratio_mc_baseline_vs_gt": results_baseline["slope_ratio_mc_vs_gt"],
        },
        "elapsed_seconds": elapsed,
    }

    # Add spread comparison
    if "spread" in results_168a and "spread" in results_baseline:
        combined["comparison"]["spread_168a"] = results_168a["spread"]["mean_std_across_mc"]
        combined["comparison"]["spread_baseline"] = results_baseline["spread"]["mean_std_across_mc"]
        combined["comparison"]["centering_ratio_168a"] = results_168a["centering_spread_ratio"]["mean_ratio"]
        combined["comparison"]["centering_ratio_baseline"] = results_baseline["centering_spread_ratio"]["mean_ratio"]

    combined = make_serializable(combined)

    out_dir = Path("results/validations/2026-04-04/analysis/168a_followup")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "centering_protocol.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Also save verification results
    ver_dir = Path("results/validations/2026-04-04/verification_results")
    ver_dir.mkdir(parents=True, exist_ok=True)
    ver_path = ver_dir / "168a_centering_protocol.json"

    # Verification summary
    verification = {
        "test": "centering_protocol",
        "date": "2026-04-04",
        "hypothesis": "K=4 improves centering/spread gradient ratio",
        "result": "TBD - see comparison metrics",
        "key_metrics": {
            "168a_K4": {
                "det_slope": results_168a["deterministic"]["slopes_mean"],
                "mc_slope": results_168a["mc_mean"]["slopes_mean"],
                "slope_ratio_vs_gt_det": results_168a["slope_ratio_det_vs_gt"],
                "slope_ratio_vs_gt_mc": results_168a["slope_ratio_mc_vs_gt"],
            },
            "baseline_K16": {
                "det_slope": results_baseline["deterministic"]["slopes_mean"],
                "mc_slope": results_baseline["mc_mean"]["slopes_mean"],
                "slope_ratio_vs_gt_det": results_baseline["slope_ratio_det_vs_gt"],
                "slope_ratio_vs_gt_mc": results_baseline["slope_ratio_mc_vs_gt"],
            },
        },
    }

    if "spread" in results_168a and "spread" in results_baseline:
        verification["key_metrics"]["168a_K4"]["spread"] = results_168a["spread"]["mean_std_across_mc"]
        verification["key_metrics"]["baseline_K16"]["spread"] = results_baseline["spread"]["mean_std_across_mc"]
        verification["key_metrics"]["168a_K4"]["centering_ratio"] = results_168a["centering_spread_ratio"]["mean_ratio"]
        verification["key_metrics"]["baseline_K16"]["centering_ratio"] = results_baseline["centering_spread_ratio"]["mean_ratio"]

    # Determine result
    det_closer = (
        abs(results_168a["slope_ratio_det_vs_gt"] - 1.0)
        < abs(results_baseline["slope_ratio_det_vs_gt"] - 1.0)
    )
    mc_closer = (
        abs(results_168a["slope_ratio_mc_vs_gt"] - 1.0)
        < abs(results_baseline["slope_ratio_mc_vs_gt"] - 1.0)
    )

    if det_closer and mc_closer:
        verification["result"] = "CONFIRMED: K=4 has better centering (closer to GT slope)"
    elif not det_closer and not mc_closer:
        verification["result"] = "REFUTED: K=4 has worse centering (further from GT slope)"
    else:
        verification["result"] = f"MIXED: det_closer={det_closer}, mc_closer={mc_closer}"

    verification = make_serializable(verification)
    with open(ver_path, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"  Verification saved to {ver_path}")


if __name__ == "__main__":
    main()
