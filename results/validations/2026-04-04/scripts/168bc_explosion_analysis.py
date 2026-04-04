#!/usr/bin/env python
"""
168b/168c Explosion Analysis: Why do lower K models fail S1?

Compares K=4 (168c), K=8 (168b), K=16 (baseline) on:
1. Explosion patterns: which cells, which horizons, floor vs ceiling
2. Delta magnitudes per step
3. Training floor barrier convergence
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

# ─── Imports from training script ──────────────────────────────────────────

from experiments.backfill.block_ar.train_164a_v3_percell_bptt_softplus import (
    ARSpatialTransformerModel,
    normalize_iv,
    denormalize_iv,
)
from diffusion.block_ar.gru_encoder import EncoderConfig


def load_model(model_path, device="cuda"):
    """Load a 164a-family checkpoint."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    enc_cfg = cfg["encoder"]
    dec_cfg = cfg["decoder"]

    encoder_config = EncoderConfig(**enc_cfg)
    model = ARSpatialTransformerModel(
        encoder_config=encoder_config,
        decoder_config=dec_cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, cfg


def load_data(data_path="data/vol_surface_with_ret.npz"):
    """Load and prepare test split windows."""
    data = np.load(data_path)
    surfaces = data["surface"]  # (N, 5, 5)
    N = len(surfaces)

    # Test split: windows starting at index 4540+
    test_start = 4540
    window_size = 60  # 30 hist + 30 future

    windows = []
    for i in range(test_start, N - window_size + 1):
        hist = surfaces[i:i+30]  # (30, 5, 5)
        future = surfaces[i+30:i+60]  # (30, 5, 5)
        windows.append((hist, future))

    return windows


def generate_with_deltas(model, history_norm, n_samples=50, device="cuda"):
    """Generate samples AND capture per-step deltas.

    Returns:
        samples: (B, n_samples, 30, 5, 5) in [0,1]
        all_deltas: (B, n_samples, 30, 25) raw delta from decoder (pre-tanh)
        all_tanh_deltas: (B, n_samples, 30, 25) tanh(delta) applied
        all_frames: (B, n_samples, 30, 25) frame values in [0,1]
    """
    B = history_norm.shape[0]
    T = 30
    CHUNK = 10

    with torch.no_grad():
        last_frame = denormalize_iv(history_norm[:, -1]).reshape(B, 25)
        hist_flat = history_norm.reshape(B, history_norm.shape[1], -1)
        gru_outputs_base, h_last_base = model.encoder.gru(hist_flat)

        all_samples = []
        all_deltas = []
        all_tanh_deltas = []
        all_frames = []

        for start in range(0, n_samples, CHUNK):
            k = min(CHUNK, n_samples - start)

            last_k = last_frame.unsqueeze(1).expand(B, k, -1).reshape(B * k, 25)
            gru_out_k = gru_outputs_base.unsqueeze(1).expand(
                B, k, -1, -1).reshape(B * k, -1, model.encoder_config.gru_hidden_dim)
            h_last_k = h_last_base.unsqueeze(2).expand(
                1, B, k, -1).reshape(1, B * k, -1)

            # Initial condition
            attn_logits = model.encoder.attn_proj(gru_out_k).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_out_k).sum(dim=1)
            cond_init = model.encoder.bottleneck(h_pooled)

            # Manual AR loop to capture deltas
            noise_dim = model.decoder.noise_dim
            prev = last_k
            gru_state = h_last_k.contiguous()
            gru_outputs = gru_out_k
            chunk_frames = []
            chunk_deltas = []
            chunk_tanh_deltas = []

            for t in range(T):
                z_t = torch.randn(B * k, noise_dim, device=device)

                # Recompute condition from GRU
                attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
                attn_weights = F.softmax(attn_logits, dim=1)
                h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
                cond_t = model.encoder.bottleneck(h_pooled)

                delta = model.decoder(cond_t, prev, z_t)
                tanh_delta = torch.tanh(delta)
                frame_t = prev + tanh_delta

                chunk_deltas.append(delta)
                chunk_tanh_deltas.append(tanh_delta)
                chunk_frames.append(frame_t)

                # GRU feedback
                frame_norm = normalize_iv(frame_t).unsqueeze(1)
                gru_out, gru_state = model.encoder.gru(frame_norm, gru_state)
                gru_outputs = torch.cat([gru_outputs, gru_out], dim=1)

                prev = frame_t

            # Stack: (B*k, T, 25)
            chunk_frames = torch.stack(chunk_frames, dim=1)
            chunk_deltas = torch.stack(chunk_deltas, dim=1)
            chunk_tanh_deltas = torch.stack(chunk_tanh_deltas, dim=1)

            # Reshape to (B, k, T, 25)
            all_frames.append(chunk_frames.reshape(B, k, T, 25))
            all_deltas.append(chunk_deltas.reshape(B, k, T, 25))
            all_tanh_deltas.append(chunk_tanh_deltas.reshape(B, k, T, 25))
            all_samples.append(chunk_frames.reshape(B, k, T, 5, 5))

    samples = torch.cat(all_samples, dim=1)  # (B, n_samples, T, 5, 5)
    deltas = torch.cat(all_deltas, dim=1)  # (B, n_samples, T, 25)
    tanh_deltas = torch.cat(all_tanh_deltas, dim=1)
    frames = torch.cat(all_frames, dim=1)  # (B, n_samples, T, 25)

    return samples, deltas, tanh_deltas, frames


def analyze_explosions(frames, threshold_hi=1.0, threshold_lo=0.0):
    """Analyze explosion patterns from frame data.

    Args:
        frames: (B, K, T, 25) in [0,1]
    Returns:
        dict with explosion statistics
    """
    B, K, T, C = frames.shape

    # Per-sample explosion: any cell > hi or < lo at any timestep
    any_hi = (frames > threshold_hi).any(dim=-1).any(dim=-1)  # (B, K)
    any_lo = (frames < threshold_lo).any(dim=-1).any(dim=-1)  # (B, K)
    any_explosion = any_hi | any_lo

    explosion_rate = any_explosion.float().mean().item()
    hi_rate = any_hi.float().mean().item()
    lo_rate = any_lo.float().mean().item()

    # Per-cell explosion rate
    cell_hi = (frames > threshold_hi).float().mean(dim=(0, 1, 2))  # (25,)
    cell_lo = (frames < threshold_lo).float().mean(dim=(0, 1, 2))  # (25,)

    # Per-horizon explosion rate
    horizon_hi = (frames > threshold_hi).any(dim=-1).float().mean(dim=(0, 1))  # (T,)
    horizon_lo = (frames < threshold_lo).any(dim=-1).float().mean(dim=(0, 1))  # (T,)

    # First horizon where explosion occurs (per exploding sample)
    # For high explosions
    hi_per_step = (frames > threshold_hi).any(dim=-1)  # (B, K, T) bool
    lo_per_step = (frames < threshold_lo).any(dim=-1)

    # Cumulative first occurrence
    hi_first = []
    lo_first = []
    for b in range(B):
        for k in range(K):
            if hi_per_step[b, k].any():
                hi_first.append(hi_per_step[b, k].float().argmax().item())
            if lo_per_step[b, k].any():
                lo_first.append(lo_per_step[b, k].float().argmax().item())

    # Min/max observed
    max_iv = frames.max().item()
    min_iv = frames.min().item()

    # Most extreme cells at final horizon
    final_frames = frames[:, :, -1, :]  # (B, K, 25)
    cell_max_final = final_frames.max(dim=0).values.max(dim=0).values  # (25,)
    cell_min_final = final_frames.min(dim=0).values.min(dim=0).values  # (25,)

    return {
        "explosion_rate_total": explosion_rate,
        "explosion_rate_hi": hi_rate,
        "explosion_rate_lo": lo_rate,
        "max_iv": max_iv,
        "min_iv": min_iv,
        "cell_hi_rate": cell_hi.cpu().tolist(),
        "cell_lo_rate": cell_lo.cpu().tolist(),
        "horizon_hi_rate": horizon_hi.cpu().tolist(),
        "horizon_lo_rate": horizon_lo.cpu().tolist(),
        "hi_first_horizon_mean": float(np.mean(hi_first)) if hi_first else None,
        "hi_first_horizon_median": float(np.median(hi_first)) if hi_first else None,
        "lo_first_horizon_mean": float(np.mean(lo_first)) if lo_first else None,
        "lo_first_horizon_median": float(np.median(lo_first)) if lo_first else None,
        "n_hi_explosions": len(hi_first),
        "n_lo_explosions": len(lo_first),
        "cell_max_at_h30": cell_max_final.cpu().tolist(),
        "cell_min_at_h30": cell_min_final.cpu().tolist(),
    }


def analyze_deltas(deltas, tanh_deltas, frames):
    """Analyze delta magnitude patterns.

    Args:
        deltas: (B, K, T, 25) raw deltas (pre-tanh)
        tanh_deltas: (B, K, T, 25) tanh(delta)
        frames: (B, K, T, 25) in [0,1]
    """
    B, K, T, C = deltas.shape

    # Mean |delta| per step
    abs_delta_per_step = deltas.abs().mean(dim=(0, 1, 3))  # (T,)
    abs_tanh_per_step = tanh_deltas.abs().mean(dim=(0, 1, 3))  # (T,)

    # Max |delta| per step
    max_delta_per_step = deltas.abs().amax(dim=(0, 1, 3))  # (T,)
    max_tanh_per_step = tanh_deltas.abs().amax(dim=(0, 1, 3))  # (T,)

    # Tanh saturation: fraction of cells where |delta| > 2 (tanh > 0.96)
    saturation_per_step = (deltas.abs() > 2.0).float().mean(dim=(0, 1, 3))  # (T,)

    # Delta per cell
    abs_delta_per_cell = deltas.abs().mean(dim=(0, 1, 2))  # (25,)

    # Mean |delta| overall
    mean_abs_delta = deltas.abs().mean().item()
    mean_abs_tanh = tanh_deltas.abs().mean().item()

    # Cumulative displacement: |frame_t - frame_0|
    # frame_0 is the last history frame, not in frames tensor
    # So we measure frame drift from h=0 to h=T
    drift_per_step = (frames - frames[:, :, 0:1, :]).abs().mean(dim=(0, 1, 3))  # (T,)

    return {
        "mean_abs_delta_per_step": abs_delta_per_step.cpu().tolist(),
        "mean_abs_tanh_per_step": abs_tanh_per_step.cpu().tolist(),
        "max_delta_per_step": max_delta_per_step.cpu().tolist(),
        "max_tanh_per_step": max_tanh_per_step.cpu().tolist(),
        "saturation_per_step": saturation_per_step.cpu().tolist(),
        "abs_delta_per_cell": abs_delta_per_cell.cpu().tolist(),
        "mean_abs_delta": mean_abs_delta,
        "mean_abs_tanh": mean_abs_tanh,
        "drift_per_step": drift_per_step.cpu().tolist(),
    }


def analyze_training_floor(log_path):
    """Extract floor barrier loss trajectory from training log."""
    lines = Path(log_path).read_text().strip().split("\n")
    epochs = []
    for line in lines:
        if line.startswith("Ep"):
            parts = line.split()
            ep = int(parts[1])
            # Parse floor= value
            for p in parts:
                if p.startswith("floor="):
                    floor_val = float(p.split("=")[1])
                    epochs.append({"epoch": ep, "floor": floor_val})
                    break
    return epochs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_windows", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    t0 = time.time()

    # Model paths
    models = {
        "168c_K4": "models/backfill/afcrps_168c/best_model.pt",
        "168b_K8": "models/backfill/afcrps_168b/best_model.pt",
        "baseline_K16": "models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt",
    }

    # Training logs
    training_logs = {
        "168c_K4": "models/backfill/afcrps_168c/training.log",
        "168b_K8": "models/backfill/afcrps_168b/training.log",
    }

    # Load data
    print("Loading data...")
    windows = load_data()
    n_win = min(args.n_windows, len(windows))
    print(f"Using {n_win} test windows, {args.n_samples} samples each")

    results = {}

    for name, path in models.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")

        model, cfg = load_model(path, device)
        K = cfg.get("n_members", "?")
        print(f"  K={K}, type={cfg['type']}")

        # Collect frames and deltas across windows
        all_frames_list = []
        all_deltas_list = []
        all_tanh_list = []

        for wi in range(n_win):
            hist, future = windows[wi]
            hist_t = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)
            hist_norm = normalize_iv(hist_t)

            samples, deltas, tanh_deltas, frames = generate_with_deltas(
                model, hist_norm, n_samples=args.n_samples, device=device
            )
            # frames: (1, n_samples, 30, 25)
            all_frames_list.append(frames.cpu())
            all_deltas_list.append(deltas.cpu())
            all_tanh_list.append(tanh_deltas.cpu())

            if (wi + 1) % 10 == 0:
                print(f"  Window {wi+1}/{n_win}")

        # Concatenate: (n_win, n_samples, 30, 25)
        all_frames = torch.cat(all_frames_list, dim=0)
        all_deltas = torch.cat(all_deltas_list, dim=0)
        all_tanh = torch.cat(all_tanh_list, dim=0)

        print(f"  Total shape: {all_frames.shape}")

        # Analyze
        explosion_stats = analyze_explosions(all_frames)
        delta_stats = analyze_deltas(all_deltas, all_tanh, all_frames)

        results[name] = {
            "K": K,
            "explosion": explosion_stats,
            "deltas": delta_stats,
        }

        print(f"  Explosion rate: {explosion_stats['explosion_rate_total']:.4f}")
        print(f"    Hi: {explosion_stats['explosion_rate_hi']:.4f}, Lo: {explosion_stats['explosion_rate_lo']:.4f}")
        print(f"    Max IV: {explosion_stats['max_iv']:.4f}, Min IV: {explosion_stats['min_iv']:.4f}")
        print(f"    Lo first horizon mean: {explosion_stats['lo_first_horizon_mean']}")
        print(f"    Hi first horizon mean: {explosion_stats['hi_first_horizon_mean']}")
        print(f"  Mean |delta|: {delta_stats['mean_abs_delta']:.6f}")
        print(f"  Mean |tanh(delta)|: {delta_stats['mean_abs_tanh']:.6f}")

    # Analyze training floor barriers
    print(f"\n{'='*60}")
    print("Training Floor Barrier Analysis")
    print(f"{'='*60}")

    training_floor_data = {}
    for name, log_path in training_logs.items():
        if Path(log_path).exists():
            floor_data = analyze_training_floor(log_path)
            training_floor_data[name] = floor_data
            if floor_data:
                print(f"\n  {name}:")
                print(f"    Epoch 1 floor: {floor_data[0]['floor']:.6f}")
                print(f"    Epoch 10 floor: {floor_data[9]['floor']:.6f}")
                print(f"    Final floor: {floor_data[-1]['floor']:.6f}")

    results["training_floor"] = training_floor_data

    # ─── Comparative Summary ──────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*60}")

    # S1 data from summary.json (pre-extracted)
    s1_data = {
        "168c_K4": {"explosion_hi": 0.00249, "explosion_lo": 0.11097, "total": 0.11346, "pass": False},
        "168b_K8": {"explosion_hi": 0.00361, "explosion_lo": 0.06342, "total": 0.06700, "pass": False},
        "baseline_K16": {"explosion_hi": 0.00128, "explosion_lo": 0.02350, "total": 0.02473, "pass": True},
    }

    print("\nS1 Explosion Rates (from summary.json):")
    print(f"  {'Model':<15} {'Hi Rate':>10} {'Lo Rate':>10} {'Total':>10} {'Pass':>6}")
    for name in models:
        d = s1_data[name]
        print(f"  {name:<15} {d['explosion_hi']:>10.5f} {d['explosion_lo']:>10.5f} {d['total']:>10.5f} {str(d['pass']):>6}")

    print("\nThis Analysis (50 windows x 50 samples):")
    print(f"  {'Model':<15} {'Hi Rate':>10} {'Lo Rate':>10} {'Total':>10} {'Mean |delta|':>12} {'Mean |tanh|':>12}")
    for name in models:
        e = results[name]["explosion"]
        d = results[name]["deltas"]
        print(f"  {name:<15} {e['explosion_rate_hi']:>10.5f} {e['explosion_rate_lo']:>10.5f} "
              f"{e['explosion_rate_total']:>10.5f} {d['mean_abs_delta']:>12.6f} {d['mean_abs_tanh']:>12.6f}")

    # Horizon onset comparison
    print("\nExplosion First Horizon (mean, median):")
    for name in models:
        e = results[name]["explosion"]
        lo_mean = e["lo_first_horizon_mean"]
        lo_med = e["lo_first_horizon_median"]
        hi_mean = e["hi_first_horizon_mean"]
        hi_med = e["hi_first_horizon_median"]
        lo_str = f"mean={lo_mean:.1f}, med={lo_med:.1f}" if lo_mean is not None else "none"
        hi_str = f"mean={hi_mean:.1f}, med={hi_med:.1f}" if hi_mean is not None else "none"
        print(f"  {name:<15} Lo: {lo_str}  |  Hi: {hi_str}")

    # Top exploding cells (by lo rate)
    print("\nTop 5 Floor-Exploding Cells (lo rate):")
    for name in models:
        lo_rates = results[name]["explosion"]["cell_lo_rate"]
        sorted_cells = sorted(enumerate(lo_rates), key=lambda x: -x[1])[:5]
        cells_str = ", ".join([f"cell{i}({r:.5f})" for i, r in sorted_cells])
        print(f"  {name:<15} {cells_str}")

    # Delta comparison per step (first, mid, last)
    print("\nMean |delta| at horizons 1, 15, 30:")
    for name in models:
        d = results[name]["deltas"]["mean_abs_delta_per_step"]
        print(f"  {name:<15} h1={d[0]:.6f}  h15={d[14]:.6f}  h30={d[29]:.6f}")

    # Saturation comparison
    print("\nTanh Saturation (|delta|>2) at horizons 1, 15, 30:")
    for name in models:
        s = results[name]["deltas"]["saturation_per_step"]
        print(f"  {name:<15} h1={s[0]:.6f}  h15={s[14]:.6f}  h30={s[29]:.6f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    out_path = "results/validations/2026-04-04/analysis/168bc_followup/explosion_analysis.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Make numpy-safe
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return to_serializable(obj.tolist())
        return obj

    with open(out_path, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ─── Build verification result ──────────────────────────────────────────

    verification = {
        "test": "168bc_explosion_analysis",
        "date": "2026-04-04",
        "models": {
            name: {
                "K": results[name]["K"],
                "explosion_rate_total": results[name]["explosion"]["explosion_rate_total"],
                "explosion_rate_hi": results[name]["explosion"]["explosion_rate_hi"],
                "explosion_rate_lo": results[name]["explosion"]["explosion_rate_lo"],
                "max_iv": results[name]["explosion"]["max_iv"],
                "min_iv": results[name]["explosion"]["min_iv"],
                "mean_abs_delta": results[name]["deltas"]["mean_abs_delta"],
                "lo_first_horizon_mean": results[name]["explosion"]["lo_first_horizon_mean"],
            }
            for name in models
        },
        "key_findings": [],
        "s1_from_summary": s1_data,
    }

    # Determine key findings
    lo_rates = {name: results[name]["explosion"]["explosion_rate_lo"] for name in models}
    hi_rates = {name: results[name]["explosion"]["explosion_rate_hi"] for name in models}
    deltas_mean = {name: results[name]["deltas"]["mean_abs_delta"] for name in models}

    if lo_rates["168c_K4"] > lo_rates["baseline_K16"] * 2:
        verification["key_findings"].append(
            f"Floor explosions dominate: K4 lo_rate={lo_rates['168c_K4']:.4f} vs K16={lo_rates['baseline_K16']:.4f} "
            f"({lo_rates['168c_K4']/max(lo_rates['baseline_K16'], 1e-8):.1f}x)"
        )

    if deltas_mean["168c_K4"] > deltas_mean["baseline_K16"] * 1.1:
        verification["key_findings"].append(
            f"K4 produces larger deltas: {deltas_mean['168c_K4']:.6f} vs {deltas_mean['baseline_K16']:.6f}"
        )
    else:
        verification["key_findings"].append(
            f"Delta magnitudes similar: K4={deltas_mean['168c_K4']:.6f} vs K16={deltas_mean['baseline_K16']:.6f}"
        )

    # Check if it's primarily floor or ceiling
    for name in models:
        e = results[name]["explosion"]
        if e["explosion_rate_lo"] > e["explosion_rate_hi"] * 5:
            verification["key_findings"].append(
                f"{name}: Primarily FLOOR explosion (lo={e['explosion_rate_lo']:.4f} >> hi={e['explosion_rate_hi']:.4f})"
            )

    verif_path = "results/validations/2026-04-04/verification_results/168bc_explosion.json"
    with open(verif_path, "w") as f:
        json.dump(to_serializable(verification), f, indent=2)
    print(f"Verification saved to {verif_path}")


if __name__ == "__main__":
    main()
