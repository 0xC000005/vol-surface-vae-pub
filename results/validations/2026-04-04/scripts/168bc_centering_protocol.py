#!/usr/bin/env python
"""
168bc Centering Protocol: K ablation vs CORRECT same-loss control (166a).

Compares centering behavior across K=16 (166a), K=8 (168b), K=4 (168c),
all sharing identical loss weights (IS=0.005, VS=1.0).

Protocol:
  A. Deterministic-path slope: noise=zeros, regress delta vs prev_frame
  B. Sampled first-step mean slope: avg delta over 50 MC draws, regress vs prev_frame
  C. Spread: std(delta) across MC draws
  D. Centering/spread ratio: |mean(delta)| / std(delta)

All per-cell + aggregate. 200 test windows starting at idx 4540, first AR step only.
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
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_164a_v3_percell_bptt_softplus import (
    ARSpatialTransformerModel,
    SpatialTransformerDecoder,
    normalize_iv,
    denormalize_iv,
)


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
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def load_model(model_path, device):
    """Load ARSpatialTransformerModel from checkpoint."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]

    encoder_config = EncoderConfig(
        input_dim=cfg["encoder"]["input_dim"],
        gru_hidden_dim=cfg["encoder"]["gru_hidden_dim"],
        bottleneck_dim=cfg["encoder"]["bottleneck_dim"],
        dropout=cfg["encoder"]["dropout"],
    )
    decoder_config = cfg["decoder"]

    model = ARSpatialTransformerModel(encoder_config, decoder_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, cfg


def run_centering_protocol(model, history_windows, last_frames, device,
                           n_mc=50, seed=42):
    """
    Run centering protocol on first AR step for all windows.

    Args:
        model: loaded ARSpatialTransformerModel
        history_windows: (N, 30, 5, 5) raw IV in [0,1]
        last_frames: (N, 25) last frame in [0,1]
        device: torch device
        n_mc: number of MC noise draws
        seed: random seed for reproducibility

    Returns:
        dict with all measurements
    """
    N = history_windows.shape[0]
    noise_dim = model.decoder.noise_dim
    BATCH = 50  # process in batches to avoid OOM

    # Storage
    all_det_delta = []       # (N, 25) deterministic delta
    all_mc_mean_delta = []   # (N, 25) mean delta over MC draws
    all_mc_std_delta = []    # (N, 25) std delta over MC draws
    all_prev_frame = []      # (N, 25) prev frame values

    torch.manual_seed(seed)

    with torch.no_grad():
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            batch_size = end - start

            # Prepare inputs
            hist = history_windows[start:end].to(device)        # (B, 30, 5, 5) [0,1]
            hist_norm = normalize_iv(hist)                       # [-1,1]
            lf = last_frames[start:end].to(device)              # (B, 25) [0,1]

            # Encode
            hist_flat = hist_norm.reshape(batch_size, 30, 25)
            gru_outputs, h_last = model.encoder.gru(hist_flat)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)  # (B, 128)

            # A. Deterministic path: noise=zeros
            z_zero = torch.zeros(batch_size, noise_dim, device=device)
            det_delta = model.decoder(cond, lf, z_zero)  # (B, 25) raw delta
            all_det_delta.append(det_delta.cpu())

            # B+C. MC sampling: n_mc noise draws
            mc_deltas = []
            for _ in range(n_mc):
                z_t = torch.randn(batch_size, noise_dim, device=device)
                delta_t = model.decoder(cond, lf, z_t)  # (B, 25)
                mc_deltas.append(delta_t.cpu())

            mc_stack = torch.stack(mc_deltas, dim=0)  # (n_mc, B, 25)
            mc_mean = mc_stack.mean(dim=0)   # (B, 25)
            mc_std = mc_stack.std(dim=0)     # (B, 25)

            all_mc_mean_delta.append(mc_mean)
            all_mc_std_delta.append(mc_std)
            all_prev_frame.append(lf.cpu())

    # Concatenate all batches
    det_delta = torch.cat(all_det_delta, dim=0).numpy()       # (N, 25)
    mc_mean_delta = torch.cat(all_mc_mean_delta, dim=0).numpy()  # (N, 25)
    mc_std_delta = torch.cat(all_mc_std_delta, dim=0).numpy()    # (N, 25)
    prev_frame = torch.cat(all_prev_frame, dim=0).numpy()        # (N, 25)

    # --- Compute metrics ---

    # A. Deterministic slope: regress det_delta vs prev_frame, per-cell + aggregate
    det_slopes = []
    det_r2s = []
    for c in range(25):
        slope, intercept, r, p, se = stats.linregress(prev_frame[:, c], det_delta[:, c])
        det_slopes.append(slope)
        det_r2s.append(r**2)

    # Aggregate: flatten all cells
    agg_slope_det, _, agg_r_det, _, _ = stats.linregress(
        prev_frame.flatten(), det_delta.flatten()
    )

    # B. Sampled mean slope: regress mc_mean_delta vs prev_frame
    mc_slopes = []
    mc_r2s = []
    for c in range(25):
        slope, intercept, r, p, se = stats.linregress(
            prev_frame[:, c], mc_mean_delta[:, c]
        )
        mc_slopes.append(slope)
        mc_r2s.append(r**2)

    agg_slope_mc, _, agg_r_mc, _, _ = stats.linregress(
        prev_frame.flatten(), mc_mean_delta.flatten()
    )

    # C. Spread: mean and per-cell std across windows
    spread_per_cell = mc_std_delta.mean(axis=0)  # (25,) avg std per cell
    spread_aggregate = mc_std_delta.mean()

    # D. Centering/spread ratio: |mean(delta)| / std(delta), per-cell + aggregate
    # Per-window, per-cell ratio then average
    eps = 1e-8
    ratio_per_window = np.abs(mc_mean_delta) / (mc_std_delta + eps)  # (N, 25)
    ratio_per_cell = ratio_per_window.mean(axis=0)  # (25,)
    ratio_aggregate = ratio_per_window.mean()

    # Also compute: global mean |mean_delta| and global mean std_delta
    abs_mean_per_cell = np.abs(mc_mean_delta).mean(axis=0)  # (25,)
    abs_mean_aggregate = np.abs(mc_mean_delta).mean()

    # GT first-step delta for reference
    # (mean-reversion slope from data)

    # Cell labels
    moneyness = ["80%", "90%", "100%", "110%", "120%"]
    tenors = ["1M", "3M", "6M", "9M", "12M"]
    cell_labels = [f"{m}_{t}" for t in tenors for m in moneyness]

    results = {
        "protocol": "168bc_centering_K_ablation",
        "n_windows": int(N),
        "n_mc_draws": n_mc,
        "test_start_idx": 4540,
        "aggregate": {
            "det_slope": float(agg_slope_det),
            "det_r2": float(agg_r_det**2),
            "mc_mean_slope": float(agg_slope_mc),
            "mc_mean_r2": float(agg_r_mc**2),
            "spread": float(spread_aggregate),
            "abs_mean_delta": float(abs_mean_aggregate),
            "centering_spread_ratio": float(ratio_aggregate),
        },
        "per_cell": {
            "labels": cell_labels,
            "det_slopes": [float(s) for s in det_slopes],
            "det_r2": [float(r) for r in det_r2s],
            "mc_mean_slopes": [float(s) for s in mc_slopes],
            "mc_mean_r2": [float(r) for r in mc_r2s],
            "spread": [float(s) for s in spread_per_cell],
            "abs_mean_delta": [float(s) for s in abs_mean_per_cell],
            "centering_spread_ratio": [float(r) for r in ratio_per_cell],
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="168bc centering protocol: K ablation vs same-loss control"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_windows", type=int, default=200)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--n_mc", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    t0 = time.time()

    # Models to compare
    models_spec = {
        "166a_K16": {
            "path": "models/backfill/afcrps_166a/best_model.pt",
            "desc": "K=16, IS=0.005, VS=1.0 (same-loss control)",
        },
        "168b_K8": {
            "path": "models/backfill/afcrps_168b/best_model.pt",
            "desc": "K=8, IS=0.005, VS=1.0 (clean K reduction)",
        },
        "168c_K4": {
            "path": "models/backfill/afcrps_168c/best_model.pt",
            "desc": "K=4, IS=0.005, VS=1.0 (clean K reduction)",
        },
    }

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0,1]
    N_total = surfaces.shape[0]
    H = 30

    # Build test windows starting at idx 4540
    test_indices = np.arange(
        args.test_start,
        min(args.test_start + args.n_windows, N_total - 2 * H + 1)
    )
    actual_n = len(test_indices)
    print(f"Using {actual_n} test windows starting at idx {args.test_start}")

    # Pre-build history windows and last frames
    history_list = []
    last_frame_list = []
    for idx in test_indices:
        hist = surfaces[idx:idx + H]  # (30, 5, 5) in [0,1]
        last_frame = hist[-1].reshape(25)  # (25,) in [0,1]
        history_list.append(hist)
        last_frame_list.append(last_frame)

    history_windows = torch.from_numpy(
        np.stack(history_list).astype(np.float32)
    )  # (N, 30, 5, 5)
    last_frames = torch.from_numpy(
        np.stack(last_frame_list).astype(np.float32)
    )  # (N, 25)

    # Run protocol for each model
    all_results = {}
    for name, spec in models_spec.items():
        print(f"\n{'='*60}")
        print(f"Running centering protocol: {name}")
        print(f"  {spec['desc']}")
        print(f"  Path: {spec['path']}")
        print(f"{'='*60}")

        model, cfg = load_model(spec["path"], device)
        print(f"  K={cfg['n_members']}, noise_dim={cfg['decoder']['noise_dim']}")
        print(f"  IS={cfg['lambda_is']}, VS={cfg['lambda_vs']}")

        results = run_centering_protocol(
            model, history_windows, last_frames, device,
            n_mc=args.n_mc, seed=args.seed
        )
        results["model_name"] = name
        results["model_path"] = spec["path"]
        results["model_desc"] = spec["desc"]
        results["config"] = {
            "n_members": cfg["n_members"],
            "lambda_is": cfg["lambda_is"],
            "lambda_vs": cfg["lambda_vs"],
            "noise_dim": cfg["decoder"]["noise_dim"],
        }
        all_results[name] = results

        # Print summary
        agg = results["aggregate"]
        print(f"\n  Aggregate results:")
        print(f"    Det slope:             {agg['det_slope']:.4f} (R2={agg['det_r2']:.4f})")
        print(f"    MC mean slope:         {agg['mc_mean_slope']:.4f} (R2={agg['mc_mean_r2']:.4f})")
        print(f"    Spread (std):          {agg['spread']:.4f}")
        print(f"    |mean(delta)|:         {agg['abs_mean_delta']:.4f}")
        print(f"    Centering/spread:      {agg['centering_spread_ratio']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ── Comparison table ──
    print(f"\n\n{'='*80}")
    print("COMPARISON TABLE: K ablation centering protocol")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'166a (K=16)':>15} {'168b (K=8)':>15} {'168c (K=4)':>15}")
    print("-" * 80)

    for metric in ["det_slope", "mc_mean_slope", "spread", "abs_mean_delta",
                    "centering_spread_ratio", "det_r2", "mc_mean_r2"]:
        vals = [all_results[n]["aggregate"][metric] for n in
                ["166a_K16", "168b_K8", "168c_K4"]]
        print(f"  {metric:<28} {vals[0]:>15.4f} {vals[1]:>15.4f} {vals[2]:>15.4f}")

    # ── Per-cell comparison: slopes ──
    print(f"\n\nPER-CELL MC MEAN SLOPES:")
    print(f"{'Cell':<15} {'166a (K=16)':>12} {'168b (K=8)':>12} {'168c (K=4)':>12}  {'K8 improve':>12} {'K4 improve':>12}")
    print("-" * 80)

    moneyness = ["80%", "90%", "100%", "110%", "120%"]
    tenors = ["1M", "3M", "6M", "9M", "12M"]

    for c in range(25):
        t_idx = c // 5
        m_idx = c % 5
        label = f"{moneyness[m_idx]}_{tenors[t_idx]}"

        s16 = all_results["166a_K16"]["per_cell"]["mc_mean_slopes"][c]
        s8 = all_results["168b_K8"]["per_cell"]["mc_mean_slopes"][c]
        s4 = all_results["168c_K4"]["per_cell"]["mc_mean_slopes"][c]

        # Improvement = more negative slope (stronger mean-reversion)
        # But also less bias = closer to GT slope
        # For now just show raw and difference
        d8 = s8 - s16
        d4 = s4 - s16

        print(f"  {label:<13} {s16:>12.4f} {s8:>12.4f} {s4:>12.4f}  {d8:>+12.4f} {d4:>+12.4f}")

    # ── Per-cell comparison: centering/spread ratio ──
    print(f"\n\nPER-CELL CENTERING/SPREAD RATIO:")
    print(f"{'Cell':<15} {'166a (K=16)':>12} {'168b (K=8)':>12} {'168c (K=4)':>12}  {'K8 change':>12} {'K4 change':>12}")
    print("-" * 80)

    for c in range(25):
        t_idx = c // 5
        m_idx = c % 5
        label = f"{moneyness[m_idx]}_{tenors[t_idx]}"

        r16 = all_results["166a_K16"]["per_cell"]["centering_spread_ratio"][c]
        r8 = all_results["168b_K8"]["per_cell"]["centering_spread_ratio"][c]
        r4 = all_results["168c_K4"]["per_cell"]["centering_spread_ratio"][c]

        d8 = r8 - r16
        d4 = r4 - r16

        print(f"  {label:<13} {r16:>12.4f} {r8:>12.4f} {r4:>12.4f}  {d8:>+12.4f} {d4:>+12.4f}")

    # ── Per-cell comparison: spread ──
    print(f"\n\nPER-CELL SPREAD (std of delta across MC draws):")
    print(f"{'Cell':<15} {'166a (K=16)':>12} {'168b (K=8)':>12} {'168c (K=4)':>12}  {'K8 change':>12} {'K4 change':>12}")
    print("-" * 80)

    for c in range(25):
        t_idx = c // 5
        m_idx = c % 5
        label = f"{moneyness[m_idx]}_{tenors[t_idx]}"

        sp16 = all_results["166a_K16"]["per_cell"]["spread"][c]
        sp8 = all_results["168b_K8"]["per_cell"]["spread"][c]
        sp4 = all_results["168c_K4"]["per_cell"]["spread"][c]

        d8 = sp8 - sp16
        d4 = sp4 - sp16

        print(f"  {label:<13} {sp16:>12.4f} {sp8:>12.4f} {sp4:>12.4f}  {d8:>+12.4f} {d4:>+12.4f}")

    # ── Save results ──
    output_dir = Path("results/validations/2026-04-04/analysis/168bc_followup")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "protocol": "168bc_centering_K_ablation_vs_same_loss_control",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Fair comparison: all models share IS=0.005, VS=1.0. 166a is the correct K=16 control.",
        "parameters": {
            "n_windows": actual_n,
            "test_start": args.test_start,
            "n_mc": args.n_mc,
            "seed": args.seed,
        },
        "models": make_serializable(all_results),
    }

    with open(output_dir / "centering_protocol.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'centering_protocol.json'}")

    # ── Verification summary ──
    verify_dir = Path("results/validations/2026-04-04/verification_results")
    verify_dir.mkdir(parents=True, exist_ok=True)

    # Key question: does K reduction change centering/spread ratio?
    r_166a = all_results["166a_K16"]["aggregate"]["centering_spread_ratio"]
    r_168b = all_results["168b_K8"]["aggregate"]["centering_spread_ratio"]
    r_168c = all_results["168c_K4"]["aggregate"]["centering_spread_ratio"]

    s_166a = all_results["166a_K16"]["aggregate"]["mc_mean_slope"]
    s_168b = all_results["168b_K8"]["aggregate"]["mc_mean_slope"]
    s_168c = all_results["168c_K4"]["aggregate"]["mc_mean_slope"]

    sp_166a = all_results["166a_K16"]["aggregate"]["spread"]
    sp_168b = all_results["168b_K8"]["aggregate"]["spread"]
    sp_168c = all_results["168c_K4"]["aggregate"]["spread"]

    verification = {
        "key_question": "Does K reduction improve centering vs same-loss K=16 (166a)?",
        "control": "166a (K=16, IS=0.005, VS=1.0) — CORRECT same-loss control",
        "previous_error": "168a was compared to 164a (wrong control, different IS/VS weights)",
        "results": {
            "centering_spread_ratio": {
                "166a_K16": r_166a,
                "168b_K8": r_168b,
                "168c_K4": r_168c,
                "K8_vs_K16_change_pct": (r_168b - r_166a) / r_166a * 100 if r_166a != 0 else None,
                "K4_vs_K16_change_pct": (r_168c - r_166a) / r_166a * 100 if r_166a != 0 else None,
            },
            "mc_mean_slope": {
                "166a_K16": s_166a,
                "168b_K8": s_168b,
                "168c_K4": s_168c,
                "K8_vs_K16_change": s_168b - s_166a,
                "K4_vs_K16_change": s_168c - s_166a,
            },
            "spread": {
                "166a_K16": sp_166a,
                "168b_K8": sp_168b,
                "168c_K4": sp_168c,
                "K8_vs_K16_change_pct": (sp_168b - sp_166a) / sp_166a * 100 if sp_166a != 0 else None,
                "K4_vs_K16_change_pct": (sp_168c - sp_166a) / sp_166a * 100 if sp_166a != 0 else None,
            },
        },
        "interpretation": {
            "lower_ratio_is_better": "centering/spread ratio < 1 means spread dominates bias; lower = better centering relative to spread",
            "more_negative_slope_is_better": "more negative slope = stronger mean-reversion centering",
        },
    }

    with open(verify_dir / "168bc_centering.json", "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"Saved verification to {verify_dir / '168bc_centering.json'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
