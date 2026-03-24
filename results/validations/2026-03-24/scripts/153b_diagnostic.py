#!/usr/bin/env python
"""
153b Diagnostic Persistence Script
Reproduces inline diagnostics from autoresearch session for 153b
(conditional one-shot flow matching with RESIDUAL prediction).

Computes:
1. Full metric table (eff_rank, PC1-5, KS, kurtosis, Frobenius)
2. Level bias: per-cell mean(gen) - mean(GT) after persistence reconstruction
3. Kurtosis per-horizon (h=1, h=15, h=30)
4. KS on levels AND changes
5. Spread analysis: h1 vs h30
"""
import sys
sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")

import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import ks_2samp, kurtosis

from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv
)
from experiments.backfill.block_ar.train_oneshot_flow import evaluate_samples

# ─── Configuration ───
MODEL_PATH = "models/backfill/flow_153b/final_model.pt"
ENCODER_PATH = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_DIR = Path("results/validations/2026-03-24/analysis/153b_diagnostic")
RESULT_PATH = Path("results/validations/2026-03-24/verification_results/153b_diagnostic.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EVAL = 512
EVAL_BATCH = 64
H, F_LEN, DIM = 30, 30, 750
TRAIN_END = 4040
SEED = 42


def make_serializable(obj):
    """Convert numpy types for JSON serialization."""
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("153b Diagnostic: Conditional One-Shot FM — Residual Prediction")
    print("=" * 60)

    # ─── Load model ───
    print(f"Loading model from {MODEL_PATH}...")
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    cfg = ckpt["config"]
    print(f"  Config: {cfg}")

    model = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], cond_dim=cfg["cond_dim"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Residual stats from checkpoint
    res_mean = ckpt["res_mean"]  # (1, 750) numpy
    res_std = ckpt["res_std"]    # (1, 750) numpy
    print(f"  res_mean range: [{res_mean.min():.6f}, {res_mean.max():.6f}]")
    print(f"  res_std range: [{res_std.min():.6f}, {res_std.max():.6f}]")

    # ─── Load encoder ───
    print(f"Loading encoder from {ENCODER_PATH}...")
    encoder, cond_dim = load_encoder(ENCODER_PATH, DEVICE)
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  cond_dim={cond_dim}")

    # ─── Load data ───
    print("Loading data...")
    data = np.load(DATA_PATH)
    surfaces = data["surface"]

    # Build training windows (for GT comparison)
    train_histories = []
    train_futures = []
    train_persist = []
    for i in range(TRAIN_END - H - F_LEN + 1):
        history = surfaces[i:i+H]
        future = surfaces[i+H:i+H+F_LEN].reshape(-1)
        last_frame = surfaces[i+H-1]
        persistence = np.tile(last_frame.reshape(1, -1), (F_LEN, 1)).reshape(-1)
        train_histories.append(history)
        train_futures.append(future)
        train_persist.append(persistence)

    train_hist = np.array(train_histories, dtype=np.float32)
    train_data = np.array(train_futures, dtype=np.float32)
    train_persist_data = np.array(train_persist, dtype=np.float32)

    print(f"  Training windows: {train_data.shape}")
    print(f"  Persistence shape: {train_persist_data.shape}")

    # ─── Pre-compute encoder conditions ───
    print("Pre-computing encoder conditions...")
    train_conds = []
    with torch.no_grad():
        for i in range(0, len(train_hist), 256):
            bh = torch.from_numpy(train_hist[i:i+256]).to(DEVICE)
            c = encoder(normalize_iv(bh))
            train_conds.append(c.cpu().numpy())
    train_conds = np.concatenate(train_conds)
    print(f"  Conditions: {train_conds.shape}")

    # ─── Generate 512 samples ───
    print(f"Generating {N_EVAL} samples with Euler ODE (n_steps={cfg['n_steps']})...")
    n_steps = cfg["n_steps"]
    dt = 1.0 / n_steps

    all_samp = []
    all_persist = []
    all_idx = []
    with torch.no_grad():
        for si in range(0, N_EVAL, EVAL_BATCH):
            eb = min(EVAL_BATCH, N_EVAL - si)
            x = torch.randn(eb, DIM, device=DEVICE)
            idx = np.random.choice(len(train_conds), eb, replace=True)
            c = torch.from_numpy(train_conds[idx]).to(DEVICE)
            persist_batch = train_persist_data[idx]

            for step in range(n_steps):
                tt = torch.full((eb,), step * dt, device=DEVICE)
                x = x + model(x, tt, cond=c) * dt

            # Denormalize residual and add persistence
            res_raw = x.cpu().numpy() * res_std + res_mean
            samples_batch = persist_batch + res_raw
            samples_batch_clipped = np.clip(samples_batch, 0, 1)
            all_samp.append(samples_batch_clipped)
            all_persist.append(persist_batch)
            all_idx.append(idx)

    samples = np.concatenate(all_samp)       # (512, 750) absolute IV
    persist_all = np.concatenate(all_persist) # (512, 750) persistence
    sample_idx = np.concatenate(all_idx)
    gt_matched = train_data[sample_idx]       # matched GT futures

    print(f"  Generated samples shape: {samples.shape}")
    print(f"  Sample range: [{samples.min():.4f}, {samples.max():.4f}]")

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSTIC 1: Full metric table (evaluate_samples)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1: Full Metric Table")
    print("=" * 60)

    m = evaluate_samples(samples, train_data)
    for k, v in m.items():
        print(f"  {k}: {v}")

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSTIC 2: Level bias (per-cell)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: Level Bias (per-cell mean(gen) - mean(GT))")
    print("=" * 60)

    # Reshape to (N, 30, 25) for per-cell analysis
    samples_3d = samples.reshape(-1, F_LEN, 25)
    gt_3d = train_data.reshape(-1, F_LEN, 25)

    # Mean across samples and time for each cell
    gen_cell_mean = samples_3d.mean(axis=(0, 1))  # (25,)
    gt_cell_mean = gt_3d.mean(axis=(0, 1))        # (25,)
    bias = gen_cell_mean - gt_cell_mean

    print(f"  Mean absolute bias: {np.abs(bias).mean():.6f}")
    print(f"  Max absolute bias: {np.abs(bias).max():.6f}")
    print(f"  Bias range: [{bias.min():.6f}, {bias.max():.6f}]")
    print(f"  Per-cell bias (5x5 grid):")
    bias_grid = bias.reshape(5, 5)
    for row in range(5):
        print(f"    {' '.join(f'{bias_grid[row, c]:+.5f}' for c in range(5))}")

    level_bias_results = {
        "mean_abs_bias": float(np.abs(bias).mean()),
        "max_abs_bias": float(np.abs(bias).max()),
        "bias_per_cell": bias.tolist(),
        "gen_cell_mean": gen_cell_mean.tolist(),
        "gt_cell_mean": gt_cell_mean.tolist(),
    }

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSTIC 3: Kurtosis per-horizon
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3: Kurtosis per-horizon (h=1, h=15, h=30)")
    print("=" * 60)

    # Daily changes: (N, 29, 25)
    gen_changes = np.diff(samples_3d, axis=1)
    gt_changes = np.diff(gt_3d, axis=1)

    horizons = {"h1": 0, "h15": 14, "h30": 28}
    kurt_results = {}
    for name, idx in horizons.items():
        gen_k = kurtosis(gen_changes[:, idx, :].flatten(), fisher=True)
        gt_k = kurtosis(gt_changes[:, idx, :].flatten(), fisher=True)
        ratio = gen_k / (gt_k + 1e-6)
        print(f"  {name}: gen={gen_k:.3f}, GT={gt_k:.3f}, ratio={ratio:.3f}")
        kurt_results[name] = {"gen": float(gen_k), "gt": float(gt_k), "ratio": float(ratio)}

    # Overall kurtosis (all horizons combined)
    gen_k_all = kurtosis(gen_changes.flatten(), fisher=True)
    gt_k_all = kurtosis(gt_changes.flatten(), fisher=True)
    ratio_all = gen_k_all / (gt_k_all + 1e-6)
    print(f"  ALL: gen={gen_k_all:.3f}, GT={gt_k_all:.3f}, ratio={ratio_all:.3f}")
    kurt_results["all"] = {"gen": float(gen_k_all), "gt": float(gt_k_all), "ratio": float(ratio_all)}

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSTIC 4: KS on levels AND changes
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: KS Tests (levels + changes)")
    print("=" * 60)

    # KS on daily changes (per cell)
    ks_changes = {}
    ks_changes_pass = 0
    for c in range(25):
        stat, pval = ks_2samp(gen_changes[:, :, c].flatten(), gt_changes[:, :, c].flatten())
        ks_changes[f"cell_{c}"] = {"stat": float(stat), "pval": float(pval)}
        if stat < 0.15:
            ks_changes_pass += 1
    print(f"  KS on changes: {ks_changes_pass}/25 pass (threshold < 0.15)")

    # KS on IV levels (per cell)
    ks_levels = {}
    ks_levels_pass = 0
    for c in range(25):
        stat, pval = ks_2samp(samples_3d[:, :, c].flatten(), gt_3d[:, :, c].flatten())
        ks_levels[f"cell_{c}"] = {"stat": float(stat), "pval": float(pval)}
        if stat < 0.15:
            ks_levels_pass += 1
    print(f"  KS on levels: {ks_levels_pass}/25 pass (threshold < 0.15)")

    # Worst KS stats
    change_stats = [ks_changes[f"cell_{c}"]["stat"] for c in range(25)]
    level_stats = [ks_levels[f"cell_{c}"]["stat"] for c in range(25)]
    print(f"  KS changes: mean={np.mean(change_stats):.4f}, max={np.max(change_stats):.4f}")
    print(f"  KS levels: mean={np.mean(level_stats):.4f}, max={np.max(level_stats):.4f}")

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSTIC 5: Spread analysis (h1 vs h30)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 5: Spread Analysis (h1 vs h30)")
    print("=" * 60)

    # Spread = std across samples, per horizon/cell
    spreads = samples_3d.std(axis=0)  # (30, 25)
    spread_per_h = spreads.mean(axis=1)  # (30,)

    spread_h1 = float(spread_per_h[0])
    spread_h15 = float(spread_per_h[14])
    spread_h30 = float(spread_per_h[29])

    print(f"  Spread h1:  {spread_h1:.6f}")
    print(f"  Spread h15: {spread_h15:.6f}")
    print(f"  Spread h30: {spread_h30:.6f}")
    print(f"  h30/h1 ratio: {spread_h30 / (spread_h1 + 1e-10):.3f}")

    # Monotonicity check
    mono = all(spread_per_h[i+1] >= spread_per_h[i] * 0.99 for i in range(len(spread_per_h)-1))
    print(f"  Monotonic (with 1% tolerance): {mono}")

    # Full per-horizon spread
    print(f"  Per-horizon spread: {[f'{s:.5f}' for s in spread_per_h]}")

    spread_results = {
        "h1": spread_h1, "h15": spread_h15, "h30": spread_h30,
        "h30_h1_ratio": float(spread_h30 / (spread_h1 + 1e-10)),
        "monotonic": bool(mono),
        "per_horizon": spread_per_h.tolist(),
    }

    # ═══════════════════════════════════════════════════════════
    # EXTRA: PC alignment (PC1-5)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXTRA: PC Alignment (PC1-5)")
    print("=" * 60)

    gen_ch = np.diff(samples_3d, axis=1).reshape(-1, 25)
    gt_ch = np.diff(gt_3d, axis=1).reshape(-1, 25)
    gen_corr = np.corrcoef(gen_ch.T)
    gt_corr = np.corrcoef(gt_ch.T)

    gt_vecs = np.linalg.eigh(gt_corr)[1][:, ::-1]
    gen_vecs = np.linalg.eigh(gen_corr)[1][:, ::-1]

    pc_alignments = {}
    for k in range(min(5, 25)):
        alignment = abs(float(np.dot(gt_vecs[:, k], gen_vecs[:, k])))
        pc_alignments[f"PC{k+1}"] = alignment
        print(f"  PC{k+1}: {alignment:.4f}")

    # ═══════════════════════════════════════════════════════════
    # EXTRA: Effective rank details
    # ═══════════════════════════════════════════════════════════
    def eff_rank(corr):
        ev = np.linalg.eigvalsh(corr)[::-1]
        ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10)
        p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    gen_er = eff_rank(gen_corr)
    gt_er = eff_rank(gt_corr)
    print(f"\n  Effective rank: gen={gen_er:.3f}, GT={gt_er:.3f}, ratio={gen_er/gt_er:.3f}")

    # ═══════════════════════════════════════════════════════════
    # Compare with 153a (load training_history if available)
    # ═══════════════════════════════════════════════════════════
    comparison = {}
    try:
        import glob
        for p in ["models/backfill/flow_153a/training_history.json"]:
            if Path(p).exists():
                with open(p) as f:
                    hist = json.load(f)
                last = hist[-1] if hist else {}
                comparison["153a_final"] = last
                print(f"\n  153a comparison (final epoch): {last}")
    except Exception as e:
        print(f"  Could not load 153a for comparison: {e}")

    # ═══════════════════════════════════════════════════════════
    # Assemble and save results
    # ═══════════════════════════════════════════════════════════
    results = {
        "model": "153b",
        "model_path": MODEL_PATH,
        "config": cfg,
        "n_samples": N_EVAL,
        "n_steps": n_steps,
        "device": DEVICE,
        "seed": SEED,
        "prediction_mode": "residual",
        "metrics": {
            "evaluate_samples": m,
            "level_bias": level_bias_results,
            "kurtosis_per_horizon": kurt_results,
            "ks_changes": {
                "pass_count": ks_changes_pass,
                "total": 25,
                "per_cell": ks_changes,
                "mean_stat": float(np.mean(change_stats)),
                "max_stat": float(np.max(change_stats)),
            },
            "ks_levels": {
                "pass_count": ks_levels_pass,
                "total": 25,
                "per_cell": ks_levels,
                "mean_stat": float(np.mean(level_stats)),
                "max_stat": float(np.max(level_stats)),
            },
            "spread": spread_results,
            "pc_alignment": pc_alignments,
            "effective_rank": {
                "gen": gen_er, "gt": gt_er, "ratio": gen_er / gt_er,
            },
        },
        "comparison_153a": comparison,
    }

    results = make_serializable(results)

    # Save detailed results
    with open(OUTPUT_DIR / "full_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full diagnostics saved to {OUTPUT_DIR / 'full_diagnostics.json'}")

    # Save verification result
    verification = {
        "task": "153b_diagnostic_persist",
        "status": "COMPLETE",
        "model": "153b (conditional one-shot FM, residual prediction)",
        "model_path": MODEL_PATH,
        "n_samples": N_EVAL,
        "key_metrics": {
            "eff_rank": m["eff_rank"],
            "gt_eff_rank": m["gt_eff_rank"],
            "eff_rank_ratio": m["eff_rank"] / m["gt_eff_rank"],
            "pc1": m["pc1"],
            "pc2": m["pc2"],
            "frob": m["frob"],
            "ks_changes_pass": f"{ks_changes_pass}/25",
            "ks_levels_pass": f"{ks_levels_pass}/25",
            "kurt_ratio_all": kurt_results["all"]["ratio"],
            "kurt_h1": kurt_results["h1"]["ratio"],
            "kurt_h15": kurt_results["h15"]["ratio"],
            "kurt_h30": kurt_results["h30"]["ratio"],
            "spread_h1": spread_h1,
            "spread_h30": spread_h30,
            "spread_h30_h1_ratio": spread_results["h30_h1_ratio"],
            "spread_monotonic": mono,
            "mean_abs_bias": level_bias_results["mean_abs_bias"],
            "max_abs_bias": level_bias_results["max_abs_bias"],
        },
        "output_files": [
            str(OUTPUT_DIR / "full_diagnostics.json"),
        ],
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"  Verification result saved to {RESULT_PATH}")

    return results


if __name__ == "__main__":
    main()
