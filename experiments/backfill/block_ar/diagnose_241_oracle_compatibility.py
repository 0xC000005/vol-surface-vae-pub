#!/usr/bin/env python
"""
Oracle-compatibility test: are per-cell KS, MR slope, and kurtosis fundamentally
compatible constraints, or does satisfying one inherently break another?

Method: take 183c's ensemble samples (they have reasonable per-cell spread and
kurtosis), re-center each window's ensemble so the ENSEMBLE MEAN = GT for that
window. This yields an "oracle-centered" ensemble that by construction:
  - has ensemble_mean = GT at every horizon (MR ratio must be 1.0 exactly)
  - preserves the per-cell spread + tail shape (kurt and level-KS should be close
    to 183c's since tail shape is unchanged)
  - preserves the change-KS pattern (changes = differences of levels; the same
    additive shift is applied at consecutive time steps, so Δ_samples - mean(Δ_samples)
    is identical to 183c's; the mean shift cancels in Δ)

Key question: does this ORACLE pass level/change KS AND kurtosis AND MR simultaneously?
  - YES → constraints are compatible. Our model just can't predict pred_mean well.
    The multi-CRPS fight is a MODEL-CAPACITY/OPTIMISATION issue, not a fundamental
    constraint conflict.
  - NO → the metrics are inherently in tension (e.g. shifting ensemble center moves
    the distribution in a way that breaks marginal match). Suggests some metrics
    need per-cell targeting.

Also run "183c-as-is" for comparison baseline and "GT-replicate-K" for degenerate
test (no spread) as sanity checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


def build_model(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = StateMetricTransportModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"], flow_config=cfg["flow"], path_config=cfg["path"],
        prior_config=cfg["prior"], integrated_config=cfg["integrated"],
        state_config=cfg["state"], metric_config=cfg["metric"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_model_samples(model, val_hist, eval_limit, n_samples, batch_size=16):
    device = next(model.parameters()).device
    out = []
    for i in range(0, min(eval_limit, len(val_hist)), batch_size):
        hist_raw = val_hist[i : i + batch_size].to(device)
        hist = normalize_iv(hist_raw)
        samples = model.sample_batched(hist, n_samples=n_samples)
        if samples.dim() == 4:
            samples = samples.view(samples.shape[0], samples.shape[1], samples.shape[2], 5, 5)
        out.append(samples.cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_metrics(samples: np.ndarray, gt: np.ndarray, hist: np.ndarray, label: str):
    """Run per-horizon MR, kurt, level-KS, change-KS on (N, K, T, 5, 5) samples."""
    from scipy.stats import ks_2samp
    N, K, T, H, W = samples.shape
    prev = hist[:, -1]
    pred_mean = samples.mean(axis=1)

    def slope(x, y):
        x = x.reshape(-1).astype(np.float64)
        y = y.reshape(-1).astype(np.float64)
        xm, ym = x.mean(), y.mean()
        vx = ((x - xm) ** 2).mean()
        return float(((x - xm) * (y - ym)).mean() / vx) if vx > 1e-12 else 0.0

    # MR per-horizon
    mr_rows = []
    for h in [1, 7, 14, 30]:
        gt_d = gt[:, h - 1] - prev
        pd_d = pred_mean[:, h - 1] - prev
        gs = slope(prev, gt_d); ps = slope(prev, pd_d)
        ratio = ps / gs if abs(gs) > 1e-12 else float("nan")
        mr_rows.append({"h": h, "gt_slope": gs, "pred_slope": ps, "ratio": ratio})

    # Kurtosis: compute from daily changes, per-cell, compare to GT.
    sample_changes = np.diff(samples, axis=2)  # (N, K, T-1, 5, 5)
    gt_changes = np.diff(gt, axis=1)  # (N, T-1, 5, 5)
    def kurt(x, axis):
        m = x.mean(axis=axis, keepdims=True)
        s = x.std(axis=axis, keepdims=True) + 1e-12
        return ((x - m) / s) ** 4
    samp_flat_k = sample_changes.reshape(-1, 5, 5)  # (N*K*(T-1), 5, 5)
    gt_flat_k   = gt_changes.reshape(-1, 5, 5)
    samp_kurt = np.mean((samp_flat_k - samp_flat_k.mean(axis=0)) ** 4, axis=0) / (samp_flat_k.std(axis=0) ** 4 + 1e-12)
    gt_kurt   = np.mean((gt_flat_k - gt_flat_k.mean(axis=0)) ** 4, axis=0) / (gt_flat_k.std(axis=0) ** 4 + 1e-12)
    kurt_ratio_per_cell = samp_kurt / np.clip(gt_kurt, 1e-12, None)
    kurt_ratio_mean = float(np.mean(samp_kurt) / max(np.mean(gt_kurt), 1e-12))

    # Level KS at h30: per-cell distribution comparison
    level_h30_ks = []
    for i in range(5):
        for j in range(5):
            samp_dist = samples[:, :, -1, i, j].reshape(-1)
            gt_dist = gt[:, -1, i, j]
            stat, _ = ks_2samp(samp_dist, gt_dist, alternative="two-sided")
            level_h30_ks.append(float(stat))
    level_h30_ks = np.array(level_h30_ks)
    level_h30_n_pass = int((level_h30_ks < 0.20).sum())

    # Change KS: concat across all horizons per cell
    change_ks = []
    for i in range(5):
        for j in range(5):
            samp_chg = sample_changes[:, :, :, i, j].reshape(-1)
            gt_chg = gt_changes[:, :, i, j].reshape(-1)
            stat, _ = ks_2samp(samp_chg, gt_chg, alternative="two-sided")
            change_ks.append(float(stat))
    change_ks = np.array(change_ks)
    change_n_pass = int((change_ks < 0.20).sum())

    # Coverage at h30: 90% CI
    q05 = np.quantile(samples[:, :, -1], 0.05, axis=1)  # (N, 5, 5)
    q95 = np.quantile(samples[:, :, -1], 0.95, axis=1)
    gt_h30 = gt[:, -1]
    inside = ((gt_h30 >= q05) & (gt_h30 <= q95)).astype(float)
    cov_h30 = float(inside.mean())

    return {
        "label": label,
        "mr": mr_rows,
        "kurt_ratio_mean": kurt_ratio_mean,
        "kurt_per_cell_min": float(kurt_ratio_per_cell.min()),
        "kurt_per_cell_max": float(kurt_ratio_per_cell.max()),
        "level_h30_KS_median": float(np.median(level_h30_ks)),
        "level_h30_KS_max": float(level_h30_ks.max()),
        "level_h30_n_pass_out_of_25": level_h30_n_pass,
        "change_KS_median": float(np.median(change_ks)),
        "change_KS_max": float(change_ks.max()),
        "change_n_pass_out_of_25": change_n_pass,
        "cov90_h30": cov_h30,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", type=str, required=True,
                    help="183c checkpoint for obtaining real ensemble with reasonable spread")
    ap.add_argument("--eval_limit", type=int, default=441)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--history_len", type=int, default=30)
    ap.add_argument("--future_len", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf = torch.from_numpy(surfaces).to(args.device)
    val_hist, val_future = build_multistep_windows(val_indices, surf, args.history_len, args.future_len)
    print(f"Val windows: {len(val_hist)}")

    model = build_model(args.model_checkpoint, args.device)
    print("Collecting 183c ensemble samples...")
    samples = collect_model_samples(model, val_hist, args.eval_limit, args.n_samples)
    n = min(samples.shape[0], args.eval_limit)
    samples = samples[:n]
    gt = val_future[:n].cpu().numpy().reshape(n, args.future_len, 5, 5)
    hist_np = val_hist[:n].cpu().numpy()

    results = []

    # 1. BASELINE: 183c as-is
    print("\n[1/4] 183c as-is baseline...")
    results.append(compute_metrics(samples, gt, hist_np, "183c_baseline"))

    # 2. ORACLE 1: perfect-mean, preserved-spread. Subtract ensemble mean, add GT.
    # This gives ensemble_mean = GT exactly at every horizon, with 183c's spread/tails.
    print("[2/4] Oracle A: 183c spread, GT centered...")
    ens_mean = samples.mean(axis=1, keepdims=True)  # (N, 1, T, 5, 5)
    oracle_a = samples - ens_mean + gt[:, np.newaxis]  # ensemble_mean now = GT exactly
    results.append(compute_metrics(oracle_a, gt, hist_np, "oracle_gt_centered_183c_spread"))

    # 3. ORACLE 2: perfect-mean + matched-variance. Rescale spread per-cell per-horizon
    # to match GT innovation standard deviation across the val set.
    print("[3/4] Oracle B: 183c center, GT-matched spread...")
    gt_std_per_cell_per_h = gt.std(axis=0)  # (T, 5, 5) marginal std across windows
    # 183c spread per-window per-cell per-h
    samp_std_per_window = samples.std(axis=1)  # (N, T, 5, 5)
    # Rescale each window's spread so mean across windows matches gt_std_per_cell_per_h
    samp_std_mean = samp_std_per_window.mean(axis=0)  # (T, 5, 5)
    scale = gt_std_per_cell_per_h / np.clip(samp_std_mean, 1e-8, None)
    oracle_b_noise = (samples - samples.mean(axis=1, keepdims=True)) * scale[np.newaxis, np.newaxis]
    oracle_b = oracle_b_noise + gt[:, np.newaxis]
    results.append(compute_metrics(oracle_b, gt, hist_np, "oracle_gt_centered_gt_spread"))

    # 4. ORACLE 3: degenerate zero-spread — GT replicated K times. Must fail coverage.
    print("[4/4] Oracle C: GT replicated K times (zero spread)...")
    oracle_c = np.tile(gt[:, np.newaxis], (1, args.n_samples, 1, 1, 1))
    results.append(compute_metrics(oracle_c, gt, hist_np, "oracle_gt_replicated"))

    # Pretty print
    print("\n\n" + "="*115)
    print(f"{'Metric':<32} {'183c baseline':>18} {'oracle_centered':>18} {'oracle_gt_spread':>18} {'oracle_gt_repl':>18}")
    print("="*115)

    def fmt(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"{v:.4f}"
        return str(v)
    keys = [
        'kurt_ratio_mean',
        'kurt_per_cell_min', 'kurt_per_cell_max',
        'level_h30_n_pass_out_of_25', 'level_h30_KS_median',
        'change_n_pass_out_of_25', 'change_KS_median',
        'cov90_h30',
    ]
    for k in keys:
        vals = [r[k] for r in results]
        print(f"{k:<32} " + "  ".join(f"{fmt(v):>18}" for v in vals))

    print("\n-- MR ratios per horizon --")
    for h_idx, h in enumerate([1, 7, 14, 30]):
        row_str = f"h={h:>2} ratio:"
        for r in results:
            ratio = r['mr'][h_idx]['ratio']
            row_str += f"  {ratio:>15.4f}"
        print(f"{row_str:<32}")

    print("\n-- Interpretation --")
    oracle_a = results[1]
    if (oracle_a['level_h30_n_pass_out_of_25'] >= 20
        and oracle_a['change_n_pass_out_of_25'] >= 20
        and 0.8 <= oracle_a['kurt_ratio_mean'] <= 1.25
        and abs(oracle_a['mr'][3]['ratio'] - 1.0) < 0.05):
        print("  ORACLE PASSES ALL: constraints are COMPATIBLE. The 241b-vs-241c trade-off")
        print("  is a model-capacity/optimization problem, NOT a fundamental tension.")
    else:
        print("  ORACLE FAILS: constraints may have fundamental tension.")
        print("  Check which of {level_KS, change_KS, kurtosis, MR} the oracle fails.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
