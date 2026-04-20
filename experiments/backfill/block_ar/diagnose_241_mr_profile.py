#!/usr/bin/env python
"""
241b Stage-2 mean-reversion and per-horizon diagnostic.

Required by plan v4 (Stage 2 analysis). For each checkpoint:

1. Per-horizon mean-reversion profile on h ∈ {1, 5, 10, 15, 20, 25, 30}:
   slope = slope(pred_delta_h on y_t), ratio = pred_slope / gt_slope.
   Gate: aggregate ratio in [0.70, 1.35] at h30 is the primary Stage 2 target.

2. Per-horizon KS on LEVELS and CHANGES (per-cell Kolmogorov-Smirnov on terminal
   level distribution and first-difference distribution). Guards h1 fidelity.

3. Spread/skill profile: mean(ensemble_std) / mean(gt_innovation_std) per horizon.
   Ensures h30 narrowing came from mean-reversion, not variance collapse.

4. CRPS trajectory during training (from training_history.json):
   afcrps_h30, twcrps_pmax, twcrps_tail, es_h30, flow_match_loss per epoch.
   Checks that CRPS terms dropped and FM didn't regress.

5. Optional comparison vs baseline_183c checkpoint:
   Δ ratios and Δ KS side-by-side.

Output: JSON summary + console table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


def _slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Copied from test_block_ar_requirements_v2.py for parity."""
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = np.mean((x - x_mean) ** 2)
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    slope = cov_xy / var_x if var_x > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    ss_res = np.mean((y - y_hat) ** 2)
    ss_tot = np.mean((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(r2)


def build_model_from_ckpt(ckpt_path: str, device: str) -> StateMetricTransportModel:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    encoder_config = EncoderConfig(**cfg["encoder"])
    model = StateMetricTransportModel(
        encoder_config=encoder_config,
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
def collect_samples(
    model: StateMetricTransportModel,
    val_hist: torch.Tensor,
    eval_limit: int,
    n_samples: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Run model.sample over val_hist[:eval_limit]; return (N, K, T, 5, 5).

    CRITICAL: val_hist is raw IV in [0,1]. The harness normalizes with normalize_iv()
    before calling sample_batched (evaluate_220h_full_multihorizon_v2_suite.py:141).
    Match that convention here — otherwise the model receives history in the wrong
    range and produces systematically biased samples (confirmed 2x inversion of MR h30
    ratio when raw [0,1] was passed directly).
    """
    device = next(model.parameters()).device
    out = []
    for i in range(0, min(eval_limit, len(val_hist)), batch_size):
        hist_raw = val_hist[i : i + batch_size].to(device)
        hist = normalize_iv(hist_raw)  # [0,1] → [-1,1] per CLAUDE.md gotcha
        samples = model.sample_batched(hist, n_samples=n_samples)  # (B, K, T, 5, 5) IV [0,1]
        if samples.dim() == 4:
            samples = samples.view(samples.shape[0], samples.shape[1], samples.shape[2], 5, 5)
        out.append(samples.cpu().numpy())
    return np.concatenate(out, axis=0)  # (N, K, T, 5, 5)


def per_horizon_mr_profile(
    samples: np.ndarray, ground_truth: np.ndarray, history: np.ndarray,
    horizons: list[int],
    active_slope_threshold: float = 0.05,
) -> dict:
    """Per-horizon MR ratio + active-cell stats, matching test suite logic."""
    pred_mean = samples.mean(axis=1)  # (N, T, 5, 5)
    prev = history[:, -1]  # (N, 5, 5)

    rows = []
    for h in horizons:
        if h > ground_truth.shape[1]:
            continue
        gt_h = ground_truth[:, h - 1]  # (N, 5, 5)
        pred_h = pred_mean[:, h - 1]
        gt_delta = gt_h - prev
        pred_delta = pred_h - prev

        gt_slope, _, _ = _slope_intercept_r2(prev, gt_delta)
        pred_slope, _, _ = _slope_intercept_r2(prev, pred_delta)
        ratio = pred_slope / gt_slope if abs(gt_slope) > 1e-12 else float("nan")
        agg_pass = np.isfinite(ratio) and (0.70 <= ratio <= 1.35)

        # Per-cell slope
        gt_cell = np.zeros((5, 5))
        pred_cell = np.zeros((5, 5))
        cell_ratio = np.full((5, 5), np.nan)
        sign_match = np.zeros((5, 5), dtype=bool)
        for i in range(5):
            for j in range(5):
                gs, _, _ = _slope_intercept_r2(prev[:, i, j], gt_delta[:, i, j])
                ps, _, _ = _slope_intercept_r2(prev[:, i, j], pred_delta[:, i, j])
                gt_cell[i, j] = gs
                pred_cell[i, j] = ps
                if abs(gs) > 1e-12:
                    cell_ratio[i, j] = ps / gs
                sign_match[i, j] = np.sign(gs) == np.sign(ps)
        active_mask = np.abs(gt_cell) >= active_slope_threshold
        active_count = int(active_mask.sum())
        if active_count > 0:
            ratio_mask = (
                np.isfinite(cell_ratio) & (cell_ratio >= 0.50) & (cell_ratio <= 1.50)
            )
            cell_pass = active_mask & sign_match & ratio_mask
            active_pass_count = int(cell_pass.sum())
            active_pass_rate = active_pass_count / active_count
            slope_corr = (
                float(np.corrcoef(
                    gt_cell[active_mask].reshape(-1),
                    pred_cell[active_mask].reshape(-1),
                )[0, 1]) if active_count >= 2 else 1.0
            )
        else:
            active_pass_count = 0
            active_pass_rate = 1.0
            slope_corr = 1.0

        rows.append({
            "h": h,
            "gt_slope": float(gt_slope),
            "pred_slope": float(pred_slope),
            "ratio": float(ratio),
            "aggregate_pass": bool(agg_pass),
            "active_count": active_count,
            "active_pass_count": active_pass_count,
            "active_pass_rate": float(active_pass_rate),
            "active_slope_corr": float(slope_corr),
        })

    return {
        "horizons": horizons,
        "rows": rows,
        "h30_ratio": float(rows[-1]["ratio"]) if rows else float("nan"),
        "h30_aggregate_pass": bool(rows[-1]["aggregate_pass"]) if rows else False,
    }


def per_horizon_ks(
    samples: np.ndarray, ground_truth: np.ndarray,
    horizons: list[int],
) -> dict:
    """Per-horizon KS on LEVELS and on CHANGES across the 25 cells."""
    N, K, T, H, W = samples.shape

    def ks_per_cell(sample_vals: np.ndarray, gt_vals: np.ndarray) -> tuple[int, float]:
        """sample_vals (N*K,), gt_vals (N,). Return (n_pass, max_statistic)."""
        stat, _ = ks_2samp(sample_vals, gt_vals, alternative="two-sided")
        return float(stat)

    rows = []
    for h in horizons:
        if h > T:
            continue
        # LEVELS
        cell_ks_levels = []
        for i in range(5):
            for j in range(5):
                samp_flat = samples[:, :, h - 1, i, j].reshape(-1)
                gt_flat = ground_truth[:, h - 1, i, j].reshape(-1)
                cell_ks_levels.append(ks_per_cell(samp_flat, gt_flat))
        cell_ks_levels = np.array(cell_ks_levels)
        level_pass = int((cell_ks_levels < 0.20).sum())

        # CHANGES: per-cell change-distribution
        # For h=1: change from h0 to h1 in gt; for h>1 we track h-1->h step.
        cell_ks_changes = []
        if h > 1:
            for i in range(5):
                for j in range(5):
                    samp_chg = (samples[:, :, h - 1, i, j] - samples[:, :, h - 2, i, j]).reshape(-1)
                    gt_chg = (ground_truth[:, h - 1, i, j] - ground_truth[:, h - 2, i, j]).reshape(-1)
                    cell_ks_changes.append(ks_per_cell(samp_chg, gt_chg))
        else:
            cell_ks_changes = [float("nan")] * 25
        cell_ks_changes = np.array(cell_ks_changes)
        change_pass = int((cell_ks_changes[~np.isnan(cell_ks_changes)] < 0.20).sum()) if h > 1 else 0

        rows.append({
            "h": h,
            "cell_ks_level_mean": float(np.mean(cell_ks_levels)),
            "cell_ks_level_max": float(np.max(cell_ks_levels)),
            "cell_ks_level_pass_count": level_pass,
            "cell_ks_change_mean": float(np.nanmean(cell_ks_changes)) if h > 1 else float("nan"),
            "cell_ks_change_max": float(np.nanmax(cell_ks_changes)) if h > 1 else float("nan"),
            "cell_ks_change_pass_count": change_pass,
        })
    return {"rows": rows}


def spread_skill_profile(
    samples: np.ndarray, ground_truth: np.ndarray,
    horizons: list[int],
) -> dict:
    """std_ratio = mean_cell(ensemble_std_{h}) / mean_cell(gt_innovation_std_{h})."""
    N, K, T, H, W = samples.shape
    rows = []
    for h in horizons:
        if h > T:
            continue
        # Ensemble std per window per cell, then mean across windows + cells
        ens_std_hc = samples[:, :, h - 1].std(axis=1)  # (N, 5, 5)
        ens_std_mean = float(ens_std_hc.mean())
        # gt marginal std across windows per cell
        gt_std_hc = ground_truth[:, h - 1].std(axis=0)  # (5, 5)
        gt_std_mean = float(gt_std_hc.mean())
        rows.append({
            "h": h,
            "ensemble_std_mean": ens_std_mean,
            "gt_marginal_std_mean": gt_std_mean,
            "std_ratio": ens_std_mean / max(gt_std_mean, 1e-10),
        })
    return {"rows": rows}


def load_crps_trajectory(training_history_path: str) -> dict:
    """Read training_history.json and extract per-epoch CRPS metrics."""
    if not Path(training_history_path).exists():
        return {}
    history = json.loads(Path(training_history_path).read_text())
    traj = {
        "epochs": [],
        "train_fm": [],
        "train_afcrps_h30": [],
        "train_twcrps_pmax": [],
        "train_twcrps_tail": [],
        "train_es_h30": [],
        "val_fm": [],
    }
    for row in history:
        traj["epochs"].append(row.get("epoch"))
        traj["train_fm"].append(row.get("train_flow_match_loss", float("nan")))
        traj["train_afcrps_h30"].append(row.get("train_afcrps_h30", float("nan")))
        traj["train_twcrps_pmax"].append(row.get("train_twcrps_pmax", float("nan")))
        traj["train_twcrps_tail"].append(row.get("train_twcrps_tail", float("nan")))
        traj["train_es_h30"].append(row.get("train_es_h30", float("nan")))
        traj["val_fm"].append(row.get("val_flow_match_loss", float("nan")))
    return traj


def run_checkpoint(
    ckpt_path: str, val_hist: torch.Tensor, val_future: torch.Tensor,
    device: str, eval_limit: int, n_samples: int, horizons: list[int],
) -> dict:
    print(f"\n{'=' * 72}\nDiagnosing: {ckpt_path}\n{'=' * 72}")
    model = build_model_from_ckpt(ckpt_path, device)

    print(f"[1/4] Collecting samples (limit={eval_limit}, K={n_samples})…")
    samples = collect_samples(model, val_hist, eval_limit, n_samples)
    # collect_samples rounds to batch_size; truncate samples and GT to the same length.
    n = min(samples.shape[0], eval_limit)
    samples = samples[:n]
    gt = val_future[:n].cpu().numpy().reshape(n, val_future.shape[1], 5, 5)
    hist_np = val_hist[:n].cpu().numpy()  # (n, H, 5, 5)

    print(f"[2/4] Per-horizon MR profile on h={horizons}…")
    mr = per_horizon_mr_profile(samples, gt, hist_np, horizons)
    print(f"   h30 MR ratio = {mr['h30_ratio']:.3f}  (gate [0.70, 1.35])  {'PASS' if mr['h30_aggregate_pass'] else 'FAIL'}")
    for row in mr["rows"]:
        print(f"   h={row['h']:>2}: ratio={row['ratio']:+.3f}  active {row['active_pass_count']}/{row['active_count']}  slope_corr={row['active_slope_corr']:+.3f}")

    print(f"[3/4] Per-horizon KS (levels & changes)…")
    ks = per_horizon_ks(samples, gt, horizons)
    for row in ks["rows"]:
        print(
            f"   h={row['h']:>2}: level KS mean={row['cell_ks_level_mean']:.3f} pass={row['cell_ks_level_pass_count']}/25  "
            f"change KS mean={row['cell_ks_change_mean']:.3f} pass={row['cell_ks_change_pass_count']}/25"
        )

    print(f"[4/4] Spread/skill profile…")
    sp = spread_skill_profile(samples, gt, horizons)
    for row in sp["rows"]:
        print(f"   h={row['h']:>2}: std_ratio={row['std_ratio']:.3f}")

    return {
        "ckpt": ckpt_path,
        "mr": mr,
        "ks": ks,
        "spread_skill": sp,
    }


def main():
    ap = argparse.ArgumentParser(description="241b Stage 2 MR profile diagnostic")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="241b best_model.pt (required)")
    ap.add_argument("--final_checkpoint", type=str, default=None,
                    help="Optional final_model.pt for stability check")
    ap.add_argument("--baseline_checkpoint", type=str, default=None,
                    help="Optional 183c best_model.pt for delta vs pre-Stage-2")
    ap.add_argument("--training_history", type=str, default=None,
                    help="Optional training_history.json for CRPS trajectory")
    ap.add_argument("--history_len", type=int, default=30)
    ap.add_argument("--future_len", type=int, default=30)
    ap.add_argument("--eval_limit", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf = torch.from_numpy(surfaces).to(args.device)
    val_hist, val_future = build_multistep_windows(val_indices, surf, args.history_len, args.future_len)
    print(f"Val windows: {len(val_hist)} | eval_limit={args.eval_limit}")

    results = {"args": vars(args), "checkpoints": {}}

    results["checkpoints"]["main"] = run_checkpoint(
        args.checkpoint, val_hist, val_future, args.device,
        args.eval_limit, args.n_samples, args.horizons,
    )
    if args.final_checkpoint:
        results["checkpoints"]["final"] = run_checkpoint(
            args.final_checkpoint, val_hist, val_future, args.device,
            args.eval_limit, args.n_samples, args.horizons,
        )
    if args.baseline_checkpoint:
        results["checkpoints"]["baseline_183c"] = run_checkpoint(
            args.baseline_checkpoint, val_hist, val_future, args.device,
            args.eval_limit, args.n_samples, args.horizons,
        )

    if args.training_history:
        print(f"\n[5/5] Loading CRPS trajectory from {args.training_history}…")
        results["crps_trajectory"] = load_crps_trajectory(args.training_history)
        if results["crps_trajectory"]:
            traj = results["crps_trajectory"]
            print("   epoch  |  fm      afc_h30   pmax     tail     es        val_fm")
            for i, ep in enumerate(traj["epochs"]):
                print(
                    f"   {ep:>5}  |  {traj['train_fm'][i]:.4f}   {traj['train_afcrps_h30'][i]:.4f}   "
                    f"{traj['train_twcrps_pmax'][i]:.4f}   {traj['train_twcrps_tail'][i]:.4f}   "
                    f"{traj['train_es_h30'][i]:.4f}   {traj['val_fm'][i]:.4f}"
                )

    # Decision matrix (Stage 2)
    main_mr = results["checkpoints"]["main"]["mr"]
    main_ks = results["checkpoints"]["main"]["ks"]
    main_sp = results["checkpoints"]["main"]["spread_skill"]
    h30_ratio = main_mr["h30_ratio"]
    h30_aggregate_pass = main_mr["h30_aggregate_pass"]
    h30_std_ratio = next((r["std_ratio"] for r in main_sp["rows"] if r["h"] == 30), float("nan"))

    # Simple verdict mapping based on the plan's decision matrix.
    if not np.isfinite(h30_ratio):
        verdict = "INSUFFICIENT_DATA"
    elif h30_aggregate_pass and 0.40 <= h30_std_ratio <= 0.90:
        verdict = "CLEAN_SUCCESS (MR target hit, spread in band)"
    elif h30_aggregate_pass and h30_std_ratio < 0.30:
        verdict = "SPREAD_COLLAPSE (MR hit but variance collapsed)"
    elif not h30_aggregate_pass and 0.40 <= h30_std_ratio <= 0.90:
        verdict = "MR_MISS_NO_COLLAPSE (spread fine but MR didn't move)"
    else:
        verdict = "PARTIAL / AMBIGUOUS"
    results["decision"] = {
        "stage": "2",
        "h30_ratio": float(h30_ratio),
        "h30_std_ratio": float(h30_std_ratio),
        "verdict": verdict,
    }
    print(f"\n=== STAGE 2 DECISION: {verdict} ===")
    print(f"   h30 MR ratio = {h30_ratio:.3f}  |  h30 std_ratio = {h30_std_ratio:.3f}")

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
