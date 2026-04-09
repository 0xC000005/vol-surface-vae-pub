#!/usr/bin/env python
"""
Persistence diagnostics for multi-day recursive rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    OneDayKernelRolloutWrapper,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    rollout_samples_in_batches,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import compute_acf


REGIME_NAMES = ["calm", "normal", "turb"]


def _offdiag_mean(corr: np.ndarray) -> float:
    n = corr.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = corr[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def _eff_rank(corr: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(corr)[::-1]
    eigvals = np.maximum(eigvals, 0)
    p = eigvals / max(eigvals.sum(), 1e-10)
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def _corr_summary(flat: np.ndarray) -> dict[str, float]:
    corr = np.corrcoef(flat.T)
    eigvals = np.linalg.eigvalsh(corr)[::-1]
    eigvals = np.maximum(eigvals, 0)
    return {
        "mean_corr": _offdiag_mean(corr),
        "eff_rank": _eff_rank(corr),
        "pc1_var": float(eigvals[0] / max(eigvals.sum(), 1e-10)),
    }


def _mean_acf(seqs: np.ndarray, max_lag: int) -> np.ndarray:
    acfs = []
    for seq in seqs:
        if np.std(seq) > 1e-10:
            acfs.append(compute_acf(seq, max_lag))
    if not acfs:
        return np.zeros(max_lag + 1, dtype=np.float64)
    min_len = min(len(a) for a in acfs)
    return np.mean([a[:min_len] for a in acfs], axis=0)


def _run_lengths(states_2d: np.ndarray, target_state: int) -> list[int]:
    lengths: list[int] = []
    for seq in states_2d:
        run = 0
        for val in seq:
            if int(val) == target_state:
                run += 1
            elif run > 0:
                lengths.append(run)
                run = 0
        if run > 0:
            lengths.append(run)
    return lengths


def _transition_stats(states_2d: np.ndarray) -> dict[str, Any]:
    counts = np.zeros((3, 3), dtype=np.float64)
    for seq in states_2d:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[int(a), int(b)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, np.maximum(row_sums, 1e-12))
    dwell = {
        REGIME_NAMES[s]: _run_lengths(states_2d, s)
        for s in range(3)
    }
    return {
        "transition_counts": counts.tolist(),
        "transition_probs": probs.tolist(),
        "state_share": (np.bincount(states_2d.reshape(-1), minlength=3) / max(states_2d.size, 1)).tolist(),
        "self_transition": {
            REGIME_NAMES[s]: float(probs[s, s]) if np.isfinite(probs[s, s]) else 0.0
            for s in range(3)
        },
        "dwell_mean": {
            name: float(np.mean(vals)) if vals else 0.0 for name, vals in dwell.items()
        },
        "dwell_q90": {
            name: float(np.quantile(vals, 0.9)) if vals else 0.0 for name, vals in dwell.items()
        },
    }


def _label_regimes(daily_move_score: np.ndarray, q20: float, q80: float) -> np.ndarray:
    labels = np.ones_like(daily_move_score, dtype=np.int64)
    labels[daily_move_score <= q20] = 0
    labels[daily_move_score >= q80] = 2
    return labels


def _sign_run_stats(delta_2d: np.ndarray) -> dict[str, float]:
    runs: list[int] = []
    for seq in delta_2d:
        signs = np.sign(seq)
        signs[np.abs(seq) < 1e-8] = 0
        last = 0
        run = 0
        for val in signs:
            if val == 0:
                if run > 0:
                    runs.append(run)
                    run = 0
                last = 0
                continue
            if val == last:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 1
                last = val
        if run > 0:
            runs.append(run)
    if not runs:
        return {"mean": 0.0, "q90": 0.0}
    arr = np.asarray(runs, dtype=np.float64)
    return {"mean": float(arr.mean()), "q90": float(np.quantile(arr, 0.9))}


def _jump_cluster_stats(move_score_2d: np.ndarray, threshold: float, max_lag: int = 5) -> dict[str, Any]:
    indicator = (move_score_2d > threshold).astype(np.float64)
    acf = _mean_acf(indicator, max_lag)
    return {
        "threshold": float(threshold),
        "acf": acf.tolist(),
        "lag1": float(acf[1]) if len(acf) > 1 else 0.0,
        "lag1_5_mean": float(np.mean(acf[1:])) if len(acf) > 1 else 0.0,
        "incidence": float(indicator.mean()),
    }


def _boundary_persistence(level_paths: np.ndarray, floor: float = 0.001, ceiling: float = 0.99) -> dict[str, Any]:
    any_floor = (level_paths <= floor).any(axis=(-1, -2)).astype(np.int64)
    any_ceiling = (level_paths >= ceiling).any(axis=(-1, -2)).astype(np.int64)

    def persistence(mask_2d: np.ndarray) -> tuple[float, float]:
        cur = mask_2d[:, :-1].reshape(-1)
        nxt = mask_2d[:, 1:].reshape(-1)
        active = cur == 1
        if active.sum() == 0:
            return 0.0, float(mask_2d.mean())
        return float(nxt[active].mean()), float(mask_2d.mean())

    floor_persist, floor_incidence = persistence(any_floor)
    ceil_persist, ceil_incidence = persistence(any_ceiling)
    return {
        "floor_next_day_persistence": floor_persist,
        "ceiling_next_day_persistence": ceil_persist,
        "floor_day_incidence": floor_incidence,
        "ceiling_day_incidence": ceil_incidence,
    }


def _state_conditioned_profiles(
    history_01: np.ndarray,
    ground_truth: np.ndarray,
    cond_samples: np.ndarray,
    horizons: list[int],
) -> dict[str, Any]:
    prev = history_01[:, -1]
    init_level = prev.mean(axis=(1, 2))
    hist_mean = history_01.mean(axis=(2, 3))
    init_vov = np.diff(hist_mean, axis=1).std(axis=1)
    floor_dist = prev.min(axis=(1, 2))

    gt_mean_levels = ground_truth.mean(axis=(2, 3))
    gen_mean_levels = cond_samples.mean(axis=(-1, -2))
    gen_level_mean = gen_mean_levels.mean(axis=1)
    gen_level_lo = np.quantile(gen_mean_levels, 0.05, axis=1)
    gen_level_hi = np.quantile(gen_mean_levels, 0.95, axis=1)

    level_bins = np.quantile(init_level, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    vov_bins = np.quantile(init_vov, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    out: dict[str, Any] = {
        "features": {
            "init_level_mean": float(init_level.mean()),
            "init_vov_mean": float(init_vov.mean()),
            "floor_dist_mean": float(floor_dist.mean()),
        },
        "per_horizon": {},
    }
    for h in horizons:
        idx = h - 1
        gt_drift = gt_mean_levels[:, idx] - init_level
        gen_drift = gen_level_mean[:, idx] - init_level
        gt_abs = np.abs(gt_drift)
        gen_width = gen_level_hi[:, idx] - gen_level_lo[:, idx]

        out["per_horizon"][str(h)] = {
            "spearman": {
                "level_vs_gt_drift": float(spearmanr(init_level, gt_drift).statistic),
                "level_vs_gen_drift": float(spearmanr(init_level, gen_drift).statistic),
                "vov_vs_gt_abs_move": float(spearmanr(init_vov, gt_abs).statistic),
                "vov_vs_gen_width": float(spearmanr(init_vov, gen_width).statistic),
                "floor_vs_gt_drift": float(spearmanr(floor_dist, gt_drift).statistic),
                "floor_vs_gen_drift": float(spearmanr(floor_dist, gen_drift).statistic),
            },
            "level_quintiles": [],
            "vov_quintiles": [],
        }

        for bins, feature, name in [
            (level_bins, init_level, "level_quintiles"),
            (vov_bins, init_vov, "vov_quintiles"),
        ]:
            bucket_rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                if hi == bins[-1]:
                    mask = (feature >= lo) & (feature <= hi)
                else:
                    mask = (feature >= lo) & (feature < hi)
                if not mask.any():
                    continue
                bucket_rows.append(
                    {
                        "lo": float(lo),
                        "hi": float(hi),
                        "n": int(mask.sum()),
                        "gt_drift": float(gt_drift[mask].mean()),
                        "gen_drift": float(gen_drift[mask].mean()),
                        "gt_abs_move": float(gt_abs[mask].mean()),
                        "gen_width": float(gen_width[mask].mean()),
                    }
                )
            out["per_horizon"][str(h)][name] = bucket_rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistence diagnostics for multi-day rollouts")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    wrapper = OneDayKernelRolloutWrapper(model).eval()

    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    cond_samples = rollout_samples_in_batches(
        wrapper=wrapper,
        history_norm=batch.history_norm,
        n_samples=args.samples,
        n_steps=batch.future_01.shape[1],
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    history_01 = batch.history_01.detach().cpu().numpy()
    ground_truth = batch.future_01.detach().cpu().numpy()
    prev = history_01[:, -1]

    gt_levels = np.concatenate([prev[:, None], ground_truth], axis=1)
    prev_expanded = np.broadcast_to(prev[:, None, None], (prev.shape[0], cond_samples.shape[1], 1, 5, 5))
    gen_levels = np.concatenate([prev_expanded, cond_samples], axis=2)

    gt_mean_level = gt_levels.mean(axis=(2, 3))
    gen_mean_level = gen_levels.mean(axis=(-1, -2))
    gt_atm_level = gt_levels[:, :, 2, 2]
    gen_atm_level = gen_levels[:, :, :, 2, 2]

    gt_mean_delta = np.diff(gt_mean_level, axis=1)
    gen_mean_delta = np.diff(gen_mean_level, axis=2)
    gt_abs_mean_delta = np.abs(gt_mean_delta)
    gen_abs_mean_delta = np.abs(gen_mean_delta)
    gt_atm_delta = np.diff(gt_atm_level, axis=1)
    gen_atm_delta = np.diff(gen_atm_level, axis=2)

    acf_profiles = {}
    for name, gt_seq, gen_seq in [
        ("mean_level", gt_mean_level, gen_mean_level.reshape(-1, gt_mean_level.shape[1])),
        ("mean_delta", gt_mean_delta, gen_mean_delta.reshape(-1, gt_mean_delta.shape[1])),
        ("abs_mean_delta", gt_abs_mean_delta, gen_abs_mean_delta.reshape(-1, gt_abs_mean_delta.shape[1])),
        ("atm_level", gt_atm_level, gen_atm_level.reshape(-1, gt_atm_level.shape[1])),
        ("atm_delta", gt_atm_delta, gen_atm_delta.reshape(-1, gt_atm_delta.shape[1])),
    ]:
        max_lag = min(10, gt_seq.shape[1] - 1)
        gt_acf = _mean_acf(gt_seq, max_lag)
        gen_acf = _mean_acf(gen_seq, max_lag)
        min_len = min(len(gt_acf), len(gen_acf))
        acf_profiles[name] = {
            "gt_acf": gt_acf[:min_len].tolist(),
            "gen_acf": gen_acf[:min_len].tolist(),
            "acf_correlation": float(np.corrcoef(gt_acf[:min_len], gen_acf[:min_len])[0, 1]),
            "acf_mae": float(np.mean(np.abs(gt_acf[:min_len] - gen_acf[:min_len]))),
        }

    gt_move_score = np.abs(np.diff(gt_levels, axis=1)).mean(axis=(2, 3))
    gen_move_score = np.abs(np.diff(gen_levels, axis=2)).mean(axis=(-1, -2))
    q20 = float(np.quantile(gt_move_score.reshape(-1), 0.20))
    q80 = float(np.quantile(gt_move_score.reshape(-1), 0.80))
    gt_regimes = _label_regimes(gt_move_score, q20, q80)
    gen_regimes = _label_regimes(gen_move_score.reshape(-1, gt_move_score.shape[1]), q20, q80)
    regime = {
        "thresholds": {"q20": q20, "q80": q80},
        "gt": _transition_stats(gt_regimes),
        "gen": _transition_stats(gen_regimes),
    }

    temporal_spatial = {"delta": {}, "level": {}}
    horizons = [1, 5, 10, 20, 30]
    for h in horizons:
        idx = h - 1
        if idx >= ground_truth.shape[1]:
            continue
        if h == 1:
            gt_delta_h = (ground_truth[:, 0] - prev).reshape(ground_truth.shape[0], -1)
            gen_delta_h = (cond_samples[:, :, 0] - prev[:, None]).reshape(-1, 25)
        else:
            gt_delta_h = (ground_truth[:, idx] - ground_truth[:, idx - 1]).reshape(ground_truth.shape[0], -1)
            gen_delta_h = (cond_samples[:, :, idx] - cond_samples[:, :, idx - 1]).reshape(-1, 25)
        gt_level_h = ground_truth[:, idx].reshape(ground_truth.shape[0], -1)
        gen_level_h = cond_samples[:, :, idx].reshape(-1, 25)
        gt_delta_summary = _corr_summary(gt_delta_h)
        gen_delta_summary = _corr_summary(gen_delta_h)
        gt_level_summary = _corr_summary(gt_level_h)
        gen_level_summary = _corr_summary(gen_level_h)
        temporal_spatial["delta"][str(h)] = {
            "gt": gt_delta_summary,
            "gen": gen_delta_summary,
            "corr_ratio": float(gen_delta_summary["mean_corr"] / max(gt_delta_summary["mean_corr"], 1e-10)),
            "rank_ratio": float(gen_delta_summary["eff_rank"] / max(gt_delta_summary["eff_rank"], 1e-10)),
        }
        temporal_spatial["level"][str(h)] = {
            "gt": gt_level_summary,
            "gen": gen_level_summary,
            "corr_ratio": float(gen_level_summary["mean_corr"] / max(gt_level_summary["mean_corr"], 1e-10)),
            "rank_ratio": float(gen_level_summary["eff_rank"] / max(gt_level_summary["eff_rank"], 1e-10)),
        }

    gt_sign_runs = _sign_run_stats(gt_mean_delta)
    gen_sign_runs = _sign_run_stats(gen_mean_delta.reshape(-1, gt_mean_delta.shape[1]))
    jump_threshold = float(np.quantile(gt_move_score.reshape(-1), 0.95))
    gt_jump_cluster = _jump_cluster_stats(gt_move_score, jump_threshold)
    gen_jump_cluster = _jump_cluster_stats(gen_move_score.reshape(-1, gt_move_score.shape[1]), jump_threshold)
    gt_max_exc = np.max(np.abs(gt_mean_level[:, 1:] - gt_mean_level[:, [0]]), axis=1)
    gen_max_exc = np.max(np.abs(gen_mean_level[:, :, 1:] - gen_mean_level[:, :, [0]]), axis=2).reshape(-1)
    gt_boundary = _boundary_persistence(gt_levels)
    gen_boundary = _boundary_persistence(gen_levels.reshape(-1, gen_levels.shape[2], 5, 5))
    path_realism = {
        "sign_runs": {"gt": gt_sign_runs, "gen": gen_sign_runs},
        "jump_cluster": {"gt": gt_jump_cluster, "gen": gen_jump_cluster},
        "max_excursion": {
            "gt_q50": float(np.quantile(gt_max_exc, 0.50)),
            "gt_q90": float(np.quantile(gt_max_exc, 0.90)),
            "gen_q50": float(np.quantile(gen_max_exc, 0.50)),
            "gen_q90": float(np.quantile(gen_max_exc, 0.90)),
            "q90_ratio": float(np.quantile(gen_max_exc, 0.90) / max(np.quantile(gt_max_exc, 0.90), 1e-10)),
        },
        "boundary": {"gt": gt_boundary, "gen": gen_boundary},
    }

    state_conditioned = _state_conditioned_profiles(history_01, ground_truth, cond_samples, horizons)

    out = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(history_01.shape[0]),
            "n_samples": int(args.samples),
        },
        "acf_profiles": acf_profiles,
        "regime_persistence": regime,
        "temporal_spatial": temporal_spatial,
        "path_realism": path_realism,
        "state_conditioned": state_conditioned,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(out), indent=2))

    lines = [
        f"- model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{history_01.shape[0]}`",
        f"- samples: `{args.samples}`",
        "",
        "**ACF**",
        f"- mean-level ACF corr: `{acf_profiles['mean_level']['acf_correlation']:.3f}`",
        f"- mean-delta ACF corr: `{acf_profiles['mean_delta']['acf_correlation']:.3f}`",
        f"- abs-mean-delta ACF corr: `{acf_profiles['abs_mean_delta']['acf_correlation']:.3f}`",
        "",
        "**Regime Persistence**",
        f"- GT calm self-transition: `{regime['gt']['self_transition']['calm']:.3f}`",
        f"- Gen calm self-transition: `{regime['gen']['self_transition']['calm']:.3f}`",
        f"- GT turb self-transition: `{regime['gt']['self_transition']['turb']:.3f}`",
        f"- Gen turb self-transition: `{regime['gen']['self_transition']['turb']:.3f}`",
        "",
        "**Temporal-Spatial h30**",
        f"- delta corr ratio h30: `{temporal_spatial['delta'].get('30', {}).get('corr_ratio', float('nan')):.3f}`",
        f"- delta rank ratio h30: `{temporal_spatial['delta'].get('30', {}).get('rank_ratio', float('nan')):.3f}`",
        f"- level corr ratio h30: `{temporal_spatial['level'].get('30', {}).get('corr_ratio', float('nan')):.3f}`",
        "",
        "**Path Realism**",
        f"- jump cluster lag1 GT/gen: `{gt_jump_cluster['lag1']:.3f}` / `{gen_jump_cluster['lag1']:.3f}`",
        f"- max excursion q90 ratio: `{path_realism['max_excursion']['q90_ratio']:.3f}`",
        f"- floor day incidence GT/gen: `{gt_boundary['floor_day_incidence']:.3%}` / `{gen_boundary['floor_day_incidence']:.3%}`",
        "",
        "**State Conditioned**",
        f"- h30 level-vs-drift Spearman GT/gen: "
        f"`{state_conditioned['per_horizon']['30']['spearman']['level_vs_gt_drift']:.3f}` / "
        f"`{state_conditioned['per_horizon']['30']['spearman']['level_vs_gen_drift']:.3f}`",
        f"- h30 vov-vs-width Spearman gen: "
        f"`{state_conditioned['per_horizon']['30']['spearman']['vov_vs_gen_width']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "220 Persistence Diagnostics", lines)
    print(json.dumps(make_serializable({"status": "ok", "model_type": args.model_type}), indent=2))


if __name__ == "__main__":
    main()
