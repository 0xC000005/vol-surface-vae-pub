#!/usr/bin/env python
"""376a: recent-window monotone quantile calibration feasibility.

This is a policy-calibration feasibility experiment, not a new learned core.
It freezes 340c, fits small monotone per-horizon/cell quantile maps on the
recent pre-validation calibration window, applies them to validation samples,
and evaluates whether the remaining coverage/level-law failures are repairable
without validation leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_multistep_windows,
    load_one_day_kernel,
    make_serializable,
)
from experiments.backfill.block_ar.analyze_374a_frozen_340c_calibration_ladder import (  # noqa: E402
    evaluate_samples,
    metric_digest,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


EPS = 1e-5


@dataclass
class WindowBlock:
    history_01: torch.Tensor
    future_01: torch.Tensor
    start_index: int


@dataclass
class QuantileMap:
    source_q: np.ndarray
    target_q: np.ndarray


def build_window_block(
    data_path: str,
    indices: np.ndarray,
    history_len: int,
    future_len: int,
    device: torch.device,
) -> WindowBlock:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, future_01 = build_multistep_windows(indices, surf_tensor, history_len, future_len)
    future_01 = future_01.view(history_01.shape[0], future_len, 5, 5)
    return WindowBlock(history_01=history_01, future_01=future_01, start_index=int(indices[0]))


def generate_samples(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist_batch = normalize_iv(history_01[start:end])
        with torch.no_grad():
            samples = model.sample_batched(
                hist_batch,
                n_samples=n_samples,
                n_steps=n_steps,
                chunk_size=chunk_size,
            )
        outputs.append(samples.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_quantile_map(
    generated: np.ndarray,
    target: np.ndarray,
    q_levels: np.ndarray,
) -> QuantileMap:
    source_q = np.quantile(generated, q_levels, axis=(0, 1))
    target_q = np.quantile(target, q_levels, axis=0)
    return QuantileMap(source_q=source_q, target_q=target_q)


def apply_quantile_map(samples: np.ndarray, qmap: QuantileMap, alpha: float) -> np.ndarray:
    mapped = np.empty_like(samples)
    for t in range(samples.shape[2]):
        for r in range(samples.shape[3]):
            for c in range(samples.shape[4]):
                vals = samples[:, :, t, r, c].reshape(-1)
                xp = qmap.source_q[:, t, r, c]
                yp = qmap.target_q[:, t, r, c]
                xp_unique, idx = np.unique(xp, return_index=True)
                yp_unique = yp[idx]
                if xp_unique.size < 2:
                    out = vals
                else:
                    out = np.interp(vals, xp_unique, yp_unique, left=yp_unique[0], right=yp_unique[-1])
                mapped[:, :, t, r, c] = out.reshape(samples.shape[0], samples.shape[1])
    blended = (1.0 - float(alpha)) * samples + float(alpha) * mapped
    return np.clip(blended, EPS, 1.0 - EPS)


def history_vov(history: np.ndarray) -> np.ndarray:
    mean_iv = history.mean(axis=(2, 3))
    return np.diff(mean_iv, axis=1).std(axis=1)


def regime_labels(
    calibration_history: np.ndarray,
    target_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cal_vov = history_vov(calibration_history)
    target_vov = history_vov(target_history)
    q20 = np.quantile(cal_vov, 0.20)
    q80 = np.quantile(cal_vov, 0.80)

    def label(vov: np.ndarray) -> np.ndarray:
        out = np.full(vov.shape[0], 1, dtype=np.int64)
        out[vov <= q20] = 0
        out[vov >= q80] = 2
        return out

    return label(cal_vov), label(target_vov)


def fit_regime_quantile_maps(
    cal_samples: np.ndarray,
    cal_gt: np.ndarray,
    cal_history: np.ndarray,
    val_history: np.ndarray,
    q_levels: np.ndarray,
) -> tuple[dict[int, QuantileMap], np.ndarray]:
    cal_labels, val_labels = regime_labels(cal_history, val_history)
    global_map = fit_quantile_map(cal_samples, cal_gt, q_levels)
    maps: dict[int, QuantileMap] = {}
    for label in [0, 1, 2]:
        mask = cal_labels == label
        if int(mask.sum()) < 24:
            maps[label] = global_map
        else:
            maps[label] = fit_quantile_map(cal_samples[mask], cal_gt[mask], q_levels)
    return maps, val_labels


def apply_regime_quantile_maps(
    samples: np.ndarray,
    maps: dict[int, QuantileMap],
    labels: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out = np.empty_like(samples)
    for label in [0, 1, 2]:
        mask = labels == label
        if not mask.any():
            continue
        out[mask] = apply_quantile_map(samples[mask], maps[label], alpha)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/340c_v0_s42/best_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", default="results/validations/2026-04-24/analysis/376a_recent_quantile_policy_calibration")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--calibration_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel("340c", args.checkpoint, device)
    model.eval()

    max_train_idx = args.test_start - args.history_len - args.future_len
    val_start = max_train_idx - args.val_size
    n_val = min(args.max_windows, args.val_size)
    n_cal = min(args.calibration_windows, val_start)
    cal_indices = np.arange(val_start - n_cal, val_start)
    val_indices = np.arange(val_start, val_start + n_val)

    cal_block = build_window_block(
        args.data_path, cal_indices, args.history_len, args.future_len, device
    )
    val_block = build_window_block(
        args.data_path, val_indices, args.history_len, args.future_len, device
    )
    cal_samples = generate_samples(
        model, cal_block.history_01, args.samples, args.future_len, args.batch_size, args.chunk_size
    )
    val_raw = generate_samples(
        model, val_block.history_01, args.samples, args.future_len, args.batch_size, args.chunk_size
    )
    cal_gt = cal_block.future_01.detach().cpu().numpy()
    val_gt = val_block.future_01.detach().cpu().numpy()
    val_history = val_block.history_01.detach().cpu().numpy()
    cal_history = cal_block.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    q_levels = np.r_[0.0, np.linspace(0.01, 0.99, 99), 1.0]
    global_map = fit_quantile_map(cal_samples, cal_gt, q_levels)
    regime_maps, val_labels = fit_regime_quantile_maps(
        cal_samples, cal_gt, cal_history, val_history, q_levels
    )

    variants = {"raw": val_raw}
    for alpha in [0.25, 0.50, 0.75, 1.00]:
        variants[f"recent_hc_qmap_a{alpha:.2f}"] = apply_quantile_map(val_raw, global_map, alpha)
    for alpha in [0.50, 1.00]:
        variants[f"recent_regime_hc_qmap_a{alpha:.2f}"] = apply_regime_quantile_maps(
            val_raw, regime_maps, val_labels, alpha
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "calibration_start_index": int(cal_indices[0]),
        "calibration_end_index": int(cal_indices[-1]),
        "validation_start_index": int(val_indices[0]),
        "validation_end_index": int(val_indices[-1]),
        "n_calibration_windows": int(n_cal),
        "n_validation_windows": int(n_val),
        "samples": int(args.samples),
        "variants": {},
    }

    for name, samples in variants.items():
        results, cond_proxy, n_pass, failed, logs = evaluate_samples(
            samples,
            val_gt,
            val_history,
            returns,
            int(val_indices[0]),
        )
        summary["variants"][name] = {
            "n_pass_proxy11": int(n_pass),
            "failed_proxy11": failed,
            "conditionality_proxy": cond_proxy,
            "digest": metric_digest(results, cond_proxy),
            "official_sample_array_results": results,
        }
        (out_dir / f"{name}_suite_stdout.txt").write_text(logs, encoding="utf-8")

    best_name = max(summary["variants"], key=lambda k: summary["variants"][k]["n_pass_proxy11"])
    summary["best_variant"] = best_name
    summary["best_n_pass_proxy11"] = summary["variants"][best_name]["n_pass_proxy11"]
    (out_dir / "summary.json").write_text(json.dumps(make_serializable(summary), indent=2), encoding="utf-8")

    lines = [
        "# 376a Recent Quantile Policy Calibration",
        "",
        "Frozen 340c samples are calibrated with monotone horizon/cell quantile maps fitted on the immediately preceding calibration window, then evaluated on validation. Conditionality is a sample-array proxy.",
        "",
        "| variant | proxy score | failed | coverage90 | h30 worst/best | cond MAE red | regime L2 | level KS | daily KS | MR | coint worst | maxjump KS |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for name, item in summary["variants"].items():
        d = item["digest"]
        lines.append(
            f"| `{name}` | {item['n_pass_proxy11']}/11 | {', '.join(item['failed_proxy11']) or 'none'} "
            f"| {d['coverage90']:.3f} | {d['worst_h30_cell_coverage']:.3f}/{d['best_h30_cell_coverage']:.3f} "
            f"| {d['conditionality_proxy_mae_reduction']:.1f}% | {d['regime_layer2'][0]}/{d['regime_layer2'][1]} "
            f"| {d['level_ks_cells']}/25 | {d['daily_ks_cells']}/25 | {d['mean_reversion_pass']} "
            f"| {d['cointegration_worst_cell']:.3f} | {d['pathwise_maxjump_ks']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Best variant: `{best_name}` with proxy score `{summary['best_n_pass_proxy11']}/11`.",
            "",
            "Interpretation guard: this is a rolling policy-calibration feasibility test. It is not evidence that the learned 340c base model learned the mapped marginal law.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["best_variant", "best_n_pass_proxy11"]}, indent=2))


if __name__ == "__main__":
    main()
