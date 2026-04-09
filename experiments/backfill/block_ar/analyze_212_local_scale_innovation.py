#!/usr/bin/env python
"""
Audit local-scale models by separating raw-delta kurtosis from innovation-law
kurtosis.

For each split:
  - compute causal local scale `s_t(c)` from the history window
  - compare GT raw delta and GT innovation `delta / s_t`
  - compare generated raw delta and generated innovation `delta / s_t`
  - quantify whether generated raw kurtosis is driven by innovation shape or by
    scale variation across windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import load_model as load_212x_model
from experiments.backfill.block_ar.train_212y_h1_minimal_direct_stochastic_delta_local_scale_asinh import load_model as load_212y_model
from experiments.backfill.block_ar.train_212z_h1_minimal_direct_stochastic_delta_local_scale_asinh_crps import load_model as load_212z_model
from experiments.backfill.block_ar.train_212aa_h1_minimal_direct_stochastic_delta_local_scale_asinh_tailcrps import load_model as load_212aa_model
from experiments.backfill.block_ar.train_212ac_h1_rectified_flow_local_scale_asinh import load_model as load_212ac_model
from experiments.backfill.block_ar.train_212ad_h1_conditional_diffusion_local_scale_asinh_terminal_es import load_model as load_212ad_model
from experiments.backfill.block_ar.train_212ae_h1_conditional_flow_local_scale_asinh import load_model as load_212ae_model
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import load_model as load_212af_model
from experiments.backfill.block_ar.train_212ag_h1_conditional_flow_local_scale_asinh_mean_aux import load_model as load_212ag_model
from experiments.backfill.block_ar.train_212ah_h1_conditional_flow_local_scale_asinh_nll_mean import load_model as load_212ah_model
from experiments.backfill.block_ar.train_212ai_h1_conditional_flow_local_scale_asinh_staged_nll import load_model as load_212ai_model
from experiments.backfill.block_ar.train_212aj_h1_conditional_flow_local_scale_asinh_locscale_staged_nll import load_model as load_212aj_model
from experiments.backfill.block_ar.train_212ak_h1_conditional_flow_local_scale_asinh_location_staged_nll import load_model as load_212ak_model
from experiments.backfill.block_ar.train_212al_h1_conditional_flow_local_scale_asinh_location_frozen_stage import load_model as load_212al_model
from experiments.backfill.block_ar.train_212ab_h1_conditional_diffusion_local_scale_asinh import load_model as load_212ab_model


LOADERS: dict[str, Callable[[str, torch.device], tuple[torch.nn.Module, dict[str, Any]]]] = {
    "212x": load_212x_model,
    "212y": load_212y_model,
    "212z": load_212z_model,
    "212aa": load_212aa_model,
    "212ac": load_212ac_model,
    "212ad": load_212ad_model,
    "212ae": load_212ae_model,
    "212af": load_212af_model,
    "212ag": load_212ag_model,
    "212ah": load_212ah_model,
    "212ai": load_212ai_model,
    "212aj": load_212aj_model,
    "212ak": load_212ak_model,
    "212al": load_212al_model,
    "212ab": load_212ab_model,
}


def _pearson_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = x.mean()
    v = ((x - m) ** 2).mean()
    if v <= 1e-12:
        return float("nan")
    return float(((x - m) ** 4).mean() / (v ** 2))


def _safe_corr(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit innovation-vs-scale behavior for 212x/212y/212z/212aa/212ac/212ad/212ae/212af/212ag/212ah/212ai/212aj/212ak/212al/212ab local-scale models")
    parser.add_argument("--model_type", type=str, required=True, choices=sorted(LOADERS))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--future_gap", type=int, default=30)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = LOADERS[args.model_type](args.checkpoint, device)
    model.eval()

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = args.test_start - args.history_len - args.future_gap
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    test_indices = np.arange(args.test_start, len(surfaces) - args.history_len - args.future_gap + 1)
    split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}[args.split]
    history_01, target_01 = build_one_step_windows(split_indices, surf_tensor, args.history_len)

    scales = []
    pred_delta = []
    for start in range(0, history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, history_01.shape[0])
        batch_hist = history_01[start:end]
        with torch.no_grad():
            _state, local_scale = model.encode_with_scale(batch_hist)
            delta = model.sample_delta(batch_hist, n_samples=args.eval_samples)
        scales.append(local_scale.detach().cpu().numpy())
        pred_delta.append(delta.detach().cpu().numpy())

    local_scale = np.concatenate(scales, axis=0).reshape(-1, 5, 5)
    pred_delta = np.concatenate(pred_delta, axis=0).reshape(-1, args.eval_samples, 5, 5)

    prev = history_01[:, -1].reshape(history_01.shape[0], 25).detach().cpu().numpy()
    gt_delta = (target_01.detach().cpu().numpy().reshape(history_01.shape[0], 25) - prev).reshape(-1, 5, 5)
    gt_innov = gt_delta / np.maximum(local_scale, model.scale_floor)
    pred_innov = pred_delta / np.maximum(local_scale[:, None], model.scale_floor)

    per_cell = []
    raw_pred_kurt = []
    pred_innov_kurt = []
    scale_cv = []
    scale_q99q50 = []

    for r in range(5):
        for c in range(5):
            gt_raw_k = _pearson_kurtosis(gt_delta[:, r, c])
            gt_innov_k = _pearson_kurtosis(gt_innov[:, r, c])
            pred_raw_k = _pearson_kurtosis(pred_delta[:, :, r, c])
            pred_innov_k = _pearson_kurtosis(pred_innov[:, :, r, c])
            s = local_scale[:, r, c]
            cv = float(s.std() / max(s.mean(), 1e-8))
            q99q50 = float(np.quantile(s, 0.99) / max(np.quantile(s, 0.50), 1e-8))

            raw_pred_kurt.append(pred_raw_k)
            pred_innov_kurt.append(pred_innov_k)
            scale_cv.append(cv)
            scale_q99q50.append(q99q50)

            per_cell.append(
                {
                    "cell": [r, c],
                    "gt_raw_kurtosis": gt_raw_k,
                    "gt_innovation_kurtosis": gt_innov_k,
                    "pred_raw_kurtosis": pred_raw_k,
                    "pred_innovation_kurtosis": pred_innov_k,
                    "raw_kurtosis_ratio": pred_raw_k / max(gt_raw_k, 1e-8),
                    "innovation_kurtosis_ratio": pred_innov_k / max(gt_innov_k, 1e-8),
                    "scale_mean": float(s.mean()),
                    "scale_cv": cv,
                    "scale_q99_q50": q99q50,
                    "abs_gt_delta_q99": float(np.quantile(np.abs(gt_delta[:, r, c]), 0.99)),
                    "abs_gt_innovation_q99": float(np.quantile(np.abs(gt_innov[:, r, c]), 0.99)),
                }
            )

    summary = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "split": args.split,
        "n_windows": int(history_01.shape[0]),
        "eval_samples": args.eval_samples,
        "innovation_summary": {
            "gt_innov_kurtosis_median": float(np.median([r["gt_innovation_kurtosis"] for r in per_cell])),
            "gt_innov_kurtosis_min": float(np.min([r["gt_innovation_kurtosis"] for r in per_cell])),
            "gt_innov_kurtosis_max": float(np.max([r["gt_innovation_kurtosis"] for r in per_cell])),
            "pred_innov_kurtosis_median": float(np.median([r["pred_innovation_kurtosis"] for r in per_cell])),
            "pred_innov_kurtosis_min": float(np.min([r["pred_innovation_kurtosis"] for r in per_cell])),
            "pred_innov_kurtosis_max": float(np.max([r["pred_innovation_kurtosis"] for r in per_cell])),
        },
        "correlations": {
            "raw_pred_kurtosis_vs_pred_innovation_kurtosis": _safe_corr(raw_pred_kurt, pred_innov_kurt),
            "raw_pred_kurtosis_vs_scale_cv": _safe_corr(raw_pred_kurt, scale_cv),
            "raw_pred_kurtosis_vs_scale_q99_q50": _safe_corr(raw_pred_kurt, scale_q99q50),
        },
        "per_cell": per_cell,
    }

    worst_under = sorted(per_cell, key=lambda r: r["innovation_kurtosis_ratio"])[:5]
    md = [
        f"# {args.model_type} Local-Scale Innovation Audit",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Epoch: `{payload.get('epoch', -1)}`",
        f"- Split: `{args.split}`",
        f"- Windows: `{history_01.shape[0]}`",
        f"- Samples/window: `{args.eval_samples}`",
        "",
        "## Innovation Summary",
        "",
        f"- GT innovation kurtosis median: `{summary['innovation_summary']['gt_innov_kurtosis_median']:.3f}`",
        f"- Pred innovation kurtosis median: `{summary['innovation_summary']['pred_innov_kurtosis_median']:.3f}`",
        "",
        "## Correlations",
        "",
        f"- raw pred kurtosis vs pred innovation kurtosis: `{summary['correlations']['raw_pred_kurtosis_vs_pred_innovation_kurtosis']:.3f}`",
        f"- raw pred kurtosis vs scale CV: `{summary['correlations']['raw_pred_kurtosis_vs_scale_cv']:.3f}`",
        f"- raw pred kurtosis vs scale q99/q50: `{summary['correlations']['raw_pred_kurtosis_vs_scale_q99_q50']:.3f}`",
        "",
        "## Worst Innovation Under-Kurtosis Cells",
        "",
    ]
    for row in worst_under:
        md.append(
            f"- cell `{tuple(row['cell'])}`: GT innov kurt `{row['gt_innovation_kurtosis']:.3f}`, "
            f"pred innov kurt `{row['pred_innovation_kurtosis']:.3f}`, "
            f"ratio `{row['innovation_kurtosis_ratio']:.3f}`"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text("\n".join(md))
    print(json.dumps({"innovation_summary": summary["innovation_summary"], "correlations": summary["correlations"]}, indent=2))


if __name__ == "__main__":
    main()
