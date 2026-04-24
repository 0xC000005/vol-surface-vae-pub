#!/usr/bin/env python
"""415a: bounded diagnostic mixture of 392a AR and 413a direct path generators."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    HistoryFutureDictDataset,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (  # noqa: E402
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


SUITE_ORDER = [
    "surface",
    "coverage",
    "conditionality",
    "time_series",
    "block_ar",
    "cointegration",
    "regime_coverage",
    "distributional_fidelity",
    "cross_cell_correlation",
    "mean_reversion",
    "pathwise_jump_realism",
]


class FixedMixtureSampler:
    def __init__(self, ar_model: torch.nn.Module, path_model: torch.nn.Module, path_weight: float):
        self.ar_model = ar_model
        self.path_model = path_model
        self.path_weight = float(path_weight)

    def eval(self) -> "FixedMixtureSampler":
        self.ar_model.eval()
        self.path_model.eval()
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        n_path = int(round(float(n_samples) * self.path_weight))
        n_path = max(0, min(int(n_samples), n_path))
        n_ar = int(n_samples) - n_path
        pieces: list[torch.Tensor] = []
        if n_ar > 0:
            pieces.append(
                self.ar_model.sample_batched(
                    history,
                    n_samples=n_ar,
                    n_steps=n_steps,
                    chunk_size=max(1, min(chunk_size, n_ar)),
                    **kwargs,
                )
            )
        if n_path > 0:
            pieces.append(
                self.path_model.sample_batched(
                    history,
                    n_samples=n_path,
                    n_steps=n_steps,
                    chunk_size=max(1, min(chunk_size, n_path)),
                    **kwargs,
                )
            )
        return torch.cat(pieces, dim=1)


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    failed = [name for name in SUITE_ORDER if not bool(results[name]["overall_pass"])]
    return len(SUITE_ORDER) - len(failed), failed


def generate_samples(
    model: FixedMixtureSampler,
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


def get_metric(mapping: dict[Any, Any], key: Any) -> Any:
    if key in mapping:
        return mapping[key]
    str_key = str(key)
    if str_key in mapping:
        return mapping[str_key]
    raise KeyError(key)


def digest(results: dict[str, Any]) -> dict[str, Any]:
    cov = []
    for horizon in (1, 7, 14, 30):
        cov.extend(np.asarray(get_metric(results["coverage"]["per_cell_coverage"], horizon)).ravel())
    cov_arr = np.asarray(cov, dtype=float)
    return {
        "n_pass": results["summary"]["n_pass"],
        "failed": results["summary"]["failed_suites"],
        "coverage90": get_metric(results["coverage"]["overall"], 0.9),
        "under70": int((cov_arr < 0.70).sum()),
        "over95": int((cov_arr > 0.95).sum()),
        "conditionality_mae_reduction": results["conditionality"]["mae_reduction_pct"],
        "time_series_pass": results["time_series"]["overall_pass"],
        "regime_layer2": [
            results["regime_coverage"]["layer2_n_passing"],
            results["regime_coverage"]["layer2_n_total"],
        ],
        "daily_ks": results["distributional_fidelity"]["ks_test"]["n_pass"],
        "level_ks": results["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "median_bias": results["distributional_fidelity"]["median_bias"]["n_pass"],
        "cointegration_worst": results["cointegration"]["worst_cell_ratio"],
        "corr_ratio": results["cross_cell_correlation"]["corr_ratio"],
        "mean_reversion_pass": results["mean_reversion"]["overall_pass"],
        "pathwise_maxjump_ks": results["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ar_checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument(
        "--path_checkpoint",
        default="models/backfill/413a_recent_score_path_fm_s42/best_model.pt",
    )
    parser.add_argument("--path_weight", type=float, default=0.125)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ar_model, ar_payload = load_one_day_kernel("340c", args.ar_checkpoint, device)
    path_model, path_payload = load_one_day_kernel("339a", args.path_checkpoint, device)
    model = FixedMixtureSampler(ar_model, path_model, path_weight=args.path_weight).eval()

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
    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    logs = io.StringIO()
    cond_samples = generate_samples(
        model,
        batch.history_01,
        args.samples,
        batch.future_01.shape[1],
        args.batch_size,
        args.chunk_size,
    )
    gt = batch.future_01.detach().cpu().numpy()
    history = batch.history_01.detach().cpu().numpy()
    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    with contextlib.redirect_stdout(logs):
        results = {
            "surface": run_surface_validity_tests(cond_samples, gt),
            "coverage": run_ci_coverage_tests(cond_samples, gt),
            "conditionality": run_conditionality_tests(
                model,
                cond_loader,
                n_samples=args.conditionality_samples,
                max_batches=args.conditionality_max_batches,
                device=str(device),
            ),
            "time_series": run_time_series_tests(cond_samples, gt),
            "block_ar": run_block_ar_tests(cond_samples),
            "cointegration": run_cointegration_tests(
                cond_samples,
                gt,
                returns=returns,
                test_start=rollout_start,
                history_len=args.history_len,
                future_len=args.future_len,
            ),
            "regime_coverage": run_regime_coverage_tests(cond_samples, gt, history),
            "distributional_fidelity": run_distributional_fidelity_tests(cond_samples, gt, history),
            "cross_cell_correlation": run_cross_cell_correlation_tests(cond_samples, gt),
            "mean_reversion": run_mean_reversion_tests(cond_samples, gt, history),
            "pathwise_jump_realism": run_pathwise_jump_realism_tests(cond_samples, gt),
        }
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}
    results["config"] = {
        "ar_checkpoint": args.ar_checkpoint,
        "ar_epoch": int(ar_payload.get("epoch", -1)),
        "path_checkpoint": args.path_checkpoint,
        "path_epoch": int(path_payload.get("epoch", -1)),
        "path_weight": args.path_weight,
        "samples": args.samples,
        "conditionality_samples": args.conditionality_samples,
    }
    results["digest"] = digest(results)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    (out_json.parent / "stdout.txt").write_text(logs.getvalue(), encoding="utf-8")

    d = results["digest"]
    lines = [
        "# 415a 392a/413a Mixture",
        "",
        f"- path weight: `{args.path_weight}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        f"- cov90: `{d['coverage90']:.3f}`; under/over: `{d['under70']}/{d['over95']}`",
        f"- conditionality MAE reduction: `{d['conditionality_mae_reduction']:.2f}%`",
        f"- cointegration worst: `{d['cointegration_worst']:.3f}`",
        f"- regime layer2: `{d['regime_layer2'][0]}/{d['regime_layer2'][1]}`",
        f"- daily/level KS: `{d['daily_ks']}/25` / `{d['level_ks']}/25`",
        f"- corr ratio: `{d['corr_ratio']:.3f}`",
        f"- mean reversion pass: `{d['mean_reversion_pass']}`",
        f"- max-jump KS: `{d['pathwise_maxjump_ks']:.3f}`",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
