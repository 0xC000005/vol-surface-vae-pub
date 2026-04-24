#!/usr/bin/env python
"""380a: official-suite mixture of base 340c and recent-adapted 377a.

This tests a clean nonstationarity hedge: sample from both the long-history
base generator and the recent-adapted generator, without post-hoc sample maps.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

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


class MixtureSampleModel:
    def __init__(self, base: torch.nn.Module, adapted: torch.nn.Module, adapted_weight: float):
        self.base = base
        self.adapted = adapted
        self.adapted_weight = float(adapted_weight)

    def eval(self) -> "MixtureSampleModel":
        self.base.eval()
        self.adapted.eval()
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        n_adapted = int(round(float(n_samples) * self.adapted_weight))
        n_adapted = max(0, min(int(n_samples), n_adapted))
        n_base = int(n_samples) - n_adapted
        pieces = []
        if n_adapted > 0:
            pieces.append(
                self.adapted.sample_batched(
                    history,
                    n_samples=n_adapted,
                    n_steps=n_steps,
                    chunk_size=max(1, min(chunk_size, n_adapted)),
                )
            )
        if n_base > 0:
            pieces.append(
                self.base.sample_batched(
                    history,
                    n_samples=n_base,
                    n_steps=n_steps,
                    chunk_size=max(1, min(chunk_size, n_base)),
                )
            )
        return torch.cat(pieces, dim=1)


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    failed = [name for name in SUITE_ORDER if not bool(results[name]["overall_pass"])]
    return len(SUITE_ORDER) - len(failed), failed


def generate_samples(
    model: MixtureSampleModel,
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


def evaluate_weight(
    model: MixtureSampleModel,
    batch: Any,
    returns: np.ndarray,
    rollout_start: int,
    samples: int,
    conditionality_samples: int,
    batch_size: int,
    chunk_size: int,
    conditionality_max_batches: int,
    device: str,
) -> tuple[dict[str, Any], str]:
    logs = io.StringIO()
    cond_samples = generate_samples(
        model,
        batch.history_01,
        samples,
        batch.future_01.shape[1],
        batch_size,
        chunk_size,
    )
    gt = batch.future_01.detach().cpu().numpy()
    history = batch.history_01.detach().cpu().numpy()
    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=batch_size,
        shuffle=False,
    )
    with contextlib.redirect_stdout(logs):
        results = {
            "surface": run_surface_validity_tests(cond_samples, gt),
            "coverage": run_ci_coverage_tests(cond_samples, gt),
            "conditionality": run_conditionality_tests(
                model,
                cond_loader,
                n_samples=conditionality_samples,
                max_batches=conditionality_max_batches,
                device=device,
            ),
            "time_series": run_time_series_tests(cond_samples, gt),
            "block_ar": run_block_ar_tests(cond_samples),
            "cointegration": run_cointegration_tests(
                cond_samples,
                gt,
                returns=returns,
                test_start=rollout_start,
                history_len=30,
                future_len=30,
            ),
            "regime_coverage": run_regime_coverage_tests(cond_samples, gt, history),
            "distributional_fidelity": run_distributional_fidelity_tests(cond_samples, gt, history),
            "cross_cell_correlation": run_cross_cell_correlation_tests(cond_samples, gt),
            "mean_reversion": run_mean_reversion_tests(cond_samples, gt, history),
            "pathwise_jump_realism": run_pathwise_jump_realism_tests(cond_samples, gt),
        }
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}
    return results, logs.getvalue()


def get_metric(mapping: dict[Any, Any], key: Any) -> Any:
    if key in mapping:
        return mapping[key]
    str_key = str(key)
    if str_key in mapping:
        return mapping[str_key]
    raise KeyError(key)


def digest(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_pass": results["summary"]["n_pass"],
        "failed": results["summary"]["failed_suites"],
        "coverage90": get_metric(results["coverage"]["overall"], 0.9),
        "h30_worst": get_metric(results["coverage"]["worst_cell_per_horizon"], 30),
        "h30_best": get_metric(results["coverage"]["best_cell_per_horizon"], 30),
        "conditionality_mae_reduction": results["conditionality"]["mae_reduction_pct"],
        "conditionality_pass": results["conditionality"]["overall_pass"],
        "regime_layer2": [
            results["regime_coverage"]["layer2_n_passing"],
            results["regime_coverage"]["layer2_n_total"],
        ],
        "daily_ks": results["distributional_fidelity"]["ks_test"]["n_pass"],
        "level_ks": results["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "median_bias": results["distributional_fidelity"]["median_bias"]["n_pass"],
        "cointegration_worst": results["cointegration"]["worst_cell_ratio"],
        "pathwise_maxjump_ks": results["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", default="models/backfill/340c_v0_s42/best_model.pt")
    parser.add_argument("--adapted_checkpoint", default="models/backfill/377a_recent_fm_s42/best_model.pt")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.50, 0.75, 0.90])
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
    parser.add_argument("--output_dir", default="results/block_ar/380a_340c_377a_mixture")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base, base_payload = load_one_day_kernel("340c", args.base_checkpoint, device)
    adapted, adapted_payload = load_one_day_kernel("340c", args.adapted_checkpoint, device)
    base.eval()
    adapted.eval()

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "base_checkpoint": args.base_checkpoint,
        "base_epoch": int(base_payload.get("epoch", -1)),
        "adapted_checkpoint": args.adapted_checkpoint,
        "adapted_epoch": int(adapted_payload.get("epoch", -1)),
        "weights": args.weights,
        "variants": {},
    }
    for weight in args.weights:
        name = f"adapted_weight_{weight:.2f}"
        model = MixtureSampleModel(base, adapted, adapted_weight=weight).eval()
        results, logs = evaluate_weight(
            model=model,
            batch=batch,
            returns=returns,
            rollout_start=rollout_start,
            samples=args.samples,
            conditionality_samples=args.conditionality_samples,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            conditionality_max_batches=args.conditionality_max_batches,
            device=str(device),
        )
        summary["variants"][name] = {
            "weight": float(weight),
            "digest": digest(results),
            "results": results,
        }
        (out_dir / f"{name}_stdout.txt").write_text(logs, encoding="utf-8")

    best_name = max(summary["variants"], key=lambda k: summary["variants"][k]["digest"]["n_pass"])
    summary["best_variant"] = best_name
    summary["best_n_pass"] = summary["variants"][best_name]["digest"]["n_pass"]
    (out_dir / "summary.json").write_text(json.dumps(make_serializable(summary), indent=2), encoding="utf-8")

    lines = [
        "# 380a 340c/377a Mixture",
        "",
        "| variant | score | failed | cov90 | h30 worst/best | cond MAE | regime L2 | level KS | daily KS | bias | coint worst | maxjump KS |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["variants"].items():
        d = item["digest"]
        lines.append(
            f"| `{name}` | {d['n_pass']}/11 | {', '.join(d['failed']) or 'none'} "
            f"| {d['coverage90']:.3f} | {d['h30_worst']:.3f}/{d['h30_best']:.3f} "
            f"| {d['conditionality_mae_reduction']:.2f}% | {d['regime_layer2'][0]}/{d['regime_layer2'][1]} "
            f"| {d['level_ks']}/25 | {d['daily_ks']}/25 | {d['median_bias']}/25 "
            f"| {d['cointegration_worst']:.3f} | {d['pathwise_maxjump_ks']:.3f} |"
        )
    lines.extend(["", f"Best: `{best_name}` at `{summary['best_n_pass']}/11`."])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["best_variant", "best_n_pass"]}, indent=2))


if __name__ == "__main__":
    main()
