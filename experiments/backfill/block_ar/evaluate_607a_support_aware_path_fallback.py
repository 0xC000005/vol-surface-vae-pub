#!/usr/bin/env python
"""607a: support-aware path-location fallback around frozen 510a."""

from __future__ import annotations

import argparse
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
    write_markdown_summary,
)
from experiments.backfill.block_ar.audit_605a_hard_window_support import (  # noqa: E402
    build_iv_features,
    build_windows,
    nearest_distance,
    nearest_self_distance,
    split_indices,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (  # noqa: E402
    suite_summary,
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv  # noqa: E402


def select_increment_pool(
    train_history: np.ndarray,
    train_future: np.ndarray,
    *,
    stress_quantile: float,
) -> np.ndarray:
    """Return training future increments from high path-severity windows."""
    hist = np.asarray(train_history, dtype=np.float32)
    fut = np.asarray(train_future, dtype=np.float32)
    increments = fut - hist[:, -1:, :, :]
    severity = np.max(np.abs(increments), axis=(1, 2, 3))
    threshold = float(np.quantile(severity, float(stress_quantile)))
    pool = increments[severity >= threshold]
    if pool.shape[0] == 0:
        return increments
    return pool.astype(np.float32)


def inject_increment_fallback(
    *,
    base_samples: np.ndarray,
    history_01: np.ndarray,
    support_scores: np.ndarray,
    support_threshold: float,
    increment_pool: np.ndarray,
    fallback_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a bounded fraction of samples for low-support histories."""
    base = np.asarray(base_samples, dtype=np.float32)
    hist = np.asarray(history_01, dtype=np.float32)
    scores = np.asarray(support_scores, dtype=np.float64)
    pool = np.asarray(increment_pool, dtype=np.float32)
    if base.ndim != 5:
        raise ValueError("base_samples must have shape [windows, samples, horizon, rows, cols]")
    if hist.shape[0] != base.shape[0]:
        raise ValueError("history and samples must share window dimension")
    if pool.ndim != 4 or pool.shape[1:] != base.shape[2:]:
        raise ValueError("increment_pool must have shape [pool, horizon, rows, cols]")
    out = base.copy()
    replace_count = int(round(float(fallback_fraction) * base.shape[1]))
    replace_count = max(0, min(replace_count, base.shape[1]))
    if replace_count == 0:
        return out
    low_support = scores >= float(support_threshold)
    for window_idx in np.where(low_support)[0]:
        chosen = rng.integers(0, pool.shape[0], size=replace_count)
        injected = hist[window_idx, -1][None, :, :] + pool[chosen]
        out[window_idx, :replace_count] = np.clip(injected, 0.0, 1.0)
    return out.astype(np.float32)


class SupportAwarePathFallbackSampler:
    def __init__(
        self,
        base_model: torch.nn.Module,
        train_feature_z: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        support_threshold: float,
        increment_pool: np.ndarray,
        fallback_fraction: float,
        seed: int,
        device: torch.device,
    ):
        self.base_model = base_model
        self.train_feature_z = train_feature_z
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.support_threshold = float(support_threshold)
        self.increment_pool = increment_pool
        self.fallback_fraction = float(fallback_fraction)
        self.rng = np.random.default_rng(int(seed))
        self.device = device
        self.last_support_scores: list[float] = []

    def eval(self) -> "SupportAwarePathFallbackSampler":
        self.base_model.eval()
        return self

    def _support_scores(self, history_01: np.ndarray) -> np.ndarray:
        features = build_iv_features(history_01)
        z = (features - self.feature_mean) / self.feature_scale
        return nearest_distance(self.train_feature_z, z)

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        history_device = history.device
        history_base = history.to(self.device)
        with torch.no_grad():
            raw = self.base_model.sample_batched(
                history_base,
                n_samples=n_samples,
                n_steps=n_steps,
                chunk_size=chunk_size,
                history_is_normalized=history_is_normalized,
                **kwargs,
            )
        if history_is_normalized:
            history_01 = denormalize_iv(history_base).detach().cpu().numpy()
        else:
            history_01 = history_base.detach().cpu().numpy()
        scores = self._support_scores(history_01)
        self.last_support_scores.extend(float(x) for x in scores)
        out = inject_increment_fallback(
            base_samples=raw.detach().cpu().numpy(),
            history_01=history_01,
            support_scores=scores,
            support_threshold=self.support_threshold,
            increment_pool=self.increment_pool,
            fallback_fraction=self.fallback_fraction,
            rng=self.rng,
        )
        return torch.from_numpy(out).to(history_device)


def _fit_support_objects(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    support_quantile: float,
    stress_quantile: float,
) -> dict[str, Any]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_idx, _val_idx = split_indices(test_start, history_len, future_len, val_size)
    train_history, train_future = build_windows(surfaces, train_idx, history_len, future_len)
    train_features = build_iv_features(train_history)
    mean = train_features.mean(axis=0, keepdims=True)
    scale = train_features.std(axis=0, keepdims=True) + 1e-8
    train_z = (train_features - mean) / scale
    train_self = nearest_self_distance(train_z)
    support_threshold = float(np.quantile(train_self, float(support_quantile)))
    increment_pool = select_increment_pool(
        train_history,
        train_future,
        stress_quantile=stress_quantile,
    )
    return {
        "train_feature_z": train_z,
        "feature_mean": mean,
        "feature_scale": scale,
        "support_threshold": support_threshold,
        "increment_pool": increment_pool,
        "n_train": int(train_history.shape[0]),
        "pool_size": int(increment_pool.shape[0]),
        "train_self_p50_p95_p99": [float(x) for x in np.quantile(train_self, [0.50, 0.95, 0.99])],
    }


def _evaluate_suite(
    model: SupportAwarePathFallbackSampler,
    batch: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    outputs: list[np.ndarray] = []
    for start in range(0, batch.history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, batch.history_01.shape[0])
        with torch.no_grad():
            samples = model.sample_batched(
                batch.history_norm[start:end],
                n_samples=args.samples,
                n_steps=args.future_len,
                chunk_size=args.chunk_size,
                history_is_normalized=True,
            )
        outputs.append(samples.detach().cpu().numpy())
    cond_samples = np.concatenate(outputs, axis=0)
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return {
        "surface": run_surface_validity_tests(cond_samples, ground_truth),
        "coverage": run_ci_coverage_tests(cond_samples, ground_truth),
        "conditionality": run_conditionality_tests(
            model,
            cond_loader,
            n_samples=args.conditionality_samples,
            max_batches=args.conditionality_max_batches,
            device=str(device),
        ),
        "time_series": run_time_series_tests(cond_samples, ground_truth),
        "block_ar": run_block_ar_tests(cond_samples),
        "cointegration": run_cointegration_tests(
            cond_samples,
            ground_truth,
            returns=returns,
            test_start=rollout_start,
            history_len=args.history_len,
            future_len=args.future_len,
        ),
        "regime_coverage": run_regime_coverage_tests(cond_samples, ground_truth, history_01),
        "distributional_fidelity": run_distributional_fidelity_tests(cond_samples, ground_truth, history_01),
        "cross_cell_correlation": run_cross_cell_correlation_tests(cond_samples, ground_truth),
        "mean_reversion": run_mean_reversion_tests(cond_samples, ground_truth, history_01),
        "pathwise_jump_realism": run_pathwise_jump_realism_tests(cond_samples, ground_truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument("--checkpoint", default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--support_quantile", type=float, default=0.95)
    parser.add_argument("--stress_quantile", type=float, default=0.80)
    parser.add_argument("--fallback_fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=607)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    support = _fit_support_objects(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        support_quantile=args.support_quantile,
        stress_quantile=args.stress_quantile,
    )
    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    model = SupportAwarePathFallbackSampler(
        base_model=base_model,
        train_feature_z=support["train_feature_z"],
        feature_mean=support["feature_mean"],
        feature_scale=support["feature_scale"],
        support_threshold=support["support_threshold"],
        increment_pool=support["increment_pool"],
        fallback_fraction=args.fallback_fraction,
        seed=args.seed,
        device=device,
    ).eval()
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
    results = _evaluate_suite(model, batch, args, device)
    support_scores = np.asarray(model.last_support_scores[: batch.history_01.shape[0]], dtype=np.float64)
    n_fallback = int(np.sum(support_scores >= support["support_threshold"]))
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}
    results["config"] = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": args.samples,
        "support_quantile": args.support_quantile,
        "support_threshold": float(support["support_threshold"]),
        "stress_quantile": args.stress_quantile,
        "fallback_fraction": args.fallback_fraction,
        "n_train": support["n_train"],
        "increment_pool_size": support["pool_size"],
        "train_self_p50_p95_p99": support["train_self_p50_p95_p99"],
        "validation_fallback_windows": n_fallback,
        "validation_fallback_fraction": float(n_fallback / max(1, batch.history_01.shape[0])),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    lines = [
        f"- base model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        "- fallback: `support-aware anchored historical increments`",
        f"- support quantile / threshold: `{args.support_quantile:.3f}` / `{support['support_threshold']:.3f}`",
        f"- stress quantile / pool size: `{args.stress_quantile:.3f}` / `{support['pool_size']}`",
        f"- fallback fraction per low-support window: `{args.fallback_fraction:.2f}`",
        f"- validation fallback windows: `{n_fallback}/{batch.history_norm.shape[0]}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{results['coverage']['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{results['coverage']['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditional MAE reduction: `{results['conditionality'].get('mae_reduction_pct', float('nan')):.3f}`",
        f"- turb/calm ratio: `{results['conditionality'].get('turb_calm_ratio', float('nan')):.3f}`",
        "",
        "**Hard Suites**",
        f"- regime coverage layer2: `{results['regime_coverage']['layer2_n_passing']}/{results['regime_coverage']['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{results['distributional_fidelity']['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{results['distributional_fidelity']['ks_level_test']['n_pass']}/25`",
        f"- max-jump KS: `{results['pathwise_jump_realism']['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "607a Support-Aware Path Fallback", lines)
    print(json.dumps(make_serializable(results["summary"] | {"fallback_windows": n_fallback}), indent=2))


if __name__ == "__main__":
    main()
