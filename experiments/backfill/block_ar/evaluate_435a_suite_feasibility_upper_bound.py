#!/usr/bin/env python
"""435a: explicit suite feasibility upper-bound diagnostic.

This is not a learned model. It uses validation futures to construct a controlled
oracle sample law with realistic residual shapes and targeted miss rates. The purpose is
to test whether the 11-suite is logically satisfiable by any risk-system sample law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (  # noqa: E402
    suite_summary,
)
from experiments.backfill.block_ar.evaluate_403a_calibrated_risk_system import sample_native  # noqa: E402
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


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def history_key(history_norm: np.ndarray) -> bytes:
    return np.round(history_norm.astype(np.float32), 6).tobytes()


def compute_vov(history_01: np.ndarray) -> np.ndarray:
    mean_iv = history_01.mean(axis=(2, 3))
    return np.diff(mean_iv, axis=1).std(axis=1)


def build_controlled_oracle_samples(
    base_samples: np.ndarray,
    ground_truth: np.ndarray,
    history_01: np.ndarray,
    miss_rate: float,
    residual_scale: float,
    center_jitter: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    median = np.median(base_samples, axis=1, keepdims=True)
    residual = float(residual_scale) * (base_samples - median)
    n_windows, _n_samples, horizon, rows, cols = base_samples.shape
    offsets = np.zeros((n_windows, horizon, rows, cols), dtype=np.float32)
    miss_mask = np.zeros((n_windows, horizon, rows, cols), dtype=bool)

    vov = compute_vov(history_01)
    q20 = float(np.quantile(vov, 0.20))
    q80 = float(np.quantile(vov, 0.80))
    groups = [
        np.where(vov <= q20)[0],
        np.where((vov > q20) & (vov < q80))[0],
        np.where(vov >= q80)[0],
    ]

    # Horizon-distributed misses prevent persistent-undercoverage failures while
    # preserving exact per-regime/cell/horizon coverage rates.
    abs_q = np.maximum(
        np.abs(np.quantile(residual, 0.05, axis=1)),
        np.abs(np.quantile(residual, 0.95, axis=1)),
    ) + 1e-4

    # Tiny balanced jitter avoids the degenerate "median exactly equals GT" case in
    # the median-bias fraction check, without changing material coverage.
    n_idx = np.arange(n_windows)[:, None, None, None]
    t_idx = np.arange(horizon)[None, :, None, None]
    r_idx = np.arange(rows)[None, None, :, None]
    c_idx = np.arange(cols)[None, None, None, :]
    jitter_sign = np.where((n_idx + t_idx + r_idx + c_idx) % 2 == 0, 1.0, -1.0)
    offsets += (float(center_jitter) * jitter_sign).astype(np.float32)

    for r in range(rows):
        for c in range(cols):
            cell_id = r * cols + c
            for t in range(horizon):
                for group_id, idx in enumerate(groups):
                    if idx.size == 0:
                        continue
                    n_miss = int(round(float(miss_rate) * idx.size))
                    n_miss = max(0, min(n_miss, idx.size))
                    chosen = rng.choice(idx, size=n_miss, replace=False) if n_miss else []
                    for k, w in enumerate(chosen):
                        sign = 1.0 if ((k + cell_id + group_id + t) % 2 == 0) else -1.0
                        offsets[w, t, r, c] = sign * 1.25 * float(abs_q[w, t, r, c])
                        miss_mask[w, t, r, c] = True

    samples = ground_truth[:, None] + offsets[:, None] + residual
    samples = np.clip(samples, 0.0, 1.0).astype(np.float32)
    meta = {
        "miss_rate_target": float(miss_rate),
        "miss_rate_realized": float(miss_mask.mean()),
        "residual_scale": float(residual_scale),
        "center_jitter": float(center_jitter),
        "vov_q20": q20,
        "vov_q80": q80,
        "offset_min": float(offsets.min()),
        "offset_median": float(np.median(offsets)),
        "offset_max": float(offsets.max()),
    }
    return samples, meta


class FixedOracleSampler:
    def __init__(self, samples_by_key: dict[bytes, np.ndarray]):
        self.samples_by_key = samples_by_key

    def eval(self) -> "FixedOracleSampler":
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        del chunk_size, history_is_normalized
        hist_np = history.detach().cpu().numpy()
        rows = []
        fallback = next(iter(self.samples_by_key.values()))
        for hist in hist_np:
            arr = self.samples_by_key.get(history_key(hist), fallback)
            if n_samples <= arr.shape[0]:
                out = arr[:n_samples, :n_steps]
            else:
                reps = int(np.ceil(n_samples / arr.shape[0]))
                out = np.tile(arr, (reps, 1, 1, 1))[:n_samples, :n_steps]
            rows.append(out)
        return torch.from_numpy(np.stack(rows, axis=0)).to(history.device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
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
    parser.add_argument("--miss_rate", type=float, default=0.12)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--center_jitter", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=435)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    base_model.eval()

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

    print("Sampling 392a residual shapes")
    base_samples = sample_native(
        model=base_model,
        history_norm=batch.history_norm,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()
    cond_samples, oracle_meta = build_controlled_oracle_samples(
        base_samples=base_samples,
        ground_truth=ground_truth,
        history_01=history_01,
        miss_rate=args.miss_rate,
        residual_scale=args.residual_scale,
        center_jitter=args.center_jitter,
        seed=args.seed,
    )
    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(hist_norm_np.shape[0])}
    model = FixedOracleSampler(samples_by_key).eval()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    conditionality = run_conditionality_tests(
        model,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    cointegration = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    regime_coverage = run_regime_coverage_tests(cond_samples, ground_truth, history_01)
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "diagnostic": {
                "kind": "suite_feasibility_upper_bound",
                "oracle_uses_validation_future": True,
                "not_deployable": True,
                "not_learned_conditional_law": True,
                "base_residual_source": "392a_samples_minus_sample_median",
                "miss_rate": args.miss_rate,
                "residual_scale": args.residual_scale,
                "seed": args.seed,
                **oracle_meta,
            },
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
            "rollout_start": int(rollout_start),
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "regime_coverage": regime_coverage,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")

    lines = [
        "- diagnostic: `suite feasibility upper bound; oracle; not deployable; not learned`",
        f"- base residual source: `{args.model_type}` at `{args.checkpoint}`",
        f"- miss rate target/realized: `{args.miss_rate:.3f}` / `{oracle_meta['miss_rate_realized']:.3f}`",
        f"- residual scale: `{args.residual_scale:.3f}`",
        f"- center jitter: `{args.center_jitter:.1e}`",
        f"- offset range: `{oracle_meta['offset_min']:.4f} / {oracle_meta['offset_median']:.4f} / {oracle_meta['offset_max']:.4f}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Feasibility Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- regime layer2: `{regime_coverage['layer2_n_passing']}/{regime_coverage['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- cointegration worst-cell ratio: `{cointegration.get('worst_cell_ratio', float('nan')):.3f}`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_reversion.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "435a Suite Feasibility Upper Bound", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
