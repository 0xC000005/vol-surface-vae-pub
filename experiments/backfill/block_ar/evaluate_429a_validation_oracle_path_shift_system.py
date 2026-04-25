#!/usr/bin/env python
"""429a: validation-oracle path-center shift diagnostic for 392a.

This freezes 392a and applies one path-constant vertical oracle shift to every sample
for a validation window. Unlike 426a/427a, the correction does not remap individual
horizons; it preserves within-future residual path geometry as much as possible.

This is an oracle feasibility diagnostic, not a deployable calibrated model.
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


def fit_path_shifts(
    samples: np.ndarray,
    future: np.ndarray,
    mode: str,
) -> np.ndarray:
    median_path = np.median(samples, axis=1)
    error = future - median_path
    if mode == "global_scalar":
        return np.median(error, axis=(1, 2, 3)).astype(np.float32)
    if mode == "per_cell":
        return np.median(error, axis=1).astype(np.float32)
    raise ValueError(f"Unknown shift mode: {mode}")


def apply_path_shift(
    samples: np.ndarray,
    shifts: np.ndarray,
    mode: str,
    shift_alpha: float,
    residual_scale: float,
) -> np.ndarray:
    median = np.median(samples, axis=1, keepdims=True)
    scaled = median + float(residual_scale) * (samples - median)
    if mode == "global_scalar":
        shift = shifts[:, None, None, None, None]
    elif mode == "per_cell":
        shift = shifts[:, None, None, :, :]
    else:
        raise ValueError(f"Unknown shift mode: {mode}")
    return np.clip(scaled + float(shift_alpha) * shift, 0.0, 1.0).astype(np.float32)


class PathShiftOracleSampler:
    def __init__(
        self,
        base_model: torch.nn.Module,
        shift_by_key: dict[bytes, np.ndarray],
        mode: str,
        shift_alpha: float,
        residual_scale: float,
        device: torch.device,
    ):
        self.base_model = base_model
        self.shift_by_key = shift_by_key
        self.mode = mode
        self.shift_alpha = float(shift_alpha)
        self.residual_scale = float(residual_scale)
        self.device = device

    def eval(self) -> "PathShiftOracleSampler":
        self.base_model.eval()
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
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
        raw_np = raw.detach().cpu().numpy()
        hist_np = history.detach().cpu().numpy()
        if self.mode == "global_scalar":
            shifts = np.array(
                [self.shift_by_key.get(history_key(h), np.float32(0.0)) for h in hist_np],
                dtype=np.float32,
            )
        elif self.mode == "per_cell":
            zero = np.zeros((5, 5), dtype=np.float32)
            shifts = np.stack(
                [self.shift_by_key.get(history_key(h), zero) for h in hist_np],
                axis=0,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown shift mode: {self.mode}")
        shifted = apply_path_shift(
            raw_np,
            shifts=shifts,
            mode=self.mode,
            shift_alpha=self.shift_alpha,
            residual_scale=self.residual_scale,
        )
        return torch.from_numpy(shifted).to(history_device)


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
    parser.add_argument("--oracle_samples", type=int, default=48)
    parser.add_argument("--shift_mode", choices=("global_scalar", "per_cell"), default="per_cell")
    parser.add_argument("--shift_alpha", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=429)
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

    print("Fitting validation-oracle path shifts")
    oracle_samples = sample_native(
        model=base_model,
        history_norm=batch.history_norm,
        n_samples=args.oracle_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    shifts = fit_path_shifts(
        samples=oracle_samples,
        future=batch.future_01.detach().cpu().numpy(),
        mode=args.shift_mode,
    )
    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    shift_by_key = {history_key(h): shifts[i] for i, h in enumerate(hist_norm_np)}
    print("Shift summary:", float(np.min(shifts)), float(np.median(shifts)), float(np.max(shifts)))

    model = PathShiftOracleSampler(
        base_model=base_model,
        shift_by_key=shift_by_key,
        mode=args.shift_mode,
        shift_alpha=args.shift_alpha,
        residual_scale=args.residual_scale,
        device=device,
    ).eval()

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
                "kind": "validation_oracle_path_center_shift",
                "oracle_uses_validation_future": True,
                "claim": "feasibility_upper_bound_not_deployable_model",
                "oracle_samples": args.oracle_samples,
                "shift_mode": args.shift_mode,
                "shift_alpha": args.shift_alpha,
                "residual_scale": args.residual_scale,
                "shift_min": float(np.min(shifts)),
                "shift_median": float(np.median(shifts)),
                "shift_max": float(np.max(shifts)),
                "seed": args.seed,
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
        f"- base model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        "- diagnostic: `validation oracle path-center shift; not deployable`",
        f"- shift mode: `{args.shift_mode}`",
        f"- shift range: `{float(np.min(shifts)):.4f} / {float(np.median(shifts)):.4f} / {float(np.max(shifts)):.4f}`",
        f"- residual scale: `{args.residual_scale:.3f}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Fidelity / Structure**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- median-bias fraction cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- median-bias magnitude cells: `{distributional['median_bias']['n_mag_pass']}/25`",
        f"- regime layer2: `{regime_coverage['layer2_n_passing']}/{regime_coverage['layer2_n_total']}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- cointegration worst-cell ratio: `{cointegration.get('worst_cell_ratio', float('nan')):.3f}`",
        f"- mean-reversion active pass: `{mean_reversion.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "429a Validation-Oracle Path-Shift Diagnostic", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()

