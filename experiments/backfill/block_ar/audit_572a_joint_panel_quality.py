#!/usr/bin/env python
"""572a: audit joint IV+factor panel scenario quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.panel_daily_cholesky_transition_model import load_model  # noqa: E402
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    build_panel_block,
    load_aligned_iv_factor_panel,
    panel_summary,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402


def ks_statistic_1d(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Two-sample empirical KS statistic without SciPy dependency."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    cand = cand[np.isfinite(cand)]
    if ref.size == 0 or cand.size == 0:
        return float("nan")
    ref = np.sort(ref)
    cand = np.sort(cand)
    points = np.sort(np.concatenate([ref, cand]))
    ref_cdf = np.searchsorted(ref, points, side="right") / float(ref.size)
    cand_cdf = np.searchsorted(cand, points, side="right") / float(cand.size)
    return float(np.max(np.abs(ref_cdf - cand_cdf)))


def quantile_scale_ratio(reference: np.ndarray, candidate: np.ndarray, q: float = 0.99) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ref_q = float(np.nanquantile(np.abs(ref), q))
    cand_q = float(np.nanquantile(np.abs(cand), q))
    return cand_q / max(ref_q, 1e-12)


def _is_return_like(column: str) -> bool:
    return column.endswith("_logret") or column.endswith("_diff") or "return:" in column


def reconstruct_factor_levels_from_returns(
    history_panel: np.ndarray,
    samples: np.ndarray,
    *,
    columns: list[str],
    iv_count: int = 25,
) -> np.ndarray:
    """Derive factor level channels from generated return/diff channels.

    This enforces the accounting identity between factor levels and their
    generated one-day increments without changing IV channels or return channels.
    """
    history = np.asarray(history_panel, dtype=np.float64)
    out = np.asarray(samples, dtype=np.float64).copy()
    if history.ndim != 3 or out.ndim != 4:
        raise ValueError("Expected history_panel (B,H,D) and samples (B,K,T,D)")
    if history.shape[0] != out.shape[0] or history.shape[2] != out.shape[3]:
        raise ValueError(f"Shape mismatch: history={history.shape}, samples={out.shape}")
    name_to_idx = {name: idx for idx, name in enumerate(columns)}
    for level_idx in range(iv_count, len(columns)):
        level_name = columns[level_idx]
        if _is_return_like(level_name):
            continue
        base_name = level_name.removeprefix("factor:")
        logret_name = f"factor:{base_name}_logret"
        diff_name = f"factor:{base_name}_diff"
        last_level = history[:, -1, level_idx][:, None, None]
        if logret_name in name_to_idx:
            ret_idx = name_to_idx[logret_name]
            cumulative = np.cumsum(out[..., ret_idx], axis=2)
            out[..., level_idx] = last_level * np.exp(cumulative)
        elif diff_name in name_to_idx:
            ret_idx = name_to_idx[diff_name]
            cumulative = np.cumsum(out[..., ret_idx], axis=2)
            out[..., level_idx] = last_level + cumulative
    return out.astype(np.float32)


def _safe_median(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _safe_worst_ks(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _safe_worst_scale(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.maximum(finite, 1.0 / np.maximum(finite, 1e-12))))


def summarize_factor_quality(
    gt_future: np.ndarray,
    gen_samples: np.ndarray,
    *,
    columns: list[str],
    iv_count: int = 25,
) -> dict[str, Any]:
    """Compare generated and realized factor panels.

    Shapes:
    - gt_future: (B, T, D)
    - gen_samples: (B, K, T, D)
    """
    gt = np.asarray(gt_future, dtype=np.float64)
    gen = np.asarray(gen_samples, dtype=np.float64)
    if gt.ndim != 3 or gen.ndim != 4:
        raise ValueError("Expected gt_future (B,T,D) and gen_samples (B,K,T,D)")
    if gt.shape[0] != gen.shape[0] or gt.shape[1] != gen.shape[2] or gt.shape[2] != gen.shape[3]:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, gen={gen.shape}")
    if len(columns) != gt.shape[-1]:
        raise ValueError("Column count does not match panel dimension")

    factor_indices = list(range(iv_count, gt.shape[-1]))
    level_indices = [idx for idx in factor_indices if not _is_return_like(columns[idx])]
    return_indices = [idx for idx in factor_indices if _is_return_like(columns[idx])]

    gt_flat = gt.reshape(-1, gt.shape[-1])
    gen_flat = gen.reshape(-1, gen.shape[-1])
    level_ks = [ks_statistic_1d(gt_flat[:, idx], gen_flat[:, idx]) for idx in level_indices]
    return_ks = [ks_statistic_1d(gt_flat[:, idx], gen_flat[:, idx]) for idx in return_indices]

    gt_level_delta = np.diff(gt[:, :, level_indices], axis=1) if level_indices else np.empty((0,))
    gen_level_delta = np.diff(gen[:, :, :, level_indices], axis=2) if level_indices else np.empty((0,))
    level_delta_scale = [
        quantile_scale_ratio(gt_level_delta[..., i], gen_level_delta[..., i], q=0.99)
        for i in range(len(level_indices))
    ]
    return_scale = [
        quantile_scale_ratio(gt[:, :, idx], gen[:, :, :, idx], q=0.99)
        for idx in return_indices
    ]

    return {
        "finite_rate": float(np.isfinite(gen).mean()),
        "factor_level_count": int(len(level_indices)),
        "factor_return_count": int(len(return_indices)),
        "level_ks_median": _safe_median(level_ks),
        "level_ks_worst": _safe_worst_ks(level_ks),
        "return_ks_median": _safe_median(return_ks),
        "return_ks_worst": _safe_worst_ks(return_ks),
        "level_delta_q99_scale_median": _safe_median(level_delta_scale),
        "level_delta_q99_scale_worst_fold": _safe_worst_scale(level_delta_scale),
        "return_q99_scale_median": _safe_median(return_scale),
        "return_q99_scale_worst_fold": _safe_worst_scale(return_scale),
        "level_ks_by_column": {
            columns[idx]: float(val) for idx, val in zip(level_indices, level_ks, strict=True)
        },
        "return_ks_by_column": {
            columns[idx]: float(val) for idx, val in zip(return_indices, return_ks, strict=True)
        },
        "level_delta_q99_scale_by_column": {
            columns[idx]: float(val) for idx, val in zip(level_indices, level_delta_scale, strict=True)
        },
        "return_q99_scale_by_column": {
            columns[idx]: float(val) for idx, val in zip(return_indices, return_scale, strict=True)
        },
    }


def summarize_iv_factor_comovement(
    gt_future: np.ndarray,
    gen_samples: np.ndarray,
    *,
    columns: list[str],
    iv_count: int = 25,
) -> dict[str, Any]:
    gt = np.asarray(gt_future, dtype=np.float64)
    gen = np.asarray(gen_samples, dtype=np.float64)
    factor_indices = list(range(iv_count, gt.shape[-1]))
    gt_iv_move = gt[:, -1, :iv_count].mean(axis=1) - gt[:, 0, :iv_count].mean(axis=1)
    gen_iv_move = gen[:, :, -1, :iv_count].mean(axis=-1) - gen[:, :, 0, :iv_count].mean(axis=-1)
    rows = []
    abs_errors = []
    for idx in factor_indices:
        gt_factor_move = gt[:, -1, idx] - gt[:, 0, idx]
        gen_factor_move = gen[:, :, -1, idx] - gen[:, :, 0, idx]
        gt_corr = float(np.corrcoef(gt_iv_move, gt_factor_move)[0, 1])
        gen_corr = float(np.corrcoef(gen_iv_move.ravel(), gen_factor_move.ravel())[0, 1])
        if not np.isfinite(gt_corr):
            gt_corr = 0.0
        if not np.isfinite(gen_corr):
            gen_corr = 0.0
        rows.append(
            {
                "column": columns[idx],
                "gt_corr_with_avg_iv_move": gt_corr,
                "gen_corr_with_avg_iv_move": gen_corr,
                "abs_error": abs(gen_corr - gt_corr),
            }
        )
        abs_errors.append(abs(gen_corr - gt_corr))
    return {
        "iv_factor_corr_mae": _safe_median(abs_errors),
        "iv_factor_corr_worst_abs_error": _safe_worst_ks(abs_errors),
        "by_column": rows,
    }


def risk_manager_gate(
    factor_quality: dict[str, Any],
    comovement: dict[str, Any],
    iv_full11: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "finite_panel": factor_quality["finite_rate"] >= 0.999,
        "factor_level_marginals": factor_quality["level_ks_median"] <= 0.30
        and factor_quality["level_ks_worst"] <= 0.60,
        "factor_return_marginals": factor_quality["return_ks_median"] <= 0.30
        and factor_quality["return_ks_worst"] <= 0.60,
        "factor_level_move_scale": factor_quality["level_delta_q99_scale_worst_fold"] <= 4.0,
        "factor_return_scale": factor_quality["return_q99_scale_worst_fold"] <= 4.0,
        "iv_factor_comovement": comovement["iv_factor_corr_mae"] <= 0.35,
    }
    if iv_full11 is not None:
        checks["iv_surface_validity"] = bool(iv_full11["surface"]["overall_pass"])
        checks["iv_conditionality"] = float(iv_full11["conditionality"]["mae_reduction_pct"]) >= 5.0
        checks["iv_cross_cell_structure"] = bool(iv_full11["cross_cell_correlation"]["overall_pass"])
        checks["iv_pathwise_realism"] = bool(iv_full11["pathwise_jump_realism"]["overall_pass"])
        checks["iv_lower_coverage"] = all(
            float(hvals["0.9"] if "0.9" in hvals else hvals[0.9]) >= gate
            for hvals, gate in [
                (iv_full11["coverage"]["per_horizon"]["1"], 0.80),
                (iv_full11["coverage"]["per_horizon"]["7"], 0.75),
                (iv_full11["coverage"]["per_horizon"]["14"], 0.70),
                (iv_full11["coverage"]["per_horizon"]["30"], 0.65),
            ]
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "acceptable": len(failed) == 0,
        "failed_checks": failed,
        "checks": checks,
    }


def sample_panel(
    model: Any,
    history_panel: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
) -> np.ndarray:
    outs = []
    for start in range(0, history_panel.shape[0], batch_size):
        end = min(start + batch_size, history_panel.shape[0])
        samples = model.sample_batched(
            history_panel[start:end],
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
        )
        outs.append(samples.detach().cpu().numpy())
        print(f"  sampled panel windows {end}/{history_panel.shape[0]}", flush=True)
    return np.concatenate(outs, axis=0).astype(np.float32)


def load_optional_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    gate = result["risk_manager_gate"]
    factor = result["factor_quality"]
    comove = result["iv_factor_comovement"]
    lines = [
        "# 572a Joint Panel Scenario Quality Audit",
        "",
        f"- checkpoint: `{result['config']['checkpoint']}`",
        f"- future horizon: `{result['config']['future_len']}`",
        f"- windows: `{result['config']['n_windows']}`",
        f"- samples/window: `{result['config']['samples']}`",
        f"- risk-manager acceptable: `{gate['acceptable']}`",
        f"- failed checks: `{', '.join(gate['failed_checks']) if gate['failed_checks'] else 'none'}`",
        "",
        "## Factor Quality",
        "",
        f"- finite rate: `{factor['finite_rate']:.6f}`",
        f"- level KS median/worst: `{factor['level_ks_median']:.3f}` / `{factor['level_ks_worst']:.3f}`",
        f"- return KS median/worst: `{factor['return_ks_median']:.3f}` / `{factor['return_ks_worst']:.3f}`",
        f"- level delta q99 worst-fold error: `{factor['level_delta_q99_scale_worst_fold']:.3f}`",
        f"- return q99 worst-fold error: `{factor['return_q99_scale_worst_fold']:.3f}`",
        "",
        "## IV-Factor Co-Movement",
        "",
        f"- median abs corr error: `{comove['iv_factor_corr_mae']:.3f}`",
        f"- worst abs corr error: `{comove['iv_factor_corr_worst_abs_error']:.3f}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, required=True)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=572)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iv_full11_json", default=None)
    parser.add_argument("--reconstruct_factor_levels", action="store_true")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    panel, columns, _ = load_aligned_iv_factor_panel()
    _, val_indices = official_train_val_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    val_indices = val_indices[: args.max_windows]
    panel_block = build_panel_block(
        panel,
        columns,
        val_indices,
        args.history_len,
        args.future_len,
        device,
    )
    samples = sample_panel(
        model,
        panel_block.history_panel,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    if args.reconstruct_factor_levels:
        samples = reconstruct_factor_levels_from_returns(
            panel_block.history_panel.detach().cpu().numpy(),
            samples,
            columns=columns,
            iv_count=25,
        )
    gt_future = panel_block.future_panel.detach().cpu().numpy()
    factor_quality = summarize_factor_quality(gt_future, samples, columns=columns, iv_count=25)
    comovement = summarize_iv_factor_comovement(gt_future, samples, columns=columns, iv_count=25)
    iv_full11 = load_optional_json(args.iv_full11_json)
    gate = risk_manager_gate(factor_quality, comovement, iv_full11)
    result = {
        "risk_manager_gate": gate,
        "factor_quality": factor_quality,
        "iv_factor_comovement": comovement,
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
            "history_len": int(args.history_len),
            "future_len": int(args.future_len),
            "n_windows": int(panel_block.history_panel.shape[0]),
            "samples": int(args.samples),
            "seed": int(args.seed),
            "reconstruct_factor_levels": bool(args.reconstruct_factor_levels),
            "panel": panel_summary(panel_block),
            "iv_full11_json": args.iv_full11_json,
        },
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(result), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(json.dumps(make_serializable(gate), indent=2))


if __name__ == "__main__":
    main()
