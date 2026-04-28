#!/usr/bin/env python
"""718a: root-cause audit for OAS stickiness and IV coverage regressions.

This is a post-experiment analysis after 717a. It does not propose a new model.
It asks whether the repeated AAA/BBB OAS failures are caused by:

- train/validation distribution shift,
- support-coordinate choice,
- zero/sticky increment structure,
- or recent model variants changing the same failure in different ways.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.normalized_innovation_662_utils import (  # noqa: E402
    normalize_increment_windows,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)


FOCUS_FACTORS = {"factor:aaa_oas", "factor:bbb_oas"}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ks_2samp(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    data = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, data, side="right") / x.size
    cdf_y = np.searchsorted(y, data, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def rankdata(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    sorted_x = x[order]
    n = x.size
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def raw_future_delta(history_state: np.ndarray, future_state: np.ndarray) -> np.ndarray:
    base = np.concatenate([history_state[:, -1:, :], future_state], axis=1)
    return np.diff(base, axis=1)


def raw_history_delta(history_state: np.ndarray) -> np.ndarray:
    return np.diff(history_state, axis=1)


def build_policy_blocks(args: argparse.Namespace, policy: str) -> tuple[Any, Any]:
    block_args = SimpleNamespace(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        iv_count=int(args.iv_count),
        max_train_windows=int(args.max_train_windows),
        clean_nonpositive_log_levels=True,
        positive_level_policy=policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=float(args.iv_lower_bound),
        iv_upper_bound=float(args.iv_upper_bound),
    )
    _columns, _metadata, train_block, val_block = build_blocks(block_args)
    return train_block, val_block


def anchor_arrays(block: Any, iv_count: int) -> dict[str, Any]:
    specs = block.specs[int(iv_count) :]
    history_state = block.history_state[..., int(iv_count) :].astype(np.float64)
    future_state = block.future_state[..., int(iv_count) :].astype(np.float64)
    history_increment = block.history_increment[..., int(iv_count) :].astype(np.float64)
    future_increment = block.future_increment[..., int(iv_count) :].astype(np.float64)
    history_norm, future_norm, _center, scale = normalize_increment_windows(
        history_increment.astype(np.float32),
        future_increment.astype(np.float32),
        half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
    )
    return {
        "names": [spec.name for spec in specs],
        "transforms": {spec.name: spec.transform for spec in specs},
        "history_state": history_state,
        "future_state": future_state,
        "history_raw_delta": raw_history_delta(history_state),
        "future_raw_delta": raw_future_delta(history_state, future_state),
        "history_increment": history_increment,
        "future_increment": future_increment,
        "history_norm": history_norm.astype(np.float64),
        "future_norm": future_norm.astype(np.float64),
        "scale": scale.astype(np.float64),
    }


def qabs(x: np.ndarray, q: float) -> float:
    return float(np.quantile(np.abs(np.asarray(x, dtype=np.float64).ravel()), q))


def factor_data_summary(
    factor: str,
    train: dict[str, Any],
    val: dict[str, Any],
) -> dict[str, Any]:
    idx = train["names"].index(factor)
    train_raw = train["future_raw_delta"][..., idx]
    val_raw = val["future_raw_delta"][..., idx]
    train_enc = train["future_increment"][..., idx]
    val_enc = val["future_increment"][..., idx]
    train_norm = train["future_norm"][..., idx]
    val_norm = val["future_norm"][..., idx]
    train_hist_raw = train["history_raw_delta"][..., idx]
    val_hist_raw = val["history_raw_delta"][..., idx]
    train_future_zero_window = np.mean(np.abs(train_raw) <= 1e-10, axis=1)
    val_future_zero_window = np.mean(np.abs(val_raw) <= 1e-10, axis=1)
    train_history_zero_window = np.mean(np.abs(train_hist_raw) <= 1e-10, axis=1)
    val_history_zero_window = np.mean(np.abs(val_hist_raw) <= 1e-10, axis=1)
    train_history_activity = np.quantile(np.abs(train_hist_raw), 0.95, axis=1)
    val_history_activity = np.quantile(np.abs(val_hist_raw), 0.95, axis=1)
    train_future_activity = np.quantile(np.abs(train_raw), 0.95, axis=1)
    val_future_activity = np.quantile(np.abs(val_raw), 0.95, axis=1)
    jump_threshold = max(qabs(train_raw, 0.95), 1e-12)
    return {
        "factor": factor,
        "transform": train["transforms"][factor],
        "train_zero_rate": float(np.mean(np.abs(train_raw) <= 1e-10)),
        "val_zero_rate": float(np.mean(np.abs(val_raw) <= 1e-10)),
        "train_raw_abs_q95": qabs(train_raw, 0.95),
        "val_raw_abs_q95": qabs(val_raw, 0.95),
        "train_raw_abs_q99": qabs(train_raw, 0.99),
        "val_raw_abs_q99": qabs(val_raw, 0.99),
        "val_train_raw_q99_ratio": qabs(val_raw, 0.99) / max(qabs(train_raw, 0.99), 1e-12),
        "raw_delta_train_val_ks": ks_2samp(train_raw, val_raw),
        "encoded_increment_train_val_ks": ks_2samp(train_enc, val_enc),
        "normalized_increment_train_val_ks": ks_2samp(train_norm, val_norm),
        "train_norm_abs_q99": qabs(train_norm, 0.99),
        "val_norm_abs_q99": qabs(val_norm, 0.99),
        "val_train_norm_q99_ratio": qabs(val_norm, 0.99) / max(qabs(train_norm, 0.99), 1e-12),
        "train_scale_median": float(np.median(train["scale"][..., idx])),
        "val_scale_median": float(np.median(val["scale"][..., idx])),
        "train_scale_floor_rate": float(np.mean(train["scale"][..., idx] <= 1.0001e-4)),
        "val_scale_floor_rate": float(np.mean(val["scale"][..., idx] <= 1.0001e-4)),
        "train_history_future_activity_spearman": spearman(train_history_activity, train_future_activity),
        "val_history_future_activity_spearman": spearman(val_history_activity, val_future_activity),
        "train_history_zero_future_zero_spearman": spearman(
            train_history_zero_window, train_future_zero_window
        ),
        "val_history_zero_future_zero_spearman": spearman(
            val_history_zero_window, val_future_zero_window
        ),
        "train_history_activity_future_jump_spearman": spearman(
            train_history_activity, np.mean(np.abs(train_raw) > jump_threshold, axis=1)
        ),
        "val_history_activity_future_jump_spearman": spearman(
            val_history_activity, np.mean(np.abs(val_raw) > jump_threshold, axis=1)
        ),
    }


def panel_factor_rows(path: str) -> dict[str, Any]:
    payload = load_json(Path(path))
    if payload is None:
        return {"path": path, "exists": False}
    rows = {}
    for row in payload.get("summary", {}).get("per_factor", []):
        if row.get("name") in FOCUS_FACTORS:
            rows[row["name"]] = {
                "ks_delta": float(row.get("ks_delta", float("nan"))),
                "q99_abs_delta_ratio": float(row.get("q99_abs_delta_ratio", float("nan"))),
                "gt_min": float(row.get("gt_min", float("nan"))),
                "gt_max": float(row.get("gt_max", float("nan"))),
                "gen_min": float(row.get("gen_min", float("nan"))),
                "gen_max": float(row.get("gen_max", float("nan"))),
            }
    return {
        "path": path,
        "exists": True,
        "factor_delta_ks_mean": float(payload.get("summary", {}).get("factor_delta_ks_mean", float("nan"))),
        "factor_delta_ks_pass_020": int(payload.get("summary", {}).get("factor_delta_ks_pass_020", -1)),
        "iv_factor_corr": payload.get("summary", {}).get("iv_factor_corr", {}),
        "conditional_panel": payload.get("summary", {}).get("conditional_panel", {}),
        "focus_rows": rows,
    }


def iv_summary(path: str) -> dict[str, Any]:
    payload = load_json(Path(path))
    if payload is None:
        return {"path": path, "exists": False}
    coverage = payload.get("coverage", {})
    dist = payload.get("distributional_fidelity", {})
    regime = payload.get("regime_coverage", {})
    return {
        "path": path,
        "exists": True,
        "n_pass": int(payload.get("summary", {}).get("n_pass", -1)),
        "failed_suites": payload.get("summary", {}).get("failed_suites", []),
        "cov90": float(coverage.get("overall", {}).get("0.9", float("nan"))),
        "h30_worst": float(coverage.get("worst_cell_per_horizon", {}).get("30", float("nan"))),
        "level_ks_pass": int(dist.get("ks_level_test", {}).get("n_pass", -1)),
        "median_bias_pass": int(dist.get("median_bias", {}).get("n_pass", -1)),
        "regime_layer2_pass": bool(regime.get("layer2_pass", False)),
        "risk_state_pass": bool(payload.get("risk_state_allocation", {}).get("overall_pass", False)),
        "max_jump_ks": float(
            payload.get("pathwise_jump_realism", {})
            .get("pathwise_max_jump", {})
            .get("ks_stat", float("nan"))
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    policy_reports: dict[str, Any] = {}
    for policy in ["reference_based", "observed_positive"]:
        train_block, val_block = build_policy_blocks(args, policy)
        train = anchor_arrays(train_block, int(args.iv_count))
        val = anchor_arrays(val_block, int(args.iv_count))
        policy_reports[policy] = {
            factor: factor_data_summary(factor, train, val)
            for factor in sorted(FOCUS_FACTORS)
        }

    panel_paths = {
        "688_anchor_reference_baseline": "results/validations/2026-04-28/688a_framework_lock/anchor_val_panel.json",
        "676_joint_reference_baseline": "results/validations/2026-04-27/676a_joint38_channel_level_alltrain/panel_audit.json",
        "714_anchor_mcrps": "results/block_ar/714a_mcrps010_frozen_scorecard/anchor_val_panel_s64.json",
        "714_joint_mcrps": "results/block_ar/714a_mcrps010_frozen_scorecard/joint_val_panel_s64.json",
        "716_anchor_innovscore": "results/block_ar/716a_innovscore_frozen_scorecard/anchor_val_panel_s64.json",
        "716_joint_innovscore": "results/block_ar/716a_innovscore_frozen_scorecard/joint_val_panel_s64.json",
        "717_anchor_observed_positive": "results/block_ar/717a_obspos_frozen_scorecard/anchor_val_panel_s64.json",
        "717_joint_observed_positive": "results/block_ar/717a_obspos_frozen_scorecard/joint_val_panel_s64.json",
    }
    iv_paths = {
        "674_iv_reference_baseline": "results/validations/2026-04-27/674a_channel_level_alltrain_e3/iv_val_full11.json",
        "711_674_iv_current": "results/block_ar/711a_norminnov_active_family_audit/674a_iv_val_s64_current.json",
        "714_iv_mcrps": "results/block_ar/714a_mcrps010_frozen_scorecard/iv_val_full11_s64.json",
        "716_iv_innovscore": "results/block_ar/716a_innovscore_frozen_scorecard/iv_val_full11_s64.json",
        "717_iv_observed_positive": "results/block_ar/717a_obspos_frozen_scorecard/iv_val_full11_s64.json",
    }
    model_reports = {name: panel_factor_rows(path) for name, path in panel_paths.items()}
    iv_reports = {name: iv_summary(path) for name, path in iv_paths.items()}

    bbb_ref = policy_reports["reference_based"]["factor:bbb_oas"]
    bbb_obs = policy_reports["observed_positive"]["factor:bbb_oas"]
    decision = {
        "failure_class": "sticky spread coordinate/objective failure plus separate IV long-horizon undercoverage",
        "support_policy_read": (
            "Observed-positive log-level support changes the coordinate but does not remove "
            "the OAS failure; the raw OAS series has high no-change mass and large "
            "state-dependent jumps, so a continuous-only innovation law smears an atom at "
            "zero into small and occasional excessive moves."
        ),
        "train_val_read": (
            "OAS raw-increment drift is not the dominant explanation if raw train/val KS "
            "stays moderate while generated OAS KS remains high across trainable variants. "
            "The issue is primarily data-object mismatch inside the continuous law."
        ),
        "next_step": (
            "Do not add another global loss or backend switch. The next decisive move is "
            "a generic sticky/mixed discrete-continuous innovation coordinate for channels "
            "with empirical no-change atoms, frozen across IV, anchor, and joint scopes. "
            "It should be formulated as a variable-type data coordinate, not an OAS-only "
            "special case."
        ),
        "bbb_reference_zero_rate": bbb_ref["val_zero_rate"],
        "bbb_observed_positive_zero_rate": bbb_obs["val_zero_rate"],
    }
    return {
        "iteration": "718a",
        "purpose": "post-experiment root-cause audit after 717a observed-positive falsifier",
        "data_policy_reports": policy_reports,
        "model_oas_reports": model_reports,
        "iv_reports": iv_reports,
        "decision": decision,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 718a OAS Sticky-Increment And IV Coverage Attribution",
        "",
        "## Data Object Audit",
        "",
        "| policy | factor | transform | train zero | val zero | raw KS | norm KS | val/train raw q99 | val hist/fut activity rho | val zero/zero rho |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, factors in report["data_policy_reports"].items():
        for factor, row in factors.items():
            lines.append(
                f"| `{policy}` | `{factor}` | `{row['transform']}` | "
                f"`{fmt(row['train_zero_rate'])}` | `{fmt(row['val_zero_rate'])}` | "
                f"`{fmt(row['raw_delta_train_val_ks'])}` | "
                f"`{fmt(row['normalized_increment_train_val_ks'])}` | "
                f"`{fmt(row['val_train_raw_q99_ratio'])}` | "
                f"`{fmt(row['val_history_future_activity_spearman'])}` | "
                f"`{fmt(row['val_history_zero_future_zero_spearman'])}` |"
            )
    lines.extend(
        [
            "",
            "## Recent Model OAS Behavior",
            "",
            "| run | pass | mean KS | AAA KS | BBB KS | AAA gen range | BBB gen range | IV-factor corr | conditional panel |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for name, payload in report["model_oas_reports"].items():
        if not payload.get("exists"):
            continue
        aaa = payload["focus_rows"].get("factor:aaa_oas", {})
        bbb = payload["focus_rows"].get("factor:bbb_oas", {})
        iv_corr = payload.get("iv_factor_corr", {}).get("matrix_corr", float("nan"))
        cond = payload.get("conditional_panel", {}).get(
            "median_mae_reduction_vs_rolled_pct", float("nan")
        )
        lines.append(
            f"| `{name}` | `{payload['factor_delta_ks_pass_020']}/13` | "
            f"`{fmt(payload['factor_delta_ks_mean'])}` | "
            f"`{fmt(aaa.get('ks_delta', float('nan')))}` | "
            f"`{fmt(bbb.get('ks_delta', float('nan')))}` | "
            f"`[{fmt(aaa.get('gen_min', float('nan')))}, {fmt(aaa.get('gen_max', float('nan')))}]` | "
            f"`[{fmt(bbb.get('gen_min', float('nan')))}, {fmt(bbb.get('gen_max', float('nan')))}]` | "
            f"`{fmt(iv_corr)}` | `{fmt(cond)}` |"
        )
    lines.extend(
        [
            "",
            "## IV Coverage Side",
            "",
            "| run | full 11 | cov90 | h30 worst | level KS | max-jump KS | risk-state | failed suites |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, row in report["iv_reports"].items():
        if not row.get("exists"):
            continue
        failed = ", ".join(row["failed_suites"])
        lines.append(
            f"| `{name}` | `{row['n_pass']}/11` | `{fmt(row['cov90'])}` | "
            f"`{fmt(row['h30_worst'])}` | `{row['level_ks_pass']}/25` | "
            f"`{fmt(row['max_jump_ks'])}` | `{row['risk_state_pass']}` | {failed} |"
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Mechanism Read",
            "",
            f"- failure class: `{decision['failure_class']}`",
            f"- support-policy read: {decision['support_policy_read']}",
            f"- train/validation read: {decision['train_val_read']}",
            "",
            "## Decision",
            "",
            decision["next_step"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    report = build_report(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(out_md, report)
    print(json.dumps(make_serializable(report["decision"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
