#!/usr/bin/env python
"""720a: residual analysis for sticky-zero readout.

Generate once, then sweep the train-derived no-change threshold to determine
whether the remaining BBB OAS failure is an atom-threshold issue or a deeper
nonzero-tail/correlation tradeoff.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    build_history_future,
    generate_panel_samples,
    load_native_model,
    panel_daily_changes,
    select_raw_state_scope,
    summarize_joint_quality,
)
from experiments.backfill.block_ar.audit_719a_sticky_zero_readout import (  # noqa: E402
    apply_sticky_readout,
    build_sticky_policy,
    train_raw_block,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)


FOCUS = ("factor:aaa_oas", "factor:bbb_oas")


def factor_zero_rows(
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    samples_raw: np.ndarray,
    factor_names: list[str],
    *,
    iv_count: int,
) -> dict[str, dict[str, float]]:
    gt_delta = panel_daily_changes(raw_history, raw_future)
    sample_prev = np.concatenate(
        [
            np.repeat(raw_history[:, None, -1:, :], samples_raw.shape[1], axis=1),
            samples_raw[:, :, :-1, :],
        ],
        axis=2,
    )
    gen_delta = samples_raw - sample_prev
    rows: dict[str, dict[str, float]] = {}
    for rel_idx, name in enumerate(factor_names):
        if name not in FOCUS:
            continue
        idx = int(iv_count) + rel_idx
        gt = gt_delta[..., idx]
        gen = gen_delta[..., idx]
        gt_nonzero = np.abs(gt.reshape(-1)) > 1e-10
        gen_nonzero = np.abs(gen.reshape(-1)) > 1e-10
        rows[name] = {
            "gt_zero_rate": float(np.mean(~gt_nonzero)),
            "gen_zero_rate": float(np.mean(~gen_nonzero)),
            "gt_abs_q99": float(np.quantile(np.abs(gt).reshape(-1), 0.99)),
            "gen_abs_q99": float(np.quantile(np.abs(gen).reshape(-1), 0.99)),
            "gt_nonzero_abs_q50": float(np.quantile(np.abs(gt.reshape(-1)[gt_nonzero]), 0.50))
            if np.any(gt_nonzero)
            else 0.0,
            "gen_nonzero_abs_q50": float(np.quantile(np.abs(gen.reshape(-1)[gen_nonzero]), 0.50))
            if np.any(gen_nonzero)
            else 0.0,
        }
    return rows


def focus_ks(summary: dict[str, Any]) -> dict[str, float]:
    out = {}
    for row in summary["per_factor"]:
        if row["name"] in FOCUS:
            out[row["name"]] = float(row["ks_delta"])
    return out


def run_scope(args: argparse.Namespace, scope: str, checkpoint: str, seed: int) -> dict[str, Any]:
    scope_args = argparse.Namespace(**vars(args))
    scope_args.state_scope = scope
    set_seed(int(seed))
    device = torch.device(scope_args.device if torch.cuda.is_available() or scope_args.device == "cpu" else "cpu")
    model, payload = load_native_model(scope_args.model_type, checkpoint, device)
    history, _future, specs, block, alignment = build_history_future(scope_args, payload)
    state_scope = payload.get("state_scope", scope)
    raw_history_full, raw_future_full = select_raw_state_scope(block, state_scope, int(scope_args.iv_count))
    n_history = int(history[0].shape[0] if isinstance(history, tuple) else history.shape[0])
    n_windows = min(int(scope_args.max_windows), n_history)
    if isinstance(history, tuple):
        history = tuple(item[:n_windows] for item in history)
    else:
        history = history[:n_windows]
    raw_history = raw_history_full[:n_windows]
    raw_future = raw_future_full[:n_windows]
    train_history, train_future, train_specs, train_alignment = train_raw_block(scope_args, payload, state_scope)
    expected_names = [spec.name for spec in specs]
    if expected_names != [spec.name for spec in train_specs]:
        raise RuntimeError("train/validation specs differ")
    if state_scope == "joint38":
        audit_iv_count = int(scope_args.iv_count)
        factor_names = [spec.name for spec in specs[audit_iv_count:]]
    else:
        audit_iv_count = 0
        factor_names = [spec.name for spec in specs]

    t0 = time.time()
    samples_raw = generate_panel_samples(
        model,
        history,
        specs,
        raw_history,
        samples=int(scope_args.samples),
        n_steps=int(scope_args.n_steps),
        batch_size=int(scope_args.batch_size),
        chunk_size=int(scope_args.chunk_size),
        device=device,
        sample_temperature=float(scope_args.sample_temperature),
        value_coordinate=payload.get("value_coordinate", scope_args.value_coordinate),
        model_coordinate=payload.get("model_coordinate", "state"),
    )
    variants: dict[str, Any] = {}
    base_summary = summarize_joint_quality(raw_history, raw_future, samples_raw, factor_names, iv_count=audit_iv_count)
    variants["identity"] = {
        "summary": {
            "factor_delta_ks_mean": base_summary["factor_delta_ks_mean"],
            "factor_delta_ks_pass_020": base_summary["factor_delta_ks_pass_020"],
            "factor_factor_abs_ratio": base_summary["factor_factor_corr"]["gen_mean_abs"]
            / max(base_summary["factor_factor_corr"]["gt_mean_abs"], 1e-12),
            "iv_factor_abs_ratio": base_summary["iv_factor_corr"]["gen_mean_abs"]
            / max(base_summary["iv_factor_corr"]["gt_mean_abs"], 1e-12),
            "conditional_reduction": base_summary["conditional_panel"]["median_mae_reduction_vs_rolled_pct"],
            "focus_ks": focus_ks(base_summary),
            "focus_zero": factor_zero_rows(raw_history, raw_future, samples_raw, factor_names, iv_count=audit_iv_count),
        }
    }
    for quantile in scope_args.quantiles:
        policy = build_sticky_policy(
            train_history,
            train_future,
            expected_names,
            zero_eps=float(scope_args.sticky_zero_eps),
            zero_rate_gate=float(scope_args.sticky_zero_rate_gate),
            nonzero_quantile=float(quantile),
        )
        sticky = apply_sticky_readout(samples_raw, raw_history, policy)
        summary = summarize_joint_quality(raw_history, raw_future, sticky, factor_names, iv_count=audit_iv_count)
        variants[f"q{quantile:g}"] = {
            "selected_names": policy["selected_names"],
            "thresholds": {
                row["name"]: row["threshold"]
                for row in policy["rows"]
                if row["selected"]
            },
            "summary": {
                "factor_delta_ks_mean": summary["factor_delta_ks_mean"],
                "factor_delta_ks_pass_020": summary["factor_delta_ks_pass_020"],
                "factor_factor_abs_ratio": summary["factor_factor_corr"]["gen_mean_abs"]
                / max(summary["factor_factor_corr"]["gt_mean_abs"], 1e-12),
                "iv_factor_abs_ratio": summary["iv_factor_corr"]["gen_mean_abs"]
                / max(summary["iv_factor_corr"]["gt_mean_abs"], 1e-12),
                "conditional_reduction": summary["conditional_panel"]["median_mae_reduction_vs_rolled_pct"],
                "focus_ks": focus_ks(summary),
                "focus_zero": factor_zero_rows(raw_history, raw_future, sticky, factor_names, iv_count=audit_iv_count),
            },
        }
    return {
        "scope": state_scope,
        "checkpoint": checkpoint,
        "seed": int(seed),
        "n_windows": int(n_windows),
        "samples": int(scope_args.samples),
        "generation_time_s": float(time.time() - t0),
        "alignment": alignment,
        "train_alignment": train_alignment,
        "variants": variants,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 720a Sticky-Zero Residual Sweep",
        "",
        "## Variant Summary",
        "",
        "| scope | variant | pass | mean KS | AAA KS | BBB KS | AAA zero gen/gt | BBB zero gen/gt | factor corr abs ratio | conditional |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope_name, scope in report["scopes"].items():
        for variant, payload in scope["variants"].items():
            s = payload["summary"]
            focus_ks = s["focus_ks"]
            focus_zero = s["focus_zero"]
            aaa_zero = focus_zero.get("factor:aaa_oas", {})
            bbb_zero = focus_zero.get("factor:bbb_oas", {})
            lines.append(
                f"| `{scope_name}` | `{variant}` | `{s['factor_delta_ks_pass_020']}/13` | "
                f"`{s['factor_delta_ks_mean']:.3f}` | "
                f"`{focus_ks.get('factor:aaa_oas', float('nan')):.3f}` | "
                f"`{focus_ks.get('factor:bbb_oas', float('nan')):.3f}` | "
                f"`{aaa_zero.get('gen_zero_rate', float('nan')):.3f}/{aaa_zero.get('gt_zero_rate', float('nan')):.3f}` | "
                f"`{bbb_zero.get('gen_zero_rate', float('nan')):.3f}/{bbb_zero.get('gt_zero_rate', float('nan')):.3f}` | "
                f"`{s['factor_factor_abs_ratio']:.3f}` | `{s['conditional_reduction']:.2f}` |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="662a")
    parser.add_argument("--anchor_checkpoint", required=True)
    parser.add_argument("--joint_checkpoint", required=True)
    parser.add_argument("--value_coordinate", choices=["raw", "encoded"], default="raw")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--sticky_zero_eps", type=float, default=1e-10)
    parser.add_argument("--sticky_zero_rate_gate", type=float, default=0.25)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.10, 0.50, 0.90])
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    scopes = {
        "anchor": run_scope(args, "anchor_only", args.anchor_checkpoint, int(args.seed) + 1),
        "joint": run_scope(args, "joint38", args.joint_checkpoint, int(args.seed) + 2),
    }
    decision = (
        "If stronger train-derived thresholds move BBB below the KS gate without "
        "destroying correlation amplitude, sticky readout is a viable data-coordinate "
        "repair. If zero rates remain below GT or correlation collapses, the next model "
        "should use an explicit mixed discrete-continuous innovation variable rather "
        "than more threshold tuning."
    )
    report = {
        "iteration": "720a",
        "purpose": "sticky-zero residual threshold and atom-mass attribution",
        "scopes": scopes,
        "decision": decision,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(out_md, report)
    print(json.dumps(make_serializable({"decision": decision}), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
