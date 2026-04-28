#!/usr/bin/env python
"""719a: joint-panel audit with a generic sticky-zero readout.

The model and sampler are unchanged. The only added rule is a deterministic
data-coordinate readout for channels whose training data has a large exact
no-change atom: very small generated raw daily moves are decoded as no-change.

This is meant to test whether OAS failures are caused by forcing sticky quoted
series through a fully continuous innovation readout.
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

from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    clean_nonpositive_log_level_factors,
)
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    build_history_future,
    generate_panel_samples,
    load_native_model,
    panel_daily_changes,
    select_raw_state_scope,
    state_block_alignment_diagnostics,
    summarize_joint_quality,
    write_markdown,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    build_increment_coordinate_block,
)


def payload_panel_options(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    positive_level_policy = payload.get(
        "positive_level_policy",
        payload.get("panel_metadata", {}).get(
            "positive_level_policy",
            getattr(args, "positive_level_policy", "reference_based"),
        ),
    )
    iv_transform = payload.get(
        "iv_transform",
        payload.get("normalization", {}).get(
            "iv_transform",
            payload.get("panel_metadata", {}).get(
                "iv_transform",
                getattr(args, "iv_transform", "log_level"),
            ),
        ),
    )
    iv_lower_bound = float(
        payload.get(
            "iv_lower_bound",
            payload.get("normalization", {}).get(
                "iv_lower_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_lower_bound",
                    getattr(args, "iv_lower_bound", 1e-4),
                ),
            ),
        )
    )
    iv_upper_bound = float(
        payload.get(
            "iv_upper_bound",
            payload.get("normalization", {}).get(
                "iv_upper_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_upper_bound",
                    getattr(args, "iv_upper_bound", 1.0),
                ),
            ),
        )
    )
    return {
        "positive_level_policy": positive_level_policy,
        "iv_transform": iv_transform,
        "iv_lower_bound": iv_lower_bound,
        "iv_upper_bound": iv_upper_bound,
    }


def train_raw_block(
    args: argparse.Namespace,
    payload: dict[str, Any],
    state_scope: str,
) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, float | int]]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    opts = payload_panel_options(args, payload)
    if args.clean_nonpositive_log_levels:
        panel, _cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
            positive_level_policy=opts["positive_level_policy"],
        )
    train_indices, _val_indices = official_train_val_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
    )
    if int(args.max_train_windows) > 0:
        train_indices = train_indices[-int(args.max_train_windows) :]
    block = build_increment_coordinate_block(
        panel,
        columns,
        train_indices,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        iv_count=int(args.iv_count),
        positive_level_policy=opts["positive_level_policy"],
        iv_transform=opts["iv_transform"],
        iv_lower_bound=opts["iv_lower_bound"],
        iv_upper_bound=opts["iv_upper_bound"],
    )
    raw_history, raw_future = select_raw_state_scope(block, state_scope, int(args.iv_count))
    if state_scope == "joint38":
        specs = block.specs
    elif state_scope == "iv_only":
        specs = block.specs[: int(args.iv_count)]
    elif state_scope == "anchor_only":
        specs = block.specs[int(args.iv_count) :]
    else:
        raise ValueError(f"unknown state_scope {state_scope!r}")
    alignment = state_block_alignment_diagnostics(panel, block, specs)
    return raw_history.astype(np.float32), raw_future.astype(np.float32), specs, alignment


def build_sticky_policy(
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    names: list[str],
    *,
    zero_eps: float,
    zero_rate_gate: float,
    nonzero_quantile: float,
) -> dict[str, Any]:
    history_delta = np.diff(raw_history, axis=1)
    future_delta = panel_daily_changes(raw_history, raw_future)
    all_delta = np.concatenate([history_delta, future_delta], axis=1)
    rows: list[dict[str, Any]] = []
    selected = np.zeros(len(names), dtype=bool)
    thresholds = np.zeros(len(names), dtype=np.float64)
    for idx, name in enumerate(names):
        delta = np.asarray(all_delta[..., idx], dtype=np.float64).reshape(-1)
        finite = delta[np.isfinite(delta)]
        zero = np.abs(finite) <= float(zero_eps)
        zero_rate = float(np.mean(zero)) if finite.size else 0.0
        nonzero_abs = np.abs(finite[~zero])
        threshold = 0.0
        if zero_rate >= float(zero_rate_gate) and nonzero_abs.size:
            threshold = float(np.quantile(nonzero_abs, float(nonzero_quantile)))
            selected[idx] = True
            thresholds[idx] = threshold
        rows.append(
            {
                "name": name,
                "selected": bool(selected[idx]),
                "zero_rate": zero_rate,
                "threshold": threshold,
                "nonzero_abs_q10": float(np.quantile(nonzero_abs, 0.10)) if nonzero_abs.size else 0.0,
                "nonzero_abs_q50": float(np.quantile(nonzero_abs, 0.50)) if nonzero_abs.size else 0.0,
                "nonzero_abs_q90": float(np.quantile(nonzero_abs, 0.90)) if nonzero_abs.size else 0.0,
            }
        )
    return {
        "zero_eps": float(zero_eps),
        "zero_rate_gate": float(zero_rate_gate),
        "nonzero_quantile": float(nonzero_quantile),
        "selected_mask": selected,
        "thresholds": thresholds.astype(np.float32),
        "rows": rows,
        "selected_names": [row["name"] for row in rows if row["selected"]],
    }


def apply_sticky_readout(samples_raw: np.ndarray, raw_history: np.ndarray, policy: dict[str, Any]) -> np.ndarray:
    out = np.asarray(samples_raw, dtype=np.float32).copy()
    selected = np.asarray(policy["selected_mask"], dtype=bool)
    thresholds = np.asarray(policy["thresholds"], dtype=np.float32)
    if not np.any(selected):
        return out
    for step in range(out.shape[2]):
        prev = raw_history[:, None, -1, :] if step == 0 else out[:, :, step - 1, :]
        delta = out[:, :, step, :] - prev
        snap = np.abs(delta) <= thresholds[None, None, :]
        snap &= selected[None, None, :]
        out[:, :, step, :] = np.where(snap, prev, out[:, :, step, :])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        choices=["609a", "625a", "628a", "629a", "638a", "641a", "647a", "652a", "658a", "661a", "662a"],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state_scope", choices=["iv_only", "anchor_only", "joint38"], default="joint38")
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
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--sticky_zero_eps", type=float, default=1e-10)
    parser.add_argument("--sticky_zero_rate_gate", type=float, default=0.25)
    parser.add_argument("--sticky_nonzero_quantile", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_native_model(args.model_type, args.checkpoint, device)
    history, _future, specs, block, alignment = build_history_future(args, payload)
    if alignment["history_max_abs_error"] > 1e-6 or alignment["future_max_abs_error"] > 1e-6:
        raise RuntimeError(f"validation alignment failed: {alignment}")
    state_scope = payload.get("state_scope", args.state_scope)
    raw_history_full, raw_future_full = select_raw_state_scope(block, state_scope, int(args.iv_count))
    history_n = int(history[0].shape[0] if isinstance(history, tuple) else history.shape[0])
    n_windows = min(int(args.max_windows), history_n)
    if isinstance(history, tuple):
        history = tuple(item[:n_windows] for item in history)
    else:
        history = history[:n_windows]
    raw_history = raw_history_full[:n_windows]
    raw_future = raw_future_full[:n_windows]

    train_history, train_future, train_specs, train_alignment = train_raw_block(args, payload, state_scope)
    expected_names = [spec.name for spec in specs]
    train_names = [spec.name for spec in train_specs]
    if expected_names != train_names:
        raise RuntimeError("train/validation sticky specs differ")
    policy = build_sticky_policy(
        train_history,
        train_future,
        expected_names,
        zero_eps=float(args.sticky_zero_eps),
        zero_rate_gate=float(args.sticky_zero_rate_gate),
        nonzero_quantile=float(args.sticky_nonzero_quantile),
    )
    if state_scope == "joint38":
        audit_iv_count = int(args.iv_count)
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
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        sample_temperature=float(args.sample_temperature),
        value_coordinate=payload.get("value_coordinate", args.value_coordinate),
        model_coordinate=payload.get("model_coordinate", "state"),
    )
    sticky_samples = apply_sticky_readout(samples_raw, raw_history, policy)
    summary = summarize_joint_quality(
        raw_history,
        raw_future,
        sticky_samples,
        factor_names,
        iv_count=int(audit_iv_count),
    )
    summary["sticky_zero_readout"] = {
        "selected_names": policy["selected_names"],
        "zero_rate_gate": policy["zero_rate_gate"],
        "nonzero_quantile": policy["nonzero_quantile"],
        "rows": policy["rows"],
    }
    result = {
        "summary": summary,
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
            "state_scope": state_scope,
            "value_coordinate": payload.get("value_coordinate", args.value_coordinate),
            "model_coordinate": payload.get("model_coordinate", "state"),
            "audit_iv_count": int(audit_iv_count),
            "n_windows": int(n_windows),
            "samples": int(args.samples),
            "n_steps": int(args.n_steps),
            "sample_temperature": float(args.sample_temperature),
            "generation_time_s": float(time.time() - t0),
            "seed": int(args.seed),
            "alignment": alignment,
            "train_alignment": train_alignment,
        },
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(result), indent=2), encoding="utf-8")
    write_markdown(out_md, "719a Sticky-Zero Readout Joint-Panel Audit", summary)
    with out_md.open("a", encoding="utf-8") as handle:
        handle.write("\n## Sticky-Zero Readout\n\n")
        handle.write(f"- selected names: `{policy['selected_names']}`\n")
        handle.write(f"- zero-rate gate: `{policy['zero_rate_gate']}`\n")
        handle.write(f"- nonzero quantile: `{policy['nonzero_quantile']}`\n")
        handle.write("\n| channel | selected | zero rate | threshold | nonzero q10 | nonzero q50 | nonzero q90 |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in policy["rows"]:
            if row["selected"] or row["zero_rate"] >= 0.10:
                handle.write(
                    f"| {row['name']} | `{row['selected']}` | `{row['zero_rate']:.3f}` | "
                    f"`{row['threshold']:.5g}` | `{row['nonzero_abs_q10']:.5g}` | "
                    f"`{row['nonzero_abs_q50']:.5g}` | `{row['nonzero_abs_q90']:.5g}` |\n"
                )
    print(json.dumps(make_serializable(result["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
