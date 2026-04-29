#!/usr/bin/env python
"""725a: attribute sticky-channel failures to nonzero-update scale.

This post-experiment analysis uses train/validation raw panels plus the saved
721a/724a atom-gate diagnostics. It does not train or sample; it asks whether
the sticky OAS residual is mostly no-update mass, nonzero update scale, or
train/validation distribution shift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    build_history_future,
    load_native_model,
    panel_daily_changes,
    select_raw_state_scope,
)
from experiments.backfill.block_ar.audit_719a_sticky_zero_readout import train_raw_block  # noqa: E402


FOCUS_NAMES = [
    "factor:spx",
    "factor:us2y",
    "factor:us10y",
    "factor:aaa_oas",
    "factor:bbb_oas",
]


def split_delta_stats(
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    names: list[str],
    *,
    zero_eps: float,
) -> dict[str, dict[str, float]]:
    delta = panel_daily_changes(raw_history, raw_future)
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        values = np.asarray(delta[..., idx], dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        zero = np.abs(values) <= float(zero_eps)
        nonzero_abs = np.abs(values[~zero])
        out[name] = {
            "zero_rate": float(np.mean(zero)),
            "signed_mean": float(np.mean(values)),
            "abs_q50": float(np.quantile(nonzero_abs, 0.50)) if nonzero_abs.size else 0.0,
            "abs_q90": float(np.quantile(nonzero_abs, 0.90)) if nonzero_abs.size else 0.0,
            "abs_q95": float(np.quantile(nonzero_abs, 0.95)) if nonzero_abs.size else 0.0,
            "abs_q99": float(np.quantile(nonzero_abs, 0.99)) if nonzero_abs.size else 0.0,
        }
    return out


def extract_generated_focus(
    path: Path,
    *,
    label: str,
    variant: str,
    scope: str,
    val_stats: dict[str, dict[str, float]],
    train_stats: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    focus = report["scopes"][scope]["variants"][variant]["focus_zero"]
    rows: list[dict[str, Any]] = []
    for name, stats in focus.items():
        val = val_stats[name]
        train = train_stats[name]
        gen_q99 = float(stats["gen_abs_q99"])
        rows.append(
            {
                "source": label,
                "scope": scope,
                "variant": variant,
                "name": name,
                "ks": float(report["scopes"][scope]["variants"][variant]["focus_ks"][name]),
                "gt_zero_rate": float(stats["gt_zero_rate"]),
                "gen_zero_rate": float(stats["gen_zero_rate"]),
                "gt_abs_q99": float(stats["gt_abs_q99"]),
                "gen_abs_q99": gen_q99,
                "gen_to_val_q99": gen_q99 / max(float(val["abs_q99"]), 1e-12),
                "gen_to_train_q99": gen_q99 / max(float(train["abs_q99"]), 1e-12),
                "gen_nonzero_abs_q50": float(stats["gen_nonzero_abs_q50"]),
                "val_nonzero_abs_q50": float(val["abs_q50"]),
                "train_nonzero_abs_q50": float(train["abs_q50"]),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| source | scope | variant | name | KS | zero gen/val | q99 gen/val/train | gen/val q99 | gen/train q99 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{source}` | `{scope}` | `{variant}` | `{name}` | {ks:.3f} | {gen_zero_rate:.3f}/{gt_zero_rate:.3f} | {gen_abs_q99:.3f}/{gt_abs_q99:.3f}/{train_q99:.3f} | {gen_to_val_q99:.2f} | {gen_to_train_q99:.2f} |".format(
                train_q99=row["train_abs_q99"],
                **row,
            )
        )
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 725a Nonzero-Update Scale Attribution",
        "",
        "## Split Diagnostics",
        "",
        "| name | train zero | val zero | train q99 | val q99 | val/train q99 | train mean | val mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in FOCUS_NAMES:
        if name not in report["train_stats"]:
            continue
        train = report["train_stats"][name]
        val = report["val_stats"][name]
        lines.append(
            f"| `{name}` | {train['zero_rate']:.3f} | {val['zero_rate']:.3f} | "
            f"{train['abs_q99']:.3f} | {val['abs_q99']:.3f} | "
            f"{val['abs_q99'] / max(train['abs_q99'], 1e-12):.2f} | "
            f"{train['signed_mean']:.4f} | {val['signed_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Generated OAS Attribution",
            "",
            *markdown_table(report["generated_rows"]),
            "",
            "## Mechanism Read",
            "",
            report["mechanism_read"],
            "",
            "## Decision",
            "",
            report["decision"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/724a_anchor_sticky_obs_nonzero_mask_e8_w2048_s7242/best_model.pt")
    parser.add_argument("--model_type", default="662a")
    parser.add_argument("--state_scope", choices=["anchor_only"], default="anchor_only")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--value_coordinate", choices=["raw", "encoded"], default="raw")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--zero_eps", type=float, default=1e-10)
    parser.add_argument("--analysis_721", default="results/block_ar/721a_empirical_atom_gate/analysis.json")
    parser.add_argument("--analysis_724", default="results/block_ar/724a_sticky_observation_nonzero_mask/atom_gate_analysis.json")
    parser.add_argument("--variant", default="history_bin_atom")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    _model, payload = load_native_model(args.model_type, args.checkpoint, device)
    history, _future, specs, block, alignment = build_history_future(args, payload)
    n_windows = min(int(args.max_windows), int(history[0].shape[0] if isinstance(history, tuple) else history.shape[0]))
    raw_history, raw_future = select_raw_state_scope(block, "anchor_only", int(args.iv_count))
    raw_history = raw_history[:n_windows]
    raw_future = raw_future[:n_windows]
    train_history, train_future, train_specs, train_alignment = train_raw_block(args, payload, "anchor_only")
    names = [spec.name for spec in train_specs]
    if names != [spec.name for spec in specs]:
        raise RuntimeError("anchor validation specs do not match train specs")

    train_stats = split_delta_stats(train_history, train_future, names, zero_eps=float(args.zero_eps))
    val_stats = split_delta_stats(raw_history, raw_future, names, zero_eps=float(args.zero_eps))
    generated_rows: list[dict[str, Any]] = []
    for source, path in [("721", Path(args.analysis_721)), ("724", Path(args.analysis_724))]:
        for scope in ["anchor", "joint"]:
            rows = extract_generated_focus(
                path,
                label=source,
                variant=args.variant,
                scope=scope,
                val_stats=val_stats,
                train_stats=train_stats,
            )
            for row in rows:
                row["train_abs_q99"] = train_stats[row["name"]]["abs_q99"]
                generated_rows.append(row)

    aaa_train = train_stats["factor:aaa_oas"]["abs_q99"]
    aaa_val = val_stats["factor:aaa_oas"]["abs_q99"]
    bbb_train = train_stats["factor:bbb_oas"]["abs_q99"]
    bbb_val = val_stats["factor:bbb_oas"]["abs_q99"]
    mechanism_read = (
        "Validation OAS is a calmer regime than the 2048-window training tail: "
        f"AAA nonzero q99 is {aaa_val:.3f} versus train {aaa_train:.3f}, and "
        f"BBB nonzero q99 is {bbb_val:.3f} versus train {bbb_train:.3f}. "
        "The 724 masked-zero loss removes the damping effect of exact-zero targets, so the continuous anchor-only law learns too much of the broader train-tail scale. "
        "721 stayed closer to validation tails because the continuous loss still mixed zeros and nonzeros, but that is an accidental dampener rather than a clean observation model."
    )
    decision = (
        "Do not add another atom-probability knob. The next principled repair should target nonzero-update scale allocation under regime shift, preferably through a shared, history-conditioned scale or frequency-balanced objective that is deterministic from data statistics and applies to IV-only, anchor-only, and joint without scope-specific recipes."
    )
    report = {
        "iteration": "725a",
        "purpose": "nonzero OAS update scale attribution after 724a",
        "alignment": alignment,
        "train_alignment": train_alignment,
        "focus_names": FOCUS_NAMES,
        "train_stats": {name: train_stats[name] for name in FOCUS_NAMES if name in train_stats},
        "val_stats": {name: val_stats[name] for name in FOCUS_NAMES if name in val_stats},
        "generated_rows": generated_rows,
        "mechanism_read": mechanism_read,
        "decision": decision,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(output_md, report)
    print(json.dumps(make_serializable({"mechanism_read": mechanism_read, "decision": decision}), indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
