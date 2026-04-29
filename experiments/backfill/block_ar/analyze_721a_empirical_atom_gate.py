#!/usr/bin/env python
"""721a: empirical atom-gate diagnostic for sticky channels.

This diagnostic separates the no-change atom from the continuous sample without
training a new network. If this works, the principled next implementation is a
learned hurdle gate plus continuous nonzero innovation law. If it does not work,
the residual failure is in the nonzero movement distribution rather than only
the atom probability.
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
from experiments.backfill.block_ar.audit_719a_sticky_zero_readout import train_raw_block  # noqa: E402
from experiments.backfill.block_ar.analyze_720a_sticky_residual_sweep import (  # noqa: E402
    factor_zero_rows,
    focus_ks,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)


def sticky_mask_from_train(
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    *,
    zero_eps: float,
    zero_rate_gate: float,
) -> tuple[np.ndarray, np.ndarray]:
    history_delta = np.diff(raw_history, axis=1)
    future_delta = panel_daily_changes(raw_history, raw_future)
    all_delta = np.concatenate([history_delta, future_delta], axis=1)
    zero_rate = np.mean(np.abs(all_delta) <= float(zero_eps), axis=(0, 1))
    return zero_rate >= float(zero_rate_gate), zero_rate.astype(np.float32)


def history_zero_rate(raw_history: np.ndarray, *, zero_eps: float) -> np.ndarray:
    delta = np.diff(raw_history, axis=1)
    return np.mean(np.abs(delta) <= float(zero_eps), axis=1).astype(np.float32)


def future_zero_rate(raw_history: np.ndarray, raw_future: np.ndarray, *, zero_eps: float) -> np.ndarray:
    delta = panel_daily_changes(raw_history, raw_future)
    return np.mean(np.abs(delta) <= float(zero_eps), axis=1).astype(np.float32)


def global_atom_probability(
    train_history: np.ndarray,
    train_future: np.ndarray,
    selected: np.ndarray,
    *,
    zero_eps: float,
) -> np.ndarray:
    p = np.zeros(train_history.shape[-1], dtype=np.float32)
    fz = future_zero_rate(train_history, train_future, zero_eps=zero_eps)
    p[selected] = np.mean(fz[:, selected], axis=0)
    return np.clip(p, 0.0, 0.98)


def history_bin_atom_probability(
    train_history: np.ndarray,
    train_future: np.ndarray,
    val_history: np.ndarray,
    selected: np.ndarray,
    *,
    zero_eps: float,
    n_bins: int,
) -> np.ndarray:
    train_hz = history_zero_rate(train_history, zero_eps=zero_eps)
    train_fz = future_zero_rate(train_history, train_future, zero_eps=zero_eps)
    val_hz = history_zero_rate(val_history, zero_eps=zero_eps)
    out = np.zeros((val_history.shape[0], val_history.shape[-1]), dtype=np.float32)
    global_p = np.mean(train_fz, axis=0)
    for ch in np.flatnonzero(selected):
        x = train_hz[:, ch]
        y = train_fz[:, ch]
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, int(n_bins) + 1)))
        if edges.size <= 2:
            out[:, ch] = global_p[ch]
            continue
        assigned = np.full(val_history.shape[0], np.nan, dtype=np.float32)
        for lo, hi in zip(edges[:-1], edges[1:]):
            train_keep = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
            val_keep = (val_hz[:, ch] >= lo) & (val_hz[:, ch] <= hi if hi == edges[-1] else val_hz[:, ch] < hi)
            if np.any(train_keep):
                assigned[val_keep] = float(np.mean(y[train_keep]))
        assigned[~np.isfinite(assigned)] = global_p[ch]
        out[:, ch] = assigned
    return np.clip(out, 0.0, 0.98)


def apply_atom_gate(
    samples_raw: np.ndarray,
    raw_history: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.asarray(samples_raw, dtype=np.float32).copy()
    if probabilities.ndim == 1:
        p = np.repeat(probabilities[None, :], out.shape[0], axis=0)
    else:
        p = probabilities
    p = np.asarray(p, dtype=np.float32)
    p = np.where(selected[None, :], p, 0.0)
    for step in range(out.shape[2]):
        prev = raw_history[:, None, -1, :] if step == 0 else out[:, :, step - 1, :]
        gate = rng.random((out.shape[0], out.shape[1], out.shape[-1])) < p[:, None, :]
        out[:, :, step, :] = np.where(gate, prev, out[:, :, step, :])
    return out


def summarize_variant(
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    samples: np.ndarray,
    factor_names: list[str],
    *,
    iv_count: int,
) -> dict[str, Any]:
    summary = summarize_joint_quality(raw_history, raw_future, samples, factor_names, iv_count=iv_count)
    return {
        "factor_delta_ks_mean": summary["factor_delta_ks_mean"],
        "factor_delta_ks_pass_020": summary["factor_delta_ks_pass_020"],
        "factor_factor_abs_ratio": summary["factor_factor_corr"]["gen_mean_abs"]
        / max(summary["factor_factor_corr"]["gt_mean_abs"], 1e-12),
        "iv_factor_abs_ratio": summary["iv_factor_corr"]["gen_mean_abs"]
        / max(summary["iv_factor_corr"]["gt_mean_abs"], 1e-12),
        "conditional_reduction": summary["conditional_panel"]["median_mae_reduction_vs_rolled_pct"],
        "focus_ks": focus_ks(summary),
        "focus_zero": factor_zero_rows(raw_history, raw_future, samples, factor_names, iv_count=iv_count),
    }


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
    names = [spec.name for spec in specs]
    if names != [spec.name for spec in train_specs]:
        raise RuntimeError("train/validation specs differ")
    selected, train_zero_rate = sticky_mask_from_train(
        train_history,
        train_future,
        zero_eps=float(scope_args.atom_zero_eps),
        zero_rate_gate=float(scope_args.atom_zero_rate_gate),
    )
    if state_scope == "joint38":
        audit_iv_count = int(scope_args.iv_count)
        factor_names = [spec.name for spec in specs[audit_iv_count:]]
    else:
        audit_iv_count = 0
        factor_names = names

    t0 = time.time()
    samples = generate_panel_samples(
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
    global_p = global_atom_probability(
        train_history,
        train_future,
        selected,
        zero_eps=float(scope_args.atom_zero_eps),
    )
    histbin_p = history_bin_atom_probability(
        train_history,
        train_future,
        raw_history,
        selected,
        zero_eps=float(scope_args.atom_zero_eps),
        n_bins=int(scope_args.atom_bins),
    )
    variants = {
        "identity": summarize_variant(raw_history, raw_future, samples, factor_names, iv_count=audit_iv_count),
        "global_atom": summarize_variant(
            raw_history,
            raw_future,
            apply_atom_gate(samples, raw_history, global_p, selected, seed=int(seed) + 1000),
            factor_names,
            iv_count=audit_iv_count,
        ),
        "history_bin_atom": summarize_variant(
            raw_history,
            raw_future,
            apply_atom_gate(samples, raw_history, histbin_p, selected, seed=int(seed) + 2000),
            factor_names,
            iv_count=audit_iv_count,
        ),
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
        "selected_names": [name for name, flag in zip(names, selected) if flag],
        "train_zero_rate": {name: float(rate) for name, rate in zip(names, train_zero_rate) if selected[names.index(name)]},
        "global_atom_probability": {name: float(global_p[idx]) for idx, name in enumerate(names) if selected[idx]},
        "history_bin_probability_mean": {
            name: float(np.mean(histbin_p[:, idx])) for idx, name in enumerate(names) if selected[idx]
        },
        "variants": variants,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 721a Empirical Atom-Gate Diagnostic",
        "",
        "| scope | variant | pass | mean KS | AAA KS | BBB KS | AAA zero gen/gt | BBB zero gen/gt | factor corr abs ratio | conditional |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope_name, scope in report["scopes"].items():
        for variant, s in scope["variants"].items():
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
    lines.extend(["", "## Selected Atom Channels", ""])
    for scope_name, scope in report["scopes"].items():
        lines.append(f"- `{scope_name}` selected: `{scope['selected_names']}`")
        lines.append(f"- `{scope_name}` global p: `{scope['global_atom_probability']}`")
        lines.append(f"- `{scope_name}` history-bin mean p: `{scope['history_bin_probability_mean']}`")
    lines.extend(["", "## Decision", "", report["decision"], ""])
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
    parser.add_argument("--atom_zero_eps", type=float, default=1e-10)
    parser.add_argument("--atom_zero_rate_gate", type=float, default=0.25)
    parser.add_argument("--atom_bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    scopes = {
        "anchor": run_scope(args, "anchor_only", args.anchor_checkpoint, int(args.seed) + 1),
        "joint": run_scope(args, "joint38", args.joint_checkpoint, int(args.seed) + 2),
    }
    decision = (
        "If empirical atom gates pass BBB while preserving correlation, implement a "
        "learned hurdle/sticky gate. If they do not, the nonzero continuous path "
        "distribution must be repaired before adding a gate to the production model."
    )
    report = {
        "iteration": "721a",
        "purpose": "empirical hurdle atom-gate feasibility diagnostic",
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
