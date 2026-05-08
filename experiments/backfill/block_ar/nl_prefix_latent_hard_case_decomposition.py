"""Decompose narrative prefix-latent hard-case warnings.

This diagnostic is aimed at the product validation question:

When a narrative-conditioned rollout warns, should the system accept the run
with a visible warning, rerank the starting state, or ask the risk manager for
an explicit starting state?

It consumes existing story-smoke reports and arrays; no OpenAI calls or model
reruns are made.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_START_POLICY_AUDIT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_policy_audit_810a_condition_only/"
    "start_policy_audit_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_hard_case_decomposition_811a_defensive"
)
IV_MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
IV_MONEYNESS_LABELS = ["0.70", "0.85", "1.00", "1.15", "1.30"]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def operational_case(report: dict[str, Any]) -> dict[str, Any]:
    for row in report.get("validation_gate", {}).get("cases", []):
        if bool(row.get("is_operational", False)):
            return row
    return {}


def diagnostic_original_case(report: dict[str, Any]) -> dict[str, Any]:
    for row in report.get("validation_gate", {}).get("cases", []):
        if str(row.get("variant")) == "original":
            return row
    return {}


def support_alignment(report: dict[str, Any]) -> dict[str, Any]:
    prior = report.get("cached_query", {}).get("memory_prior", {})
    alignment = prior.get("support_alignment", {})
    return alignment if isinstance(alignment, dict) else {}


def _status_from_thresholds(
    value: float,
    *,
    warn: float,
    fail: float,
    lower_is_bad: bool = False,
) -> str:
    if lower_is_bad:
        if value < fail:
            return "fail"
        if value < warn:
            return "warning"
        return "pass"
    if value > fail:
        return "fail"
    if value > warn:
        return "warning"
    return "pass"


def factor_labels_from_checkpoint(checkpoint_path: str | Path | None, dim: int) -> list[str]:
    names: list[str] = []
    if checkpoint_path:
        try:
            import torch

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            specs = checkpoint.get("state_specs", [])
            for spec in specs:
                if isinstance(spec, dict):
                    names.append(str(spec.get("name", f"factor_{len(names):02d}")))
                else:
                    names.append(str(getattr(spec, "name", spec)))
        except Exception:
            names = []
    if len(names) < dim:
        names.extend(f"factor_{idx:02d}" for idx in range(len(names), dim))
    return [display_factor_name(idx, name) for idx, name in enumerate(names[:dim])]


def display_factor_name(index: int, raw_name: str) -> str:
    if raw_name.startswith("iv:"):
        try:
            iv_index = int(raw_name.split(":", 1)[1])
        except ValueError:
            return raw_name
        row, col = divmod(iv_index, len(IV_MONEYNESS_LABELS))
        if row < len(IV_MATURITY_LABELS):
            return f"IV {IV_MATURITY_LABELS[row]} K={IV_MONEYNESS_LABELS[col]}"
        return raw_name
    if raw_name.startswith("factor:"):
        return raw_name.split(":", 1)[1].upper()
    return raw_name or f"factor_{index:02d}"


def top_rollout_shift_factors(
    *,
    arrays_path: Path,
    report: dict[str, Any],
    labels: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    with np.load(arrays_path) as payload:
        samples = np.asarray(payload["samples"], dtype=np.float32)
        scale = np.maximum(np.asarray(payload["delta_scale"], dtype=np.float32), 1e-8)
    if samples.ndim != 4 or samples.shape[0] < 2:
        return []
    cases = report.get("validation_gate", {}).get("cases", [])
    selected = operational_case(report)
    original = diagnostic_original_case(report)
    if not selected or not original:
        return []
    selected_idx = int(selected.get("case_index", 1))
    original_idx = int(original.get("case_index", 0))
    if selected_idx >= samples.shape[0] or original_idx >= samples.shape[0]:
        return []
    sample_z = samples / scale[None, None, :, :]
    diff = sample_z[selected_idx].mean(axis=0) - sample_z[original_idx].mean(axis=0)
    terminal = np.abs(diff[-1])
    path = np.mean(np.abs(diff), axis=0)
    order = np.argsort(-terminal)[: int(top_k)]
    rows = []
    for idx in order:
        label = labels[int(idx)] if int(idx) < len(labels) else f"factor_{idx:02d}"
        rows.append(
            {
                "factor_index": int(idx),
                "factor": label,
                "terminal_abs_shift_z": float(terminal[int(idx)]),
                "mean_path_abs_shift_z": float(path[int(idx)]),
                "signed_terminal_shift_z": float(diff[-1, int(idx)]),
            }
        )
    return rows


def decompose_report(report_path: Path, *, top_factors: int) -> dict[str, Any]:
    report = load_json(report_path)
    selected = operational_case(report)
    original = diagnostic_original_case(report)
    gate = report.get("validation_gate", {})
    thresholds = gate.get("thresholds", {})
    alignment = support_alignment(report)
    artifact_paths = report.get("artifact_paths", {})
    arrays_path = Path(artifact_paths.get("arrays", report_path.with_suffix(".npz")))
    checkpoint = report.get("artifact_inputs", {}).get("checkpoint")
    labels = factor_labels_from_checkpoint(checkpoint, dim=39)

    memory_cosine = float(selected.get("input_memory_cosine", 0.0))
    start_distance = float(selected.get("start_distance_z", 0.0))
    terminal_shift = float(selected.get("terminal_mean_abs_delta_z", 0.0))
    mean_shift = float(selected.get("mean_abs_delta_z", 0.0))
    endpoint_error = float(gate.get("endpoint_max_abs_error", 0.0))
    warnings = [str(item) for item in selected.get("warnings", [])]
    failures = [str(item) for item in selected.get("failures", [])]

    components = {
        "support_prior": {
            "status": str(alignment.get("status", "unknown")),
            "checked_count": int(alignment.get("checked_count", 0) or 0),
            "mismatch_count": int(alignment.get("mismatch_count", 0) or 0),
        },
        "memory_compatibility": {
            "status": _status_from_thresholds(
                memory_cosine,
                warn=float(thresholds.get("memory_cosine_warn", 0.80)),
                fail=float(thresholds.get("memory_cosine_fail", 0.65)),
                lower_is_bad=True,
            ),
            "input_memory_cosine": memory_cosine,
        },
        "start_distance": {
            "status": _status_from_thresholds(
                start_distance,
                warn=float(thresholds.get("start_distance_warn", 15.0)),
                fail=float(thresholds.get("start_distance_fail", 32.0)),
            ),
            "start_distance_z": start_distance,
        },
        "decoder_endpoint": {
            "status": _status_from_thresholds(
                endpoint_error,
                warn=float(thresholds.get("endpoint_abs_fail", 1e-6)),
                fail=float(thresholds.get("endpoint_abs_fail", 1e-6)),
            ),
            "endpoint_max_abs_error": endpoint_error,
        },
        "rollout_shift": {
            "status": _status_from_thresholds(
                max(mean_shift, terminal_shift),
                warn=float(thresholds.get("rollout_shift_warn", 1.0)),
                fail=float(thresholds.get("rollout_shift_fail", 2.0)),
            ),
            "mean_abs_delta_z": mean_shift,
            "terminal_mean_abs_delta_z": terminal_shift,
        },
    }
    return {
        "start_mode": str(selected.get("variant", "")),
        "selected_status": str(selected.get("status", "unknown")),
        "selected_warnings": warnings,
        "selected_failures": failures,
        "selected_start_window_index": selected.get("start_window_index"),
        "diagnostic_original_status": str(original.get("status", "unknown")),
        "diagnostic_original_warnings": list(original.get("warnings", [])),
        "overall_status": str(gate.get("overall_status", "unknown")),
        "operational_status": str(gate.get("operational_status", "unknown")),
        "report_path": str(report_path),
        "arrays_path": str(arrays_path),
        "components": components,
        "top_rollout_shift_factors": top_rollout_shift_factors(
            arrays_path=arrays_path,
            report=report,
            labels=labels,
            top_k=top_factors,
        ),
    }


def hard_case_rows(summary: dict[str, Any], case_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in summary.get("rows", [])
        if str(row.get("case_name")) == case_name
    ]


def production_decision(decompositions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decompositions:
        return {"decision": "no_case_rows", "reason": "No matching case rows found."}
    failures = [
        item
        for item in decompositions
        if item.get("selected_failures") or item.get("selected_status") == "fail"
    ]
    if failures:
        return {
            "decision": "reject_or_require_explicit_start",
            "reason": "At least one audited policy produced a hard validation failure.",
        }
    viable = [
        item
        for item in decompositions
        if item["components"]["support_prior"]["mismatch_count"] == 0
        and item["components"]["support_prior"]["status"] == "pass"
        and item["components"]["memory_compatibility"]["status"] == "pass"
        and item["components"]["start_distance"]["status"] == "pass"
    ]
    rejected = [
        {
            "start_mode": item["start_mode"],
            "support_status": item["components"]["support_prior"]["status"],
            "memory_status": item["components"]["memory_compatibility"]["status"],
            "start_status": item["components"]["start_distance"]["status"],
            "warnings": item.get("selected_warnings", []),
        }
        for item in decompositions
        if item not in viable
    ]
    warning_counts = Counter()
    for item in viable:
        warning_counts.update(item.get("selected_warnings", []))
    if viable and not warning_counts:
        return {
            "decision": "accept_for_narrative_only",
            "reason": (
                "At least one audited policy has clean support, memory, start "
                "distance, and rollout-shift gates."
            ),
            "viable_start_modes": [item["start_mode"] for item in viable],
            "rejected_start_modes": rejected,
            "ui_guidance": (
                "The narrative condition is supported by the analogue pool and "
                "the selected start passes the current validation gates."
            ),
        }
    only_rollout_warning = set(warning_counts) <= {"large_rollout_shift"}
    if viable and only_rollout_warning:
        return {
            "decision": "warn_and_continue_for_narrative_only",
            "reason": (
                "At least one audited policy has clean support, memory, and start "
                "distance. Among viable policies, the remaining warning is rollout "
                "sensitivity, so the demo should continue while making the warning "
                "visible."
            ),
            "viable_start_modes": [item["start_mode"] for item in viable],
            "rejected_start_modes": rejected,
            "ui_guidance": (
                "This stress narrative is supported by the analogue pool, but the "
                "model-chosen start is rollout-sensitive. For production, show the "
                "warning and let the risk manager supply today's starting state."
            ),
        }
    return {
        "decision": "needs_rerank_or_user_start",
        "reason": (
            "No audited policy has clean support, memory, and start distance with "
            "warnings isolated to rollout sensitivity."
        ),
        "rejected_start_modes": rejected,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hard-Case Decomposition",
        "",
        f"Case: `{summary['case_name']}`",
        f"Decision: `{summary['production_decision']['decision']}`",
        "",
        summary["production_decision"].get("reason", ""),
        "",
        "## Policy Rows",
        "",
    ]
    for item in summary["decompositions"]:
        rollout = item["components"]["rollout_shift"]
        start = item["components"]["start_distance"]
        memory = item["components"]["memory_compatibility"]
        support = item["components"]["support_prior"]
        lines.append(
            "- `{mode}`: status `{status}`, warnings `{warnings}`, "
            "start `{start_status}` z `{start_z:.3f}`, memory `{memory_status}` "
            "cos `{cosine:.3f}`, support `{support_status}` mismatches "
            "`{mismatch}`, rollout `{rollout_status}` terminal z `{terminal:.3f}`.".format(
                mode=item["start_mode"],
                status=item["selected_status"],
                warnings=", ".join(item["selected_warnings"]) or "none",
                start_status=start["status"],
                start_z=float(start["start_distance_z"]),
                memory_status=memory["status"],
                cosine=float(memory["input_memory_cosine"]),
                support_status=support["status"],
                mismatch=int(support["mismatch_count"]),
                rollout_status=rollout["status"],
                terminal=float(rollout["terminal_mean_abs_delta_z"]),
            )
        )
    lines.extend(["", "## Largest Rollout-Shift Factors", ""])
    for item in summary["decompositions"]:
        lines.append(f"### `{item['start_mode']}`")
        for factor in item["top_rollout_shift_factors"]:
            lines.append(
                "- {factor}: terminal abs z `{terminal:.3f}`, signed terminal z "
                "`{signed:.3f}`, path abs z `{path:.3f}`.".format(
                    factor=factor["factor"],
                    terminal=float(factor["terminal_abs_shift_z"]),
                    signed=float(factor["signed_terminal_shift_z"]),
                    path=float(factor["mean_path_abs_shift_z"]),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-policy-summary", default=str(DEFAULT_START_POLICY_AUDIT))
    parser.add_argument("--case-name", default="defensive_risk_off")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-factors", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source = load_json(args.start_policy_summary)
    rows = hard_case_rows(source, str(args.case_name))
    decompositions = [
        decompose_report(Path(row["report_path"]), top_factors=int(args.top_factors))
        for row in rows
    ]
    summary = {
        "status": "ok",
        "case_name": str(args.case_name),
        "source_summary": str(args.start_policy_summary),
        "policy_count": len(decompositions),
        "production_decision": production_decision(decompositions),
        "decompositions": decompositions,
        "artifact_paths": {
            "summary_json": str(Path(args.output_dir) / "hard_case_decomposition.json"),
            "summary_markdown": str(
                Path(args.output_dir) / "hard_case_decomposition.md"
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hard_case_decomposition.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, output_dir / "hard_case_decomposition.md")
    print(json.dumps(summary["production_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
