from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TAXONOMY = Path("results/world/scale_baseline_target_taxonomy_head134.json")
DEFAULT_REGIME = Path("results/world/scale_regime_probe_gap_head133.json")
DEFAULT_EXACT_STATE = Path("results/world/scale_exact_state_gap_head132.json")


FEATURE_SURFACES = ["raw_only", "learned_only", "raw_plus_learned"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_from(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(f"Missing one of {names}")


def _path_shape_layer(taxonomy: dict[str, Any]) -> dict[str, Any]:
    family = taxonomy["family_summaries"]["path_shape_or_risk_width"]
    n_targets = int(family["n_targets"])
    wins = int(family["barlow_best_raw_wins"])
    adds = int(family["barlow_adds_to_raw_last"])
    status = "PASS" if wins == n_targets and adds == n_targets else "PARTIAL"
    if wins == 0 and adds == 0:
        status = "FAIL"
    return {
        "status": status,
        "n_targets": n_targets,
        "learned_only_wins": wins,
        "raw_plus_learned_improvements": adds,
    }


def _persistence_layer(taxonomy: dict[str, Any]) -> dict[str, Any]:
    family = taxonomy["family_summaries"]["persistence_or_exact_state_dominated"]
    n_targets = int(family["n_targets"])
    wins = int(family["barlow_best_raw_wins"])
    adds = int(family["barlow_adds_to_raw_last"])
    status = "PASS" if adds == n_targets else "PARTIAL" if adds > 0 else "FAIL"
    return {
        "status": status,
        "n_targets": n_targets,
        "learned_only_wins": wins,
        "raw_plus_learned_improvements": adds,
    }


def _exact_state_guardrail(exact_state: dict[str, Any]) -> dict[str, Any]:
    gap = exact_state["iv_surface_gap"]
    raw_mse = _float_from(gap, "raw_surface_mse", "raw_surface_iv_mse")
    learned_mse = _float_from(gap, "scale_barlow_mse", "scale_barlow_iv_mse")
    ratio = _float_from(gap, "scale_to_raw_ratio", "scale_to_raw_iv_mse_ratio")
    gap_confirmed = bool(exact_state["decision"]["exact_iv_retention_gap_confirmed"])
    return {
        "status": "FAIL" if gap_confirmed else "PASS",
        "raw_only_iv_mse": raw_mse,
        "learned_only_iv_mse": learned_mse,
        "learned_to_raw_ratio": ratio,
    }


def _regime_layer(regime: dict[str, Any]) -> dict[str, Any]:
    decision = regime["decision"]
    accuracy_gate = bool(decision["accuracy_gate_passed"])
    balanced_signal = bool(decision["balanced_signal_present"])
    status = "PASS" if accuracy_gate and balanced_signal else "PARTIAL"
    if not accuracy_gate and not balanced_signal:
        status = "FAIL"
    deltas = regime.get("scaled_barlow_vs_raw_last", {})
    return {
        "status": status,
        "accuracy_gate_passed": accuracy_gate,
        "balanced_signal_present": balanced_signal,
        "macro_recall_delta": float(deltas.get("macro_recall_delta", 0.0)),
        "accuracy_delta": float(deltas.get("accuracy_delta", 0.0)),
    }


def summarize_additive_signal_gate(
    *,
    taxonomy: dict[str, Any],
    regime: dict[str, Any],
    exact_state: dict[str, Any],
) -> dict[str, Any]:
    gate_layers = {
        "exact_state_guardrail": _exact_state_guardrail(exact_state),
        "path_shape_risk_width": _path_shape_layer(taxonomy),
        "persistence_guardrail": _persistence_layer(taxonomy),
        "regime_balanced_signal": _regime_layer(regime),
    }
    additive_signal_present = (
        gate_layers["path_shape_risk_width"]["status"] == "PASS"
        or gate_layers["regime_balanced_signal"]["balanced_signal_present"]
    )
    gate_passed = (
        gate_layers["exact_state_guardrail"]["status"] != "FAIL"
        and gate_layers["path_shape_risk_width"]["status"] == "PASS"
    )
    return {
        "analysis": "world_model_additive_signal_gate",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_additive_signal_gate",
        "feature_surfaces": FEATURE_SURFACES,
        "sources": {
            "target_taxonomy": str(DEFAULT_TAXONOMY),
            "regime_gap": str(DEFAULT_REGIME),
            "exact_state_gap": str(DEFAULT_EXACT_STATE),
        },
        "gate_layers": gate_layers,
        "decision": {
            "additive_signal_present": additive_signal_present,
            "gate_passed": gate_passed,
            "promotion_decision": "DO_NOT_PROMOTE",
            "part_b_blocked": True,
            "interpretation": (
                "Existing frozen probes show additive path-shape/risk-width and "
                "balanced-regime signal, but exact-state and persistence guardrails "
                "still block Part 1 promotion."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`post_experiment_analysis`",
        "",
        "## Objective Family",
        "",
        "`downstream_probe_additive_signal_gate`; frozen probes only.",
        "",
        "## Hypothesis",
        "",
        "If the learned Part 1 embedding is useful as abstract market-state",
        "information, it should add value to explicit raw state on path-shape,",
        "risk-width, or balanced state probes without being treated as a",
        "replacement for exact raw conditioning.",
        "",
        "## Falsifier",
        "",
        "The additive framing would fail if raw-plus-learned features add no",
        "value on the abstract target families, or if exact-state guardrails",
        "are treated as solved by a learned-only embedding that still loses to",
        "raw current state.",
        "",
        "## Feature Surfaces",
        "",
        "- `raw_only`: explicit current-state floor.",
        "- `learned_only`: frozen scaled Barlow embedding.",
        "- `raw_plus_learned`: explicit raw state plus frozen embedding.",
        "",
        "## Gate Layers",
        "",
        "| layer | status | key evidence |",
        "| --- | --- | --- |",
    ]
    layers = result["gate_layers"]
    exact = layers["exact_state_guardrail"]
    lines.append(
        "| exact-state guardrail | {status} | learned/raw IV MSE ratio `{ratio}` |".format(
            status=exact["status"],
            ratio=_fmt(exact["learned_to_raw_ratio"]),
        )
    )
    path = layers["path_shape_risk_width"]
    lines.append(
        "| path-shape/risk-width | {status} | learned wins `{wins}/{n}` and raw+learned improves `{adds}/{n}` |".format(
            status=path["status"],
            wins=path["learned_only_wins"],
            adds=path["raw_plus_learned_improvements"],
            n=path["n_targets"],
        )
    )
    persistence = layers["persistence_guardrail"]
    lines.append(
        "| persistence guardrail | {status} | learned wins `{wins}/{n}` and raw+learned improves `{adds}/{n}` |".format(
            status=persistence["status"],
            wins=persistence["learned_only_wins"],
            adds=persistence["raw_plus_learned_improvements"],
            n=persistence["n_targets"],
        )
    )
    regime = layers["regime_balanced_signal"]
    lines.append(
        "| regime balanced signal | {status} | macro-recall delta `{macro}`, accuracy delta `{acc}` |".format(
            status=regime["status"],
            macro=_fmt(regime["macro_recall_delta"]),
            acc=_fmt(regime["accuracy_delta"]),
        )
    )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Additive signal present: `{decision['additive_signal_present']}`.",
            f"- Gate passed: `{decision['gate_passed']}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            f"- Part B blocked: `{decision['part_b_blocked']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the additive-signal gate for frozen Part 1 features"
    )
    parser.add_argument("--taxonomy-json", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--regime-json", type=Path, default=DEFAULT_REGIME)
    parser.add_argument("--exact-state-json", type=Path, default=DEFAULT_EXACT_STATE)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/additive_signal_gate_head167.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head167_additive_signal_gate_audit.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD167: Additive-Signal Gate Audit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = summarize_additive_signal_gate(
        taxonomy=_load_json(args.taxonomy_json),
        regime=_load_json(args.regime_json),
        exact_state=_load_json(args.exact_state_json),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(
        render_markdown(result, title=args.report_title),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
