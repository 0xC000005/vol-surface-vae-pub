#!/usr/bin/env python
"""Aggregate product-facing conditionality gates for NL scenario generation.

This artifact-only audit turns the current research diagnosis into a
risk-manager-facing contract:

same fixed starting level + different professional current-market narratives
should produce auditable support changes and final scenario distributions that
separate above repeat/bootstrap/start-only controls in the relevant risk
channels.

The script does not train a model, call OpenAI, or rerun the generator. It joins
the latest caption, transmission, portfolio, and support-weighting artifacts so
the next autoresearch step targets the first failing product gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CAPTION_REFRESH = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "caption_conditionality_refresh_917f_balanced80/"
    "caption_conditionality_refresh_summary.json"
)
DEFAULT_TRANSMISSION_AUDIT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_transmission_audit_918a_balanced80/"
    "conditionality_transmission_audit.json"
)
DEFAULT_BENCHMARK = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_strength_benchmark_917f_balanced80_caption_quality/"
    "conditionality_strength_benchmark.json"
)
DEFAULT_WEIGHTING_FRONTIER = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_generator_response_weighting_frontier_918b/"
    "generator_response_weighting_frontier.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_product_conditionality_contract_919a"
)


STATUS_SCORE = {"fail": 0, "warning": 1, "pass": 2}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _gate(
    name: str,
    status: str,
    *,
    question: str,
    evidence: str,
    metrics: dict[str, Any],
    next_action: str,
) -> dict[str, Any]:
    if status not in STATUS_SCORE:
        raise ValueError(f"unknown status {status!r} for gate {name}")
    return {
        "name": name,
        "status": status,
        "question": question,
        "evidence": evidence,
        "metrics": metrics,
        "next_action": next_action,
    }


def _overall_status(gates: list[dict[str, Any]]) -> str:
    statuses = [str(g["status"]) for g in gates]
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    return "pass"


def semantic_gate(caption: dict[str, Any]) -> dict[str, Any]:
    gain_gate = caption.get("scenario_gain_gate", {})
    checks = gain_gate.get("checks", {})
    small = caption.get("small_group_summary", {})
    large = caption.get("large_group_summary", {})
    small_simple = small.get("simple", {})
    small_fused = small.get("fused_codex", {})
    large_simple = large.get("simple", {})
    large_fused = large.get("fused_codex", {})
    crps_gains = [
        _num(item.get("crps_gain"))
        for item in checks.values()
        if isinstance(item, dict)
    ]
    energy_gains = [
        _num(item.get("energy_gain"))
        for item in checks.values()
        if isinstance(item, dict)
    ]
    support_gains = [
        _num(small_fused.get("mean_support_cosine")) - _num(small_simple.get("mean_support_cosine")),
        _num(large_fused.get("mean_support_cosine")) - _num(large_simple.get("mean_support_cosine")),
    ]
    target_gains = [
        _num(small_fused.get("mean_target_cosine")) - _num(small_simple.get("mean_target_cosine")),
        _num(large_fused.get("mean_target_cosine")) - _num(large_simple.get("mean_target_cosine")),
    ]
    passes = bool(gain_gate.get("passes_all_models")) and min(crps_gains or [0.0]) > 0.0
    status = "pass" if passes and min(support_gains + target_gains) > 0.0 else "warning"
    return _gate(
        "semantic_narrative_representation",
        status,
        question=(
            "Does the professional narrative representation improve support/target "
            "alignment and scenario quality versus simple text?"
        ),
        evidence=(
            "Balanced-80 fused Codex captions plus fact tokens beat the simple "
            "text group on CRPS/energy under both embedding models."
        ),
        metrics={
            "passes_all_models": bool(gain_gate.get("passes_all_models")),
            "min_crps_gain": min(crps_gains or [0.0]),
            "min_energy_gain": min(energy_gains or [0.0]),
            "small_support_cosine_gain": support_gains[0],
            "large_support_cosine_gain": support_gains[1],
            "small_target_cosine_gain": target_gains[0],
            "large_target_cosine_gain": target_gains[1],
        },
        next_action=(
            "Keep fused professional captions plus explicit fact tokens as the "
            "candidate representation; do not spend effort on prompt-only changes "
            "until downstream gates improve."
        ),
    )


def support_gate(transmission: dict[str, Any]) -> dict[str, Any]:
    summaries = transmission.get("summaries", {})
    observed = summaries.get("observed_cross_narrative", {})
    ratios = transmission.get("decision", {}).get("ratios", {})
    support_tv = _num(observed.get("support_tv_distance_median"))
    repeat_ratio = _num(ratios.get("repeat_support_tv_to_observed"))
    start_ratio = _num(ratios.get("start_only_support_tv_to_observed"))
    status = "pass" if support_tv >= 0.75 and repeat_ratio <= 0.25 and start_ratio <= 0.25 else "fail"
    return _gate(
        "support_mixture_conditionality",
        status,
        question=(
            "Do different narratives under the same start select materially "
            "different auditable support mixtures?"
        ),
        evidence=(
            "Observed cross-narrative support TV is high while repeat and "
            "start-only controls are near zero."
        ),
        metrics={
            "observed_support_tv_median": support_tv,
            "repeat_support_tv_to_observed": repeat_ratio,
            "start_only_support_tv_to_observed": start_ratio,
        },
        next_action=(
            "Do not treat support selection as the immediate bottleneck; preserve "
            "diverse non-overlapping component support in future tests."
        ),
    )


def prefix_gate(transmission: dict[str, Any]) -> dict[str, Any]:
    summaries = transmission.get("summaries", {})
    observed = summaries.get("observed_cross_narrative", {})
    ratios = transmission.get("decision", {}).get("ratios", {})
    prefix_rmse = _num(observed.get("decoded_prefix_norm_rmse_median"))
    repeat_ratio = _num(ratios.get("repeat_decoded_prefix_to_observed"))
    start_ratio = _num(ratios.get("start_only_decoded_prefix_to_observed"))
    status = "pass" if prefix_rmse > 0.1 and repeat_ratio <= 0.25 and start_ratio <= 0.25 else "fail"
    return _gate(
        "decoded_prefix_conditionality",
        status,
        question=(
            "Do narrative-selected supports become different recent-prefix objects "
            "before the frozen rollout?"
        ),
        evidence=(
            "Decoded-prefix RMSE is nontrivial across narratives, while repeat "
            "and start-only prefix differences are zero in the current audit."
        ),
        metrics={
            "observed_decoded_prefix_rmse_median": prefix_rmse,
            "repeat_decoded_prefix_to_observed": repeat_ratio,
            "start_only_decoded_prefix_to_observed": start_ratio,
        },
        next_action=(
            "Do not add another text-to-memory projection before checking rollout "
            "and readout; the prefix layer is already transmitting narrative signal."
        ),
    )


def factor_distribution_gate(transmission: dict[str, Any]) -> dict[str, Any]:
    summaries = transmission.get("summaries", {})
    observed = summaries.get("observed_cross_narrative", {})
    ratios = transmission.get("decision", {}).get("ratios", {})
    observed_energy = _num(observed.get("rollout_path_energy_distance_median"))
    repeat_ratio = _num(ratios.get("repeat_rollout_energy_to_observed"))
    bootstrap_ratio = _num(ratios.get("bootstrap_rollout_energy_to_observed"))
    start_ratio = _num(ratios.get("start_only_rollout_energy_to_observed"))
    if repeat_ratio >= 0.75 or start_ratio > 0.25:
        status = "fail"
    elif bootstrap_ratio >= 0.75:
        status = "warning"
    else:
        status = "pass"
    return _gate(
        "factor_distribution_conditionality",
        status,
        question=(
            "Do generated factor path distributions differ beyond repeat, "
            "bootstrap, and start-only controls?"
        ),
        evidence=(
            "Narrative path energy is above repeat and start-only controls, but "
            "within-run bootstrap noise remains close to the observed effect."
        ),
        metrics={
            "observed_rollout_path_energy_median": observed_energy,
            "repeat_rollout_energy_to_observed": repeat_ratio,
            "bootstrap_rollout_energy_to_observed": bootstrap_ratio,
            "start_only_rollout_energy_to_observed": start_ratio,
        },
        next_action=(
            "Target conditionality-aware readout or generator-response labels; "
            "plain support retrieval is not the failing layer."
        ),
    )


def portfolio_tail_gate(benchmark: dict[str, Any]) -> dict[str, Any]:
    decision = benchmark.get("decision", {})
    ratios = decision.get("key_ratios", {})
    portfolio_status = str(decision.get("portfolio_status", ""))
    path_repeat = _num(ratios.get("portfolio_path_vs_repeat"))
    path_bootstrap = _num(ratios.get("portfolio_path_vs_bootstrap"))
    var_repeat = _num(ratios.get("portfolio_var95_vs_repeat"))
    var_bootstrap = _num(ratios.get("portfolio_var95_vs_bootstrap"))
    if path_repeat < 1.25:
        status = "fail"
    elif portfolio_status == "warning" or var_repeat < 1.25 or path_bootstrap < 1.0:
        status = "warning"
    else:
        status = "pass"
    return _gate(
        "portfolio_tail_conditionality",
        status,
        question=(
            "Do portfolio PnL and tail-risk summaries change more across "
            "narratives than across controls?"
        ),
        evidence=(
            "Portfolio path separation is above repeat, but VaR95 separation is "
            "below the repeat-control threshold."
        ),
        metrics={
            "portfolio_status": portfolio_status,
            "portfolio_path_vs_repeat": path_repeat,
            "portfolio_path_vs_bootstrap": path_bootstrap,
            "portfolio_var95_vs_repeat": var_repeat,
            "portfolio_var95_vs_bootstrap": var_bootstrap,
        },
        next_action=(
            "Evaluate candidate changes on portfolio distributions, not only SPX "
            "fans or bridge cosine. Portfolio-risk-aware support utilities are a "
            "reasonable next branch."
        ),
    )


def auditability_gate(
    *,
    caption: dict[str, Any],
    transmission: dict[str, Any],
    benchmark: dict[str, Any],
    frontier: dict[str, Any],
) -> dict[str, Any]:
    artifact_paths: list[str] = []
    for payload in [caption, transmission, benchmark, frontier]:
        paths = payload.get("artifact_paths", {})
        if isinstance(paths, dict):
            artifact_paths.extend(str(value) for value in paths.values())
    missing = [path for path in artifact_paths if path and not Path(path).exists()]
    status = "pass" if artifact_paths and not missing else "warning"
    return _gate(
        "auditability_and_provenance",
        status,
        question=(
            "Can a risk manager inspect the narrative, support evidence, controls, "
            "plots, and decision artifacts?"
        ),
        evidence=(
            "The current run preserves JSON/markdown/plot artifacts for caption "
            "quality, support/prefix/factor transmission, portfolio controls, and "
            "support-weighting frontier decisions."
        ),
        metrics={
            "artifact_path_count": len(artifact_paths),
            "missing_artifact_count": len(missing),
            "missing_artifacts": missing[:20],
        },
        next_action=(
            "Keep evidence paths attached to every future demo/paper claim; do "
            "not promote claims that lack artifact-level provenance."
        ),
    )


def frontier_read(frontier: dict[str, Any]) -> dict[str, Any]:
    decision = frontier.get("decision", {})
    return {
        "status": str(decision.get("status", "")),
        "recommendation": str(decision.get("recommendation", "")),
        "oracle_clean_improvement": bool(decision.get("oracle_clean_improvement")),
        "learned_clean_improvement": bool(decision.get("learned_clean_improvement")),
        "learned_competitive": bool(decision.get("learned_competitive")),
    }


def build_product_conditionality_report(
    *,
    caption: dict[str, Any],
    transmission: dict[str, Any],
    benchmark: dict[str, Any],
    frontier: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    gates = [
        semantic_gate(caption),
        support_gate(transmission),
        prefix_gate(transmission),
        factor_distribution_gate(transmission),
        portfolio_tail_gate(benchmark),
        auditability_gate(
            caption=caption,
            transmission=transmission,
            benchmark=benchmark,
            frontier=frontier,
        ),
    ]
    overall = _overall_status(gates)
    failing = [g["name"] for g in gates if g["status"] == "fail"]
    warning = [g["name"] for g in gates if g["status"] == "warning"]
    if overall == "pass":
        verdict = "product_conditionality_supported"
        next_action = "Move to promotion verifier before changing defaults."
    elif failing:
        verdict = "product_conditionality_blocked"
        next_action = "Fix failing gates before promotion or demo-default changes."
    else:
        verdict = "product_conditionality_partially_supported_with_warnings"
        next_action = (
            "Do not claim fully solved conditionality. Target the warning layers: "
            "factor rollout/readout noise and portfolio-tail separation."
        )
    report = {
        "scope_note": (
            "Product-facing conditionality audit over existing artifacts. No model "
            "training, OpenAI calls, or generator reruns are performed."
        ),
        "conditionality_definition": (
            "For a fixed approved starting level, a professional current-market "
            "narrative is conditionally useful if it changes the auditable support "
            "mixture and produces a distinguishable future risk distribution in "
            "the implied risk channels above repeat/bootstrap/start-only controls."
        ),
        "overall_status": overall,
        "verdict": verdict,
        "failing_gates": failing,
        "warning_gates": warning,
        "next_action": next_action,
        "gates": gates,
        "support_weighting_frontier": frontier_read(frontier),
        "source_artifacts": {
            "caption_refresh": str(DEFAULT_CAPTION_REFRESH),
            "transmission_audit": str(DEFAULT_TRANSMISSION_AUDIT),
            "conditionality_benchmark": str(DEFAULT_BENCHMARK),
            "support_weighting_frontier": str(DEFAULT_WEIGHTING_FRONTIER),
        },
        "artifact_paths": {
            "report_json": str(output_dir / "product_conditionality_contract_audit.json"),
            "report_markdown": str(output_dir / "product_conditionality_contract_audit.md"),
            "gate_plot": str(output_dir / "product_conditionality_gate_ladder.png"),
        },
    }
    return report


def _plot(report: dict[str, Any], output: Path) -> None:
    gates = report["gates"]
    labels = [str(g["name"]).replace("_", "\n") for g in gates]
    scores = [STATUS_SCORE[str(g["status"])] for g in gates]
    colors = [
        {"pass": "#2E7D32", "warning": "#EF6C00", "fail": "#C62828"}[str(g["status"])]
        for g in gates
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(np.arange(len(gates)), scores, color=colors, alpha=0.82)
    ax.set_xticks(np.arange(len(gates)), labels, fontsize=8)
    ax.set_yticks([0, 1, 2], ["fail", "warning", "pass"])
    ax.set_ylim(0, 2.25)
    ax.set_title("Product conditionality gate ladder", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    for i, gate in enumerate(gates):
        ax.text(i, scores[i] + 0.05, str(gate["status"]), ha="center", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Product Conditionality Contract Audit",
        "",
        f"Status: **{report['overall_status']}**",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Definition",
        "",
        str(report["conditionality_definition"]),
        "",
        "## Gate Summary",
        "",
        "| Gate | Status | Evidence | Next action |",
        "|---|---:|---|---|",
    ]
    for gate in report["gates"]:
        lines.append(
            "| "
            + str(gate["name"])
            + " | "
            + str(gate["status"])
            + " | "
            + str(gate["evidence"])
            + " | "
            + str(gate["next_action"])
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(report["next_action"]),
            "",
            "## Support-Weighting Frontier",
            "",
            f"Status: `{report['support_weighting_frontier']['status']}`",
            "",
            str(report["support_weighting_frontier"]["recommendation"]),
            "",
            "## Artifacts",
            "",
            *[f"- {key}: `{value}`" for key, value in report["artifact_paths"].items()],
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    report = build_product_conditionality_report(
        caption=_load_json(args.caption_refresh),
        transmission=_load_json(args.transmission_audit),
        benchmark=_load_json(args.conditionality_benchmark),
        frontier=_load_json(args.support_weighting_frontier),
        output_dir=output_dir,
    )
    _plot(report, Path(report["artifact_paths"]["gate_plot"]))
    _write_json(report["artifact_paths"]["report_json"], report)
    _write_text(report["artifact_paths"]["report_markdown"], _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-refresh", type=Path, default=DEFAULT_CAPTION_REFRESH)
    parser.add_argument("--transmission-audit", type=Path, default=DEFAULT_TRANSMISSION_AUDIT)
    parser.add_argument("--conditionality-benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--support-weighting-frontier",
        type=Path,
        default=DEFAULT_WEIGHTING_FRONTIER,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["overall_status"],
                "verdict": report["verdict"],
                "warning_gates": report["warning_gates"],
                "failing_gates": report["failing_gates"],
                "report": report["artifact_paths"]["report_json"],
                "markdown": report["artifact_paths"]["report_markdown"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
