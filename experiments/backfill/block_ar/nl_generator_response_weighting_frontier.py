#!/usr/bin/env python
"""Summarize the generator-response-aware support-weighting frontier.

This artifact-only diagnostic joins:

- the current conditionality transmission audit;
- the non-deployable oracle soft-support upper bound;
- the deployable learned kernel/listwise support-weight policy.

It answers the immediate research question after the transmission audit:
should the next production push focus on generator-response-aware support
weights, or does the evidence point more strongly to rollout/readout noise?
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


DEFAULT_TRANSMISSION_AUDIT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_transmission_audit_918a_balanced80/"
    "conditionality_transmission_audit.json"
)
DEFAULT_ORACLE_COMPARISON = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_oracle_soft_support_weights_906c_fullheldout/"
    "oracle_soft_support_comparison.json"
)
DEFAULT_LEARNED_COMPARISON = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_kernel_listwise_full_906e_278train_to_66test_temp0p20/"
    "full_kernel_policy_comparison.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_generator_response_weighting_frontier_918b"
)


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


def _window_count(metrics: dict[str, Any], fallback: dict[str, Any]) -> int:
    direct = metrics.get("window_count", fallback.get("window_count", None))
    if direct is not None:
        return int(direct or 0)
    patterns = metrics.get("sample_count_patterns", fallback.get("sample_count_patterns", {}))
    if isinstance(patterns, dict):
        return int(sum(int(count) for count in patterns.values()))
    return 0


def _oracle_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    equal = summary.get("equal_top5", {})
    oracle = summary.get("oracle_weighted_top5", {})
    if not equal or not oracle:
        raise ValueError("oracle comparison missing equal_top5 or oracle_weighted_top5")
    return {
        "baseline": "equal_top5",
        "candidate": "oracle_weighted_top5",
        "window_count": _window_count(oracle, equal),
        "crps_delta_candidate_minus_baseline": float(oracle["crps_mean"]) - float(equal["crps_mean"]),
        "energy_delta_candidate_minus_baseline": float(oracle["energy_mean"]) - float(equal["energy_mean"]),
        "coverage_delta_candidate_minus_baseline": float(oracle["coverage_80_mean"])
        - float(equal["coverage_80_mean"]),
        "crps_relative_reduction": (float(equal["crps_mean"]) - float(oracle["crps_mean"]))
        / float(equal["crps_mean"]),
        "energy_relative_reduction": (float(equal["energy_mean"]) - float(oracle["energy_mean"]))
        / float(equal["energy_mean"]),
        "nonuniform_allocation_patterns": dict(oracle.get("sample_count_patterns", {})),
    }


def _learned_summary(report: dict[str, Any]) -> dict[str, Any]:
    equal = report.get("equal_top5", {})
    learned = report.get("learned_kernel_listwise_temp0p20", {})
    delta = report.get("absolute_delta_learned_minus_equal", {})
    rel = report.get("relative_delta", {})
    if not equal or not learned:
        raise ValueError("learned comparison missing equal_top5 or learned policy")
    return {
        "baseline": "equal_top5",
        "candidate": "learned_kernel_listwise_temp0p20",
        "window_count": _window_count(learned, equal),
        "crps_delta_candidate_minus_baseline": float(
            delta.get("ensemble_crps_z_mean", float(learned["ensemble_crps_z_mean"]) - float(equal["ensemble_crps_z_mean"]))
        ),
        "energy_delta_candidate_minus_baseline": float(
            delta.get("energy_score_z_mean", float(learned["energy_score_z_mean"]) - float(equal["energy_score_z_mean"]))
        ),
        "coverage_delta_candidate_minus_baseline": float(
            delta.get("coverage_80_mean", float(learned["coverage_80_mean"]) - float(equal["coverage_80_mean"]))
        ),
        "crps_relative_reduction": float(
            rel.get("ensemble_crps_z_mean_relative_reduction_vs_equal", 0.0)
        ),
        "energy_relative_reduction": float(
            rel.get("energy_score_z_mean_relative_reduction_vs_equal", 0.0)
        ),
        "nonuniform_support_allocations": dict(report.get("learned_nonuniform_support_allocations", {})),
        "source_status": str(report.get("status", "")),
    }


def _clean_improvement(summary: dict[str, Any]) -> bool:
    return (
        float(summary["crps_delta_candidate_minus_baseline"]) < 0.0
        and float(summary["energy_delta_candidate_minus_baseline"]) < 0.0
        and float(summary["coverage_delta_candidate_minus_baseline"]) >= 0.0
    )


def decide_frontier(
    *,
    transmission: dict[str, Any],
    oracle: dict[str, Any],
    learned: dict[str, Any],
) -> dict[str, Any]:
    transmission_decision = transmission.get("decision", {})
    bottlenecks = list(transmission_decision.get("bottlenecks", []))
    oracle_clean = _clean_improvement(oracle)
    learned_clean = _clean_improvement(learned)
    learned_competitive = (
        float(learned["crps_delta_candidate_minus_baseline"]) <= 0.0
        and float(learned["coverage_delta_candidate_minus_baseline"]) >= 0.0
        and abs(float(learned["energy_delta_candidate_minus_baseline"])) <= 0.002
    )
    if oracle_clean and not learned_clean:
        status = "upper_bound_found_learned_policy_insufficient"
        recommendation = (
            "Do not promote the current learned support-weight policy. Continue "
            "support weighting only with a better generator-response surface, "
            "such as regime/prototype-aware labels or portfolio-risk-aware "
            "candidate utilities. In parallel, treat readout/rollout noise as "
            "the current product bottleneck."
        )
    elif learned_clean:
        status = "learned_support_weighting_candidate_ready_for_fixed_start_gate"
        recommendation = (
            "Run the learned policy through fixed-start conditionality and "
            "portfolio-tail gates before considering promotion."
        )
    elif learned_competitive:
        status = "learned_policy_competitive_but_not_clean"
        recommendation = (
            "The learned policy is a candidate diagnostic, not a production "
            "default. Improve the response label surface before broad rollout."
        )
    else:
        status = "support_weighting_not_current_lever"
        recommendation = (
            "Prioritize path/dependence-aware readout before more support-policy "
            "complexity."
        )
    if "within_run_rollout_bootstrap_noise_close_to_observed" in bottlenecks:
        recommendation += (
            " The transmission audit specifically shows that within-run rollout "
            "bootstrap noise remains close to the observed narrative effect."
        )
    return {
        "status": status,
        "oracle_clean_improvement": oracle_clean,
        "learned_clean_improvement": learned_clean,
        "learned_competitive": learned_competitive,
        "transmission_verdict": str(transmission_decision.get("verdict", "")),
        "transmission_bottlenecks": bottlenecks,
        "recommendation": recommendation,
    }


def _plot_frontier(report: dict[str, Any], output: Path) -> None:
    oracle = report["oracle_soft_top5"]
    learned = report["learned_kernel_listwise"]
    labels = ["Oracle soft\nupper bound", "Learned kernel\nlistwise"]
    crps = [
        -100.0 * float(oracle["crps_delta_candidate_minus_baseline"]),
        -100.0 * float(learned["crps_delta_candidate_minus_baseline"]),
    ]
    energy = [
        -100.0 * float(oracle["energy_delta_candidate_minus_baseline"]),
        -100.0 * float(learned["energy_delta_candidate_minus_baseline"]),
    ]
    coverage = [
        100.0 * float(oracle["coverage_delta_candidate_minus_baseline"]),
        100.0 * float(learned["coverage_delta_candidate_minus_baseline"]),
    ]
    x = np.arange(2)
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.0, color="#111111", linewidth=0.9)
    ax.bar(x - width, crps, width=width, label="CRPS improvement", color="#1565C0")
    ax.bar(x, energy, width=width, label="Energy improvement", color="#2E7D32")
    ax.bar(x + width, coverage, width=width, label="Coverage delta", color="#EF6C00")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Candidate minus equal top-5, shown as improvement points")
    ax.set_title("Generator-response support weighting frontier")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    oracle = report["oracle_soft_top5"]
    learned = report["learned_kernel_listwise"]
    decision = report["decision"]
    return "\n".join(
        [
            "# Generator-Response Support-Weighting Frontier",
            "",
            f"Status: **{decision['status']}**",
            "",
            "## Why This Was Run",
            "",
            (
                "The transmission audit showed that narrative signal reaches "
                "the support and decoded-prefix layers, but rollout/bootstrap "
                "noise and portfolio-tail separation limit the final scenario "
                "distribution. This report checks whether generator-response-aware "
                "support weights are already a sufficient remedy."
            ),
            "",
            "## Evidence",
            "",
            (
                "- Oracle soft top-5 upper bound: "
                f"CRPS delta `{oracle['crps_delta_candidate_minus_baseline']:+.4f}`, "
                f"energy delta `{oracle['energy_delta_candidate_minus_baseline']:+.4f}`, "
                f"coverage delta `{oracle['coverage_delta_candidate_minus_baseline']:+.4f}`."
            ),
            (
                "- Learned kernel/listwise policy: "
                f"CRPS delta `{learned['crps_delta_candidate_minus_baseline']:+.4f}`, "
                f"energy delta `{learned['energy_delta_candidate_minus_baseline']:+.4f}`, "
                f"coverage delta `{learned['coverage_delta_candidate_minus_baseline']:+.4f}`."
            ),
            (
                "- Transmission verdict: "
                f"`{decision['transmission_verdict']}` with bottlenecks "
                f"`{', '.join(decision['transmission_bottlenecks'])}`."
            ),
            "",
            "## Decision",
            "",
            decision["recommendation"],
            "",
            "## Artifacts",
            "",
            *[f"- {key}: `{value}`" for key, value in report["artifact_paths"].items()],
            "",
        ]
    )


def build_frontier_report(args: argparse.Namespace) -> dict[str, Any]:
    transmission = _load_json(args.transmission_audit)
    oracle_report = _load_json(args.oracle_comparison)
    learned_report = _load_json(args.learned_comparison)
    oracle = _oracle_summary(oracle_report)
    learned = _learned_summary(learned_report)
    decision = decide_frontier(
        transmission=transmission,
        oracle=oracle,
        learned=learned,
    )
    output_dir = Path(args.output_dir)
    plot_path = output_dir / "generator_response_weighting_frontier.png"
    report = {
        "scope_note": (
            "Artifact-only frontier decision for generator-response-aware support "
            "weighting. This does not train a new policy or call OpenAI."
        ),
        "transmission_audit": str(args.transmission_audit),
        "oracle_comparison": str(args.oracle_comparison),
        "learned_comparison": str(args.learned_comparison),
        "oracle_soft_top5": oracle,
        "learned_kernel_listwise": learned,
        "decision": decision,
        "artifact_paths": {
            "report_json": str(output_dir / "generator_response_weighting_frontier.json"),
            "report_markdown": str(output_dir / "generator_response_weighting_frontier.md"),
            "frontier_plot": str(plot_path),
        },
    }
    _plot_frontier(report, plot_path)
    _write_json(report["artifact_paths"]["report_json"], report)
    _write_text(report["artifact_paths"]["report_markdown"], _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transmission-audit", type=Path, default=DEFAULT_TRANSMISSION_AUDIT)
    parser.add_argument("--oracle-comparison", type=Path, default=DEFAULT_ORACLE_COMPARISON)
    parser.add_argument("--learned-comparison", type=Path, default=DEFAULT_LEARNED_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build_frontier_report(args)
    print(
        json.dumps(
            {
                "status": report["decision"]["status"],
                "recommendation": report["decision"]["recommendation"],
                "report": report["artifact_paths"]["report_json"],
                "markdown": report["artifact_paths"]["report_markdown"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
