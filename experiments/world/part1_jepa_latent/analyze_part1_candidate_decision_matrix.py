from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCALE_GATE = Path("results/world/scale_part1_quality_gate_head130.json")
TARGET_TAXONOMY = Path("results/world/scale_baseline_target_taxonomy_head134.json")
CONTEXT_LATENT = Path("results/world/context_target_latent_health_head142.json")
CONTEXT_CLEAN_QUALITY = Path("results/world/context_target_clean_quality_head145.json")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe(result: dict[str, Any], feature: str, target: str) -> float:
    return float(result["probe_metrics"][feature]["targets"][target]["mse"])


def _rank(result: dict[str, Any], feature: str) -> float:
    return float(result["probe_metrics"][feature]["health"]["effective_rank"])


def build_part1_candidate_decision_matrix() -> dict[str, Any]:
    scale_gate = _load(SCALE_GATE)
    taxonomy = _load(TARGET_TAXONOMY)
    latent = _load(CONTEXT_LATENT)
    clean = _load(CONTEXT_CLEAN_QUALITY)
    layer_status = {
        row["layer"]: row["status"] for row in scale_gate["layer_results"]
    }
    rows = [
        {
            "candidate": "raw_surface_last",
            "family": "raw_baseline_not_part1_model",
            "current_iv_mse": _probe(clean, "raw_surface_last", "iv_surface"),
            "effective_rank": _rank(clean, "raw_surface_last"),
            "status": "baseline_only",
            "decision": "use_as_exact_state_floor_not_representation_candidate",
        },
        {
            "candidate": "scale_barlow_head127",
            "family": "masked_multiview_invariance",
            "current_iv_mse": _probe(clean, "scale_barlow_last", "iv_surface"),
            "effective_rank": _rank(clean, "scale_barlow_last"),
            "status": "best_learned_candidate_do_not_promote",
            "decision": (
                "keep_as_active_candidate; representation and corruption pass but "
                "baseline superiority, regime, and exact-state gates remain blocked"
            ),
        },
        {
            "candidate": "context_target_head140",
            "family": "minimal_context_to_target_jepa",
            "current_iv_mse": _probe(clean, "target_only_head140_last", "iv_surface"),
            "effective_rank": _rank(clean, "target_only_head140_last"),
            "status": "demoted",
            "decision": (
                "target latent has mask-family lift "
                f"{latent['decision']['target_family_accuracy_lift']:.6f} and "
                "predicted latent is low rank"
            ),
        },
        {
            "candidate": "context_target_head144",
            "family": "minimal_context_to_target_jepa_clean_target",
            "current_iv_mse": _probe(clean, "clean_target_head144_last", "iv_surface"),
            "effective_rank": _rank(clean, "clean_target_head144_last"),
            "status": "demoted",
            "decision": "clean-target correction is worse than target-only and scaled Barlow",
        },
    ]
    return {
        "analysis": "world_model_part1_candidate_decision_matrix",
        "date": "2026-05-10",
        "objective_family": "part1_evidence_consolidation",
        "sources": {
            "scale_gate": str(SCALE_GATE),
            "target_taxonomy": str(TARGET_TAXONOMY),
            "context_latent": str(CONTEXT_LATENT),
            "context_clean_quality": str(CONTEXT_CLEAN_QUALITY),
        },
        "rows": rows,
        "scale_gate_layer_status": layer_status,
        "target_family_summary": taxonomy["decision"],
        "decision": {
            "active_candidate": "scale_barlow_head127",
            "part1_ready_for_part_b": False,
            "minimal_context_to_target_demoted": True,
            "primary_blocker": "exact_state_retention_and_baseline_superiority",
            "allowed_next_work": [
                "bounded evidence consolidation around exact-state gap",
                "new token_geometry_level_jepa_design_gate",
                "quality-gate reconciliation for scaled Barlow",
            ],
            "blocked_next_work": [
                "Part 2 decoder training",
                "minimal context-to-target knob sweep",
                "future prediction as pretraining objective",
            ],
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
        "`part1_evidence_consolidation`; no model change.",
        "",
        "## Hypothesis",
        "",
        "After demoting the minimal context-to-target route, the workflow should",
        "have a compact candidate matrix that identifies the active Part 1",
        "candidate and the remaining blocker.",
        "",
        "## Candidate Matrix",
        "",
        "| candidate | family | current-IV MSE | rank | status | decision |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {candidate} | {family} | {mse} | {rank} | {status} | {decision} |".format(
                candidate=row["candidate"],
                family=row["family"],
                mse=_fmt(row["current_iv_mse"]),
                rank=_fmt(row["effective_rank"]),
                status=row["status"],
                decision=row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Scale Gate Layers",
            "",
            "| layer | status |",
            "| --- | --- |",
        ]
    )
    for layer, status in result["scale_gate_layer_status"].items():
        lines.append(f"| {layer} | {status} |")
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Active learned candidate: `{decision['active_candidate']}`.",
            f"- Part 1 ready for Part B: `{decision['part1_ready_for_part_b']}`.",
            f"- Minimal context-to-target demoted: `{decision['minimal_context_to_target_demoted']}`.",
            f"- Primary blocker: `{decision['primary_blocker']}`.",
            "",
            "Allowed next work:",
        ]
    )
    lines.extend(f"- `{item}`" for item in decision["allowed_next_work"])
    lines.extend(["", "Blocked next work:"])
    lines.extend(f"- `{item}`" for item in decision["blocked_next_work"])
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Part 1 candidate matrix")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/part1_candidate_decision_matrix_head147.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head147_part1_candidate_decision_matrix.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD147: Part 1 Candidate Decision Matrix",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_part1_candidate_decision_matrix()
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
