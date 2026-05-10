from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCALE_REFERENCE = Path("results/world/masked_multiview_barlow_scale_head127.json")
SCALE_STATE = Path("results/world/scale_state_probe_head127.json")
SCALE_DOWNSTREAM = Path("results/world/scale_downstream_quality_head128.json")
SCALE_MASK_ARTIFACT = Path(
    "results/world/masked_multiview_mask_artifact_scale_head129.json"
)
SCALE_STRATIFIED = Path("results/world/masked_multiview_stratified_scale_head129.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _min_top10(stratified: dict[str, Any]) -> float:
    rows = list(stratified["by_mask_family_a"].values()) + list(
        stratified["by_mask_family_b"].values()
    )
    return min(float(row["retrieval_top10"]) for row in rows)


def _max_offdiag(stratified: dict[str, Any]) -> float:
    rows = list(stratified["by_mask_family_a"].values()) + list(
        stratified["by_mask_family_b"].values()
    )
    return max(float(row["offdiag_abs_mean"]) for row in rows)


def assess_scale_part1_quality_gate() -> dict[str, Any]:
    reference = _load_json(SCALE_REFERENCE)
    state = _load_json(SCALE_STATE)
    downstream = _load_json(SCALE_DOWNSTREAM)
    mask = _load_json(SCALE_MASK_ARTIFACT)
    stratified = _load_json(SCALE_STRATIFIED)

    retrieval = reference["val_metrics"]["view_alignment"]["retrieval"]
    raw_retrieval = reference["raw_val_baseline"]["retrieval"]
    health = reference["val_metrics"]["view_alignment"]["view_a_health"]
    state_decision = state["decision"]
    downstream_decision = downstream["decision"]
    regime = downstream["regime"]["scale"]

    layers = [
        {
            "layer": "representation_health",
            "status": "PASS",
            "evidence": (
                f"top10={retrieval['top10']:.6f} vs raw={raw_retrieval['top10']:.6f}; "
                f"mrr={retrieval['mrr']:.6f} vs raw={raw_retrieval['mrr']:.6f}; "
                f"effective_rank={health['effective_rank']:.3f}; "
                f"variance_min={health['variance_min']:.6f}; "
                f"offdiag_abs_mean={health['offdiag_abs_mean']:.6f}"
            ),
        },
        {
            "layer": "corruption_robustness",
            "status": "PASS",
            "evidence": (
                f"mask_artifact={mask['decision_hint']}; "
                f"stratified={stratified['decision_hint']}; "
                f"min_stratified_top10={_min_top10(stratified):.6f}; "
                f"max_stratified_offdiag={_max_offdiag(stratified):.6f}"
            ),
        },
        {
            "layer": "state_content",
            "status": "PARTIAL",
            "evidence": (
                "state probes improve versus HEAD070, but exact IV retention still "
                f"loses to raw surface; scale_beats_raw_surface_on_iv="
                f"{state_decision['scale_beats_raw_surface_on_iv']}"
            ),
        },
        {
            "layer": "baseline_superiority",
            "status": "FAIL",
            "evidence": (
                f"standalone Barlow wins {downstream['scale_barlow_best_raw_wins']}/5 "
                "IV future targets versus best raw surface baselines"
            ),
        },
        {
            "layer": "market_state_regime_probe",
            "status": "FAIL",
            "evidence": (
                f"regime_accuracy={regime['barlow_accuracy']:.6f}; "
                f"raw_last={regime['raw_surface_last_accuracy']:.6f}; "
                f"majority={regime['majority_accuracy']:.6f}"
            ),
        },
        {
            "layer": "scale_and_stability",
            "status": "PARTIAL",
            "evidence": (
                "scale improved one-seed smoke to 1024 train windows and 256 validation "
                "windows, but multi-seed and full-data stability are still not run"
            ),
        },
    ]
    passed = all(layer["status"] == "PASS" for layer in layers)
    return {
        "assessment": "world_model_part1_scaled_quality_gate",
        "date": "2026-05-10",
        "reference_run": "HEAD127_scaled_flat_barlow",
        "quality_gate_passed": passed,
        "promotion_decision": "PROMOTE" if passed else "DO_NOT_PROMOTE",
        "part1_ready_for_part_b": passed,
        "layer_results": layers,
        "artifacts": {
            "reference": str(SCALE_REFERENCE),
            "state_probe": str(SCALE_STATE),
            "downstream_quality": str(SCALE_DOWNSTREAM),
            "mask_artifact": str(SCALE_MASK_ARTIFACT),
            "stratified": str(SCALE_STRATIFIED),
        },
        "overall_interpretation": (
            "HEAD127 is the best Part 1 candidate so far and validates scale as "
            "useful, but it is not ready for Part B because baseline superiority "
            "and market-state regime probes still fail."
        ),
        "next_required_evidence": [
            "multi-seed scale stability",
            "exact-state retention improvement versus raw surface features",
            "stronger market-state probes that beat majority and raw baselines",
            "broad baseline superiority beyond 2/5 standalone IV future targets",
        ],
    }


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
        "`masked_multiview_invariance` scaled-candidate quality gate.",
        "",
        "## Verdict",
        "",
        f"- Quality gate passed: `{result['quality_gate_passed']}`.",
        f"- Promotion decision: `{result['promotion_decision']}`.",
        f"- Part 1 ready for Part B: `{result['part1_ready_for_part_b']}`.",
        "",
        "## Layer Results",
        "",
        "| layer | status | evidence |",
        "| --- | --- | --- |",
    ]
    for layer in result["layer_results"]:
        lines.append(f"| {layer['layer']} | {layer['status']} | {layer['evidence']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["overall_interpretation"],
            "",
            "## Next Required Evidence",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["next_required_evidence"])
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess the scaled Part 1 candidate quality gate"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_part1_quality_gate_head130.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head130_scale_part1_quality_gate.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD130: Scale Part 1 Quality Gate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = assess_scale_part1_quality_gate()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(
        render_markdown(result, title=args.report_title),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "quality_gate_passed": result["quality_gate_passed"],
                "promotion_decision": result["promotion_decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
