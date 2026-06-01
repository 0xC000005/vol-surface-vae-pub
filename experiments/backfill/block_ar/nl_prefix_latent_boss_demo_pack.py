#!/usr/bin/env python
"""Create a boss-ready evidence pack for the narrative scenario demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_VALIDATION_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_conditioned_bakeoff_820b_expanded_temp050_s16/"
    "start_conditioned_bakeoff.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_boss_demo_pack_821a"
)
DEFAULT_LIVE_CASEBOOK_REPORT = ""
DEFAULT_FIXED_START_CAPTION_AUDIT_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "full_corpus_fixed_start_conditionality_audit_957c_start22_s8_shared_startonly/"
    "full_corpus_fixed_start_conditionality_audit.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):+.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def validation_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    variants = report.get("variant_summary", [])
    best = variants[0] if isinstance(variants, list) and variants else {}
    if not isinstance(best, dict):
        best = {}
    rows = [row for row in report.get("rows", []) if isinstance(row, dict)]
    improved_crps = 0
    improved_energy = 0
    warning_rows = 0
    pass_rows = 0
    case_names: set[str] = set()
    start_names: set[str] = set()
    for row in rows:
        case_name = str(row.get("case_name", ""))
        start_name = str(row.get("start_name", ""))
        if case_name:
            case_names.add(case_name)
        if start_name:
            start_names.add(start_name)
        status = str(row.get("validation_operational", ""))
        if status == "pass":
            pass_rows += 1
        if status == "warning":
            warning_rows += 1
        metrics = row.get("scenario_metrics", {})
        if isinstance(metrics, dict):
            crps_value = metrics.get("ensemble_crps_z_improvement_vs_persistence")
            try:
                improved_crps += int(float(crps_value) > 0.0)
            except (TypeError, ValueError):
                pass
            energy_value = metrics.get("energy_score_z_improvement_vs_persistence")
            try:
                improved_energy += int(float(energy_value) > 0.0)
            except (TypeError, ValueError):
                pass
    return {
        "case_count": int(report.get("case_count", len(rows)) or len(rows)),
        "run_count": int(report.get("run_count", len(rows)) or len(rows)),
        "case_set": str(report.get("case_set", "")),
        "variant_set": str(report.get("variant_set", "")),
        "best_variant": str(best.get("variant_name", "")),
        "narrative_family_count": len(case_names),
        "narrative_families": sorted(case_names),
        "fixed_start_count": len(start_names),
        "fixed_starts": sorted(start_names),
        "status_counts": best.get("operational_status_counts", {}),
        "pass_rows": int(pass_rows),
        "warning_rows": int(warning_rows),
        "improved_crps_rows": int(improved_crps),
        "improved_energy_rows": int(improved_energy),
        "mean_energy_score_z": best.get("mean_energy_score_z"),
        "mean_ensemble_crps_z": best.get("mean_ensemble_crps_z"),
        "mean_energy_improvement_vs_persistence": best.get(
            "mean_energy_improvement_vs_persistence"
        ),
        "mean_crps_improvement_vs_persistence": best.get(
            "mean_crps_improvement_vs_persistence"
        ),
    }


def live_casebook_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    cases = [row for row in report.get("cases", []) if isinstance(row, dict)]
    models = sorted(
        {
            str(row.get("grounding_model", ""))
            for row in cases
            if str(row.get("grounding_model", ""))
        }
    )
    embedding_models = sorted(
        {
            str(row.get("embedding_model", ""))
            for row in cases
            if str(row.get("embedding_model", ""))
        }
    )
    case_rows = []
    for row in cases:
        case_rows.append(
            {
                "case_name": str(row.get("case_name", "")),
                "casebook_choice": str(row.get("casebook_choice", "")),
                "status": str(row.get("status", "")),
                "expected_start_index": row.get("expected_start_index"),
                "condition_only_validation_status": str(
                    row.get("condition_only_validation_status", "")
                ),
                "selected_start_status": str(row.get("selected_start_status", "")),
                "overall_status": str(row.get("overall_status", "")),
                "forward_warning_count": int(
                    row.get("condition_only_forward_warning_count", 0) or 0
                ),
                "support_candidate_count": int(
                    row.get("support_candidate_count", 0) or 0
                ),
                "support_prior_mode": str(row.get("support_prior_mode", "")),
                "summary_path": str(row.get("summary_path", "")),
            }
        )
    return {
        "status": str(report.get("status", "")),
        "case_count": int(report.get("case_count", len(cases)) or len(cases)),
        "pass_count": int(report.get("pass_count", 0) or 0),
        "total_openai_tokens": int(report.get("total_openai_tokens", 0) or 0),
        "min_support_candidate_count": int(
            report.get("min_support_candidate_count", 0) or 0
        ),
        "grounding_models": models,
        "embedding_models": embedding_models,
        "case_rows": case_rows,
        "artifact_paths": report.get("artifact_paths", {}),
    }


def fixed_start_caption_audit_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    groups = report.get("group_summaries", {})
    if not isinstance(groups, dict):
        groups = {}
    decision = report.get("decision", {})
    if not isinstance(decision, dict):
        decision = {}
    rows: list[dict[str, Any]] = []
    for name in ("professional", "simple", "start_only"):
        group = groups.get(name, {})
        if not isinstance(group, dict):
            group = {}
        ranges = group.get("terminal_mean_level_ranges", {})
        if not isinstance(ranges, dict):
            ranges = {}
        rows.append(
            {
                "condition": name,
                "mean_support_jaccard": group.get("mean_support_jaccard"),
                "mean_factor_terminal_ks": group.get("mean_factor_terminal_ks"),
                "mean_portfolio_terminal_ks": group.get("mean_portfolio_terminal_ks"),
                "mean_path_energy": group.get("mean_path_energy"),
                "spx_terminal_mean_range": ranges.get("SPX"),
                "vix_terminal_mean_range": ranges.get("VIX"),
            }
        )
    return {
        "status": str(decision.get("status", "")),
        "start_index": report.get("start_index"),
        "case_count": int(report.get("case_count", 0) or 0),
        "samples": int(report.get("samples", 0) or 0),
        "professional_minus_start_only": decision.get(
            "professional_minus_start_only", {}
        ),
        "professional_minus_simple": decision.get("professional_minus_simple", {}),
        "rows": rows,
        "artifact_paths": report.get("artifact_paths", {}),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    snapshot = summary["validation_snapshot"]
    live_snapshot = summary.get("live_casebook_snapshot")
    caption_audit = summary.get("fixed_start_caption_audit_snapshot")
    lines = [
        "# Narrative-Conditioned Scenario Generator Evidence Pack",
        "",
        "## Product Claim",
        "",
        (
            "A risk manager can enter a market narrative and provide a joint39 "
            "starting level. "
            "The system converts the narrative into grounded current/recent "
            "market implications, forms a narrative-compatible historical "
            "support mixture under that fixed start, and rolls the frozen joint39 generator "
            "forward with calibrated sampling."
        ),
        "",
        "## Workflow",
        "",
        "1. Parse the narrative into conditioning implications and non-conditioning forward-risk warnings.",
        "2. Fix the initial joint39 level before building the 30-day prefix mixture.",
        "3. Build a direction-checked support mixture conditioned on the professional narrative memory and fixed start.",
        "4. Decode the recent-prefix object and run the frozen SNI generator autoregressively.",
        "5. Show scenario fans, analogue support, IV-cell views, pass/warning/fail gates, and distributional metrics.",
        "",
        "## Current Evidence",
        "",
        f"- Validation report: `{summary['validation_report']}`",
        f"- Case set: `{snapshot['case_set']}`",
        f"- Narrative families: `{snapshot['narrative_family_count']}` "
        f"({', '.join(snapshot['narrative_families']) or 'n/a'})",
        f"- Fixed starts: `{snapshot['fixed_start_count']}` "
        f"({', '.join(snapshot['fixed_starts']) or 'n/a'})",
        f"- Runs: `{snapshot['run_count']}`",
        f"- Best variant: `{snapshot['best_variant']}`",
        f"- Status counts: `{json.dumps(snapshot['status_counts'], sort_keys=True)}`",
        f"- Rows improving CRPS vs persistence: `{snapshot['improved_crps_rows']}/{snapshot['run_count']}`",
        f"- Rows improving energy vs persistence: `{snapshot['improved_energy_rows']}/{snapshot['run_count']}`",
        f"- Mean energy z: `{_fmt_float(snapshot['mean_energy_score_z'])}`",
        f"- Mean CRPS z: `{_fmt_float(snapshot['mean_ensemble_crps_z'])}`",
        f"- Mean energy improvement vs persistence: `{_fmt_pct(snapshot['mean_energy_improvement_vs_persistence'])}`",
        f"- Mean CRPS improvement vs persistence: `{_fmt_pct(snapshot['mean_crps_improvement_vs_persistence'])}`",
        "",
    ]
    if isinstance(live_snapshot, dict):
        lines.extend(
            [
                "## Live Gradio API Casebook",
                "",
                f"- Live casebook report: `{summary.get('live_casebook_report', '')}`",
                f"- Status: `{live_snapshot['status']}`",
                f"- Cases: `{live_snapshot['case_count']}`",
                f"- Pass count: `{live_snapshot['pass_count']}`",
                f"- Total OpenAI tokens: `{live_snapshot['total_openai_tokens']}`",
                f"- Min support candidates: `{live_snapshot['min_support_candidate_count']}`",
                f"- Grounding models: `{', '.join(live_snapshot['grounding_models']) or 'n/a'}`",
                f"- Embedding models: `{', '.join(live_snapshot['embedding_models']) or 'n/a'}`",
                "",
                "| Case | Start | Status | Condition | Warnings | Support | Summary |",
                "| --- | ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in live_snapshot["case_rows"]:
            lines.append(
                "| "
                f"{row['case_name']} | "
                f"{row['expected_start_index']} | "
                f"{row['overall_status']} | "
                f"{row['condition_only_validation_status']} | "
                f"{row['forward_warning_count']} | "
                f"{row['support_candidate_count']} | "
                f"`{row['summary_path']}` |"
            )
        lines.append("")
    if isinstance(caption_audit, dict):
        prof_vs_null = caption_audit.get("professional_minus_start_only", {})
        prof_vs_simple = caption_audit.get("professional_minus_simple", {})
        lines.extend(
            [
                "## Full-Corpus Fixed-Start Conditionality Audit",
                "",
                f"- Audit report: `{summary.get('fixed_start_caption_audit_report', '')}`",
                f"- Status: `{caption_audit.get('status', '')}`",
                f"- Shared start index: `{caption_audit.get('start_index', '')}`",
                f"- Cases: `{caption_audit.get('case_count', 0)}`",
                f"- Samples per condition: `{caption_audit.get('samples', 0)}`",
                f"- Professional vs start-only factor KS delta: `{_fmt_float(prof_vs_null.get('factor_terminal_ks'))}`",
                f"- Professional vs start-only portfolio KS delta: `{_fmt_float(prof_vs_null.get('portfolio_terminal_ks'))}`",
                f"- Professional vs simple fact-token path-energy delta: `{_fmt_float(prof_vs_simple.get('path_energy'))}`",
                "",
                "| Condition | Support Jaccard | Factor KS | Portfolio KS | Path energy | SPX range | VIX range |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in caption_audit.get("rows", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                f"{row.get('condition', '')} | "
                f"{_fmt_float(row.get('mean_support_jaccard'))} | "
                f"{_fmt_float(row.get('mean_factor_terminal_ks'))} | "
                f"{_fmt_float(row.get('mean_portfolio_terminal_ks'))} | "
                f"{_fmt_float(row.get('mean_path_energy'))} | "
                f"{_fmt_float(row.get('spx_terminal_mean_range'))} | "
                f"{_fmt_float(row.get('vix_terminal_mean_range'))} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Warning Semantics",
            "",
            (
                "`pass` means the run is supported and calibrated under current "
                "gates. `warning` means the run can still be useful, but the UI "
                "must show support/shift caveats. In the current "
                "validation, warning rows can still improve CRPS versus persistence, "
                "so warning is a trust caveat rather than an automatic scenario "
                "metric failure."
            ),
            "",
            "## Demo Talking Points",
            "",
            "- This is not an LLM inventing future paths; the LLM only grounds the narrative into conditioning language.",
            "- The numerical scenario paths come from the frozen joint39 generator and calibrated rollout.",
            "- Historical analogues are support/provenance for the recent prefix, not a single nearest-neighbor replay.",
            "- The risk manager provides the starting level; the support mixture is rebuilt after that start is fixed.",
            "- The demo should show the narrative, extracted implications, warnings, analogue weights, fan charts, selected IV cells, and JSON report.",
            "",
            "## Next Validation Step",
            "",
            (
                "Use the live casebook as the boss-demo path, then keep scaling "
                "narratives and fixed starts while preserving the same "
                "condition-only warning semantics and support-weight provenance."
            ),
        ]
    )
    return "\n".join(lines)


def build_demo_pack(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _load_json(args.validation_report)
    summary = {
        "status": "ok",
        "scope_note": (
            "Boss-ready evidence pack built from existing calibrated validation "
            "artifacts. No OpenAI calls or model rollouts are made."
        ),
        "validation_report": str(args.validation_report),
        "validation_snapshot": validation_snapshot(validation),
        "artifact_paths": {
            "summary_json": str(output_dir / "boss_demo_pack.json"),
            "summary_markdown": str(output_dir / "boss_demo_pack.md"),
        },
    }
    live_casebook_report = str(getattr(args, "live_casebook_report", "") or "").strip()
    if live_casebook_report:
        live_casebook = _load_json(live_casebook_report)
        summary["live_casebook_report"] = live_casebook_report
        summary["live_casebook_snapshot"] = live_casebook_snapshot(live_casebook)
    fixed_start_caption_audit_report = str(
        getattr(args, "fixed_start_caption_audit_report", "") or ""
    ).strip()
    if fixed_start_caption_audit_report and Path(fixed_start_caption_audit_report).exists():
        caption_audit = _load_json(fixed_start_caption_audit_report)
        summary["fixed_start_caption_audit_report"] = fixed_start_caption_audit_report
        summary["fixed_start_caption_audit_snapshot"] = (
            fixed_start_caption_audit_snapshot(caption_audit)
        )
    _write_json(summary["artifact_paths"]["summary_json"], summary)
    Path(summary["artifact_paths"]["summary_markdown"]).write_text(
        render_markdown(summary).rstrip() + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--live-casebook-report", default=DEFAULT_LIVE_CASEBOOK_REPORT)
    parser.add_argument(
        "--fixed-start-caption-audit-report",
        default=DEFAULT_FIXED_START_CAPTION_AUDIT_REPORT,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_demo_pack(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary_json": summary["artifact_paths"]["summary_json"],
                "summary_markdown": summary["artifact_paths"]["summary_markdown"],
                "mean_crps_improvement_vs_persistence": summary["validation_snapshot"][
                    "mean_crps_improvement_vs_persistence"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
