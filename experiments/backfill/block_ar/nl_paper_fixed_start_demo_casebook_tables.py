#!/usr/bin/env python
"""Build demo-style fixed-start casebook tables for the NL scenario paper.

This script is artifact-only. It reads saved professional fixed-start
top3/90 reports, attaches the matching start-only baseline reports, and writes
appendix-ready LaTeX tables in the same baseline-vs-narrative format used by
the Gradio demo. It does not call OpenAI, train, or rerun the SNI generator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    apply_live_top3_90_posterior_ensemble,
    attach_start_only_baseline_report,
    scenario_table,
)

DEFAULT_CONTROL_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "posterior_ensemble_candidate_966a_professional_start22_s384_d400"
)
DEFAULT_TABLE_PATH = Path(
    "paper/narrative_grounded_scenarios/generated_tables/"
    "table_fixed_start_demo_casebook_readout.tex"
)
DEFAULT_BEAMER_TABLE_PATH = Path(
    "paper/narrative_grounded_scenarios/generated_tables/"
    "table_fixed_start_demo_casebook_readout_beamer.tex"
)
DEFAULT_SUMMARY_PATH = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "fixed_start_demo_casebook_readout_summary.json"
)
DEFAULT_VARIANT_DIR = "cohesive_support_gap30"
DEFAULT_BASELINE_DIR = "start_only_topk"

CASES = [
    ("fragile_risk_on", "Fragile risk-on rebound"),
    ("defensive_risk_off", "Defensive risk-off shock"),
    ("commodity_inflation", "Commodity inflation pressure"),
    ("dollar_liquidity", "Dollar liquidity squeeze"),
    ("rates_selloff", "Rates-led tightening fear"),
    ("safe_haven_gold", "Safe-haven gold bid"),
]

MARKET_LABELS = {
    "IV_SURFACE": "IV surface",
    "SPX": "SPX",
    "US2Y": "US2Y",
    "US10Y": "US10Y",
    "BBB_OAS": "BBB OAS",
    "AAA_OAS": "AAA OAS",
    "USDJPY": "USDJPY",
    "DXY": "DXY",
    "GOLD": "Gold",
    "CRUDE_OIL": "Crude oil",
    "VIX": "VIX",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _case_slug(label: str) -> str:
    return (
        str(label)
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _view_tex(value: Any) -> str:
    text = str(value or "").strip()
    if text == "Up":
        return r"\DemoViewUp"
    if text == "Down":
        return r"\DemoViewDown"
    if text in {"-", "Flat/mixed"}:
        return r"\DemoViewFlat"
    return _tex_escape(text or "n/a")


def _change_tex(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "n/a":
        return "n/a"
    if text == "Similar to baseline":
        return r"\DemoChangeFlat{Similar}"
    if text in {"More up than baseline", "Higher than baseline"}:
        return rf"\DemoChangeUp{{{_tex_escape(text)}}}"
    if text in {"More down than baseline", "Lower than baseline"}:
        return rf"\DemoChangeDown{{{_tex_escape(text)}}}"
    if text == "Less down than baseline":
        return rf"\DemoChangeModerateDown{{{_tex_escape(text)}}}"
    if text == "Less up than baseline":
        return rf"\DemoChangeModerateUp{{{_tex_escape(text)}}}"
    return _tex_escape(text)


def _mean_tex(value: Any) -> str:
    return _tex_escape(value).replace("σ", r"$\sigma$")


def _report_path(control_root: Path, case_name: str, leaf: str) -> Path:
    return control_root / case_name / leaf / "prefix_latent_story_smoke_report.json"


def _load_case_bundle(
    *,
    control_root: Path,
    variant_dir: str,
    baseline_dir: str,
    case_name: str,
) -> dict[str, Any]:
    report_path = _report_path(control_root, case_name, variant_dir)
    baseline_path = _report_path(control_root, case_name, baseline_dir)
    report = apply_live_top3_90_posterior_ensemble(_load_json(report_path))
    baseline = apply_live_top3_90_posterior_ensemble(_load_json(baseline_path))
    attached = attach_start_only_baseline_report(
        report,
        baseline,
        memory_prior_mode="soft_topk_start_only",
    )
    return {
        "report_path": str(report_path),
        "baseline_report_path": str(baseline_path),
        "report": attached,
    }


def _support_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    posterior = report.get("generation", {}).get("posterior_ensemble", {})
    if not isinstance(posterior, dict):
        return []
    rows: list[dict[str, Any]] = []
    for item in posterior.get("selected_support", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "rank": int(item.get("rank", len(rows) + 1)),
                "window_id": str(item.get("window_id", "")),
                "history_end_date": str(item.get("history_end_date", "")),
                "posterior_weight": float(
                    item.get("posterior_weight", item.get("weight", 0.0)) or 0.0
                ),
                "story_match": float(item.get("memory_support_cosine", 0.0) or 0.0),
                "start_gap_z": float(item.get("start_distance_z", 0.0) or 0.0),
                "required_claims": str(
                    item.get("recent_prefix_alignment_status", "")
                ),
            }
        )
    return rows


def _table_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    df = scenario_table(report)
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        market = str(row.get("Market", "")).strip()
        rows.append(
            {
                "Market": MARKET_LABELS.get(market, market.replace("_", " ")),
                "Baseline View": _view_tex(row.get("Baseline View", "")),
                "Baseline Path Count": _tex_escape(row.get("Baseline Path Count", "")),
                "Baseline Mean Move": _mean_tex(row.get("Baseline Mean Move", "")),
                "Narrative View": _view_tex(row.get("Narrative View", "")),
                "Narrative Path Count": _tex_escape(
                    row.get("Narrative Path Count", "")
                ),
                "Narrative Mean Move": _mean_tex(row.get("Narrative Mean Move", "")),
                "30d Change vs Baseline": _change_tex(
                    row.get("30d Change vs Baseline", "")
                ),
            }
        )
    return rows


def _case_table_tex(label: str, rows: list[dict[str, str]]) -> str:
    slug = _case_slug(label)
    body = "\n".join(
        " & ".join(
            [
                _tex_escape(row["Market"]),
                row["Baseline View"],
                row["Baseline Path Count"],
                row["Baseline Mean Move"],
                row["Narrative View"],
                row["Narrative Path Count"],
                row["Narrative Mean Move"],
                row["30d Change vs Baseline"],
            ]
        )
        + r" \\"
        for row in rows
    )
    return rf"""
\begin{{table}}[H]
\centering
\caption{{Fixed-start demo-style scenario readout for { _tex_escape(label) }. The start-only baseline uses the same accepted start but omits the narrative condition. Views compare the day-30 terminal distribution with the accepted starting level; path counts report the dominant terminal sign share; mean moves report raw market units and standardized size.}}
\label{{tab:fixed_start_demo_casebook_{slug}}}
\begingroup
\tiny
\setlength{{\tabcolsep}}{{2.2pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llllllll}}
\toprule
Market & \shortstack{{Baseline\\View}} & \shortstack{{Baseline\\Path\\Count}} & \shortstack{{Baseline\\Mean\\Move}} & \shortstack{{Narrative\\View}} & \shortstack{{Narrative\\Path\\Count}} & \shortstack{{Narrative\\Mean\\Move}} & \shortstack{{30d Change\\vs Baseline}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\endgroup
\end{{table}}
""".strip()


def _case_beamer_frame_tex(label: str, rows: list[dict[str, str]]) -> str:
    body = "\n".join(
        " & ".join(
            [
                _tex_escape(row["Market"]),
                row["Baseline View"],
                row["Baseline Path Count"],
                row["Baseline Mean Move"],
                row["Narrative View"],
                row["Narrative Path Count"],
                row["Narrative Mean Move"],
                row["30d Change vs Baseline"],
            ]
        )
        + r" \\"
        for row in rows
    )
    return rf"""
\begin{{frame}}{{A3. { _tex_escape(label) } readout}}
\tiny
\begin{{adjustbox}}{{max width=\textwidth,max totalheight=0.78\textheight}}
\begin{{tabular}}{{llllllll}}
\toprule
Market & \shortstack{{Baseline\\View}} & \shortstack{{Baseline\\Path\\Count}} & \shortstack{{Baseline\\Mean\\Move}} & \shortstack{{Narrative\\View}} & \shortstack{{Narrative\\Path\\Count}} & \shortstack{{Narrative\\Mean\\Move}} & \shortstack{{30d Change\\vs Baseline}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{frame}}
""".strip()


def build_fixed_start_demo_casebook_tables(
    *,
    control_root: str | Path = DEFAULT_CONTROL_ROOT,
    table_path: str | Path = DEFAULT_TABLE_PATH,
    beamer_table_path: str | Path = DEFAULT_BEAMER_TABLE_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    variant_dir: str = DEFAULT_VARIANT_DIR,
    baseline_dir: str = DEFAULT_BASELINE_DIR,
) -> dict[str, Any]:
    control_root = Path(control_root)
    cases: list[dict[str, Any]] = []
    table_parts = [
        "% Auto-generated by experiments/backfill/block_ar/"
        "nl_paper_fixed_start_demo_casebook_tables.py",
        "% Do not edit by hand; rerun the script after refreshing casebook artifacts.",
    ]
    beamer_parts = [
        "% Auto-generated by experiments/backfill/block_ar/"
        "nl_paper_fixed_start_demo_casebook_tables.py",
        "% Do not edit by hand; rerun the script after refreshing casebook artifacts.",
    ]
    for case_name, label in CASES:
        bundle = _load_case_bundle(
            control_root=control_root,
            variant_dir=variant_dir,
            baseline_dir=baseline_dir,
            case_name=case_name,
        )
        report = bundle["report"]
        rows = _table_rows(report)
        table_parts.append(_case_table_tex(label, rows))
        beamer_parts.append(_case_beamer_frame_tex(label, rows))
        posterior = report.get("generation", {}).get("posterior_ensemble", {})
        cases.append(
            {
                "case_name": case_name,
                "label": label,
                "report_path": bundle["report_path"],
                "baseline_report_path": bundle["baseline_report_path"],
                "posterior_mode": (
                    str(posterior.get("mode", "")) if isinstance(posterior, dict) else ""
                ),
                "posterior_sample_count": (
                    int(posterior.get("posterior_sample_count", 0) or 0)
                    if isinstance(posterior, dict)
                    else 0
                ),
                "selected_support": _support_summary(report),
                "table_rows": rows,
            }
        )

    table_text = "\n\n".join(table_parts)
    _write_text(table_path, table_text)
    beamer_text = "\n\n".join(beamer_parts)
    _write_text(beamer_table_path, beamer_text)
    summary = {
        "status": "ok",
        "control_root": str(control_root),
        "variant_dir": str(variant_dir),
        "baseline_dir": str(baseline_dir),
        "table_path": str(table_path),
        "beamer_table_path": str(beamer_table_path),
        "case_count": len(cases),
        "cases": cases,
        "scope_note": (
            "Fixed-start appendix readout generated from saved professional "
            "casebook reports. The scenario tables use the same top3/90 "
            "posterior view and start-only baseline attachment as the Gradio "
            "demo summary table."
        ),
    }
    _write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", default=str(DEFAULT_CONTROL_ROOT))
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--table-path", default=str(DEFAULT_TABLE_PATH))
    parser.add_argument("--beamer-table-path", default=str(DEFAULT_BEAMER_TABLE_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_fixed_start_demo_casebook_tables(
        control_root=args.control_root,
        table_path=args.table_path,
        beamer_table_path=args.beamer_table_path,
        summary_path=args.summary_path,
        variant_dir=args.variant_dir,
        baseline_dir=args.baseline_dir,
    )
    print(json.dumps({"status": summary["status"], "table_path": summary["table_path"], "beamer_table_path": summary["beamer_table_path"], "case_count": summary["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
