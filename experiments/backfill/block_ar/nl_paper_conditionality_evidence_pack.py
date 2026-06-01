#!/usr/bin/env python
"""Build paper-facing fixed-start conditionality evidence artifacts.

This script is intentionally artifact-only.  It consolidates the current
fixed-start conditionality benchmark, transmission audit, qualitative casebook,
and portfolio-impact readout into one figure and two LaTeX tables for the
narrative-grounded scenario paper.  It does not call OpenAI, train a model, or
rerun the frozen SNI generator.
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


DEFAULT_BENCHMARK_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_strength_benchmark_917f_balanced80_caption_quality/"
    "conditionality_strength_benchmark.json"
)
DEFAULT_TRANSMISSION_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_transmission_audit_918a_balanced80/"
    "conditionality_transmission_audit.json"
)
DEFAULT_CASEBOOK_SUMMARY = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_qualitative_casebook_summary.json"
)
DEFAULT_PORTFOLIO_SUMMARY = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_portfolio_impact_summary.json"
)
DEFAULT_FIGURE_DIR = Path("paper/narrative_grounded_scenarios/figures")
DEFAULT_TABLE_DIR = Path("paper/narrative_grounded_scenarios/generated_tables")
DEFAULT_SUMMARY_JSON = (
    DEFAULT_FIGURE_DIR / "narrative_conditionality_evidence_pack_summary.json"
)
DEFAULT_SUMMARY_MD = (
    DEFAULT_FIGURE_DIR / "narrative_conditionality_evidence_pack_summary.md"
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
        return value if math.isfinite(value) else str(value)
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


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


def _summary_value(summary: dict[str, Any], metric: str, stat: str = "median") -> float | None:
    value = summary.get(f"{metric}_{stat}")
    if value is None:
        return None
    return float(value)


def _support_labels(rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        for key in ("left_label", "right_label"):
            label = str(row[key])
            if label not in labels:
                labels.append(label)
    return labels


def _support_matrix(rows: list[dict[str, Any]], labels: list[str]) -> np.ndarray:
    matrix = np.eye(len(labels), dtype=np.float64)
    index = {label: i for i, label in enumerate(labels)}
    for row in rows:
        i = index[str(row["left_label"])]
        j = index[str(row["right_label"])]
        matrix[i, j] = matrix[j, i] = float(row["jaccard"])
    return matrix


def _case_maps(
    casebook: dict[str, Any],
    portfolio: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = {
        str(row["label"]): row
        for row in casebook.get("case_summaries", [])
        if isinstance(row, dict)
    }
    ports = {
        str(row["label"]): row
        for row in portfolio.get("case_summaries", [])
        if isinstance(row, dict)
    }
    return cases, ports


def _public_support_label(item: dict[str, Any]) -> str:
    date = str(item.get("history_end_date", "")).strip()
    try:
        weight = float(item.get("weight", 0.0))
    except (TypeError, ValueError):
        weight = 0.0
    if date:
        return f"historical support ending {date} ({weight:.0%})"
    return f"historical support ({weight:.0%})"


def _portfolio_case_rows(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in portfolio.get("case_summaries", []):
        stats = row.get("portfolio_stats", {})
        top_support = row.get("top_support", [])
        support = "; ".join(
            _public_support_label(item)
            for item in top_support[:2]
            if isinstance(item, dict)
        )
        rows.append(
            {
                "label": str(row.get("label", "")),
                "terminal_p50": float(stats.get("terminal_p50", 0.0)),
                "terminal_p10": float(stats.get("terminal_p10", 0.0)),
                "terminal_p90": float(stats.get("terminal_p90", 0.0)),
                "var95_loss": float(stats.get("terminal_var95_loss", 0.0)),
                "es95_loss": float(stats.get("terminal_expected_shortfall95_loss", 0.0)),
                "top_support": support,
            }
        )
    return rows


def _factor_change_matrix(casebook: dict[str, Any]) -> tuple[list[str], list[str], np.ndarray]:
    factors = ["SPX", "VIX", "BBB OAS", "US10Y", "DXY", "Gold", "Crude", "1Y ATM IV"]
    labels: list[str] = []
    raw = []
    for row in casebook.get("case_summaries", []):
        labels.append(str(row["label"]))
        terminal = row.get("terminal_raw_level_summary", {})
        raw.append(
            [
                float(terminal.get(factor, {}).get("median_change_from_start", 0.0))
                for factor in factors
            ]
        )
    arr = np.asarray(raw, dtype=np.float64)
    denom = np.maximum(np.max(np.abs(arr), axis=0, keepdims=True), 1e-12)
    return labels, factors, arr / denom


def _plot_evidence_pack(report: dict[str, Any], output: str | Path) -> None:
    support_rows = report["benchmark_report"]["support_overlap_rows"]
    labels = _support_labels(support_rows)
    support_matrix = _support_matrix(support_rows, labels)
    case_labels, factors, factor_matrix = _factor_change_matrix(report["casebook_summary"])
    portfolio_rows = _portfolio_case_rows(report["portfolio_summary"])

    transmission = report["transmission_report"]
    summaries = transmission["summaries"]
    observed = summaries["observed_cross_narrative"]
    repeat = summaries["same_narrative_repeat"]
    bootstrap = summaries["within_run_bootstrap"]

    portfolio = report["benchmark_report"]["portfolio_audit_summary"]["summaries"]
    portfolio_obs = portfolio["observed_cross_narrative"]
    portfolio_repeat = portfolio["same_narrative_repeat"]
    portfolio_bootstrap = portfolio["within_run_bootstrap"]

    metrics = [
        (
            "Rollout\nenergy",
            _summary_value(observed, "rollout_path_energy_distance"),
            _summary_value(repeat, "rollout_path_energy_distance"),
            _summary_value(bootstrap, "rollout_path_energy_distance"),
        ),
        (
            "Portfolio\npath",
            portfolio_obs["median_path_wasserstein"],
            portfolio_repeat["median_path_wasserstein"],
            portfolio_bootstrap["median_path_wasserstein"],
        ),
        (
            "Portfolio\nterminal",
            portfolio_obs["median_terminal_wasserstein"],
            portfolio_repeat["median_terminal_wasserstein"],
            portfolio_bootstrap["median_terminal_wasserstein"],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Fixed-start narrative conditionality evidence pack",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    image = ax.imshow(support_matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_title("A. Distinct historical support by narrative")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{support_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.72, label="Jaccard overlap")

    ax = axes[0, 1]
    x = np.arange(len(metrics))
    width = 0.26
    observed_vals = np.ones(len(metrics), dtype=np.float64)
    repeat_vals = np.asarray([m[2] / m[1] if m[1] else 0.0 for m in metrics], dtype=np.float64)
    boot_vals = np.asarray([m[3] / m[1] if m[1] else 0.0 for m in metrics], dtype=np.float64)
    ax.bar(x - width, observed_vals, width, color="#1565C0", label="Observed narrative effect")
    ax.bar(x, repeat_vals, width, color="#78909C", label="Same-narrative repeat / observed")
    ax.bar(x + width, boot_vals, width, color="#EF6C00", label="Bootstrap / observed")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [m[0] for m in metrics])
    ax.set_ylim(0, max(1.35, float(max(repeat_vals.max(), boot_vals.max(), 1.0)) + 0.15))
    ax.set_ylabel("Ratio to observed cross-narrative effect")
    ax.set_title("B. Narrative effect versus controls")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1, 0]
    names = [row["label"] for row in portfolio_rows]
    p50 = np.asarray([row["terminal_p50"] for row in portfolio_rows], dtype=np.float64)
    p10 = np.asarray([row["terminal_p10"] for row in portfolio_rows], dtype=np.float64)
    p90 = np.asarray([row["terminal_p90"] for row in portfolio_rows], dtype=np.float64)
    y = np.arange(len(names))
    ax.hlines(y, p10, p90, color="#90A4AE", linewidth=4, label="10-90% terminal range")
    ax.scatter(p50, y, color="#1565C0", s=42, zorder=3, label="Median")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Exposure-weighted terminal risk units")
    ax.set_title("C. Portfolio-impact readout by narrative")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.18)

    ax = axes[1, 1]
    heat = ax.imshow(factor_matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    ax.set_title("D. Terminal median market moves by narrative")
    ax.set_xticks(range(len(factors)), factors, rotation=35, ha="right")
    ax.set_yticks(range(len(case_labels)), case_labels)
    fig.colorbar(heat, ax=ax, shrink=0.72, label="Change from start, normalized by factor")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _conditionality_table(report: dict[str, Any]) -> str:
    benchmark = report["benchmark_report"]
    transmission = report["transmission_report"]
    support = benchmark["support_overlap_summary"]
    summaries = transmission["summaries"]
    observed = summaries["observed_cross_narrative"]
    repeat = summaries["same_narrative_repeat"]
    bootstrap = summaries["within_run_bootstrap"]
    portfolio = benchmark["portfolio_audit_summary"]["summaries"]
    p_obs = portfolio["observed_cross_narrative"]
    p_repeat = portfolio["same_narrative_repeat"]
    p_boot = portfolio["within_run_bootstrap"]
    p_case = report["portfolio_summary"]["cross_narrative_range"]

    rows = [
        (
            "Support provenance",
            f"median Jaccard {_fmt(support['median_jaccard'])}; max {_fmt(support['max_jaccard'])}",
            "15 pairwise narrative contrasts",
            "Distinct historical support pools",
        ),
        (
            "Decoded prefix",
            f"RMSE {_fmt(_summary_value(observed, 'decoded_prefix_norm_rmse'))}",
            f"repeat {_fmt(_summary_value(repeat, 'decoded_prefix_norm_rmse'))}; start-only 0.000",
            "Narrative changes the recent-condition object",
        ),
        (
            "Generated path family",
            f"energy {_fmt(_summary_value(observed, 'rollout_path_energy_distance'))}",
            (
                f"repeat {_fmt(_summary_value(repeat, 'rollout_path_energy_distance'))}; "
                f"bootstrap {_fmt(_summary_value(bootstrap, 'rollout_path_energy_distance'))}"
            ),
            "Narrative effect exceeds same-narrative repeat",
        ),
        (
            "Portfolio path distribution",
            f"Wasserstein {_fmt(p_obs['median_path_wasserstein'])}",
            (
                f"repeat {_fmt(p_repeat['median_path_wasserstein'])}; "
                f"bootstrap {_fmt(p_boot['median_path_wasserstein'])}"
            ),
            "Observed path spread is 2.44x repeat",
        ),
        (
            "Portfolio casebook spread",
            (
                f"median range {_fmt(p_case['terminal_p50_range'])}; "
                f"95% loss range {_fmt(p_case['var95_loss_range'])}"
            ),
            "same start across all narratives",
            "Risk readout varies in business units",
        ),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fixed-start narrative conditionality evidence. All rows use the same accepted starting level; only the current-market narrative changes.}",
        r"\label{tab:conditionality_evidence_pack}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}p{0.21\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Layer & Observed narrative effect & Control or scope & Product interpretation \\",
        r"\midrule",
    ]
    for layer, observed_text, control, interpretation in rows:
        lines.append(
            f"{_tex_escape(layer)} & {_tex_escape(observed_text)} & "
            f"{_tex_escape(control)} & {_tex_escape(interpretation)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(lines)


def _portfolio_table(report: dict[str, Any]) -> str:
    rows = _portfolio_case_rows(report["portfolio_summary"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Illustrative portfolio-impact readout at the same accepted start. Values are exposure-weighted normalized risk units. Higher 95\% loss and ES-style loss indicate a worse downside tail.}",
        r"\label{tab:portfolio_casebook}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}Xrrrr>{\raggedright\arraybackslash}p{0.31\linewidth}}",
        r"\toprule",
        r"Narrative & Median & 10th pct. & 95\% loss & ES-style loss & Main historical support \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row['label'])} & "
            f"{_fmt(row['terminal_p50'], 2)} & "
            f"{_fmt(row['terminal_p10'], 2)} & "
            f"{_fmt(row['var95_loss'], 2)} & "
            f"{_fmt(row['es95_loss'], 2)} & "
            f"{_tex_escape(row['top_support'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    metrics = report["headline_metrics"]
    return "\n".join(
        [
            "# Fixed-Start Narrative Conditionality Evidence Pack",
            "",
            "This artifact consolidates current paper-facing conditionality evidence.",
            "It is artifact-only: no OpenAI calls, training, or new generator rollout.",
            "",
            "## Headline",
            "",
            (
                "- Different professional narratives at the same start produce "
                "different support pools, decoded-prefix objects, path families, "
                "and portfolio-impact readouts."
            ),
            (
                f"- Median support Jaccard is `{float(metrics['support_median_jaccard']):.3f}` "
                f"across `{int(metrics['support_pair_count'])}` pairwise narrative contrasts."
            ),
            (
                "- Portfolio path Wasserstein observed/repeat ratio is "
                f"`{float(metrics['portfolio_path_vs_repeat']):.3f}`."
            ),
            (
                "- Terminal median portfolio range is "
                f"`{float(metrics['terminal_p50_range']):.3f}` risk units; "
                f"95% loss range is `{float(metrics['var95_loss_range']):.3f}`."
            ),
            "",
            "## Outputs",
            "",
        ]
        + [f"- {key}: `{value}`" for key, value in report["artifact_paths"].items()]
        + [""]
    )


def build_evidence_pack(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = _load_json(args.benchmark_report)
    transmission = _load_json(args.transmission_report)
    casebook = _load_json(args.casebook_summary)
    portfolio = _load_json(args.portfolio_summary)

    figure_path = Path(args.figure_dir) / "narrative_conditionality_evidence_pack.png"
    cond_table_path = Path(args.table_dir) / "table_fixed_start_conditionality_evidence.tex"
    portfolio_table_path = Path(args.table_dir) / "table_portfolio_casebook_readout.tex"

    report = {
        "scope_note": (
            "Paper-facing conditionality evidence package. Consolidates existing "
            "fixed-start support, path, and portfolio artifacts without new model calls."
        ),
        "benchmark_report_path": str(args.benchmark_report),
        "transmission_report_path": str(args.transmission_report),
        "casebook_summary_path": str(args.casebook_summary),
        "portfolio_summary_path": str(args.portfolio_summary),
        "benchmark_report": benchmark,
        "transmission_report": transmission,
        "casebook_summary": casebook,
        "portfolio_summary": portfolio,
        "artifact_paths": {
            "evidence_figure": str(figure_path),
            "conditionality_table": str(cond_table_path),
            "portfolio_casebook_table": str(portfolio_table_path),
            "summary_json": str(args.summary_json),
            "summary_markdown": str(args.summary_markdown),
        },
    }

    _plot_evidence_pack(report, figure_path)
    _write_text(cond_table_path, _conditionality_table(report))
    _write_text(portfolio_table_path, _portfolio_table(report))

    slim = {
        "scope_note": report["scope_note"],
        "benchmark_report_path": report["benchmark_report_path"],
        "transmission_report_path": report["transmission_report_path"],
        "casebook_summary_path": report["casebook_summary_path"],
        "portfolio_summary_path": report["portfolio_summary_path"],
        "artifact_paths": report["artifact_paths"],
        "headline_metrics": {
            "support_pair_count": benchmark["support_overlap_summary"]["pair_count"],
            "support_median_jaccard": benchmark["support_overlap_summary"]["median_jaccard"],
            "support_max_jaccard": benchmark["support_overlap_summary"]["max_jaccard"],
            "decoded_prefix_rmse_median": transmission["summaries"]["observed_cross_narrative"][
                "decoded_prefix_norm_rmse_median"
            ],
            "rollout_path_energy_median": transmission["summaries"]["observed_cross_narrative"][
                "rollout_path_energy_distance_median"
            ],
            "same_narrative_repeat_energy_median": transmission["summaries"][
                "same_narrative_repeat"
            ]["rollout_path_energy_distance_median"],
            "portfolio_path_vs_repeat": benchmark["decision"]["key_ratios"][
                "portfolio_path_vs_repeat"
            ],
            "portfolio_path_vs_bootstrap": benchmark["decision"]["key_ratios"][
                "portfolio_path_vs_bootstrap"
            ],
            "terminal_p50_range": portfolio["cross_narrative_range"][
                "terminal_p50_range"
            ],
            "var95_loss_range": portfolio["cross_narrative_range"]["var95_loss_range"],
        },
    }
    _write_json(args.summary_json, slim)
    _write_text(args.summary_markdown, _markdown_summary(slim))
    return slim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", type=Path, default=DEFAULT_BENCHMARK_REPORT)
    parser.add_argument("--transmission-report", type=Path, default=DEFAULT_TRANSMISSION_REPORT)
    parser.add_argument("--casebook-summary", type=Path, default=DEFAULT_CASEBOOK_SUMMARY)
    parser.add_argument("--portfolio-summary", type=Path, default=DEFAULT_PORTFOLIO_SUMMARY)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-markdown", type=Path, default=DEFAULT_SUMMARY_MD)
    args = parser.parse_args()
    report = build_evidence_pack(args)
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
