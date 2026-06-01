#!/usr/bin/env python
"""Build a public-facing narrative-caption appendix example.

This script is artifact-only. It selects one historical caption case, plots raw
SPX and VIX levels over the 30-day conditioning window plus the realized next
30 trading days, and writes a LaTeX appendix snippet comparing the earlier
simple caption with the professional risk-manager narrative.
"""

from __future__ import annotations

import argparse
import json
import re
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

from experiments.backfill.block_ar.nl_caption_conditionality_refresh import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR as DEFAULT_CAPTION_REFRESH_DIR,
    load_history_future,
)

DEFAULT_PIPELINE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_full_906b_all_windows/narrative_pipeline_report.json"
)
DEFAULT_CODEX_CAPTION_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_codex_batch_917e_balanced80/captions"
)
DEFAULT_FIGURE_DIR = Path("paper/narrative_grounded_scenarios/figures")
DEFAULT_GENERATED_DIR = Path("paper/narrative_grounded_scenarios/generated_tables")

MARKETS = [("SPX", 25), ("VIX", 38)]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _find_bundle(pipeline_report: dict[str, Any], window_id: str) -> dict[str, Any]:
    for bundle in pipeline_report.get("narrative_bundles", []):
        if isinstance(bundle, dict) and str(bundle.get("window_id")) == str(window_id):
            return bundle
    raise KeyError(f"window not found in pipeline report: {window_id}")


def _select_narrative(bundle: dict[str, Any], narrative_id: str) -> str:
    for row in bundle.get("narratives", []):
        if isinstance(row, dict) and str(row.get("id")) == str(narrative_id):
            return str(row.get("text", "")).strip()
    raise KeyError(f"narrative id not found: {narrative_id}")


def _selected_window(default_window: str, refresh_dir: Path) -> str:
    summary = refresh_dir / "caption_conditionality_refresh_summary.json"
    if not summary.exists():
        return default_window
    payload = _load_json(summary)
    return str(
        payload.get("qualitative_casebook", {})
        .get("selected_window", {})
        .get("window_id", default_window)
    )


def _caption_path(caption_dir: Path, window_id: str) -> Path:
    return caption_dir / f"codex_gpt55_caption_{window_id}.json"


def _plot_market_paths(
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    *,
    block_window_index: int,
    calendar: dict[str, Any],
    output: Path,
) -> dict[str, dict[str, float]]:
    history = np.asarray(history_raw[block_window_index], dtype=np.float64)
    future = np.asarray(future_raw[block_window_index], dtype=np.float64)
    x_hist = np.arange(-history.shape[0] + 1, 1)
    x_future = np.arange(1, future.shape[0] + 1)

    fig, axes = plt.subplots(len(MARKETS), 1, figsize=(9.5, 6.2), sharex=True)
    if len(MARKETS) == 1:
        axes = [axes]
    title = (
        "Historical caption example: raw market path around the conditioning date\n"
        f"history {calendar.get('calendar_start_date')} to {calendar.get('calendar_end_date')}; "
        f"realized future {calendar.get('forecast_start_date')} to {calendar.get('forecast_end_date')}"
    )
    fig.suptitle(title, fontsize=12.5, fontweight="bold")
    summary: dict[str, dict[str, float]] = {}
    for ax, (market, idx) in zip(axes, MARKETS, strict=True):
        hist_values = history[:, idx]
        future_values = future[:, idx]
        ax.plot(x_hist, hist_values, color="#1565C0", linewidth=2.2, label="observed history")
        ax.plot(
            np.r_[0, x_future],
            np.r_[hist_values[-1], future_values],
            color="#263238",
            linewidth=2.0,
            linestyle="--",
            label="realized next 30 days",
        )
        ax.axvline(0, color="#C62828", linewidth=1.1, alpha=0.9)
        ax.scatter([0], [hist_values[-1]], color="#C62828", s=36, zorder=4)
        ax.set_ylabel(f"{market}\nraw level")
        ax.grid(alpha=0.18)
        summary[market] = {
            "history_start": float(hist_values[0]),
            "conditioning_date": float(hist_values[-1]),
            "future_end": float(future_values[-1]),
            "history_change": float(hist_values[-1] - hist_values[0]),
            "future_change": float(future_values[-1] - hist_values[-1]),
        }
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Trading days relative to conditioning date")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def _field_item(label: str, value: Any) -> str:
    return f"\\item \\textbf{{{_tex_escape(label)}.}} {_tex_escape(value)}"


def _public_archetype(value: Any) -> str:
    return str(value).replace("_", " ").strip().title()


def _public_evidence_line(value: Any) -> str:
    text = str(value)
    def repl(match: re.Match[str]) -> str:
        number = float(match.group(1))
        return f"30-day change {number:.2f}"

    text = re.sub(r"\b[a-z0-9_]+_30d_change=([-+]?\d+(?:\.\d+)?)", repl, text)
    return text


def _tex_snippet(
    *,
    figure_name: str,
    bundle: dict[str, Any],
    codex: dict[str, Any],
    simple_text: str,
    market_summary: dict[str, dict[str, float]],
) -> str:
    calendar = bundle.get("calendar", {})
    evidence_items = codex.get("evidence_used", [])
    ambiguity_items = codex.get("ambiguity_flags", [])
    evidence_tex = "\n".join(
        f"  \\item {_tex_escape(_public_evidence_line(item))}" for item in evidence_items[:8]
    )
    ambiguity_tex = "\n".join(
        f"  \\item {_tex_escape(item)}" for item in ambiguity_items
    )
    market_rows = "\n".join(
        (
            f"{_tex_escape(market)} & {vals['history_start']:.2f} & "
            f"{vals['conditioning_date']:.2f} & {vals['future_end']:.2f} & "
            f"{vals['history_change']:+.2f} & {vals['future_change']:+.2f} \\\\"
        )
        for market, vals in market_summary.items()
    )
    return rf"""
\section{{Complete Caption Example}}

This appendix shows one held-out historical scenario used in the caption
representation study. The example is included to make the caption-quality
change concrete: the same market window can be described by a short directional
caption or by a professional risk-manager narrative with mechanism,
cross-asset context, ambiguity, and leakage controls. The realized future path
is shown only for reader context; it is not used when writing the narrative.

\begin{{figure}}[p]
\centering
\includegraphics[width=0.92\linewidth]{{figures/{figure_name}}}
\caption{{Raw SPX and VIX levels around the caption example. The solid blue
line is the 30-trading-day conditioning window; the dashed black line is the
realized next 30 trading days. The vertical red line is the conditioning date.
This plot is a reader-facing market context plot, not an input to the caption.}}
\label{{fig:caption_example_market_paths}}
\end{{figure}}

\begin{{table}}[p]
\centering
\caption{{Raw market levels for the caption example.}}
\label{{tab:caption_example_market_levels}}
\small
\begin{{tabularx}}{{0.92\linewidth}}{{Xrrrrr}}
\toprule
Market & History start & Conditioning date & Future end & History change & Future change \\
\midrule
{market_rows}
\bottomrule
\end{{tabularx}}
\end{{table}}

\paragraph{{Scenario window.}}
The conditioning window runs from {_tex_escape(calendar.get('calendar_start_date'))}
to {_tex_escape(calendar.get('calendar_end_date'))}; the realized future window
runs from {_tex_escape(calendar.get('forecast_start_date'))} to
{_tex_escape(calendar.get('forecast_end_date'))}.

\paragraph{{Earlier simple narrative.}}
\begin{{quote}}
{_tex_escape(simple_text)}
\end{{quote}}

\paragraph{{Professional risk-manager narrative.}}
\begin{{quote}}
{_tex_escape(codex.get('training_caption', ''))}
\end{{quote}}

\begin{{description}}[leftmargin=1.4em,style=nextline]
{_field_item('Scenario title', codex.get('scenario_title', ''))}
{_field_item('Archetype', f"{_public_archetype(codex.get('archetype', ''))} ({codex.get('archetype_confidence', '')} confidence)")}
{_field_item('Current market state', codex.get('current_market_state', ''))}
{_field_item('Mechanical summary', codex.get('mechanical_summary', ''))}
{_field_item('Trigger', codex.get('trigger', ''))}
{_field_item('Transmission', codex.get('transmission', ''))}
{_field_item('Cross-asset reaction', codex.get('cross_asset_reaction', ''))}
{_field_item('Sequence', codex.get('sequence', ''))}
{_field_item('Portfolio vulnerability', codex.get('portfolio_vulnerability', ''))}
{_field_item('Risk-manager implication', codex.get('risk_manager_implication', ''))}
{_field_item('No-forecast caveat', codex.get('no_forecast_caveat', ''))}
\end{{description}}

\paragraph{{Evidence used by the professional narrative.}}
\begin{{itemize}}[leftmargin=1.4em]
{evidence_tex}
\end{{itemize}}

\paragraph{{Ambiguity flags preserved by the professional narrative.}}
\begin{{itemize}}[leftmargin=1.4em]
{ambiguity_tex}
\end{{itemize}}
"""


def build_appendix(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_report = _load_json(args.pipeline_report)
    window_id = str(args.window_id or _selected_window("joint39_val_0378", args.caption_refresh_dir))
    bundle = _find_bundle(pipeline_report, window_id)
    codex = _load_json(_caption_path(args.codex_caption_dir, window_id))
    simple_text = _select_narrative(bundle, args.simple_narrative_id)
    history_raw, future_raw = load_history_future(str(args.checkpoint), device_name=str(args.device))

    block_window_index = int(bundle.get("window_index", 0))
    if args.block_window_index is not None:
        block_window_index = int(args.block_window_index)

    figure_path = args.figure_dir / "narrative_caption_example_market_paths.png"
    market_summary = _plot_market_paths(
        history_raw,
        future_raw,
        block_window_index=block_window_index,
        calendar=bundle.get("calendar", {}),
        output=figure_path,
    )
    tex_path = args.generated_dir / "appendix_caption_example.tex"
    tex = _tex_snippet(
        figure_name=figure_path.name,
        bundle=bundle,
        codex=codex,
        simple_text=simple_text,
        market_summary=market_summary,
    )
    _write_text(tex_path, tex)

    summary = {
        "window_public_label": "Late-July 2017 held-out scenario",
        "window_id": window_id,
        "block_window_index": block_window_index,
        "calendar": bundle.get("calendar", {}),
        "simple_narrative": simple_text,
        "professional_training_caption": codex.get("training_caption", ""),
        "market_summary": market_summary,
        "figure": str(figure_path),
        "tex": str(tex_path),
    }
    summary_path = args.figure_dir / "narrative_caption_example_summary.json"
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", type=Path, default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--codex-caption-dir", type=Path, default=DEFAULT_CODEX_CAPTION_DIR)
    parser.add_argument("--caption-refresh-dir", type=Path, default=DEFAULT_CAPTION_REFRESH_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--window-id", default=None)
    parser.add_argument("--block-window-index", type=int, default=None)
    parser.add_argument("--simple-narrative-id", default="description_terse_trader")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = build_appendix(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
