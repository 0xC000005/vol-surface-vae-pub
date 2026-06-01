#!/usr/bin/env python
"""Summarize paired base-vs-quality-guard story-smoke rollouts."""

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

from experiments.backfill.block_ar.nl_portfolio_risk_response_label_audit import (  # noqa: E402
    MARKET_INDEX,
    PORTFOLIO_BOOKS,
    portfolio_pnl,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_quality_guard_paired_rollout_925g"
)
DEFAULT_FACTORS = ["SPX", "VIX", "BBB_OAS", "GOLD", "US10Y"]
DEFAULT_BOOKS = ["equity_beta_carry", "credit_duration", "dollar_liquidity"]


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


def _support_jaccard(left: list[int], right: list[int]) -> float:
    a = set(int(item) for item in left)
    b = set(int(item) for item in right)
    if not a and not b:
        return 1.0
    return float(len(a & b) / max(len(a | b), 1))


def _operational_index(report: dict[str, Any]) -> int:
    for idx, row in enumerate(report.get("variant_rows", [])):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return int(idx)
    return max(len(report.get("variant_rows", [])) - 1, 0)


def load_story_smoke_case(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    report_path = root / "prefix_latent_story_smoke_report.json"
    arrays_path = root / "prefix_latent_story_smoke_arrays.npz"
    report = _load_json(report_path)
    arrays = np.load(arrays_path)
    op_idx = _operational_index(report)
    states = np.asarray(arrays["generated_states"][op_idx], dtype=np.float64)
    start_key = "requested_raw" if "requested_raw" in arrays.files else "requested_start"
    start = np.asarray(arrays[start_key][op_idx], dtype=np.float64)
    prior = report.get("cached_query", {}).get("memory_prior", {})
    if not isinstance(prior, dict):
        prior = {}
    return {
        "root": str(root),
        "report": report,
        "operational_index": int(op_idx),
        "states": states,
        "start": start,
        "start_array_key": start_key,
        "support_indices": [int(item) for item in prior.get("window_indices", [])],
        "support_weights": [float(item) for item in prior.get("weights", [])],
        "quality_guard_policy": prior.get("portfolio_quality_guard_policy"),
    }


def _terminal_summary(states: np.ndarray, index: int) -> dict[str, float]:
    terminal = np.asarray(states, dtype=np.float64)[:, -1, int(index)]
    return {
        "mean": float(np.mean(terminal)),
        "p10": float(np.quantile(terminal, 0.10)),
        "p90": float(np.quantile(terminal, 0.90)),
        "width_10_90": float(np.quantile(terminal, 0.90) - np.quantile(terminal, 0.10)),
    }


def build_pair_summary(
    *,
    case_name: str,
    base: dict[str, Any],
    guarded: dict[str, Any],
    factors: list[str] | None = None,
    book_names: list[str] | None = None,
) -> dict[str, Any]:
    factors = list(factors or DEFAULT_FACTORS)
    book_names = list(book_names or DEFAULT_BOOKS)
    factor_rows: list[dict[str, Any]] = []
    for factor in factors:
        idx = int(MARKET_INDEX[factor])
        base_stats = _terminal_summary(base["states"], idx)
        guarded_stats = _terminal_summary(guarded["states"], idx)
        factor_rows.append(
            {
                "factor": factor,
                "start_level": float(base["start"][idx]),
                "base_terminal_mean": base_stats["mean"],
                "quality_guard_terminal_mean": guarded_stats["mean"],
                "terminal_mean_delta_qg_minus_base": (
                    guarded_stats["mean"] - base_stats["mean"]
                ),
                "base_width_10_90": base_stats["width_10_90"],
                "quality_guard_width_10_90": guarded_stats["width_10_90"],
                "width_delta_qg_minus_base": (
                    guarded_stats["width_10_90"] - base_stats["width_10_90"]
                ),
            }
        )
    book_lookup = {str(book["name"]): book for book in PORTFOLIO_BOOKS}
    portfolio_rows: list[dict[str, Any]] = []
    for name in book_names:
        book = book_lookup[str(name)]
        base_pnl = portfolio_pnl(base["states"], base["start"], book)
        guarded_pnl = portfolio_pnl(guarded["states"], guarded["start"], book)
        base_terminal = base_pnl[:, -1]
        guarded_terminal = guarded_pnl[:, -1]
        portfolio_rows.append(
            {
                "book": str(name),
                "base_terminal_mean_pnl": float(np.mean(base_terminal)),
                "quality_guard_terminal_mean_pnl": float(np.mean(guarded_terminal)),
                "terminal_mean_pnl_delta_qg_minus_base": float(
                    np.mean(guarded_terminal) - np.mean(base_terminal)
                ),
                "terminal_p05_delta_qg_minus_base": float(
                    np.quantile(guarded_terminal, 0.05)
                    - np.quantile(base_terminal, 0.05)
                ),
                "path_loss_mean_delta_qg_minus_base": float(
                    np.mean(-np.min(guarded_pnl, axis=1))
                    - np.mean(-np.min(base_pnl, axis=1))
                ),
            }
        )
    qg_policy = guarded.get("quality_guard_policy")
    return {
        "case_name": str(case_name),
        "base_root": base["root"],
        "quality_guard_root": guarded["root"],
        "start_array_key": str(base.get("start_array_key", "")),
        "base_support_indices": base["support_indices"],
        "quality_guard_support_indices": guarded["support_indices"],
        "support_jaccard": _support_jaccard(
            base["support_indices"],
            guarded["support_indices"],
        ),
        "quality_guard_policy": qg_policy if isinstance(qg_policy, dict) else None,
        "factor_terminal_rows": factor_rows,
        "portfolio_rows": portfolio_rows,
        "generated_shape": list(base["states"].shape),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Portfolio Quality Guard Paired Rollout Summary",
        "",
        f"Case: `{report['case_name']}`",
        f"Support Jaccard: `{report['support_jaccard']:.3f}`",
        "",
        "## Support",
        "",
        f"- Base: `{report['base_support_indices']}`",
        f"- Quality guard: `{report['quality_guard_support_indices']}`",
        "",
        "## Factor Terminal Effects",
        "",
        "| Factor | Start | Mean Delta | Width Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in report["factor_terminal_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["factor"]),
                    f"{float(row['start_level']):.4f}",
                    f"{float(row['terminal_mean_delta_qg_minus_base']):.4f}",
                    f"{float(row['width_delta_qg_minus_base']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Portfolio Effects",
            "",
            "| Book | Terminal Mean PnL Delta | Terminal P05 Delta | Path Loss Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["portfolio_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["book"]),
                    f"{float(row['terminal_mean_pnl_delta_qg_minus_base']):.4f}",
                    f"{float(row['terminal_p05_delta_qg_minus_base']):.4f}",
                    f"{float(row['path_loss_mean_delta_qg_minus_base']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", default="paired_rollout")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--quality-guard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS))
    parser.add_argument("--books", default=",".join(DEFAULT_BOOKS))
    args = parser.parse_args()

    report = build_pair_summary(
        case_name=str(args.case_name),
        base=load_story_smoke_case(args.base_root),
        guarded=load_story_smoke_case(args.quality_guard_root),
        factors=[item.strip() for item in str(args.factors).split(",") if item.strip()],
        book_names=[item.strip() for item in str(args.books).split(",") if item.strip()],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "portfolio_quality_guard_paired_rollout_summary.json"
    markdown_path = args.output_dir / "portfolio_quality_guard_paired_rollout_summary.md"
    report["artifact_paths"] = {"report": str(json_path), "markdown": str(markdown_path)}
    _write_json(json_path, report)
    _write_text(markdown_path, _markdown(report))
    print(
        json.dumps(
            {
                "case_name": report["case_name"],
                "support_jaccard": report["support_jaccard"],
                "report": str(json_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
