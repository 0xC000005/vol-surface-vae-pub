#!/usr/bin/env python
"""Offline rollout-level reranker for narrative prefix-latent casebooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_rollout_reranker_803a"
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


def parse_candidate_arg(value: str) -> tuple[str, str]:
    if "=" not in str(value):
        path = str(value)
        return Path(path).parent.name or Path(path).stem, path
    name, path = str(value).split("=", 1)
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("candidate name must be non-empty")
    return clean_name, path.strip()


def _alignment_stats(case: dict[str, Any]) -> dict[str, Any]:
    alignment = case.get("market_alignment", {})
    if not isinstance(alignment, dict):
        alignment = {}
    checked = int(alignment.get("checked_count", 0) or 0)
    mismatches = int(alignment.get("mismatch_count", 0) or 0)
    return {
        "checked_count": checked,
        "mismatch_count": mismatches,
        "mismatch_rate": float(mismatches / checked) if checked else 1.0,
        "alignment_status": str(alignment.get("status", "unknown")),
    }


def load_candidate_casebook(name: str, path: str | Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    rows: dict[str, dict[str, Any]] = {}
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_name = str(case.get("case_name", ""))
        if not case_name:
            continue
        rows[case_name] = {
            "candidate_name": str(name),
            "source_path": str(path),
            "case_name": case_name,
            "selected_start_status": str(case.get("selected_start_status", "")),
            "overall_status": str(case.get("overall_status", "")),
            "selected_path_labels": [
                label
                for label in case.get("path_labels", [])
                if isinstance(label, str) and "Selected start:" in label
            ],
            "warning_counts": case.get("warning_counts", {}),
            **_alignment_stats(case),
        }
    return rows


def _status_rank(status: str) -> int:
    ranks = {"pass": 0, "warning": 1, "fail": 2}
    return ranks.get(str(status), 3)


def choose_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    return min(
        candidates,
        key=lambda row: (
            float(row.get("mismatch_rate", 1.0)),
            int(row.get("mismatch_count", 0)),
            _status_rank(str(row.get("selected_start_status", ""))),
            str(row.get("candidate_name", "")),
        ),
    )


def rerank_casebooks(candidate_args: list[str]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    candidate_paths: dict[str, str] = {}
    for value in candidate_args:
        name, path = parse_candidate_arg(value)
        candidate_paths[name] = path
        for case_name, row in load_candidate_casebook(name, path).items():
            by_case.setdefault(case_name, []).append(row)
    selected_rows: list[dict[str, Any]] = []
    for case_name in sorted(by_case):
        candidates = by_case[case_name]
        chosen = choose_best_candidate(candidates)
        selected_rows.append(
            {
                "case_name": case_name,
                "chosen_candidate": chosen["candidate_name"],
                "chosen": chosen,
                "candidate_count": len(candidates),
                "candidates": sorted(
                    candidates,
                    key=lambda row: str(row.get("candidate_name", "")),
                ),
            }
        )
    return {
        "candidate_paths": candidate_paths,
        "selected_cases": selected_rows,
    }


def summarize_rerank_result(
    rerank: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    selected_cases = rerank.get("selected_cases", [])
    if not isinstance(selected_cases, list):
        selected_cases = []
    total_checked = 0
    total_mismatches = 0
    chosen_counts: dict[str, int] = {}
    selected_status_counts: dict[str, int] = {}
    for row in selected_cases:
        if not isinstance(row, dict):
            continue
        chosen = row.get("chosen", {})
        if not isinstance(chosen, dict):
            continue
        total_checked += int(chosen.get("checked_count", 0) or 0)
        total_mismatches += int(chosen.get("mismatch_count", 0) or 0)
        candidate = str(row.get("chosen_candidate", ""))
        status = str(chosen.get("selected_start_status", ""))
        chosen_counts[candidate] = chosen_counts.get(candidate, 0) + 1
        selected_status_counts[status] = selected_status_counts.get(status, 0) + 1
    output_path = Path(output_dir) / "rollout_reranker_summary.json"
    return {
        "status": "ok",
        "scope_note": (
            "Offline rollout-level reranker over existing product casebook "
            "artifacts. No OpenAI calls are made; it chooses the candidate "
            "casebook result with the lowest generated scenario implication "
            "mismatch rate for each story."
        ),
        "case_count": len(selected_cases),
        "total_checked_implications": total_checked,
        "total_mismatches": total_mismatches,
        "mismatch_rate": (
            float(total_mismatches / total_checked) if total_checked else None
        ),
        "chosen_candidate_counts": chosen_counts,
        "selected_start_status_counts": selected_status_counts,
        "candidate_paths": rerank.get("candidate_paths", {}),
        "selected_cases": selected_cases,
        "artifact_paths": {"summary": str(output_path)},
    }


def run_rollout_reranker(args: argparse.Namespace) -> dict[str, Any]:
    rerank = rerank_casebooks([str(value) for value in args.candidate])
    summary = summarize_rerank_result(rerank, output_dir=args.output_dir)
    _write_json(summary["artifact_paths"]["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", action="append", required=True)
    args = parser.parse_args()
    summary = run_rollout_reranker(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "case_count": summary["case_count"],
                "chosen_candidate_counts": summary["chosen_candidate_counts"],
                "total_checked_implications": summary["total_checked_implications"],
                "total_mismatches": summary["total_mismatches"],
                "mismatch_rate": summary["mismatch_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
