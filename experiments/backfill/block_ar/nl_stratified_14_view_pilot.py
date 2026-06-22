#!/usr/bin/env python
"""Run stratified 14-view narrative pilots before full corpus regeneration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stratified_fourteen_view_pilot_986e"
)


def _window_number(window_id: str) -> int:
    match = re.search(r"_(\d+)$", str(window_id))
    return int(match.group(1)) if match else -1


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def is_feasible_target(card: dict[str, Any], *, cards: list[dict[str, Any]]) -> bool:
    """Return whether the current 14-view assignment layer can handle a target."""

    window_id = str(card.get("window_id", ""))
    try:
        candidates = pilot.select_negative_candidates_for_fourteen_view(
            cards=cards,
            target_window_id=window_id,
            count=80,
            min_temporal_gap=30,
        )
        assignments = pilot.assign_negative_candidates_by_view(candidates)
    except (KeyError, ValueError, IndexError):
        return False
    return len({row["candidate"]["window_id"] for row in assignments}) == len(
        pilot.EXPECTED_VIEW_NAMES
    )


def select_stratified_targets(
    *,
    cards: list[dict[str, Any]],
    target_count: int,
    max_per_archetype: int,
    is_feasible: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Select feasible targets while spreading coverage across archetypes."""

    by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        archetype = _compact(card.get("archetype")) or "unknown"
        by_archetype[archetype].append(card)
    for rows in by_archetype.values():
        rows.sort(key=lambda card: _window_number(str(card.get("window_id", ""))))

    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    archetypes = sorted(
        by_archetype,
        key=lambda name: (-len(by_archetype[name]), name),
    )
    cursors = {name: 0 for name in archetypes}
    while len(selected) < int(target_count):
        progressed = False
        for archetype in archetypes:
            if len(selected) >= int(target_count):
                break
            if counts[archetype] >= int(max_per_archetype):
                continue
            rows = by_archetype[archetype]
            while cursors[archetype] < len(rows):
                card = rows[cursors[archetype]]
                cursors[archetype] += 1
                if is_feasible(card):
                    selected.append(
                        {
                            "window_id": str(card.get("window_id", "")),
                            "archetype": _compact(card.get("archetype")) or "unknown",
                            "scenario_title": _compact(card.get("scenario_title")),
                        }
                    )
                    counts[archetype] += 1
                    progressed = True
                    break
        if not progressed:
            break
    return selected


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize stratified pilot records."""

    statuses = Counter(str(row.get("status", "unknown")) for row in records)
    archetypes = Counter(str(row.get("archetype", "unknown")) for row in records)
    fail_count = statuses.get("fail", 0) + statuses.get("missing", 0) + statuses.get(
        "unknown", 0
    )
    return {
        "target_count": len(records),
        "pass_count": statuses.get("pass", 0),
        "fail_count": fail_count,
        "dry_run_count": statuses.get("dry_run", 0),
        "status_counts": dict(sorted(statuses.items())),
        "archetype_counts": dict(sorted(archetypes.items())),
    }


def build_summary_markdown(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Build a human-readable index for all stratified pilot packets."""

    lines = [
        "# Stratified Fourteen-View Narrative Pilot",
        "",
        "Generated: 2026-06-03",
        "",
        "This packet indexes 14-positive/14-negative Codex-authored review "
        "packets across stratified historical episodes. Structured evidence "
        "and candidate assignment are local; narrative prose is not generated "
        "locally.",
        "",
        "## Summary",
        "",
        f"- Target count: `{summary['target_count']}`",
        f"- Pass count: `{summary['pass_count']}`",
        f"- Fail count: `{summary['fail_count']}`",
        f"- Dry-run count: `{summary.get('dry_run_count', 0)}`",
        f"- Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        f"- Archetype counts: `{json.dumps(summary['archetype_counts'], sort_keys=True)}`",
        "",
        "## Targets",
        "",
        "| Target | Status | Errors | Archetype | Scenario | Review |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in records:
        lines.append(
            "| "
            f"`{row['target_window_id']}` | "
            f"`{row['status']}` | "
            f"`{row.get('validation_error_count', '')}` | "
            f"`{row['archetype']}` | "
            f"{row['scenario_title']} | "
            f"{row['review_path']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def existing_pass_record(*, target: dict[str, Any], output_dir: Path) -> dict[str, Any] | None:
    """Return a previous passing target record when its artifacts are present."""

    target_id = str(target["window_id"])
    target_output_dir = output_dir / target_id
    report_path = target_output_dir / "fourteen_view_report.json"
    review_path = target_output_dir / "fourteen_view_review.md"
    if not report_path.exists() or not review_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(report.get("status", "fail")) != "pass":
        return None
    return {
        "target_window_id": target_id,
        "archetype": target["archetype"],
        "scenario_title": target["scenario_title"],
        "status": "pass",
        "validation_error_count": int(report.get("validation", {}).get("error_count", 0)),
        "review_path": str(review_path),
        "report_path": str(report_path),
        "reused_existing": True,
    }


def _run_one_target(
    *,
    target: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
    timeout_seconds: int,
    reuse_existing_pass: bool,
) -> dict[str, Any]:
    target_id = str(target["window_id"])
    target_output_dir = output_dir / target_id
    if reuse_existing_pass:
        existing_record = existing_pass_record(target=target, output_dir=output_dir)
        if existing_record is not None:
            return existing_record
    if dry_run:
        return {
            "target_window_id": target_id,
            "archetype": target["archetype"],
            "scenario_title": target["scenario_title"],
            "status": "dry_run",
            "validation_error_count": "",
            "review_path": str(target_output_dir / "fourteen_view_review.md"),
            "report_path": str(target_output_dir / "fourteen_view_report.json"),
        }
    cmd = [
        sys.executable,
        "experiments/backfill/block_ar/nl_14_view_variant_pilot.py",
        "--target-window-id",
        target_id,
        "--output-dir",
        str(target_output_dir),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds + 60,
        check=False,
    )
    report_path = target_output_dir / "fourteen_view_report.json"
    review_path = target_output_dir / "fourteen_view_review.md"
    status = "fail"
    error_count: int | str = "missing"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = str(report.get("status", "fail"))
        error_count = int(report.get("validation", {}).get("error_count", 0))
    return {
        "target_window_id": target_id,
        "archetype": target["archetype"],
        "scenario_title": target["scenario_title"],
        "status": status,
        "validation_error_count": error_count,
        "review_path": str(review_path),
        "report_path": str(report_path),
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
    }


def run_stratified_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Select targets, optionally run each pilot, and write an index."""

    output_dir = _resolve(Path(args.output_dir))
    cards = pilot._read_jsonl(Path(args.cards_jsonl))
    targets = select_stratified_targets(
        cards=cards,
        target_count=int(args.target_count),
        max_per_archetype=int(args.max_per_archetype),
        is_feasible=lambda card: is_feasible_target(card, cards=cards),
    )
    records = [
        _run_one_target(
            target=target,
            output_dir=output_dir,
            dry_run=bool(args.dry_run),
            timeout_seconds=int(args.timeout_seconds),
            reuse_existing_pass=bool(args.reuse_existing_pass),
        )
        for target in targets
    ]
    summary = build_summary(records)
    report = {
        "schema_version": "stratified_fourteen_view_pilot_report_v1",
        "status": "pass" if records and summary["fail_count"] == 0 else "fail",
        "dry_run": bool(args.dry_run),
        "target_count_requested": int(args.target_count),
        "target_count_selected": len(targets),
        "targets": targets,
        "records": records,
        "summary": summary,
        "local_prose_generated": False,
        "artifact_paths": {
            "report": str(output_dir / "stratified_fourteen_view_report.json"),
            "markdown": str(output_dir / "stratified_fourteen_view_report.md"),
        },
    }
    _write_json(output_dir / "stratified_fourteen_view_report.json", report)
    _write_text(
        output_dir / "stratified_fourteen_view_report.md",
        build_summary_markdown(records, summary),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=pilot.DEFAULT_CARDS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--max-per-archetype", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--reuse-existing-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_stratified_pilot(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dry_run": report["dry_run"],
                "target_count_selected": report["target_count_selected"],
                "pass_count": report["summary"]["pass_count"],
                "fail_count": report["summary"]["fail_count"],
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
