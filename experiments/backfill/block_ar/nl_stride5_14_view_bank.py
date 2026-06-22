#!/usr/bin/env python
"""Generate the first stride-5 14-view narrative bank with batch validation.

This runner orchestrates the existing one-episode Codex/GPT 14-positive /
14-hard-negative generator. It does not author narrative prose locally.
Instead, it selects every fifth historical episode, runs the agentic generator
per target, revalidates each generated target report, and writes a validation
checkpoint after every batch so the job can resume safely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot  # noqa: E402
from experiments.backfill.block_ar.nl_hard_negative_bank_regenerate import (  # noqa: E402
    _window_number,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_fourteen_view_bank_988a"
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def select_stride_targets(
    *,
    cards: list[dict[str, Any]],
    stride: int,
    max_targets: int | None = None,
) -> list[dict[str, Any]]:
    """Select every `stride`-th train window by window number."""

    selected: list[dict[str, Any]] = []
    for card in sorted(cards, key=lambda row: _window_number(str(row.get("window_id", "")))):
        window_id = str(card.get("window_id", ""))
        number = _window_number(window_id)
        if number < 0 or number % int(stride) != 0:
            continue
        selected.append(
            {
                "window_id": window_id,
                "window_number": number,
                "archetype": str(card.get("archetype", "")),
                "scenario_title": str(card.get("scenario_title", "")),
            }
        )
        if max_targets is not None and len(selected) >= int(max_targets):
            break
    return selected


def _target_dir(output_dir: Path, target_id: str) -> Path:
    return output_dir / "targets" / str(target_id)


def validate_generated_target_report(report_path: str | Path) -> dict[str, Any]:
    """Revalidate one generated target report and return compact status."""

    path = Path(report_path)
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "status": "fail",
            "report_path": str(path),
            "pair_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "errors": [{"code": "missing_report"}],
        }
    try:
        report = _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "fail",
            "report_path": str(path),
            "pair_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "errors": [
                {
                    "code": "report_parse_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        }

    pairs_payload = report.get("pairs", [])
    pairs = pairs_payload if isinstance(pairs_payload, list) else []
    embedded_report_status = str(report.get("status", ""))
    if report.get("dry_run"):
        errors.append({"code": "target_report_is_dry_run"})
    if report.get("local_prose_generated") is not False:
        errors.append(
            {
                "code": "local_prose_generated_not_false",
                "value": report.get("local_prose_generated"),
            }
        )
    if len(pairs) != len(pilot.EXPECTED_VIEW_NAMES):
        errors.append(
            {
                "code": "pair_count_mismatch",
                "expected": len(pilot.EXPECTED_VIEW_NAMES),
                "actual": len(pairs),
            }
        )

    try:
        batch = pilot.FourteenViewPilotBatch(
            target_window_id=str(report.get("target_window_id", "")),
            target_title=str(report.get("target", {}).get("scenario_title", "")),
            pairs=[pilot.NarrativePair.model_validate(pair) for pair in pairs],
        )
        validation = pilot.validate_batch(
            batch=batch,
            target=report.get("target", {}),
            negative_candidates=report.get("negative_candidates", []),
            assigned_negative_candidates=report.get("assigned_negative_candidates", []),
        )
    except Exception as exc:  # defensive because this protects long Codex runs
        validation = {
            "status": "fail",
            "error_count": 1,
            "errors": [
                {
                    "code": "target_revalidation_exception",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        }
    if validation.get("status") != "pass":
        errors.extend(validation.get("errors", []))

    positive_texts = {
        str(pair.get("positive_text", "")).strip()
        for pair in pairs
        if str(pair.get("positive_text", "")).strip()
    }
    negative_texts = {
        str(pair.get("negative_text", "")).strip()
        for pair in pairs
        if str(pair.get("negative_text", "")).strip()
    }
    return {
        "status": "pass" if not errors else "fail",
        "report_path": str(path),
        "review_path": str(path.with_name("fourteen_view_review.md")),
        "target_window_id": str(report.get("target_window_id", "")),
        "embedded_report_status": embedded_report_status,
        "pair_count": len(pairs),
        "positive_count": len(positive_texts),
        "negative_count": len(negative_texts),
        "validation_error_count": len(errors),
        "errors": errors,
    }


def existing_pass_record(*, target: dict[str, Any], output_dir: str | Path) -> dict[str, Any] | None:
    """Return an existing valid generated target record, if present."""

    output = Path(output_dir)
    target_id = str(target["window_id"])
    target_output_dir = _target_dir(output, target_id)
    report_path = target_output_dir / "fourteen_view_report.json"
    review_path = target_output_dir / "fourteen_view_review.md"
    if not report_path.exists() or not review_path.exists():
        return None
    validation = validate_generated_target_report(report_path)
    if validation["status"] != "pass":
        return None
    return {
        **target,
        **validation,
        "status": "pass",
        "target_output_dir": str(target_output_dir),
        "reused_existing": True,
    }


def build_batch_validation_summary(
    *,
    batch_index: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(record.get("status", "unknown")) for record in records)
    fail_count = sum(
        statuses.get(status, 0)
        for status in ("fail", "missing", "unknown", "codex_exec_failed")
    )
    return {
        "batch_index": int(batch_index),
        "target_count": len(records),
        "pass_count": statuses.get("pass", 0),
        "fail_count": fail_count,
        "dry_run_count": statuses.get("dry_run", 0),
        "status_counts": dict(sorted(statuses.items())),
        "status": "pass" if records and fail_count == 0 else "fail",
    }


def _write_batch_validation(
    *,
    output_dir: Path,
    batch_index: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = build_batch_validation_summary(batch_index=batch_index, records=records)
    payload = {
        "schema_version": "stride5_fourteen_view_batch_validation_v1",
        "summary": summary,
        "records": records,
    }
    json_path = output_dir / "batch_validations" / f"batch_{batch_index:04d}_validation.json"
    md_path = output_dir / "batch_validations" / f"batch_{batch_index:04d}_validation.md"
    _write_json(json_path, payload)
    lines = [
        f"# Batch {batch_index:04d} Validation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Target count: `{summary['target_count']}`",
        f"- Pass count: `{summary['pass_count']}`",
        f"- Fail count: `{summary['fail_count']}`",
        "",
        "| Target | Status | Pairs | Positives | Negatives | Reused | Report |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| "
            f"`{record.get('target_window_id', '')}` | "
            f"`{record.get('status', '')}` | "
            f"`{record.get('pair_count', '')}` | "
            f"`{record.get('positive_count', '')}` | "
            f"`{record.get('negative_count', '')}` | "
            f"`{record.get('reused_existing', False)}` | "
            f"{record.get('report_path', '')} |"
        )
    lines.append("")
    _write_text(md_path, "\n".join(lines))
    return {
        **summary,
        "artifact_paths": {"json": str(json_path), "markdown": str(md_path)},
    }


def load_existing_batch_validations(
    *,
    output_dir: str | Path,
    before_batch: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Load already-checkpointed batch validations before a resumed batch."""

    output = Path(output_dir)
    records: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    missing_batches: list[int] = []
    for batch_index in range(max(0, int(before_batch))):
        json_path = output / "batch_validations" / f"batch_{batch_index:04d}_validation.json"
        md_path = output / "batch_validations" / f"batch_{batch_index:04d}_validation.md"
        if not json_path.exists():
            missing_batches.append(batch_index)
            continue
        try:
            payload = _read_json(json_path)
        except (OSError, json.JSONDecodeError, ValueError):
            missing_batches.append(batch_index)
            continue
        batch_records_payload = payload.get("records", [])
        batch_records = batch_records_payload if isinstance(batch_records_payload, list) else []
        summary_payload = payload.get("summary", {})
        computed_summary = build_batch_validation_summary(
            batch_index=batch_index,
            records=batch_records,
        )
        summary = {
            **computed_summary,
            **(summary_payload if isinstance(summary_payload, dict) else {}),
            "batch_index": batch_index,
            "artifact_paths": {"json": str(json_path), "markdown": str(md_path)},
        }
        records.extend(batch_records)
        batch_summaries.append(summary)
    return records, batch_summaries, missing_batches


def _run_one_target(
    *,
    target: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_id = str(target["window_id"])
    target_output_dir = _target_dir(output_dir, target_id)
    if bool(args.reuse_existing_pass):
        existing = existing_pass_record(target=target, output_dir=output_dir)
        if existing is not None:
            return existing
    if bool(args.dry_run):
        return {
            **target,
            "status": "dry_run",
            "target_window_id": target_id,
            "target_output_dir": str(target_output_dir),
            "report_path": str(target_output_dir / "fourteen_view_report.json"),
            "review_path": str(target_output_dir / "fourteen_view_review.md"),
            "pair_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "reused_existing": False,
        }

    cmd = build_target_command(
        target_id=target_id,
        target_output_dir=target_output_dir,
        args=args,
    )
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=int(args.timeout_seconds) + 90,
        check=False,
    )
    report_path = target_output_dir / "fourteen_view_report.json"
    validation = validate_generated_target_report(report_path)
    status = validation["status"] if completed.returncode == 0 else "fail"
    record = {
        **target,
        **validation,
        "status": status,
        "target_output_dir": str(target_output_dir),
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
        "reused_existing": False,
    }
    if completed.returncode != 0:
        record.setdefault("errors", [])
        record["errors"] = list(record["errors"]) + [
            {
                "code": "codex_target_command_failed",
                "returncode": completed.returncode,
            }
        ]
    return record


def build_target_command(
    *,
    target_id: str,
    target_output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build the one-target generator command."""

    return [
        sys.executable,
        "experiments/backfill/block_ar/nl_14_view_variant_pilot.py",
        "--cards-jsonl",
        str(args.cards_jsonl),
        "--support-cards-jsonl",
        str(args.support_cards_jsonl),
        "--target-window-id",
        str(target_id),
        "--output-dir",
        str(target_output_dir),
        "--negative-candidate-count",
        str(getattr(args, "negative_candidate_count", 200)),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--validation-retries",
        str(args.validation_retries),
    ]


def _chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + int(size)] for idx in range(0, len(rows), int(size))]


def _build_summary(records: list[dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    statuses = Counter(str(record.get("status", "unknown")) for record in records)
    fail_count = sum(statuses.get(status, 0) for status in ("fail", "missing", "unknown"))
    pass_count = statuses.get("pass", 0)
    dry_count = statuses.get("dry_run", 0)
    return {
        "record_count": len(records),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "dry_run_count": dry_count,
        "status_counts": dict(sorted(statuses.items())),
        "status": (
            "dry_run"
            if dry_run
            else "pass"
            if records and fail_count == 0
            else "fail"
        ),
    }


def _write_manifest(path: Path, targets: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in targets),
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Stride-5 Fourteen-View Narrative Bank",
        "",
        "This report tracks the stride-5 production run for 14 positive and 14 "
        "hard-negative narratives per selected historical episode. Narrative "
        "text is generated by the existing Codex/GPT one-episode workflow; this "
        "runner only orchestrates, validates, checkpoints, and resumes.",
        "",
        "## Status",
        "",
        f"- Status: `{report['status']}`",
        f"- Selected targets: `{report['target_count_selected']}`",
        f"- Batch count: `{report['batch_count']}`",
        f"- Completed records: `{summary['record_count']}`",
        f"- Pass count: `{summary['pass_count']}`",
        f"- Fail count: `{summary['fail_count']}`",
        f"- Dry-run count: `{summary['dry_run_count']}`",
        f"- Local prose generated: `{report['local_prose_generated']}`",
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{report['artifact_paths']['manifest']}`",
        f"- JSON report: `{report['artifact_paths']['report']}`",
        f"- Markdown report: `{report['artifact_paths']['markdown']}`",
        "",
        "## Batch Validations",
        "",
        "| Batch | Status | Targets | Pass | Fail | Validation JSON |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for batch in report.get("batch_summaries", []):
        lines.append(
            f"| {batch['batch_index']} | `{batch['status']}` | "
            f"{batch['target_count']} | {batch['pass_count']} | "
            f"{batch['fail_count']} | {batch['artifact_paths']['json']} |"
        )
    lines.append("")
    _write_text(path, "\n".join(lines))


def run_bank_generation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = pilot._read_jsonl(Path(args.cards_jsonl))
    targets = select_stride_targets(
        cards=cards,
        stride=int(args.stride),
        max_targets=(None if args.max_targets is None else int(args.max_targets)),
    )
    batches = _chunked(targets, int(args.batch_size))
    if args.max_batches is not None:
        runnable_batches = batches[int(args.start_batch) : int(args.start_batch) + int(args.max_batches)]
    else:
        runnable_batches = batches[int(args.start_batch) :]

    manifest_path = output_dir / "stride5_targets_manifest.jsonl"
    report_path = output_dir / "stride5_fourteen_view_bank_report.json"
    markdown_path = output_dir / "stride5_fourteen_view_bank_report.md"
    _write_manifest(manifest_path, targets)

    (
        all_records,
        batch_summaries,
        missing_prior_batch_validations,
    ) = load_existing_batch_validations(
        output_dir=output_dir,
        before_batch=int(args.start_batch),
    )
    stopped_on_failure = False
    started = time.time()
    for local_batch_index, batch_targets in enumerate(runnable_batches):
        batch_index = int(args.start_batch) + local_batch_index
        print(
            f"[stride5-bank] starting batch {batch_index:04d} "
            f"with {len(batch_targets)} targets",
            flush=True,
        )
        records: list[dict[str, Any]] = []
        for target_pos, target in enumerate(batch_targets, 1):
            print(
                f"[stride5-bank] batch {batch_index:04d} "
                f"target {target_pos}/{len(batch_targets)} "
                f"{target['window_id']}",
                flush=True,
            )
            record = _run_one_target(target=target, output_dir=output_dir, args=args)
            records.append(record)
            print(
                f"[stride5-bank] target {target['window_id']} "
                f"status={record.get('status')} pairs={record.get('pair_count')}",
                flush=True,
            )
        all_records.extend(records)
        batch_summary = _write_batch_validation(
            output_dir=output_dir,
            batch_index=batch_index,
            records=records,
        )
        batch_summaries.append(batch_summary)
        print(
            f"[stride5-bank] batch {batch_index:04d} "
            f"validation={batch_summary['status']} "
            f"pass={batch_summary['pass_count']} fail={batch_summary['fail_count']}",
            flush=True,
        )
        if bool(args.stop_on_failure) and batch_summary["fail_count"] > 0:
            stopped_on_failure = True
            break

        summary = _build_summary(all_records, dry_run=bool(args.dry_run))
        checkpoint = _build_report(
            args=args,
            output_dir=output_dir,
            targets=targets,
            batches=batches,
            records=all_records,
            batch_summaries=batch_summaries,
            summary=summary,
            stopped_on_failure=stopped_on_failure,
            missing_prior_batch_validations=missing_prior_batch_validations,
            elapsed_seconds=round(time.time() - started, 3),
            manifest_path=manifest_path,
            report_path=report_path,
            markdown_path=markdown_path,
        )
        _write_json(report_path, checkpoint)
        _write_markdown(markdown_path, checkpoint)

    summary = _build_summary(all_records, dry_run=bool(args.dry_run))
    report = _build_report(
        args=args,
        output_dir=output_dir,
        targets=targets,
        batches=batches,
        records=all_records,
        batch_summaries=batch_summaries,
        summary=summary,
        stopped_on_failure=stopped_on_failure,
        missing_prior_batch_validations=missing_prior_batch_validations,
        elapsed_seconds=round(time.time() - started, 3),
        manifest_path=manifest_path,
        report_path=report_path,
        markdown_path=markdown_path,
    )
    _write_json(report_path, report)
    _write_markdown(markdown_path, report)
    return report


def _build_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    targets: list[dict[str, Any]],
    batches: list[list[dict[str, Any]]],
    records: list[dict[str, Any]],
    batch_summaries: list[dict[str, Any]],
    summary: dict[str, Any],
    stopped_on_failure: bool,
    missing_prior_batch_validations: list[int],
    elapsed_seconds: float,
    manifest_path: Path,
    report_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    processed_all = len(records) >= len(targets)
    status = summary["status"]
    if not bool(args.dry_run) and not processed_all and summary["fail_count"] == 0:
        status = "partial"
    if stopped_on_failure:
        status = "fail"
    return {
        "schema_version": "stride5_fourteen_view_bank_report_v1",
        "status": status,
        "dry_run": bool(args.dry_run),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "target_count_selected": len(targets),
        "batch_count": len(batches),
        "processed_all_targets": processed_all,
        "stopped_on_failure": stopped_on_failure,
        "missing_prior_batch_validations": missing_prior_batch_validations,
        "summary": summary,
        "targets": targets,
        "records": records,
        "batch_summaries": batch_summaries,
        "local_prose_generated": False,
        "elapsed_seconds": elapsed_seconds,
        "artifact_paths": {
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            "report": str(report_path),
            "markdown": str(markdown_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=pilot.DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-cards-jsonl", type=Path, default=pilot.DEFAULT_SUPPORT_CARDS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument("--negative-candidate-count", type=int, default=200)
    parser.add_argument("--reuse-existing-pass", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    parser.add_argument("--continue-on-failure", dest="stop_on_failure", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_bank_generation(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dry_run": report["dry_run"],
                "target_count_selected": report["target_count_selected"],
                "batch_count": report["batch_count"],
                "processed_records": report["summary"]["record_count"],
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
