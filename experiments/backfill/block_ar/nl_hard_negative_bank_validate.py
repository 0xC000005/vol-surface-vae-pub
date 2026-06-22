#!/usr/bin/env python
"""Validate a generated hard-negative narrative bank."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_hard_negative_codex_batch import (  # noqa: E402
    DEFAULT_MANIFEST_JSONL,
    DEFAULT_OUTPUT_DIR,
    HardNegativeNarrative,
    read_jsonl,
    row_id,
    validate_generated,
)


DEFAULT_BANK_JSONL = DEFAULT_OUTPUT_DIR / "hard_negative_bank.jsonl"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_bank(
    *,
    manifest_jsonl: Path,
    bank_jsonl: Path,
    expected_count: int,
    allow_partial: bool,
) -> dict[str, Any]:
    manifest = read_jsonl(manifest_jsonl)
    manifest_by_row_id = {row_id(row): row for row in manifest}
    bank_rows = read_jsonl(bank_jsonl)
    seen: Counter[str] = Counter()
    validation_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warning_count = 0
    valid_count = 0

    for raw in bank_rows:
        rid = str(raw.get("row_id", ""))
        seen[rid] += 1
        source = manifest_by_row_id.get(rid)
        if source is None:
            errors.append({"row_id": rid, "code": "row_not_in_manifest"})
            continue
        try:
            narrative = HardNegativeNarrative.model_validate(
                {
                    "row_id": raw.get("row_id"),
                    "target_window_id": raw.get("target_window_id"),
                    "positive_view": raw.get("positive_view"),
                    "negative_window_id": raw.get("negative_window_id"),
                    "hard_negative_text": raw.get("hard_negative_text"),
                    "quality_notes": raw.get("quality_notes", []),
                }
            )
        except ValidationError as exc:
            errors.append(
                {
                    "row_id": rid,
                    "code": "schema_validation_error",
                    "message": str(exc),
                }
            )
            continue
        record = validate_generated(source, narrative)
        row_errors = list(record.validation_errors)
        row_warnings = list(record.validation_warnings)
        if str(raw.get("generated_negative_text", "")) != narrative.hard_negative_text:
            row_errors.append(
                {
                    "code": "generated_negative_text_mismatch",
                    "message": "generated_negative_text must equal hard_negative_text for downstream compatibility.",
                }
            )
        validation_rows.append(
            {
                "row_id": rid,
                "target_window_id": raw.get("target_window_id"),
                "positive_view": raw.get("positive_view"),
                "negative_window_id": raw.get("negative_window_id"),
                "errors": row_errors,
                "warnings": row_warnings,
            }
        )
        if row_errors:
            errors.append({"row_id": rid, "code": "row_validation_failed", "errors": row_errors})
        else:
            valid_count += 1
        warning_count += len(row_warnings)

    duplicate_rows = sorted([rid for rid, count in seen.items() if count > 1])
    for rid in duplicate_rows:
        errors.append({"row_id": rid, "code": "duplicate_row_id", "count": seen[rid]})

    missing_rows: list[str] = []
    if not allow_partial:
        missing_rows = sorted(set(manifest_by_row_id) - set(seen))
        for rid in missing_rows[:100]:
            errors.append({"row_id": rid, "code": "missing_generated_row"})

    if expected_count > 0 and len(bank_rows) != expected_count:
        errors.append(
            {
                "code": "expected_count_mismatch",
                "expected_count": int(expected_count),
                "actual_count": len(bank_rows),
            }
        )

    status = "pass" if not errors and bank_rows else "fail"
    return {
        "schema_version": "hard_negative_bank_validation_report_v1",
        "status": status,
        "manifest_jsonl": str(manifest_jsonl),
        "bank_jsonl": str(bank_jsonl),
        "manifest_row_count": len(manifest),
        "bank_row_count": len(bank_rows),
        "valid_row_count": valid_count,
        "error_count": len(errors),
        "warning_count": warning_count,
        "duplicate_row_count": len(duplicate_rows),
        "missing_row_count": len(missing_rows),
        "expected_count": int(expected_count),
        "allow_partial": bool(allow_partial),
        "local_prose_generated": False,
        "view_counts": dict(sorted(Counter(str(row.get("positive_view", "")) for row in bank_rows).items())),
        "errors": errors,
        "validation": validation_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST_JSONL)
    parser.add_argument("--bank-jsonl", type=Path, default=DEFAULT_BANK_JSONL)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    report = validate_bank(
        manifest_jsonl=args.manifest_jsonl,
        bank_jsonl=args.bank_jsonl,
        expected_count=int(args.expected_count),
        allow_partial=bool(args.allow_partial),
    )
    output_report = (
        _resolve(args.output_report)
        if args.output_report is not None
        else _resolve(args.bank_jsonl).parent / "hard_negative_bank_validation_report.json"
    )
    _write_json(output_report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "bank_row_count": report["bank_row_count"],
                "valid_row_count": report["valid_row_count"],
                "error_count": report["error_count"],
                "warning_count": report["warning_count"],
                "report": str(output_report),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
