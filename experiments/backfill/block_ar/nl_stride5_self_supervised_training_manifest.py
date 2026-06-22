#!/usr/bin/env python
"""Prepare the stride-5 14+14 narrative bank for self-supervised retrieval.

This utility does not generate narrative prose. It flattens already validated
Codex/GPT-authored target reports into a neutral manifest that can feed both
active retrieval families:

1. text-space narrative-to-narrative contrastive retrieval;
2. text-to-projected SNI memory contrastive retrieval.

The important labeling rule is that positive texts are labeled to the selected
historical target window, while hard-negative texts are labeled to the distinct
historical prefix they describe. The pair rows preserve the explicit
positive-vs-hard-negative relation for contrastive losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BANK_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_fourteen_view_bank_988b/stride5_fourteen_view_bank_report.json"
)
DEFAULT_SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_self_supervised_training_manifest_990a"
)
EXPECTED_PAIRS_PER_TARGET = 14


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _window_index(window_id: str) -> int:
    match = re.search(r"_(\d+)$", str(window_id))
    if not match:
        raise ValueError(f"cannot parse window index from {window_id!r}")
    return int(match.group(1))


def _memory_target_count(support_arrays_path: Path | None) -> int | None:
    if support_arrays_path is None:
        return None
    path = _resolve(support_arrays_path)
    if not path.exists():
        raise FileNotFoundError(path)
    arrays = np.load(path)
    if "memory_targets" not in arrays:
        raise ValueError(f"{path}: missing memory_targets array")
    return int(arrays["memory_targets"].shape[0])


def _text_digest(rows: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        parts = [
            str(row.get("example_id", "")),
            str(row.get("role", "")),
            str(row.get("label_window_id", "")),
            str(row.get("text", "")),
        ]
        hasher.update("\t".join(parts).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _record_error(
    errors: list[dict[str, Any]],
    *,
    code: str,
    target_window_id: str | None = None,
    **payload: Any,
) -> None:
    error = {"code": code}
    if target_window_id is not None:
        error["target_window_id"] = target_window_id
    error.update(payload)
    errors.append(error)


def _load_bank_records(bank_report_path: Path, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bank_report = _read_json(bank_report_path)
    if bank_report.get("status") != "pass":
        _record_error(errors, code="bank_status_not_pass", status=bank_report.get("status"))
    if bank_report.get("processed_all_targets") is not True:
        _record_error(
            errors,
            code="bank_not_marked_processed_all_targets",
            value=bank_report.get("processed_all_targets"),
        )
    if bank_report.get("local_prose_generated") is not False:
        _record_error(
            errors,
            code="bank_local_prose_generated_not_false",
            value=bank_report.get("local_prose_generated"),
        )
    records = bank_report.get("records", [])
    if not isinstance(records, list):
        _record_error(errors, code="bank_records_not_list")
        return []
    return [row for row in records if isinstance(row, dict)]


def _target_report_path(record: dict[str, Any], bank_report_path: Path) -> Path:
    raw = record.get("report_path")
    if raw:
        return _resolve(Path(str(raw)))
    target_id = str(record.get("target_window_id") or record.get("window_id") or "")
    return bank_report_path.parent / "targets" / target_id / "fourteen_view_report.json"


def _example_row(
    *,
    example_id: str,
    role: str,
    view_name: str,
    text: str,
    label_window_id: str,
    target_window_id: str,
    paired_window_id: str,
    source_report: Path,
    source_bank_report: Path,
) -> dict[str, Any]:
    label_index = _window_index(label_window_id)
    target_index = _window_index(target_window_id)
    row = {
        "schema_version": "nl_stride5_self_supervised_training_example_v1",
        "example_id": example_id,
        "role": role,
        "view_name": view_name,
        "text": text,
        "label_window_id": label_window_id,
        "label_window_index": label_index,
        "target_window_id": target_window_id,
        "target_window_index": target_index,
        "source_report": str(source_report),
        "source_bank_report": str(source_bank_report),
    }
    if role == "positive":
        row["paired_negative_window_id"] = paired_window_id
    else:
        row["paired_positive_window_id"] = paired_window_id
    return row


def _check_memory_label(
    *,
    errors: list[dict[str, Any]],
    memory_rows: int | None,
    window_id: str,
    target_window_id: str,
    role: str,
) -> None:
    if memory_rows is None:
        return
    index = _window_index(window_id)
    if index < 0 or index >= memory_rows:
        _record_error(
            errors,
            code="memory_label_out_of_range",
            target_window_id=target_window_id,
            role=role,
            window_id=window_id,
            window_index=index,
            memory_target_count=memory_rows,
        )


def _append_target_rows(
    *,
    record: dict[str, Any],
    bank_report_path: Path,
    memory_rows: int | None,
    examples: list[dict[str, Any]],
    pairs_out: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    if record.get("status") != "pass":
        _record_error(
            errors,
            code="bank_record_status_not_pass",
            target_window_id=str(record.get("target_window_id") or record.get("window_id") or ""),
            status=record.get("status"),
        )
        return
    report_path = _target_report_path(record, bank_report_path)
    if not report_path.exists():
        _record_error(
            errors,
            code="missing_target_report",
            target_window_id=str(record.get("target_window_id") or record.get("window_id") or ""),
            report_path=str(report_path),
        )
        return
    target_report = _read_json(report_path)
    target_window_id = str(
        target_report.get("target_window_id")
        or record.get("target_window_id")
        or record.get("window_id")
        or ""
    )
    if target_report.get("status") != "pass":
        _record_error(
            errors,
            code="target_report_status_not_pass",
            target_window_id=target_window_id,
            status=target_report.get("status"),
        )
    if target_report.get("local_prose_generated") is not False:
        _record_error(
            errors,
            code="target_local_prose_generated_not_false",
            target_window_id=target_window_id,
            value=target_report.get("local_prose_generated"),
        )
    try:
        target_index = _window_index(target_window_id)
    except ValueError as exc:
        _record_error(
            errors,
            code="target_window_index_parse_failed",
            target_window_id=target_window_id,
            message=str(exc),
        )
        return

    pairs = target_report.get("pairs", [])
    if not isinstance(pairs, list):
        _record_error(errors, code="target_pairs_not_list", target_window_id=target_window_id)
        return
    if len(pairs) != EXPECTED_PAIRS_PER_TARGET:
        _record_error(
            errors,
            code="target_pair_count_mismatch",
            target_window_id=target_window_id,
            expected=EXPECTED_PAIRS_PER_TARGET,
            actual=len(pairs),
        )

    positive_texts: list[str] = []
    negative_texts: list[str] = []
    for pos, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            _record_error(
                errors,
                code="pair_not_object",
                target_window_id=target_window_id,
                pair_position=pos,
            )
            continue
        view_name = str(pair.get("view_name", "")).strip()
        positive_text = str(pair.get("positive_text", "")).strip()
        negative_text = str(pair.get("negative_text", "")).strip()
        negative_window_id = str(pair.get("negative_window_id", "")).strip()
        if not view_name:
            _record_error(errors, code="missing_view_name", target_window_id=target_window_id, pair_position=pos)
        if not positive_text:
            _record_error(errors, code="missing_positive_text", target_window_id=target_window_id, pair_position=pos)
        if not negative_text:
            _record_error(errors, code="missing_negative_text", target_window_id=target_window_id, pair_position=pos)
        if not negative_window_id:
            _record_error(errors, code="missing_negative_window_id", target_window_id=target_window_id, pair_position=pos)
            continue
        try:
            negative_index = _window_index(negative_window_id)
        except ValueError as exc:
            _record_error(
                errors,
                code="negative_window_index_parse_failed",
                target_window_id=target_window_id,
                pair_position=pos,
                negative_window_id=negative_window_id,
                message=str(exc),
            )
            continue
        if negative_window_id == target_window_id:
            _record_error(
                errors,
                code="negative_window_matches_target",
                target_window_id=target_window_id,
                view_name=view_name,
            )
        _check_memory_label(
            errors=errors,
            memory_rows=memory_rows,
            window_id=target_window_id,
            target_window_id=target_window_id,
            role="positive",
        )
        _check_memory_label(
            errors=errors,
            memory_rows=memory_rows,
            window_id=negative_window_id,
            target_window_id=target_window_id,
            role="hard_negative",
        )
        positive_example_id = f"{target_window_id}__{view_name}__positive"
        negative_example_id = f"{target_window_id}__{view_name}__hard_negative"
        examples.append(
            _example_row(
                example_id=positive_example_id,
                role="positive",
                view_name=view_name,
                text=positive_text,
                label_window_id=target_window_id,
                target_window_id=target_window_id,
                paired_window_id=negative_window_id,
                source_report=report_path,
                source_bank_report=bank_report_path,
            )
        )
        examples.append(
            _example_row(
                example_id=negative_example_id,
                role="hard_negative",
                view_name=view_name,
                text=negative_text,
                label_window_id=negative_window_id,
                target_window_id=target_window_id,
                paired_window_id=target_window_id,
                source_report=report_path,
                source_bank_report=bank_report_path,
            )
        )
        pairs_out.append(
            {
                "schema_version": "nl_stride5_self_supervised_training_pair_v1",
                "pair_id": f"{target_window_id}__{view_name}",
                "target_window_id": target_window_id,
                "target_window_index": target_index,
                "view_name": view_name,
                "positive_example_id": positive_example_id,
                "negative_example_id": negative_example_id,
                "negative_window_id": negative_window_id,
                "negative_window_index": negative_index,
                "quality_notes": pair.get("quality_notes", []),
                "source_report": str(report_path),
            }
        )
        positive_texts.append(positive_text)
        negative_texts.append(negative_text)

    if len(set(positive_texts)) != len(positive_texts):
        _record_error(
            errors,
            code="duplicate_positive_texts",
            target_window_id=target_window_id,
            unique_count=len(set(positive_texts)),
            total_count=len(positive_texts),
        )
    if len(set(negative_texts)) != len(negative_texts):
        _record_error(
            errors,
            code="duplicate_negative_texts",
            target_window_id=target_window_id,
            unique_count=len(set(negative_texts)),
            total_count=len(negative_texts),
        )


def _summary(
    *,
    examples: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    target_count: int,
    memory_rows: int | None,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    roles = Counter(str(row.get("role", "")) for row in examples)
    labels = {int(row["label_window_index"]) for row in examples}
    return {
        "target_count": int(target_count),
        "pair_count": len(pairs),
        "example_count": len(examples),
        "positive_example_count": roles.get("positive", 0),
        "hard_negative_example_count": roles.get("hard_negative", 0),
        "unique_label_window_count": len(labels),
        "memory_target_count": memory_rows,
        "validation_error_count": len(errors),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Stride-5 Self-Supervised Narrative Training Manifest",
        "",
        "This artifact flattens the validated stride-5 14-positive / "
        "14-hard-negative bank into examples for the two retrieval families. "
        "It does not generate narrative text.",
        "",
        "## Status",
        "",
        f"- Status: `{report['status']}`",
        f"- Target count: `{summary['target_count']}`",
        f"- Pair count: `{summary['pair_count']}`",
        f"- Positive examples: `{summary['positive_example_count']}`",
        f"- Hard-negative examples: `{summary['hard_negative_example_count']}`",
        f"- Total examples: `{summary['example_count']}`",
        f"- Unique memory labels: `{summary['unique_label_window_count']}`",
        f"- Memory target rows: `{summary['memory_target_count']}`",
        f"- Text hash digest: `{report['text_hash_digest']}`",
        "",
        "## Artifacts",
        "",
        f"- Training examples JSONL: `{report['artifact_paths']['training_examples_jsonl']}`",
        f"- Training pairs JSONL: `{report['artifact_paths']['training_pairs_jsonl']}`",
        f"- Report JSON: `{report['artifact_paths']['report_json']}`",
        f"- Report Markdown: `{report['artifact_paths']['report_markdown']}`",
        "",
        "## Labeling Rule",
        "",
        "- Positive rows map to the selected target window memory.",
        "- Hard-negative rows map to the incompatible historical window they describe.",
        "- Pair rows preserve the explicit positive-vs-hard-negative relation.",
        "",
        "## Validation",
        "",
        f"- Error count: `{summary['validation_error_count']}`",
    ]
    for error in report["validation"]["errors"][:20]:
        lines.append(f"- `{error.get('code')}`: `{error}`")
    if len(report["validation"]["errors"]) > 20:
        lines.append(f"- ... {len(report['validation']['errors']) - 20} more errors")
    lines.append("")
    _write_text(path, "\n".join(lines))


def build_self_supervised_training_manifest(
    *,
    bank_report_path: str | Path = DEFAULT_BANK_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    support_arrays_path: str | Path | None = DEFAULT_SUPPORT_ARRAYS,
) -> dict[str, Any]:
    bank_path = _resolve(bank_report_path)
    output = _resolve(output_dir)
    errors: list[dict[str, Any]] = []
    try:
        memory_rows = _memory_target_count(
            None if support_arrays_path is None else Path(support_arrays_path)
        )
    except Exception as exc:
        memory_rows = None
        _record_error(
            errors,
            code="support_arrays_load_failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )

    records = _load_bank_records(bank_path, errors)
    examples: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for record in records:
        _append_target_rows(
            record=record,
            bank_report_path=bank_path,
            memory_rows=memory_rows,
            examples=examples,
            pairs_out=pairs,
            errors=errors,
        )

    report_path = output / "training_manifest_report.json"
    markdown_path = output / "training_manifest_report.md"
    examples_path = output / "training_examples.jsonl"
    pairs_path = output / "training_pairs.jsonl"
    _write_jsonl(examples_path, examples)
    _write_jsonl(pairs_path, pairs)
    report = {
        "schema_version": "nl_stride5_self_supervised_training_manifest_v1",
        "status": "pass" if not errors else "fail",
        "bank_report_path": str(bank_path),
        "support_arrays_path": (
            None if support_arrays_path is None else str(_resolve(support_arrays_path))
        ),
        "summary": _summary(
            examples=examples,
            pairs=pairs,
            target_count=len(records),
            memory_rows=memory_rows,
            errors=errors,
        ),
        "text_hash_digest": _text_digest(examples),
        "validation": {"status": "pass" if not errors else "fail", "errors": errors},
        "artifact_paths": {
            "training_examples_jsonl": str(examples_path),
            "training_pairs_jsonl": str(pairs_path),
            "report_json": str(report_path),
            "report_markdown": str(markdown_path),
        },
    }
    _write_json(report_path, report)
    _write_markdown(markdown_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-report", type=Path, default=DEFAULT_BANK_REPORT)
    parser.add_argument("--support-arrays", type=Path, default=DEFAULT_SUPPORT_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_self_supervised_training_manifest(
        bank_report_path=args.bank_report,
        output_dir=args.output_dir,
        support_arrays_path=args.support_arrays,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
