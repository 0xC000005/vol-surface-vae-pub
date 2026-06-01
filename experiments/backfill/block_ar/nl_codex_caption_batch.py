#!/usr/bin/env python
"""Resume-safe Codex caption batch runner for RiskManagerCaptionV2.

This is the premium/gold scenario-to-text lane for the NL prefix-latent
workflow. It reuses the risk-manager specialist prompt and leakage validator
from ``nl_risk_manager_caption_v2.py`` while routing generation through
``codex exec`` instead of the OpenAI API.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (  # noqa: E402
    ARCHETYPES,
    DEFAULT_PIPELINE_REPORT,
    PROMPT_VERSION,
    RiskManagerCaptionV2,
    SpecialistStandards,
    _bundle_market_lines,
    _source_text,
    build_caption_messages,
    load_pipeline_report,
    load_specialist_standards,
    validate_caption_v2,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_codex_full_corpus_917a"
)
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "xhigh"


@dataclass(frozen=True)
class CaptionRecord:
    caption: RiskManagerCaptionV2 | None
    validation_errors: list[dict[str, Any]]
    validation_warnings: list[dict[str, Any]]
    error: dict[str, Any] | None = None
    skipped_existing: bool = False


class RiskManagerCaptionV2Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captions: list[RiskManagerCaptionV2] = Field(default_factory=list)


def strict_caption_schema() -> dict[str, Any]:
    """Return a JSON schema with all RiskManagerCaptionV2 fields required."""

    schema = RiskManagerCaptionV2.model_json_schema()
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        raise ValueError("RiskManagerCaptionV2 schema has no properties")
    schema["required"] = list(properties)
    schema["additionalProperties"] = False
    return schema


def _require_all_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively make pydantic object schemas strict for Codex output."""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict) and properties:
                node["required"] = list(properties)
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


def strict_caption_batch_schema() -> dict[str, Any]:
    """Return a strict JSON schema for a batch of RiskManagerCaptionV2 objects."""

    schema = RiskManagerCaptionV2Batch.model_json_schema()
    return _require_all_properties(schema)


def write_strict_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strict_caption_schema(), indent=2, sort_keys=True) + "\n")
    return path


def select_bundles(
    pipeline_report: str | Path | dict[str, Any],
    *,
    count: int = 0,
    offset: int = 0,
    split: str = "all",
    window_ids: list[str] | None = None,
    selection_mode: str = "ordered",
) -> list[dict[str, Any]]:
    """Select ordered corpus bundles for Codex labeling.

    ``count <= 0`` means all selected bundles after offset/window filtering.
    """

    report = (
        load_pipeline_report(pipeline_report)
        if not isinstance(pipeline_report, dict)
        else pipeline_report
    )
    bundles = report.get("narrative_bundles", [])
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("pipeline report does not contain narrative_bundles")

    wanted_ids = set(window_ids or [])
    selected: list[dict[str, Any]] = []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        window_id = str(bundle.get("window_id", ""))
        if wanted_ids and window_id not in wanted_ids:
            continue
        if split != "all" and str(bundle.get("manifest_split", "")).lower() != split:
            continue
        selected.append(bundle)

    if (
        selection_mode == "split_balanced"
        and not wanted_ids
        and split == "all"
        and int(count) > 0
    ):
        split_order = ("train", "validation", "val", "test")
        grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in split_order}
        other: list[dict[str, Any]] = []
        for bundle in selected:
            key = str(bundle.get("manifest_split", "")).lower()
            if key in grouped:
                grouped[key].append(bundle)
            else:
                other.append(bundle)
        balanced: list[dict[str, Any]] = []
        row_no = 0
        while len(balanced) < int(count):
            progressed = False
            for key in split_order:
                bucket = grouped[key]
                if row_no < len(bucket) and len(balanced) < int(count):
                    balanced.append(bucket[row_no])
                    progressed = True
            if not progressed:
                break
            row_no += 1
        for bundle in other:
            if len(balanced) >= int(count):
                break
            balanced.append(bundle)
        selected = balanced
    elif selection_mode != "ordered":
        raise ValueError(f"unsupported selection_mode: {selection_mode}")

    if offset:
        selected = selected[int(offset) :]
    if int(count) > 0:
        selected = selected[: int(count)]
    if wanted_ids:
        missing = wanted_ids - {str(item.get("window_id", "")) for item in selected}
        if missing:
            raise ValueError(f"window_ids not found after filtering: {sorted(missing)}")
    return selected


def _safe_window_id(window_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in window_id)


def caption_path(output_dir: Path, window_id: str) -> Path:
    return output_dir / "captions" / f"codex_gpt55_caption_{_safe_window_id(window_id)}.json"


def prompt_path(output_dir: Path, window_id: str) -> Path:
    return output_dir / "prompts" / f"prompt_{_safe_window_id(window_id)}.txt"


def batch_prompt_path(output_dir: Path, batch_no: int, bundles: list[dict[str, Any]]) -> Path:
    first = _safe_window_id(str(bundles[0].get("window_id", "window")))
    last = _safe_window_id(str(bundles[-1].get("window_id", "window")))
    return output_dir / "prompts" / f"prompt_batch_{batch_no:06d}_{first}_to_{last}.txt"


def codex_events_path(output_dir: Path, window_id: str) -> Path:
    return output_dir / "codex_events" / f"codex_events_{_safe_window_id(window_id)}.jsonl"


def batch_codex_events_path(output_dir: Path, batch_no: int, bundles: list[dict[str, Any]]) -> Path:
    first = _safe_window_id(str(bundles[0].get("window_id", "window")))
    last = _safe_window_id(str(bundles[-1].get("window_id", "window")))
    return output_dir / "codex_events" / f"codex_events_batch_{batch_no:06d}_{first}_to_{last}.jsonl"


def batch_output_path(output_dir: Path, batch_no: int, bundles: list[dict[str, Any]]) -> Path:
    first = _safe_window_id(str(bundles[0].get("window_id", "window")))
    last = _safe_window_id(str(bundles[-1].get("window_id", "window")))
    return output_dir / "batches" / f"codex_gpt55_batch_{batch_no:06d}_{first}_to_{last}.json"


def build_codex_prompt(
    bundle: dict[str, Any],
    *,
    standards: SpecialistStandards,
) -> str:
    messages = build_caption_messages(bundle, standards=standards)
    system = next(item["content"] for item in messages if item["role"] == "system")
    user = next(item["content"] for item in messages if item["role"] == "user")
    return (
        "You are generating exactly one risk-manager-grade scenario-to-text caption. "
        "Return only JSON matching the provided output schema. Do not modify files. "
        "Do not call external APIs. Do not wrap the JSON in markdown fences.\n\n"
        "Important leakage rule: the training_caption must describe only the current "
        "or recent historical prefix, not the realized future, forecast horizon, "
        "terminal path, generated scenario, VaR, ES, or target P&L. The "
        "training_caption must avoid those leakage words even in negated form. "
        "Put exclusions in leakage_exclusions or no_forecast_caveat instead. "
        "The no_forecast_caveat must contain the exact phrase 'not a forecast'.\n\n"
        f"SYSTEM:\n{system}\n\n"
        f"USER:\n{user}\n"
    )


def _bundle_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    calendar = bundle.get("calendar", {})
    if not isinstance(calendar, dict):
        calendar = {}
    return {
        "window_id": bundle.get("window_id"),
        "manifest_split": bundle.get("manifest_split"),
        "source_index": bundle.get("source_index"),
        "window_index": bundle.get("window_index"),
        "current_recent_prefix_dates": {
            "calendar_start_date": calendar.get("calendar_start_date", ""),
            "calendar_end_date": calendar.get("calendar_end_date", ""),
        },
        "market_implication_lines": _bundle_market_lines(bundle),
        "existing_caption_to_improve": _source_text(bundle),
        "allowed_archetypes": list(ARCHETYPES),
    }


def build_codex_batch_prompt(
    bundles: list[dict[str, Any]],
    *,
    standards: SpecialistStandards,
) -> str:
    """Build one Codex prompt for several independent caption payloads."""

    if not bundles:
        raise ValueError("build_codex_batch_prompt requires at least one bundle")
    doc_summary = "\n\n".join(
        (
            f"Source document: {doc.path.name}\n"
            f"SHA256: {doc.sha256}\n"
            f"Extracted standard excerpt:\n{doc.excerpt}"
        )
        for doc in standards.documents
    )
    payloads = [_bundle_payload(bundle) for bundle in bundles]
    return (
        "You are generating risk-manager-grade scenario-to-text captions. "
        "Return only JSON matching the provided output schema. Do not modify "
        "files. Do not call external APIs. Do not wrap the JSON in markdown "
        "fences.\n\n"
        "Generate one RiskManagerCaptionV2 object per payload, preserving each "
        "input window_id exactly. Each caption must be independent. The "
        "training_caption must describe only the current or recent historical "
        "prefix, not the realized future, forecast horizon, terminal path, "
        "generated scenario, VaR, ES, or target P&L. The training_caption must "
        "avoid those leakage words even in negated form. Put exclusions in "
        "leakage_exclusions or no_forecast_caveat instead. The "
        "no_forecast_caveat must contain the exact phrase 'not a forecast'.\n\n"
        "Risk-manager standards:\n"
        "Risk-manager scenario narratives must translate numbers into meaning. "
        "They should include a scenario title, mechanical summary, archetype, "
        "trigger, transmission channel, cross-asset reaction, sequencing, "
        "portfolio vulnerability, risk-manager implication, evidence used, "
        "ambiguity flags, and a no-forecast caveat. They must use plain "
        "investment language, preserve cause-and-effect logic, avoid fake "
        "certainty, avoid unsupported real-world events, and separate current "
        "conditions from future scenario outcomes.\n\n"
        f"Specialist source documents:\n{doc_summary}\n\n"
        "Current/recent prefix payloads:\n"
        f"{json.dumps(payloads, indent=2, sort_keys=True)}"
    )


def _load_caption_json(path: Path) -> RiskManagerCaptionV2:
    text = path.read_text(encoding="utf-8").strip()
    return RiskManagerCaptionV2.model_validate_json(_extract_json_object(text))


def validate_caption_file(path: Path) -> CaptionRecord:
    try:
        caption = _load_caption_json(path)
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        return CaptionRecord(
            caption=None,
            validation_errors=[],
            validation_warnings=[],
            error={
                "error_type": type(exc).__name__,
                "message": str(exc),
                "path": str(path),
            },
        )
    issues = validate_caption_v2(caption)
    return CaptionRecord(
        caption=caption,
        validation_errors=[
            issue.model_dump() for issue in issues if issue.severity == "error"
        ],
        validation_warnings=[
            issue.model_dump() for issue in issues if issue.severity != "error"
        ],
    )


def _load_caption_batch_json(path: Path) -> RiskManagerCaptionV2Batch:
    text = path.read_text(encoding="utf-8").strip()
    return RiskManagerCaptionV2Batch.model_validate_json(_extract_json_object(text))


def _extract_json_object(text: str) -> str:
    """Accept raw JSON or a final message containing one JSON object."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Codex output did not contain a JSON object")
    return stripped[start : end + 1]


def run_codex_caption(
    bundle: dict[str, Any],
    *,
    standards: SpecialistStandards,
    output_dir: Path,
    schema_path: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    dry_run: bool = False,
) -> CaptionRecord:
    window_id = str(bundle.get("window_id", ""))
    out_path = caption_path(output_dir, window_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_path(output_dir, window_id)
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    events_file = codex_events_path(output_dir, window_id)
    events_file.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_codex_prompt(bundle, standards=standards)
    prompt_file.write_text(prompt, encoding="utf-8")
    if dry_run:
        return CaptionRecord(
            caption=None,
            validation_errors=[],
            validation_warnings=[],
            error={
                "error_type": "DryRun",
                "message": "Prompt/schema written; Codex was not invoked.",
                "path": str(prompt_file),
            },
        )

    cmd = [
        "codex",
        "exec",
        "--ephemeral",
        "--json",
        "-m",
        model,
        "-c",
        f"model_reasoning_effort='{reasoning_effort}'",
        "--sandbox",
        "read-only",
        "--cd",
        str(ROOT),
        "--output-schema",
        str(schema_path),
        "-o",
        str(out_path),
        prompt,
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    events_file.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        return CaptionRecord(
            caption=None,
            validation_errors=[],
            validation_warnings=[],
            error={
                "error_type": "CodexExecFailed",
                "returncode": completed.returncode,
                "message": completed.stderr.strip()[-4000:],
                "events_path": str(events_file),
            },
        )

    record = validate_caption_file(out_path)
    if record.caption is None or record.validation_errors:
        return record
    normalized = record.caption.model_dump()
    out_path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    return record


def run_codex_caption_batch(
    bundles: list[dict[str, Any]],
    *,
    batch_no: int,
    standards: SpecialistStandards,
    output_dir: Path,
    schema_path: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    dry_run: bool = False,
) -> list[tuple[str, CaptionRecord, dict[str, Any]]]:
    """Run one Codex call that returns several RiskManagerCaptionV2 captions."""

    if not bundles:
        return []
    out_path = batch_output_path(output_dir, batch_no, bundles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_file = batch_prompt_path(output_dir, batch_no, bundles)
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    events_file = batch_codex_events_path(output_dir, batch_no, bundles)
    events_file.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_codex_batch_prompt(bundles, standards=standards)
    prompt_file.write_text(prompt, encoding="utf-8")

    results: list[tuple[str, CaptionRecord, dict[str, Any]]] = []
    base_metadata = {
        "caption_path": "",
        "prompt_path": str(prompt_file),
        "events_path": str(events_file),
        "batch_output_path": str(out_path),
        "batch_no": int(batch_no),
        "skipped_existing": False,
    }
    if dry_run:
        for bundle in bundles:
            window_id = str(bundle.get("window_id", ""))
            results.append(
                (
                    window_id,
                    CaptionRecord(
                        caption=None,
                        validation_errors=[],
                        validation_warnings=[],
                        error={
                            "error_type": "DryRun",
                            "message": "Batch prompt/schema written; Codex was not invoked.",
                            "path": str(prompt_file),
                        },
                    ),
                    {
                        **base_metadata,
                        "window_id": window_id,
                        "caption_path": str(caption_path(output_dir, window_id)),
                    },
                )
            )
        return results

    cmd = [
        "codex",
        "exec",
        "--ephemeral",
        "--json",
        "-m",
        model,
        "-c",
        f"model_reasoning_effort='{reasoning_effort}'",
        "--sandbox",
        "read-only",
        "--cd",
        str(ROOT),
        "--output-schema",
        str(schema_path),
        "-o",
        str(out_path),
        prompt,
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    events_file.write_text(completed.stdout, encoding="utf-8")
    requested_ids = [str(bundle.get("window_id", "")) for bundle in bundles]
    if completed.returncode != 0:
        for window_id in requested_ids:
            results.append(
                (
                    window_id,
                    CaptionRecord(
                        caption=None,
                        validation_errors=[],
                        validation_warnings=[],
                        error={
                            "error_type": "CodexExecFailed",
                            "returncode": completed.returncode,
                            "message": completed.stderr.strip()[-4000:],
                            "events_path": str(events_file),
                        },
                    ),
                    {
                        **base_metadata,
                        "window_id": window_id,
                        "caption_path": str(caption_path(output_dir, window_id)),
                    },
                )
            )
        return results

    try:
        parsed = _load_caption_batch_json(out_path)
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        for window_id in requested_ids:
            results.append(
                (
                    window_id,
                    CaptionRecord(
                        caption=None,
                        validation_errors=[],
                        validation_warnings=[],
                        error={
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "path": str(out_path),
                        },
                    ),
                    {
                        **base_metadata,
                        "window_id": window_id,
                        "caption_path": str(caption_path(output_dir, window_id)),
                    },
                )
            )
        return results

    captions_by_id: dict[str, RiskManagerCaptionV2] = {}
    duplicate_ids: set[str] = set()
    for caption in parsed.captions:
        window_id = str(caption.window_id)
        if window_id in captions_by_id:
            duplicate_ids.add(window_id)
        captions_by_id[window_id] = caption

    for window_id in requested_ids:
        cap_path = caption_path(output_dir, window_id)
        metadata = {
            **base_metadata,
            "window_id": window_id,
            "caption_path": str(cap_path),
        }
        caption = captions_by_id.get(window_id)
        if caption is None:
            results.append(
                (
                    window_id,
                    CaptionRecord(
                        caption=None,
                        validation_errors=[],
                        validation_warnings=[],
                        error={
                            "error_type": "MissingCaption",
                            "message": f"Batch output did not contain window_id {window_id}",
                            "path": str(out_path),
                        },
                    ),
                    metadata,
                )
            )
            continue
        if window_id in duplicate_ids:
            results.append(
                (
                    window_id,
                    CaptionRecord(
                        caption=None,
                        validation_errors=[],
                        validation_warnings=[],
                        error={
                            "error_type": "DuplicateCaption",
                            "message": f"Batch output contained duplicate window_id {window_id}",
                            "path": str(out_path),
                        },
                    ),
                    metadata,
                )
            )
            continue
        issues = validate_caption_v2(caption)
        record = CaptionRecord(
            caption=caption,
            validation_errors=[
                issue.model_dump() for issue in issues if issue.severity == "error"
            ],
            validation_warnings=[
                issue.model_dump() for issue in issues if issue.severity != "error"
            ],
        )
        if not record.validation_errors:
            cap_path.parent.mkdir(parents=True, exist_ok=True)
            cap_path.write_text(
                json.dumps(caption.model_dump(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        results.append((window_id, record, metadata))
    return results


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    standards = load_specialist_standards()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_mode = int(getattr(args, "batch_size", 1)) > 1
    schema_payload = (
        strict_caption_batch_schema() if batch_mode else strict_caption_schema()
    )
    schema_path = output_dir / (
        "risk_manager_caption_v2_codex_batch_strict_schema.json"
        if batch_mode
        else "risk_manager_caption_v2_codex_strict_schema.json"
    )
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema_payload, indent=2, sort_keys=True) + "\n")
    bundles = select_bundles(
        args.pipeline_report,
        count=int(args.count),
        offset=int(args.offset),
        split=str(args.split),
        window_ids=list(args.window_id or []),
        selection_mode=str(args.selection_mode),
    )
    report_path = output_dir / "codex_caption_batch_report.json"
    captions_jsonl = output_dir / "codex_caption_batch_captions.jsonl"

    captions: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    def write_checkpoint() -> dict[str, Any]:
        validation_error_count = sum(len(row["errors"]) for row in validation_rows)
        codex_error_count = len(error_rows)
        if validation_error_count == 0 and codex_error_count == 0 and captions:
            status = "pass"
        elif captions:
            status = "partial_fail"
        else:
            status = "dry_run" if args.dry_run else "fail"
        report = {
            "status": status,
            "scope_note": (
                "Codex CLI premium/gold RiskManagerCaptionV2 corpus labeling lane. "
                "Outputs are candidates for reverse-direction support/scenario gates, "
                "not a production default by themselves."
            ),
            "pipeline_report": str(args.pipeline_report),
            "prompt_version": PROMPT_VERSION,
            "codex_model": str(args.model),
            "reasoning_effort": str(args.reasoning_effort),
            "dry_run": bool(args.dry_run),
            "batch_size": int(getattr(args, "batch_size", 1)),
            "selection_mode": str(args.selection_mode),
            "requested_count": int(len(bundles)),
            "caption_count": int(len(captions)),
            "validation_error_count": int(validation_error_count),
            "codex_error_count": int(codex_error_count),
            "skipped_existing_count": int(
                sum(1 for row in metadata_rows if row.get("skipped_existing"))
            ),
            "specialist_documents": [
                {"path": str(doc.path), "sha256": doc.sha256}
                for doc in standards.documents
            ],
            "captions": captions,
            "validation": validation_rows,
            "errors": error_rows,
            "metadata": metadata_rows,
            "artifact_paths": {
                "report": str(report_path),
                "captions_jsonl": str(captions_jsonl),
                "strict_schema": str(schema_path),
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with captions_jsonl.open("w", encoding="utf-8") as handle:
            for row in captions:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return report

    batch_no = 0
    bundle_index = 0
    while bundle_index < len(bundles):
        bundle = bundles[bundle_index]
        window_id = str(bundle.get("window_id", ""))
        out_path = caption_path(output_dir, window_id)
        existing = validate_caption_file(out_path) if out_path.exists() else None
        if (
            existing is not None
            and existing.caption is not None
            and not existing.validation_errors
            and bool(args.skip_existing)
        ):
            captions.append(existing.caption.model_dump())
            validation_rows.append(
                {
                    "window_id": window_id,
                    "errors": existing.validation_errors,
                    "warnings": existing.validation_warnings,
                    "skipped_existing": True,
                }
            )
            metadata_rows.append(
                {
                    "window_id": window_id,
                    "caption_path": str(out_path),
                    "prompt_path": str(prompt_path(output_dir, window_id)),
                    "events_path": str(codex_events_path(output_dir, window_id)),
                    "skipped_existing": True,
                }
            )
            write_checkpoint()
            bundle_index += 1
            continue

        if not batch_mode:
            record = run_codex_caption(
                bundle,
                standards=standards,
                output_dir=output_dir,
                schema_path=schema_path,
                model=str(args.model),
                reasoning_effort=str(args.reasoning_effort),
                timeout_seconds=int(args.timeout_seconds),
                dry_run=bool(args.dry_run),
            )
            result_rows = [
                (
                    window_id,
                    record,
                    {
                        "window_id": window_id,
                        "caption_path": str(out_path),
                        "prompt_path": str(prompt_path(output_dir, window_id)),
                        "events_path": str(codex_events_path(output_dir, window_id)),
                        "skipped_existing": False,
                    },
                )
            ]
            bundle_index += 1
        else:
            chunk: list[dict[str, Any]] = []
            while bundle_index < len(bundles) and len(chunk) < int(args.batch_size):
                candidate = bundles[bundle_index]
                candidate_id = str(candidate.get("window_id", ""))
                candidate_path = caption_path(output_dir, candidate_id)
                candidate_existing = (
                    validate_caption_file(candidate_path)
                    if candidate_path.exists()
                    else None
                )
                if (
                    candidate_existing is not None
                    and candidate_existing.caption is not None
                    and not candidate_existing.validation_errors
                    and bool(args.skip_existing)
                ):
                    captions.append(candidate_existing.caption.model_dump())
                    validation_rows.append(
                        {
                            "window_id": candidate_id,
                            "errors": candidate_existing.validation_errors,
                            "warnings": candidate_existing.validation_warnings,
                            "skipped_existing": True,
                        }
                    )
                    metadata_rows.append(
                        {
                            "window_id": candidate_id,
                            "caption_path": str(candidate_path),
                            "prompt_path": str(prompt_path(output_dir, candidate_id)),
                            "events_path": str(codex_events_path(output_dir, candidate_id)),
                            "skipped_existing": True,
                        }
                    )
                    write_checkpoint()
                    bundle_index += 1
                    continue
                chunk.append(candidate)
                bundle_index += 1
            result_rows = run_codex_caption_batch(
                chunk,
                batch_no=batch_no,
                standards=standards,
                output_dir=output_dir,
                schema_path=schema_path,
                model=str(args.model),
                reasoning_effort=str(args.reasoning_effort),
                timeout_seconds=int(args.timeout_seconds),
                dry_run=bool(args.dry_run),
            )
            batch_no += 1

        stop_after_error = False
        for result_window_id, record, metadata in result_rows:
            if record.caption is not None:
                captions.append(record.caption.model_dump())
            validation_rows.append(
                {
                    "window_id": result_window_id,
                    "errors": record.validation_errors,
                    "warnings": record.validation_warnings,
                    "skipped_existing": False,
                }
            )
            metadata_rows.append(metadata)
            if record.error:
                error_rows.append({"window_id": result_window_id, **record.error})
                if not bool(args.continue_on_error):
                    stop_after_error = True
        write_checkpoint()
        if stop_after_error:
            break

    return write_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", type=Path, default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=0, help="0 means all selected bundles.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    parser.add_argument(
        "--selection-mode",
        choices=["ordered", "split_balanced"],
        default="ordered",
        help="Use split_balanced for representative pilots; ordered is best for resumable full-corpus runs.",
    )
    parser.add_argument("--window-id", action="append", default=[])
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generate this many missing captions per Codex call; 1 preserves the original per-window path.",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_batch(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "requested_count": report["requested_count"],
                "caption_count": report["caption_count"],
                "validation_error_count": report["validation_error_count"],
                "codex_error_count": report["codex_error_count"],
                "skipped_existing_count": report["skipped_existing_count"],
                "report": report["artifact_paths"]["report"],
                "captions_jsonl": report["artifact_paths"]["captions_jsonl"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
