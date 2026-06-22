#!/usr/bin/env python
"""Generate matched hard-negative narratives from a linked-window manifest.

This is the Codex/GPT authoring step for the NL hard-negative corpus. The input
manifest links each positive view to a real incompatible historical window. This
script asks Codex/GPT to write the negative narrative text and validates the
result. It never fills failed rows with deterministic prose.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_codex_caption_batch import (  # noqa: E402
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
)


DEFAULT_MANIFEST_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "hard_negative_bank_regeneration_985b_manifest/"
    "hard_negative_generation_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "hard_negative_bank_regeneration_985c_codex_batch"
)

LEAKAGE_PATTERNS = (
    re.compile(r"\bnext\s+\d+\s+(day|days|trading\s+day|trading\s+days)\b", re.I),
    re.compile(r"\brealized\s+(future|post[- ]window|outcome|path)\b", re.I),
    re.compile(r"\bterminal\s+(path|value|level|return|move|distribution)\b", re.I),
    re.compile(r"\bforecast\s+horizon\b", re.I),
    re.compile(r"\bVaR\b"),
    re.compile(r"\bES\b"),
    re.compile(r"\bwill\s+(rally|fall|rise|drop|sell off|tighten|widen)\b", re.I),
)
INTERNAL_PATTERNS = (
    re.compile(r"\bhard[- ]negative\b", re.I),
    re.compile(r"\bnegative example\b", re.I),
    re.compile(r"\bpositive example\b", re.I),
    re.compile(r"\bcontrastive\b", re.I),
    re.compile(r"\btraining data\b", re.I),
    re.compile(r"\bembedding\b", re.I),
    re.compile(r"\bretrieval\b", re.I),
    re.compile(r"\bscenario generator\b", re.I),
)
FACTOR_ALIASES = {
    "SPX": ("SPX", "equity", "equities", "stocks", "risk assets"),
    "VIX": ("VIX", "volatility", "vol"),
    "BBB_OAS": ("BBB", "lower-quality credit", "credit spreads", "spread"),
    "AAA_OAS": ("AAA", "high-grade credit", "credit spreads", "spread"),
    "DXY": ("DXY", "dollar", "broad dollar"),
    "USDJPY": ("USDJPY", "yen", "dollar-yen", "yen cross"),
    "CRUDE_OIL": ("crude", "oil", "energy"),
    "US2Y": ("US2Y", "front-end rates", "front end", "short rates"),
    "US10Y": ("US10Y", "long-end rates", "duration", "Treasury yields"),
    "GOLD": ("gold", "safe-haven metal", "precious metal"),
}


class HardNegativeNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: str = Field(min_length=3)
    target_window_id: str = Field(min_length=3)
    positive_view: str = Field(min_length=3)
    negative_window_id: str = Field(min_length=3)
    hard_negative_text: str = Field(min_length=12)
    quality_notes: list[str] = Field(default_factory=list)


class HardNegativeNarrativeBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narratives: list[HardNegativeNarrative] = Field(default_factory=list)


@dataclass(frozen=True)
class GenerationRecord:
    narrative: HardNegativeNarrative | None
    validation_errors: list[dict[str, Any]]
    validation_warnings: list[dict[str, Any]]
    error: dict[str, Any] | None = None
    skipped_existing: bool = False


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value)


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _extract_json_object(text: str) -> str:
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


def _require_all_properties(schema: dict[str, Any]) -> dict[str, Any]:
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


def strict_batch_schema() -> dict[str, Any]:
    return _require_all_properties(HardNegativeNarrativeBatch.model_json_schema())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def row_id(row: dict[str, Any]) -> str:
    return f"{row['target_window_id']}__{row['positive_view']}"


def generated_path(output_dir: Path, row_id_value: str) -> Path:
    return output_dir / "generated" / f"hard_negative_{_safe(row_id_value)}.json"


def _load_generated(path: Path) -> HardNegativeNarrative:
    return HardNegativeNarrative.model_validate_json(
        _extract_json_object(path.read_text(encoding="utf-8"))
    )


def _snippets(patterns: tuple[re.Pattern[str], ...], text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 45)
            end = min(len(text), match.end() + 45)
            hits.append({"pattern": pattern.pattern, "snippet": text[start:end].strip()})
    return hits


def _mentions_any_channel(text: str, channels: list[str]) -> bool:
    low = text.lower()
    for channel in channels:
        for alias in FACTOR_ALIASES.get(str(channel), (str(channel),)):
            if re.search(r"\b" + re.escape(alias.lower()).replace(r"\ ", r"\s+") + r"\b", low):
                return True
    return False


def _token_jaccard(left: str, right: str) -> float:
    ignore = {"the", "a", "an", "and", "or", "of", "to", "in", "with", "is", "are"}
    left_tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", left)
        if token.lower() not in ignore
    }
    right_tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", right)
        if token.lower() not in ignore
    }
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def validate_generated(row: dict[str, Any], narrative: HardNegativeNarrative) -> GenerationRecord:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    expected = {
        "row_id": row_id(row),
        "target_window_id": str(row["target_window_id"]),
        "positive_view": str(row["positive_view"]),
        "negative_window_id": str(row["negative_window_id"]),
    }
    for key, value in expected.items():
        if str(getattr(narrative, key)) != value:
            errors.append(
                {
                    "code": "identity_mismatch",
                    "field": key,
                    "expected": value,
                    "actual": str(getattr(narrative, key)),
                }
            )
    if str(row.get("target_window_id", "")) == str(row.get("negative_window_id", "")):
        errors.append(
            {
                "code": "same_window_negative_link",
                "message": "Hard negatives must link to a different incompatible historical window.",
            }
        )
    text = _compact(narrative.hard_negative_text)
    positive_text = _compact(row.get("positive_text", ""))
    if positive_text and text.lower() == positive_text.lower():
        errors.append(
            {
                "code": "copied_positive_text",
                "message": "Hard-negative text must not copy the positive view.",
            }
        )
    words = _word_count(text)
    min_words = 18
    if narrative.positive_view in {
        "risk_manager_memo",
        "institutional_risk_committee_note",
        "full_professional",
    }:
        min_words = 45
    elif narrative.positive_view in {"weekly_risk_monitor", "mechanism_first"}:
        min_words = 28
    if words < min_words:
        errors.append(
            {
                "code": "too_short",
                "word_count": words,
                "min_words": min_words,
            }
        )
    leakage = _snippets(LEAKAGE_PATTERNS, text)
    if leakage:
        errors.append({"code": "future_or_metric_leakage", "hits": leakage[:5]})
    internal = _snippets(INTERNAL_PATTERNS, text)
    if internal:
        errors.append({"code": "internal_training_language", "hits": internal[:5]})
    if not _mentions_any_channel(text, list(row.get("contradiction_channels", []))):
        warnings.append(
            {
                "code": "does_not_name_contradiction_channel",
                "contradiction_channels": row.get("contradiction_channels", []),
            }
        )
    similarity = _token_jaccard(positive_text, text)
    if similarity >= 0.65:
        warnings.append(
            {
                "code": "high_positive_text_overlap",
                "token_jaccard": round(similarity, 4),
            }
        )
    return GenerationRecord(
        narrative=narrative,
        validation_errors=errors,
        validation_warnings=warnings,
    )


def build_batch_prompt(rows: list[dict[str, Any]]) -> str:
    payloads = []
    for row in rows:
        payloads.append(
            {
                "row_id": row_id(row),
                "target_window_id": row["target_window_id"],
                "positive_view": row["positive_view"],
                "negative_window_id": row["negative_window_id"],
                "target_title": row["target_title"],
                "negative_title": row["negative_title"],
                "positive_text": row["positive_text"],
                "target_mechanical_summary": row["target_mechanical_summary"],
                "negative_mechanical_summary": row["negative_mechanical_summary"],
                "contradiction_channels": row["contradiction_channels"],
            }
        )
    return (
        "You are writing hard-negative narratives for a financial risk "
        "scenario retrieval model. Return only JSON matching the output schema; "
        "do not wrap it in markdown.\n\n"
        "For each payload, write exactly one hard_negative_text. The text must "
        "describe the NEGATIVE historical 30-day current/recent market prefix, "
        "not the target/positive prefix. Preserve row_id, target_window_id, "
        "positive_view, and negative_window_id exactly.\n\n"
        "Rules:\n"
        "- The narrative must be Codex/GPT-authored prose, not a copied factor list.\n"
        "- Match the requested positive_view style.\n"
        "- Describe only current/recent market conditions in the negative prefix.\n"
        "- Do not forecast what happens next.\n"
        "- Do not mention future horizons, terminal moves, generated scenarios, VaR, ES, or P&L.\n"
        "- Do not mention hard negatives, positives, embeddings, retrieval, training, or scenario generators.\n"
        "- Do not invent named real-world news events.\n"
        "- Use the negative mechanical summary as evidence, but translate it into economic meaning.\n"
        "- Mention at least one contradiction channel when natural, so the text is clearly incompatible with the positive prefix.\n"
        "- quality_notes should be short notes about why the negative is incompatible.\n\n"
        "View style guide:\n"
        "- sparse_user_query: 1-3 declarative market-commentary sentences, one or two channels only.\n"
        "- weekly_risk_monitor: concise institutional monitor tone, active channel and uncertainty.\n"
        "- risk_manager_memo/full_professional: regime, trigger evidence, transmission, cross-asset confirmation, ambiguity, no forecast.\n"
        "- institutional_risk_committee_note: severity, affected exposures, transmission, committee-style risk read.\n"
        "- technical_factor_evidence/factor_list_baseline: compact evidence-oriented current-prefix read.\n"
        "- mechanism_first: mechanism and transmission first, with a few supporting markets.\n\n"
        f"Payloads:\n{json.dumps(payloads, indent=2, sort_keys=True)}\n"
    )


def _batch_name(rows: list[dict[str, Any]], batch_no: int) -> str:
    first = _safe(row_id(rows[0]))
    last = _safe(row_id(rows[-1]))
    return f"{batch_no:06d}_{first}_to_{last}"


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(Path(args.manifest_jsonl))
    selected = manifest_rows[int(args.offset) :]
    if int(args.count) > 0:
        selected = selected[: int(args.count)]
    max_new = max(0, int(getattr(args, "max_new", 0)))

    schema_path = output_dir / "hard_negative_batch_schema.json"
    schema_path.write_text(
        json.dumps(strict_batch_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "hard_negative_codex_generation_report.json"
    bank_jsonl = output_dir / "hard_negative_bank.jsonl"

    generated: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    new_generation_request_count = 0
    new_generation_valid_count = 0
    pending_due_to_max_new = 0
    started_at = time.time()

    def write_checkpoint() -> dict[str, Any]:
        validation_error_count = sum(len(row["errors"]) for row in validations)
        validation_warning_count = sum(len(row["warnings"]) for row in validations)
        status = "pass" if generated and not errors and validation_error_count == 0 else "partial_fail"
        if pending_due_to_max_new > 0 and generated and not errors and validation_error_count == 0:
            status = "partial_pass_generation_limit"
        if bool(args.dry_run):
            status = "dry_run"
        if not generated and not bool(args.dry_run):
            status = "fail"
        report = {
            "schema_version": "hard_negative_codex_generation_report_v1",
            "status": status,
            "manifest_jsonl": str(args.manifest_jsonl),
            "requested_count": len(selected),
            "generated_count": len(generated),
            "codex_error_count": len(errors),
            "validation_error_count": validation_error_count,
            "validation_warning_count": validation_warning_count,
            "codex_model": str(args.model),
            "reasoning_effort": str(args.reasoning_effort),
            "batch_size": int(args.batch_size),
            "max_new": int(max_new),
            "new_generation_request_count": int(new_generation_request_count),
            "new_generation_valid_count": int(new_generation_valid_count),
            "pending_due_to_max_new": int(pending_due_to_max_new),
            "skipped_existing_count": int(
                sum(1 for row in metadata if row.get("skipped_existing"))
            ),
            "dry_run": bool(args.dry_run),
            "local_prose_generated": False,
            "elapsed_seconds": round(time.time() - started_at, 3),
            "generated": generated,
            "validation": validations,
            "errors": errors,
            "metadata": metadata,
            "artifact_paths": {
                "report": str(report_path),
                "bank_jsonl": str(bank_jsonl),
                "schema": str(schema_path),
            },
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with bank_jsonl.open("w", encoding="utf-8") as handle:
            for row in generated:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return report

    batch_size = max(1, int(args.batch_size))
    batch_no = 0
    pos = 0
    while pos < len(selected):
        chunk: list[dict[str, Any]] = []
        while pos < len(selected) and len(chunk) < batch_size:
            row = selected[pos]
            rid = row_id(row)
            out_path = generated_path(output_dir, rid)
            if bool(args.skip_existing) and out_path.exists():
                try:
                    narrative = _load_generated(out_path)
                    record = validate_generated(row, narrative)
                    if record.narrative is not None and not record.validation_errors:
                        generated.append(
                            {
                                **row,
                                **record.narrative.model_dump(),
                                "generated_negative_text": record.narrative.hard_negative_text,
                            }
                        )
                    validations.append(
                        {
                            "row_id": rid,
                            "errors": record.validation_errors,
                            "warnings": record.validation_warnings,
                            "skipped_existing": True,
                        }
                    )
                    metadata.append(
                        {
                            "row_id": rid,
                            "generated_path": str(out_path),
                            "skipped_existing": True,
                        }
                    )
                    pos += 1
                    continue
                except (OSError, ValidationError, ValueError, json.JSONDecodeError):
                    pass
            if max_new > 0 and new_generation_request_count + len(chunk) >= max_new:
                pending_due_to_max_new = len(selected) - pos
                pos = len(selected)
                continue
            chunk.append(row)
            pos += 1
        if not chunk:
            continue

        name = _batch_name(chunk, batch_no)
        if bool(getattr(args, "progress", True)):
            print(
                json.dumps(
                    {
                        "event": "hard_negative_batch_start",
                        "batch_no": int(batch_no),
                        "chunk_size": len(chunk),
                        "generated_so_far": len(generated),
                        "new_requested_so_far": int(new_generation_request_count),
                        "row_start": row_id(chunk[0]),
                        "row_end": row_id(chunk[-1]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        prompt = build_batch_prompt(chunk)
        prompt_path = output_dir / "prompts" / f"hard_negative_prompt_{name}.txt"
        output_path = output_dir / "batches" / f"hard_negative_batch_{name}.json"
        events_path = output_dir / "codex_events" / f"hard_negative_events_{name}.jsonl"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        if bool(args.dry_run):
            new_generation_request_count += len(chunk)
            for row in chunk:
                rid = row_id(row)
                errors.append(
                    {
                        "row_id": rid,
                        "error_type": "DryRun",
                        "message": "Prompt/schema written; Codex was not invoked.",
                        "prompt_path": str(prompt_path),
                    }
                )
            batch_no += 1
            checkpoint = write_checkpoint()
            if bool(getattr(args, "progress", True)):
                print(
                    json.dumps(
                        {
                            "event": "hard_negative_batch_checkpoint",
                            "batch_no": int(batch_no - 1),
                            "status": checkpoint["status"],
                            "generated_count": checkpoint["generated_count"],
                            "new_generation_request_count": checkpoint[
                                "new_generation_request_count"
                            ],
                            "new_generation_valid_count": checkpoint[
                                "new_generation_valid_count"
                            ],
                            "codex_error_count": checkpoint["codex_error_count"],
                            "validation_error_count": checkpoint[
                                "validation_error_count"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue

        new_generation_request_count += len(chunk)
        cmd = [
            "codex",
            "exec",
            "--ephemeral",
            "--json",
            "--disable",
            "apps",
            "--disable",
            "image_generation",
            "-m",
            str(args.model),
            "-c",
            f"model_reasoning_effort='{args.reasoning_effort}'",
            "--sandbox",
            "read-only",
            "--cd",
            str(ROOT),
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            input=prompt,
            capture_output=True,
            timeout=int(args.timeout_seconds),
            check=False,
        )
        events_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            for row in chunk:
                errors.append(
                    {
                        "row_id": row_id(row),
                        "error_type": "CodexExecFailed",
                        "returncode": completed.returncode,
                        "message": completed.stderr.strip()[-4000:],
                        "events_path": str(events_path),
                    }
                )
            write_checkpoint()
            if not bool(args.continue_on_error):
                break
            batch_no += 1
            continue

        try:
            parsed = HardNegativeNarrativeBatch.model_validate_json(
                _extract_json_object(output_path.read_text(encoding="utf-8"))
            )
        except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
            for row in chunk:
                errors.append(
                    {
                        "row_id": row_id(row),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "path": str(output_path),
                    }
                )
            write_checkpoint()
            if not bool(args.continue_on_error):
                break
            batch_no += 1
            continue

        by_id = {item.row_id: item for item in parsed.narratives}
        for row in chunk:
            rid = row_id(row)
            item = by_id.get(rid)
            if item is None:
                errors.append(
                    {
                        "row_id": rid,
                        "error_type": "MissingHardNegative",
                        "message": "Codex output did not contain requested row_id.",
                        "path": str(output_path),
                    }
                )
                continue
            record = validate_generated(row, item)
            validations.append(
                {
                    "row_id": rid,
                    "errors": record.validation_errors,
                    "warnings": record.validation_warnings,
                    "skipped_existing": False,
                }
            )
            metadata.append(
                {
                    "row_id": rid,
                    "prompt_path": str(prompt_path),
                    "batch_output_path": str(output_path),
                    "events_path": str(events_path),
                    "generated_path": str(generated_path(output_dir, rid)),
                    "skipped_existing": False,
                }
            )
            if record.validation_errors:
                continue
            payload = {
                **row,
                **item.model_dump(),
                "generated_negative_text": item.hard_negative_text,
            }
            generated.append(payload)
            new_generation_valid_count += 1
            out_path = generated_path(output_dir, rid)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(item.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        batch_no += 1
        checkpoint = write_checkpoint()
        if bool(getattr(args, "progress", True)):
            print(
                json.dumps(
                    {
                        "event": "hard_negative_batch_checkpoint",
                        "batch_no": int(batch_no - 1),
                        "status": checkpoint["status"],
                        "generated_count": checkpoint["generated_count"],
                        "new_generation_request_count": checkpoint[
                            "new_generation_request_count"
                        ],
                        "new_generation_valid_count": checkpoint[
                            "new_generation_valid_count"
                        ],
                        "codex_error_count": checkpoint["codex_error_count"],
                        "validation_error_count": checkpoint[
                            "validation_error_count"
                        ],
                        "validation_warning_count": checkpoint[
                            "validation_warning_count"
                        ],
                        "elapsed_seconds": checkpoint["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return write_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help=(
            "When >0, scan the selected manifest and include existing generated "
            "rows, but author at most this many new rows. Use with --count 0 "
            "for resume-safe full-corpus scaling."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_batch(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "requested_count": report["requested_count"],
                "generated_count": report["generated_count"],
                "validation_error_count": report["validation_error_count"],
                "validation_warning_count": report["validation_warning_count"],
                "codex_error_count": report["codex_error_count"],
                "new_generation_request_count": report["new_generation_request_count"],
                "new_generation_valid_count": report["new_generation_valid_count"],
                "pending_due_to_max_new": report["pending_due_to_max_new"],
                "skipped_existing_count": report["skipped_existing_count"],
                "report": report["artifact_paths"]["report"],
                "bank_jsonl": report["artifact_paths"]["bank_jsonl"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
