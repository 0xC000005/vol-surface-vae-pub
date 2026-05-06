#!/usr/bin/env python
"""Natural-language descriptions for historical joint scenario windows.

This module is intentionally split into pure helpers plus thin OpenAI wrappers.
The pure helpers make it possible to validate prompt shape, JSONL batch payloads,
and description quality without using network calls or exposing API keys.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DescriptionStyle = Literal[
    "terse_trader",
    "risk_manager",
    "macro_narrative",
    "historical_analogy",
    "underspecified_user",
]


class FreeFormDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: DescriptionStyle | str
    text: str = Field(min_length=12)


class MarketMoveAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    magnitude: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"] | str
    evidence: list[str] = Field(default_factory=list)
    inferred: bool = False


class MarketImplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    magnitude: str = Field(min_length=1)
    horizon: str = "30d conditioning window"
    confidence: Literal["low", "medium", "high"] | str = "medium"
    evidence: list[str] = Field(default_factory=list)


class CatalystCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = ""
    url: str = ""
    published_date: str = ""
    evidence: str = ""


class NarrativeCatalyst(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    category: str = Field(min_length=1)
    grounding_status: Literal[
        "observed_market_pattern",
        "historical_analogy",
        "cited_external_event",
        "user_hypothetical",
        "unsupported",
    ] | str
    description: str = Field(min_length=1)
    market_linkage: list[str] = Field(default_factory=list)
    citations: list[CatalystCitation] = Field(default_factory=list)


class ContrastiveDescriptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opposite: str = ""
    partial: list[str] = Field(default_factory=list)
    magnitude: list[str] = Field(default_factory=list)


class ScenarioDescriptionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str = Field(min_length=1)
    panel_version: str = Field(min_length=1)
    calendar_start_date: str = ""
    calendar_end_date: str = ""
    forecast_start_date: str = ""
    forecast_end_date: str = ""
    canonical_machine_text: str = Field(min_length=8)
    market_implications: list[MarketImplication] = Field(default_factory=list)
    narrative_catalysts: list[NarrativeCatalyst] = Field(default_factory=list)
    descriptions: list[FreeFormDescription] = Field(min_length=1)
    structured_audit: list[MarketMoveAudit] = Field(default_factory=list)
    contrastive: ContrastiveDescriptions
    critique: list[str] = Field(default_factory=list)
    revised_description: str = Field(min_length=12)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: Literal["warning", "error"] = "warning"


UNSUPPORTED_NEWS_TERMS = (
    "geopolitical",
    "war",
    "election",
    "central bank",
    "federal reserve",
    "fed ",
    "fed.",
    "policy makers",
    "sanction",
    "us-china",
    "china relationship",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _summary_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, sort_keys=True)


def build_description_messages(window_summary: dict[str, Any]) -> list[dict[str, str]]:
    """Build a grounded narrative scenario description prompt."""

    panel_version = str(window_summary.get("panel_version", "unknown"))
    system = (
        "You write training labels for a text-conditioned financial scenario "
        "generator. Separate observed market facts, explicit market "
        "implications, and narrative catalysts. Use only the supplied market "
        "facts, calendar dates, and any supplied external evidence. You may use "
        "risk-manager stories such as COVID-style panic, 2008-style credit "
        "stress, war-style oil shock, tariff-policy shock, or flash-crash "
        "analogy, but you must mark them as historical_analogy or "
        "user_hypothetical unless supplied external evidence supports a real "
        "event. Never present an uncited event as a confirmed cause. Create "
        "multiple realistic risk-manager phrasings while preserving direction, "
        "magnitude, uncertainty, and missingness. Also create hard contrastive "
        "negatives for text-adapter training; artificial contrasts are negatives, "
        "not real historical outcomes."
    )
    user = (
        f"Panel version: {panel_version}\n"
        "Return a ScenarioDescriptionBundle JSON object. The canonical text must "
        "use explicit tokens such as SPX: DOWN LARGE, VIX: UP LARGE, US2Y: UP "
        "MEDIUM, BBB_OAS: WIDER SMALL. market_implications must be the explicit "
        "asset-level directions/magnitudes used for conditioning. "
        "narrative_catalysts should contain risk-manager story labels and their "
        "grounding_status. cited_external_event requires citations supplied in "
        "the input; otherwise use historical_analogy, user_hypothetical, or "
        "observed_market_pattern. Free-form descriptions may be natural and "
        "varied, but they must not collapse unsupported causes into facts. The "
        "structured audit should list only market moves supported by the numeric "
        "summary.\n\n"
        f"Window summary JSON:\n{_summary_json(window_summary)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _contains_unsupported_news(text: str) -> bool:
    lower = f" {text.lower()} "
    return any(term in lower for term in UNSUPPORTED_NEWS_TERMS)


def validate_description_bundle(
    bundle: ScenarioDescriptionBundle,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    catalyst_statuses = {
        str(item.grounding_status)
        for item in bundle.narrative_catalysts
    }
    joined = " ".join(
        [bundle.canonical_machine_text, bundle.revised_description]
        + [item.text for item in bundle.descriptions]
        + [bundle.contrastive.opposite]
        + bundle.contrastive.partial
        + bundle.contrastive.magnitude
    )
    if _contains_unsupported_news(joined):
        severity: Literal["warning", "error"] = (
            "warning"
            if catalyst_statuses
            & {"historical_analogy", "cited_external_event", "user_hypothetical"}
            else "error"
        )
        issues.append(
            ValidationIssue(
                code="external_catalyst_requires_grounding",
                message=(
                    "Description mentions news/policy/geopolitics; it must stay "
                    "in narrative_catalysts with explicit grounding status."
                ),
                severity=severity,
            )
        )
    for catalyst in bundle.narrative_catalysts:
        status = str(catalyst.grounding_status)
        if status == "unsupported":
            issues.append(
                ValidationIssue(
                    code="unsupported_catalyst",
                    message=f"Unsupported narrative catalyst: {catalyst.label}",
                    severity="warning",
                )
            )
        if status == "cited_external_event" and not catalyst.citations:
            issues.append(
                ValidationIssue(
                    code="missing_catalyst_citation",
                    message=f"Cited external event has no citation: {catalyst.label}",
                    severity="error",
                )
            )
    if not bundle.contrastive.opposite.strip():
        issues.append(
            ValidationIssue(
                code="missing_opposite_contrast",
                message="Missing hard opposite-direction contrastive description.",
                severity="error",
            )
        )
    if not bundle.contrastive.partial:
        issues.append(
            ValidationIssue(
                code="missing_partial_contrast",
                message="Missing partial-direction contrastive descriptions.",
            )
        )
    if not bundle.contrastive.magnitude:
        issues.append(
            ValidationIssue(
                code="missing_magnitude_contrast",
                message="Missing same-direction/different-magnitude contrast.",
            )
        )
    if len(bundle.descriptions) < 2:
        issues.append(
            ValidationIssue(
                code="too_few_descriptions",
                message="Need multiple free-form descriptions for robustness.",
            )
        )
    if not bundle.structured_audit:
        issues.append(
            ValidationIssue(
                code="missing_structured_audit",
                message="Structured audit is needed to debug direction and magnitude.",
                severity="error",
            )
        )
    canonical_upper = bundle.canonical_machine_text.upper()
    if ":" not in canonical_upper or not any(
        token in canonical_upper
        for token in ("UP", "DOWN", "WIDER", "TIGHTER", "FLAT", "MIXED")
    ):
        issues.append(
            ValidationIssue(
                code="weak_canonical_machine_text",
                message=(
                    "Canonical machine text should include explicit market tokens "
                    "and signed direction words."
                ),
            )
        )
    return issues


def _response_json_schema() -> dict[str, Any]:
    try:
        from openai.lib._pydantic import to_strict_json_schema
    except ImportError:  # pragma: no cover - OpenAI SDK is a runtime dependency.
        schema = ScenarioDescriptionBundle.model_json_schema()
    else:
        schema = to_strict_json_schema(ScenarioDescriptionBundle)
    return {
        "type": "json_schema",
        "name": "ScenarioDescriptionBundle",
        "schema": schema,
        "strict": True,
    }


def build_batch_request_rows(
    window_summaries: list[dict[str, Any]],
    *,
    model: str,
    max_output_tokens: int = 5000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(window_summaries):
        custom_id = str(summary.get("window_id", f"window_{index:06d}"))
        rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": build_description_messages(summary),
                    "max_output_tokens": int(max_output_tokens),
                    "text": {"format": _response_json_schema()},
                    "store": False,
                },
            }
        )
    return rows


def bundle_from_batch_response_row(row: dict[str, Any]) -> ScenarioDescriptionBundle:
    """Extract a structured description bundle from one Batch API output row."""

    custom_id = row.get("custom_id", "<unknown>")
    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{custom_id}: missing response object")
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(f"{custom_id}: non-200 response status {status_code}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"{custom_id}: missing response body")
    for output_item in body.get("output", []):
        for content_item in output_item.get("content", []):
            if content_item.get("type") != "output_text":
                continue
            text = content_item.get("text")
            if not isinstance(text, str):
                continue
            return ScenarioDescriptionBundle.model_validate_json(text)
    raise ValueError(f"{custom_id}: no output_text content found")


def load_dotenv_key(dotenv_path: str | Path = ".env") -> bool:
    """Load OPENAI_API_KEY from a simple local .env file if it is not set."""

    if os.getenv("OPENAI_API_KEY"):
        return True
    path = Path(dotenv_path)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != "OPENAI_API_KEY":
            continue
        cleaned = value.strip().strip("'").strip('"')
        if cleaned:
            os.environ["OPENAI_API_KEY"] = cleaned
            return True
    return False


def describe_window_with_openai(
    window_summary: dict[str, Any],
    *,
    model: str,
    dotenv_path: str | Path = ".env",
    max_output_tokens: int = 5000,
) -> ScenarioDescriptionBundle:
    load_dotenv_key(dotenv_path)
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=build_description_messages(window_summary),
        text_format=ScenarioDescriptionBundle,
        max_output_tokens=int(max_output_tokens),
        store=False,
    )
    return response.output_parsed


def submit_batch_with_openai(
    batch_jsonl: str | Path,
    *,
    dotenv_path: str | Path = ".env",
) -> dict[str, Any]:
    load_dotenv_key(dotenv_path)
    from openai import OpenAI

    client = OpenAI()
    with Path(batch_jsonl).open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    return {
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
    }


def _file_content_to_text(content: Any) -> str:
    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    return str(content)


def retrieve_batch_with_openai(
    batch_id: str,
    *,
    dotenv_path: str | Path = ".env",
    output_file: str | Path | None = None,
    error_file: str | Path | None = None,
) -> dict[str, Any]:
    load_dotenv_key(dotenv_path)
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    result = {
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }
    if result["output_file_id"] and output_file:
        text = _file_content_to_text(client.files.content(result["output_file_id"]))
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(text, encoding="utf-8")
    if result["error_file_id"] and error_file:
        text = _file_content_to_text(client.files.content(result["error_file_id"]))
        Path(error_file).parent.mkdir(parents=True, exist_ok=True)
        Path(error_file).write_text(text, encoding="utf-8")
    return result


def _cmd_validate(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        bundle = ScenarioDescriptionBundle.model_validate(row)
        issues = validate_description_bundle(bundle)
        record = bundle.model_dump()
        record["validation_issues"] = [issue.model_dump() for issue in issues]
        output_rows.append(record)
    write_jsonl(args.output, output_rows)


def _cmd_batch_jsonl(args: argparse.Namespace) -> None:
    summaries = read_jsonl(args.input)
    rows = build_batch_request_rows(
        summaries,
        model=args.model,
        max_output_tokens=int(args.max_output_tokens),
    )
    write_jsonl(args.output, rows)


def _cmd_describe_one(args: argparse.Namespace) -> None:
    summary = json.loads(Path(args.input).read_text(encoding="utf-8"))
    bundle = describe_window_with_openai(
        summary,
        model=args.model,
        dotenv_path=args.dotenv,
        max_output_tokens=int(args.max_output_tokens),
    )
    issues = validate_description_bundle(bundle)
    output = bundle.model_dump()
    output["validation_issues"] = [issue.model_dump() for issue in issues]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def _cmd_submit_batch(args: argparse.Namespace) -> None:
    result = submit_batch_with_openai(args.input, dotenv_path=args.dotenv)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _cmd_batch_status(args: argparse.Namespace) -> None:
    batch_id = args.batch_id
    if args.submit_result:
        payload = json.loads(Path(args.submit_result).read_text(encoding="utf-8"))
        batch_id = payload["batch_id"]
    result = retrieve_batch_with_openai(
        batch_id,
        dotenv_path=args.dotenv,
        output_file=args.download_output,
        error_file=args.download_errors,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _cmd_extract_batch_output(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        bundle = bundle_from_batch_response_row(row)
        issues = validate_description_bundle(bundle)
        record = bundle.model_dump()
        record["batch_custom_id"] = row.get("custom_id")
        record["validation_issues"] = [issue.model_dump() for issue in issues]
        output_rows.append(record)
    write_jsonl(args.output, output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate generated descriptions JSONL")
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", required=True)
    validate.set_defaults(func=_cmd_validate)

    batch = sub.add_parser("batch-jsonl", help="Build OpenAI Batch request JSONL")
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--model", default="gpt-5.4-mini")
    batch.add_argument("--max_output_tokens", type=int, default=5000)
    batch.set_defaults(func=_cmd_batch_jsonl)

    one = sub.add_parser("describe-one", help="Call OpenAI for one summary JSON file")
    one.add_argument("--input", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--model", default="gpt-5.4-mini")
    one.add_argument("--dotenv", default=".env")
    one.add_argument("--max_output_tokens", type=int, default=5000)
    one.set_defaults(func=_cmd_describe_one)

    submit = sub.add_parser("submit-batch", help="Submit a batch request JSONL")
    submit.add_argument("--input", required=True)
    submit.add_argument("--output", required=True)
    submit.add_argument("--dotenv", default=".env")
    submit.set_defaults(func=_cmd_submit_batch)

    status = sub.add_parser("batch-status", help="Retrieve batch status and files")
    status.add_argument("--batch-id")
    status.add_argument("--submit-result")
    status.add_argument("--output", required=True)
    status.add_argument("--download-output")
    status.add_argument("--download-errors")
    status.add_argument("--dotenv", default=".env")
    status.set_defaults(func=_cmd_batch_status)

    extract = sub.add_parser(
        "extract-batch-output",
        help="Extract ScenarioDescriptionBundle records from batch output JSONL",
    )
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.set_defaults(func=_cmd_extract_batch_output)

    args = parser.parse_args()
    if args.command == "batch-status" and not (args.batch_id or args.submit_result):
        parser.error("batch-status requires --batch-id or --submit-result")
    args.func(args)


if __name__ == "__main__":
    main()
