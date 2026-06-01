#!/usr/bin/env python
"""Risk-manager-grade scenario-to-text captions for NL conditioning.

This module is the scenario-to-text side of the NL workflow. It uses the two
specialist Word documents as required style sources, builds a structured prompt
for current/recent-prefix captions, validates leakage, and can run a small
OpenAI TestFlight before any larger relabeling run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_scenario_descriptions import (  # noqa: E402
    load_dotenv_key,
)


PROMPT_VERSION = "risk_manager_caption_v2_2026_05_15"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_PIPELINE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_full_906b_all_windows/narrative_pipeline_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_testflight_916a"
)
REQUIRED_SPECIALIST_DOCS = (
    Path("research/narrative_specialist/quant generated scenarios story narrative.docx"),
    Path("research/narrative_specialist/quant generated scenarios story narrative 2.docx"),
)

ARCHETYPES = (
    "demand_recession",
    "inflation_supply_shock",
    "policy_overshoot",
    "growth_upside_reflation",
    "financial_accident",
    "banking_credit_crunch",
    "sovereign_fiscal_stress",
    "geopolitical_shock",
    "global_growth_divergence",
    "liquidity_surge",
    "liquidity_withdrawal",
    "secular_stagnation",
    "productivity_tech_breakthrough",
    "mixed_ambiguous",
)


class CaptionValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: Literal["warning", "error"] = "warning"


class RiskManagerCaptionV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str = Field(min_length=1)
    schema_version: str = "risk_manager_caption_v2"
    scenario_title: str = Field(min_length=3)
    archetype: str = Field(min_length=3)
    archetype_confidence: Literal["low", "medium", "high"] | str = "medium"
    mechanical_summary: str = Field(min_length=8)
    current_market_state: str = Field(min_length=8)
    trigger: str = ""
    transmission: str = ""
    cross_asset_reaction: str = ""
    sequence: str = ""
    portfolio_vulnerability: str = ""
    risk_manager_implication: str = ""
    evidence_used: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)
    leakage_exclusions: list[str] = Field(default_factory=list)
    no_forecast_caveat: str = Field(min_length=6)
    training_caption: str = Field(min_length=20)
    contrastive_captions: list[str] = Field(default_factory=list)
    quality_self_critique: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SpecialistDocument:
    path: Path
    sha256: str
    excerpt: str


@dataclass(frozen=True)
class SpecialistStandards:
    documents: tuple[SpecialistDocument, ...]
    rubric_text: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _docx_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts: list[str] = []
        for node in para.iter():
            if node.tag == f"{{{ns['w']}}}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{{{ns['w']}}}tab":
                parts.append("\t")
            elif node.tag == f"{{{ns['w']}}}br":
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _shorten(text: str, *, max_chars: int = 2800) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit("\n", 1)[0].strip()


def load_specialist_standards(
    doc_paths: tuple[Path, ...] = REQUIRED_SPECIALIST_DOCS,
) -> SpecialistStandards:
    """Load both Word documents and derive the required caption rubric."""

    documents = tuple(
        SpecialistDocument(path=path, sha256=_sha256(path), excerpt=_shorten(_docx_text(path)))
        for path in doc_paths
    )
    rubric_text = (
        "Risk-manager scenario narratives must translate numbers into meaning. "
        "They should include a scenario title, mechanical summary, archetype, "
        "trigger, transmission channel, cross-asset reaction, sequencing, "
        "portfolio vulnerability, risk-manager implication, evidence used, "
        "ambiguity flags, and a no-forecast caveat. They must use plain "
        "investment language, preserve cause-and-effect logic, avoid fake "
        "certainty, avoid unsupported real-world events, and separate current "
        "conditions from future scenario outcomes."
    )
    return SpecialistStandards(documents=documents, rubric_text=rubric_text)


def _bundle_market_lines(bundle: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for item in bundle.get("market_implications", []):
        if not isinstance(item, dict):
            continue
        market = str(item.get("market", "")).strip()
        direction = str(item.get("direction", "")).strip()
        magnitude = str(item.get("magnitude", "")).strip()
        confidence = str(item.get("confidence", "")).strip()
        evidence = "; ".join(str(x) for x in item.get("evidence", [])[:2])
        if market and direction:
            rows.append(
                f"{market}: {direction} {magnitude} confidence={confidence}; evidence={evidence}"
            )
    return rows


def _source_text(bundle: dict[str, Any]) -> str:
    narratives = bundle.get("narratives", [])
    if isinstance(narratives, list):
        for item in narratives:
            if isinstance(item, dict) and item.get("text"):
                return str(item["text"])
    source = bundle.get("source_description_bundle", {})
    if isinstance(source, dict):
        return str(source.get("revised_description") or source.get("canonical_machine_text") or "")
    return ""


def build_caption_messages(
    bundle: dict[str, Any],
    *,
    standards: SpecialistStandards | None = None,
) -> list[dict[str, str]]:
    """Build the structured-output prompt for one historical-prefix caption."""

    standards = standards or load_specialist_standards()
    doc_summary = "\n\n".join(
        (
            f"Source document: {doc.path.name}\n"
            f"SHA256: {doc.sha256}\n"
            f"Extracted standard excerpt:\n{doc.excerpt}"
        )
        for doc in standards.documents
    )
    calendar = bundle.get("calendar", {})
    if not isinstance(calendar, dict):
        calendar = {}
    payload = {
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
    system = (
        "You are a senior market-risk narrative specialist creating training "
        "captions for a natural-language-conditioned scenario generator. Your "
        "job is scenario-to-text translation for the current/recent 30-day "
        "prefix only. Follow the two specialist Word documents below every "
        "time this narrative generation process is changed or run. Do not use "
        "realized future outcomes, generated future paths, target P&L, VaR/ES, "
        "future terminal values, or post-horizon facts in the conditioning "
        "caption. Use only supplied current/recent market facts and explicitly "
        "mark ambiguity. Do not invent real news or policy events."
    )
    user = (
        "Create a RiskManagerCaptionV2 JSON object.\n\n"
        "Scenario title should be concise and professional. The training_caption "
        "must be a self-contained current/recent-condition caption suitable for "
        "text embeddings and contrastive learning. Contrastive captions should "
        "be hard negatives that flip direction, mechanism, or magnitude without "
        "claiming to be the actual historical outcome. Put leakage exclusions "
        "only in leakage_exclusions or no_forecast_caveat; the training_caption "
        "itself must avoid even negated leakage phrases such as future path, "
        "future scenario, terminal value, VaR, ES, forecast horizon, or target "
        "P&L. The no_forecast_caveat must explicitly include the words "
        "'not a forecast'.\n\n"
        f"Risk-manager standards:\n{standards.rubric_text}\n\n"
        f"Specialist source documents:\n{doc_summary}\n\n"
        f"Current/recent prefix payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_FUTURE_LEAKAGE_PATTERNS = (
    r"\bnext\s+30\s+days?\b",
    r"\bwill\s+(fall|rise|rally|crash|drop|increase|decrease|widen|tighten|spike)\b",
    r"\bfuture\s+(path|return|terminal|outcome|scenario)\b",
    r"\brealized\s+future\b",
    r"\bterminal\s+(level|delta|return)\b",
    r"\bvar\s*95\b",
    r"\bes\s*95\b",
)


def validate_caption_v2(caption: RiskManagerCaptionV2) -> list[CaptionValidationIssue]:
    issues: list[CaptionValidationIssue] = []
    required = {
        "trigger": caption.trigger,
        "transmission": caption.transmission,
        "sequence": caption.sequence,
        "portfolio_vulnerability": caption.portfolio_vulnerability,
        "risk_manager_implication": caption.risk_manager_implication,
        "cross_asset_reaction": caption.cross_asset_reaction,
        "current_market_state": caption.current_market_state,
    }
    for key, value in required.items():
        if not str(value).strip():
            issues.append(
                CaptionValidationIssue(
                    code=f"missing_{key}",
                    message=f"Missing risk-manager narrative section: {key}.",
                    severity="error",
                )
            )
    if caption.archetype not in ARCHETYPES:
        issues.append(
            CaptionValidationIssue(
                code="unknown_archetype",
                message=f"Archetype is outside the approved taxonomy: {caption.archetype}",
            )
        )
    if not caption.evidence_used:
        issues.append(
            CaptionValidationIssue(
                code="missing_evidence_used",
                message="Caption must list evidence fields used.",
                severity="error",
            )
        )
    if not caption.contrastive_captions:
        issues.append(
            CaptionValidationIssue(
                code="missing_contrastive_caption",
                message="Need at least one hard-negative contrastive caption.",
                severity="error",
            )
        )
    if "forecast" not in caption.no_forecast_caveat.lower():
        issues.append(
            CaptionValidationIssue(
                code="missing_no_forecast_caveat",
                message="No-forecast caveat must explicitly say this is not a forecast.",
            )
        )
    training_lower = caption.training_caption.lower()
    for pattern in _FUTURE_LEAKAGE_PATTERNS:
        if re.search(pattern, training_lower):
            issues.append(
                CaptionValidationIssue(
                    code="future_target_leakage",
                    message=(
                        "Training caption contains future-target language: "
                        f"matched {pattern!r}."
                    ),
                    severity="error",
                )
            )
            break
    if len(caption.training_caption.split()) < 18:
        issues.append(
            CaptionValidationIssue(
                code="training_caption_too_short",
                message="Training caption is too short for risk-manager-grade embedding text.",
            )
        )
    return issues


def load_pipeline_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def select_testflight_bundles(
    pipeline_report: str | Path | dict[str, Any],
    *,
    count: int = 10,
) -> list[dict[str, Any]]:
    """Select a deterministic split-balanced caption TestFlight subset."""

    report = (
        load_pipeline_report(pipeline_report)
        if not isinstance(pipeline_report, dict)
        else pipeline_report
    )
    bundles = report.get("narrative_bundles", [])
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("pipeline report does not contain narrative_bundles")
    split_order = ("train", "validation", "val", "test")
    grouped: dict[str, list[dict[str, Any]]] = {split: [] for split in split_order}
    other: list[dict[str, Any]] = []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        split = str(bundle.get("manifest_split", "")).lower()
        if split in grouped:
            grouped[split].append(bundle)
        else:
            other.append(bundle)
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < int(count):
        progressed = False
        for split in split_order:
            bucket = grouped.get(split, [])
            if round_index < len(bucket) and len(selected) < int(count):
                selected.append(bucket[round_index])
                progressed = True
        if not progressed:
            break
        round_index += 1
    for bundle in other:
        if len(selected) >= int(count):
            break
        selected.append(bundle)
    if len(selected) < int(count):
        raise ValueError(f"only selected {len(selected)} bundles, requested {count}")
    return selected[: int(count)]


def _local_caption(bundle: dict[str, Any]) -> RiskManagerCaptionV2:
    lines = _bundle_market_lines(bundle)
    lines_lower = [line.lower() for line in lines]
    source = _source_text(bundle) or "Current/recent market state is mixed."
    title = "Current-market risk regime"
    if any("spx: down" in line for line in lines_lower):
        title = "Defensive risk-off setup"
        archetype = "financial_accident"
    elif any("vix: down" in line for line in lines_lower):
        title = "Risk-on volatility-compression setup"
        archetype = "growth_upside_reflation"
    else:
        archetype = "mixed_ambiguous"
    evidence = lines[:8] or [source]
    training_caption = (
        f"{title}: {source} The current/recent prefix is best read through "
        "observed cross-asset moves rather than a confirmed external event. "
        "This caption describes the conditioning state only."
    )
    return RiskManagerCaptionV2(
        window_id=str(bundle.get("window_id", "")),
        scenario_title=title,
        archetype=archetype,
        archetype_confidence="medium",
        mechanical_summary="; ".join(lines[:6]) or source,
        current_market_state=source,
        trigger="No confirmed catalyst is supplied; infer the mechanism from market moves.",
        transmission="Risk appetite, volatility, rates, credit, FX, and commodities are linked through the observed prefix pattern.",
        cross_asset_reaction="; ".join(lines[:8]) or "Cross-asset evidence is limited.",
        sequence="The caption treats the 30-day prefix as the observed sequence and avoids realized-future claims.",
        portfolio_vulnerability="Risk would concentrate in exposures aligned with the dominant equity, volatility, rate, credit, FX, and commodity moves.",
        risk_manager_implication="Use this as a current-condition label for scenario conditioning, not as a point forecast.",
        evidence_used=evidence,
        ambiguity_flags=["local_test_caption_not_llm_polished"],
        leakage_exclusions=["realized future path", "generated future distribution", "target portfolio P&L"],
        no_forecast_caveat="This is not a forecast; it describes the current/recent conditioning prefix.",
        training_caption=training_caption,
        contrastive_captions=[
            "Opposite hard negative: the same markets show a calm risk-on recovery with equities firmer, volatility lower, and credit stress easing."
        ],
        quality_self_critique=[
            "Local fallback is schema-valid but less nuanced than OpenAI risk-manager prose."
        ],
    )


def caption_one_with_openai(
    bundle: dict[str, Any],
    *,
    standards: SpecialistStandards,
    model: str,
    dotenv_path: str | Path = ".env",
    max_output_tokens: int = 2200,
) -> tuple[RiskManagerCaptionV2, dict[str, Any]]:
    load_dotenv_key(dotenv_path)
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=build_caption_messages(bundle, standards=standards),
        text_format=RiskManagerCaptionV2,
        max_output_tokens=int(max_output_tokens),
        store=False,
    )
    parsed = response.output_parsed
    if not isinstance(parsed, RiskManagerCaptionV2):
        raise TypeError("OpenAI did not return a RiskManagerCaptionV2")
    usage = getattr(response, "usage", None)
    return parsed, {
        "model": str(model),
        "prompt_version": PROMPT_VERSION,
        "response_id": str(getattr(response, "id", "")),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
    }


def run_testflight(args: argparse.Namespace) -> dict[str, Any]:
    standards = load_specialist_standards()
    bundles = select_testflight_bundles(args.pipeline_report, count=int(args.count))
    captions = []
    validation_rows = []
    metadata_rows = []
    api_error_rows = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "risk_manager_caption_v2_testflight_report.json"
    captions_path = output_dir / "risk_manager_caption_v2_captions.jsonl"

    def write_checkpoint() -> dict[str, Any]:
        validation_error_count = sum(len(row["errors"]) for row in validation_rows)
        api_error_count = len(api_error_rows)
        if validation_error_count == 0 and api_error_count == 0:
            status = "pass"
        elif captions and api_error_count:
            status = "partial_fail"
        else:
            status = "fail"
        report = {
            "status": status,
            "scope_note": (
                "Risk-manager caption V2 TestFlight. Captions describe current/recent "
                "historical prefixes only and are intended for future embedding and "
                "contrastive bridge tests."
            ),
            "backend": str(args.backend),
            "model": str(args.model) if args.backend == "openai" else "local_test",
            "prompt_version": PROMPT_VERSION,
            "specialist_documents": [
                {"path": str(doc.path), "sha256": doc.sha256}
                for doc in standards.documents
            ],
            "pipeline_report": str(args.pipeline_report),
            "requested_count": int(len(bundles)),
            "attempted_count": int(len(captions) + len(api_error_rows)),
            "caption_count": int(len(captions)),
            "error_count": int(validation_error_count),
            "api_error_count": int(api_error_count),
            "captions": captions,
            "validation": validation_rows,
            "api_errors": api_error_rows,
            "metadata": metadata_rows,
            "artifact_paths": {
                "report": str(report_path),
                "captions_jsonl": str(captions_path),
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with captions_path.open("w", encoding="utf-8") as handle:
            for row in captions:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return report

    for bundle in bundles:
        try:
            if args.backend == "openai":
                caption, metadata = caption_one_with_openai(
                    bundle,
                    standards=standards,
                    model=str(args.model),
                    dotenv_path=args.dotenv,
                    max_output_tokens=int(args.max_output_tokens),
                )
            else:
                caption = _local_caption(bundle)
                metadata = {
                    "model": "local_test",
                    "prompt_version": PROMPT_VERSION,
                    "response_id": "",
                    "usage": None,
                }
        except Exception as exc:
            if not bool(getattr(args, "continue_on_error", False)):
                raise
            api_error_rows.append(
                {
                    "window_id": str(bundle.get("window_id", "")),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            write_checkpoint()
            continue
        issues = validate_caption_v2(caption)
        captions.append(caption.model_dump())
        validation_rows.append(
            {
                "window_id": caption.window_id,
                "errors": [
                    issue.model_dump() for issue in issues if issue.severity == "error"
                ],
                "warnings": [
                    issue.model_dump() for issue in issues if issue.severity != "error"
                ],
            }
        )
        metadata_rows.append({"window_id": caption.window_id, **metadata})
        write_checkpoint()
    return write_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", type=Path, default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write partial reports and keep going when a single OpenAI call fails.",
    )
    args = parser.parse_args()
    report = run_testflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "backend": report["backend"],
                "caption_count": report["caption_count"],
                "error_count": report["error_count"],
                "report": report["artifact_paths"]["report"],
                "captions_jsonl": report["artifact_paths"]["captions_jsonl"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
