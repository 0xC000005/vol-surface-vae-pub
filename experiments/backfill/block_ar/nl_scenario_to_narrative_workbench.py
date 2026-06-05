from __future__ import annotations

import csv
import json
import math
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SPREAD_FACTORS = {
    "AAA_OAS",
    "BBB_OAS",
    "HY_OAS",
    "IG_OAS",
}
FULL_JOINT39_FACTOR_COUNT = 39
ScenarioTypeV1 = Literal[
    "historical_joint39",
    "generated_deck",
    "factor_table_full",
    "factor_table_partial",
]


class ScenarioFactorRowV1(BaseModel):
    """Normalized one-factor movement supplied by a scenario sidecar."""

    model_config = ConfigDict(extra="forbid")

    factor: str = Field(min_length=1)
    start: float
    end: float
    delta: float
    p10_delta: float | None = None
    p50_delta: float | None = None
    p90_delta: float | None = None
    direction: Literal["up", "down", "flat", "wider", "tighter"]
    magnitude: Literal["flat", "small", "medium", "large"]
    confidence: str = "medium"
    evidence: str = Field(min_length=1)


class ScenarioSidecarV1(BaseModel):
    """Machine-readable sidecar extracted from scenario input."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    scenario_type: ScenarioTypeV1
    scenario_title: str = ""
    archetype: str = "mixed_ambiguous"
    horizon_days: int = Field(default=30, gt=0)
    factor_rows: list[ScenarioFactorRowV1] = Field(default_factory=list)
    mechanical_summary: str = Field(min_length=1)
    normalization_warnings: list[str | dict[str, str]] = Field(default_factory=list)
    summary_source: str = "uploaded_csv"
    sample_count: int | None = None
    source_artifacts: dict[str, str] = Field(default_factory=dict)


class ScenarioNarrativePacketV1(BaseModel):
    """Minimal packet joining a sidecar with narrative text."""

    model_config = ConfigDict(extra="forbid")

    scenario_sidecar: ScenarioSidecarV1
    positive_narratives: list[str] = Field(default_factory=list)
    hard_negative_narratives: list[str] = Field(default_factory=list)
    paired_review: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


def _compact(text: object) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def direction_for_delta(factor: str, delta: float) -> str:
    factor_name = _compact(factor).upper()
    if delta == 0:
        return "flat"
    if (
        factor_name in SPREAD_FACTORS
        or factor_name.endswith("_OAS")
        or factor_name.endswith(" OAS")
    ):
        return "wider" if delta > 0 else "tighter"
    return "up" if delta > 0 else "down"


def magnitude_for_delta(delta: float) -> str:
    absolute_delta = abs(delta)
    if absolute_delta == 0:
        return "flat"
    if absolute_delta < 1:
        return "small"
    if absolute_delta < 10:
        return "medium"
    return "large"


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def _parse_finite_float(value: object, *, column: str, row_number: int) -> float:
    text = _compact(value)
    if not text:
        raise ValueError(f"row {row_number}: {column} must be numeric")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {column} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return number


def factor_row_from_start_end(
    factor: str,
    start: float,
    end: float,
    *,
    confidence: str = "medium",
) -> ScenarioFactorRowV1:
    factor_name = _compact(factor).upper()
    if not factor_name:
        raise ValueError("factor must be non-empty")

    delta = end - start
    evidence = (
        f"start={_format_number(start)}; "
        f"end={_format_number(end)}; "
        f"delta={_format_number(delta)}"
    )
    return ScenarioFactorRowV1(
        factor=factor_name,
        start=start,
        end=end,
        delta=delta,
        direction=direction_for_delta(factor_name, delta),
        magnitude=magnitude_for_delta(delta),
        confidence=_compact(confidence) or "medium",
        evidence=evidence,
    )


def _historical_caption_fields(card: dict[str, Any]) -> dict[str, Any]:
    fields = card.get("caption_fields")
    return fields if isinstance(fields, dict) else {}


def _historical_evidence_used(card: dict[str, Any]) -> list[str]:
    evidence = _historical_caption_fields(card).get("evidence_used")
    if isinstance(evidence, list):
        return [_compact(item) for item in evidence if _compact(item)]
    return []


def _historical_mechanical_summary(card: dict[str, Any]) -> str:
    top_level_summary = _compact(card.get("mechanical_summary"))
    if top_level_summary:
        return top_level_summary

    caption_summary = _compact(_historical_caption_fields(card).get("mechanical_summary"))
    if caption_summary:
        return caption_summary

    evidence_rows = _historical_evidence_used(card)
    if evidence_rows:
        return "Mechanical baseline: " + "; ".join(evidence_rows) + "."

    return "Mechanical summary unavailable from historical episode card."


def normalize_factor_table_csv_text(
    csv_text: str,
    *,
    scenario_id: str,
    horizon_days: int = 30,
) -> ScenarioSidecarV1:
    reader = csv.DictReader(StringIO(csv_text))
    field_lookup = {
        str(name).strip().lower(): name for name in (reader.fieldnames or [])
    }
    fieldnames = set(field_lookup)
    required_columns = {"factor", "start", "end"}
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"missing required column(s): {missing}")

    factor_rows: list[ScenarioFactorRowV1] = []
    seen_factors: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        factor = _compact(row.get(field_lookup["factor"], ""))
        if not factor:
            raise ValueError(f"row {row_number}: factor must be non-empty")
        factor_name = factor.upper()
        if factor_name in seen_factors:
            raise ValueError(f"row {row_number}: duplicate factor: {factor_name}")
        seen_factors.add(factor_name)
        start = _parse_finite_float(
            row.get(field_lookup["start"], ""), column="start", row_number=row_number
        )
        end = _parse_finite_float(
            row.get(field_lookup["end"], ""), column="end", row_number=row_number
        )
        confidence_key = field_lookup.get("confidence")
        confidence = row.get(confidence_key, "medium") if confidence_key else "medium"
        factor_rows.append(
            factor_row_from_start_end(
                factor,
                start,
                end,
                confidence=confidence or "medium",
            )
        )
    if not factor_rows:
        raise ValueError("factor table contains no factor rows")

    scenario_type: ScenarioTypeV1
    if len(factor_rows) >= FULL_JOINT39_FACTOR_COUNT:
        scenario_type = "factor_table_full"
    else:
        scenario_type = "factor_table_partial"

    pieces = [f"{row.factor} {row.direction} {row.magnitude}" for row in factor_rows]
    mechanical_summary = "Mechanical baseline: " + "; ".join(pieces)
    return ScenarioSidecarV1(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        horizon_days=horizon_days,
        factor_rows=factor_rows,
        mechanical_summary=mechanical_summary,
        source_artifacts={"input": "uploaded_csv"},
    )


def normalize_historical_joint39_card(
    card: dict[str, Any],
    *,
    source_path: str | Path,
) -> ScenarioSidecarV1:
    window_id = _compact(card.get("window_id"))
    if not window_id:
        raise ValueError("historical card is missing window_id")
    mechanical = _historical_mechanical_summary(card)
    evidence_rows = _historical_evidence_used(card)
    warnings: list[dict[str, str]] = []
    if not evidence_rows:
        warnings.append(
            {
                "code": "missing_caption_evidence",
                "message": "Historical card has no caption_fields.evidence_used rows.",
            }
        )
    return ScenarioSidecarV1(
        scenario_id=window_id,
        scenario_type="historical_joint39",
        horizon_days=30,
        scenario_title=_compact(card.get("scenario_title")),
        archetype=_compact(card.get("archetype")) or "mixed_ambiguous",
        mechanical_summary=mechanical,
        factor_rows=[],
        source_artifacts={"cards_jsonl": str(source_path)},
        normalization_warnings=warnings,
        summary_source="historical_episode_card",
    )


def load_historical_joint39_sidecar(
    cards_jsonl: str | Path,
    target_window_id: str,
) -> tuple[ScenarioSidecarV1, dict[str, Any]]:
    path = Path(cards_jsonl)
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_id[_compact(row.get("window_id"))] = row
    target_id = _compact(target_window_id)
    if target_id not in by_id:
        raise ValueError(f"target window not found: {target_window_id}")
    card = by_id[target_id]
    return normalize_historical_joint39_card(card, source_path=path), card
