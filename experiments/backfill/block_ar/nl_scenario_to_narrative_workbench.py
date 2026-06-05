from __future__ import annotations

import csv
import math
from io import StringIO
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SPREAD_FACTORS = {
    "AAA_OAS",
    "BBB_OAS",
    "HY_OAS",
    "IG_OAS",
}
FULL_JOINT39_FACTOR_COUNT = 39


class ScenarioFactorRowV1(BaseModel):
    """Normalized one-factor movement supplied by a scenario sidecar."""

    model_config = ConfigDict(extra="forbid")

    factor: str = Field(min_length=1)
    start: float
    end: float
    delta: float
    direction: Literal["up", "down", "flat", "wider", "tighter"]
    magnitude: Literal["flat", "small", "medium", "large"]
    confidence: str = "medium"
    evidence: str = Field(min_length=1)


class ScenarioSidecarV1(BaseModel):
    """Machine-readable sidecar extracted from scenario input."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    scenario_type: Literal["factor_table_full", "factor_table_partial"]
    horizon_days: int = 30
    factor_rows: list[ScenarioFactorRowV1] = Field(default_factory=list)
    mechanical_summary: str = Field(min_length=1)
    source_artifacts: dict[str, str] = Field(default_factory=dict)


class ScenarioNarrativePacketV1(BaseModel):
    """Minimal packet joining a sidecar with narrative text."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    sidecar: ScenarioSidecarV1
    mechanical_summary: str = Field(min_length=1)
    narrative: str = ""


def _compact(text: object) -> str:
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
    for row_number, row in enumerate(reader, start=2):
        factor = _compact(row.get(field_lookup["factor"], ""))
        if not factor:
            raise ValueError(f"row {row_number}: factor must be non-empty")
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

    scenario_type: Literal["factor_table_full", "factor_table_partial"]
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
