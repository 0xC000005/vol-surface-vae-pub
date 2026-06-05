# Scenario-To-Narrative Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved analyst workbench that normalizes historical Joint39 cases, generated deck artifacts, and uploaded numeric factor tables into structured sidecars, visualizes the numerical evidence, and produces validated 14-positive / 14-hard-negative narrative packets.

**Architecture:** Add a focused core module for schemas, input normalization, negative-candidate selection, and packet assembly. Refactor the existing 14-view pilot just enough to accept a precomputed sidecar target and candidate set while preserving the current historical CLI behavior. Add a Gradio workbench module that calls the core functions and displays numerical evidence before narrative text.

**Tech Stack:** Python 3.13, Pydantic, pandas, numpy, Plotly, Gradio, pytest, existing Codex CLI runner patterns.

---

## File Structure

- Create `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`
  - Owns `ScenarioSidecarV1`, `ScenarioFactorRowV1`, `ScenarioNarrativePacketV1`.
  - Parses CSV factor tables.
  - Normalizes historical cards, generated deck reports, and factor tables.
  - Builds sidecar-derived negative candidates and report summaries.
- Modify `experiments/backfill/block_ar/nl_14_view_variant_pilot.py`
  - Extract the existing Codex/validation execution into `run_pilot_from_prepared_payload`.
  - Keep `run_pilot(args)` and its CLI behavior intact.
- Create `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`
  - Gradio analyst workbench.
  - Left input rail, numerical visualizations, factor table, narrative packet, validation/audit export.
- Create `test_code/test_nl_scenario_to_narrative_workbench.py`
  - Core schema, CSV parser, normalizers, and sidecar negative selection tests.
- Create `test_code/test_nl_scenario_to_narrative_workbench_app.py`
  - App helper tests for dataframes, plot construction, and dry-run packet rendering.
- Modify `test_code/test_nl_14_view_variant_pilot.py`
  - Add regression coverage for the extracted sidecar-aware runner without invoking Codex.

---

### Task 1: Core Schemas And Factor Table Parser

**Files:**
- Create: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench.py`

- [ ] **Step 1: Write failing schema and parser tests**

Add this to `test_code/test_nl_scenario_to_narrative_workbench.py`:

```python
from __future__ import annotations

import pytest

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioSidecarV1,
    normalize_factor_table_csv_text,
)


def test_factor_table_requires_numeric_start_and_end() -> None:
    csv_text = "factor,start,confidence\nSPX,1294,medium\n"

    with pytest.raises(ValueError, match="missing required column"):
        normalize_factor_table_csv_text(csv_text, scenario_id="demo_upload")


def test_factor_table_normalizes_numeric_rows() -> None:
    csv_text = (
        "factor,start,end,confidence\n"
        "SPX,1294.0,1311.0,medium\n"
        "DXY,90.3,87.2,high\n"
        "BBB_OAS,1.42,1.50,medium\n"
    )

    sidecar = normalize_factor_table_csv_text(csv_text, scenario_id="demo_upload")

    assert isinstance(sidecar, ScenarioSidecarV1)
    assert sidecar.scenario_id == "demo_upload"
    assert sidecar.scenario_type == "factor_table_partial"
    assert sidecar.horizon_days == 30
    rows = {row.factor: row for row in sidecar.factor_rows}
    assert rows["SPX"].delta == pytest.approx(17.0)
    assert rows["SPX"].direction == "up"
    assert rows["DXY"].direction == "down"
    assert rows["BBB_OAS"].direction == "wider"
    assert rows["SPX"].evidence == "start=1294; end=1311; delta=17"
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_factor_table_requires_numeric_start_and_end test_code/test_nl_scenario_to_narrative_workbench.py::test_factor_table_normalizes_numeric_rows -q
```

Expected: both tests fail with `ModuleNotFoundError` or missing symbol errors because the module does not exist yet.

- [ ] **Step 3: Implement schemas and parser**

Create `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py` with:

```python
#!/usr/bin/env python
"""Scenario-to-narrative workbench core utilities.

Local code normalizes numeric scenario evidence and validates artifacts. It does
not author narrative prose.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ScenarioType = Literal[
    "historical_joint39",
    "generated_deck",
    "factor_table_full",
    "factor_table_partial",
]

SPREAD_FACTORS = {"AAA_OAS", "BBB_OAS"}
FULL_JOINT39_FACTOR_COUNT = 39


class ScenarioFactorRowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: str = Field(min_length=1)
    start: float
    end: float
    delta: float
    direction: Literal["up", "down", "flat", "wider", "tighter"]
    magnitude: Literal["flat", "small", "medium", "large"]
    confidence: str = "medium"
    evidence: str = Field(min_length=1)
    p10_delta: float | None = None
    p50_delta: float | None = None
    p90_delta: float | None = None


class ScenarioSidecarV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    scenario_type: ScenarioType
    horizon_days: int = 30
    scenario_title: str = ""
    archetype: str = "mixed_ambiguous"
    mechanical_summary: str = ""
    factor_rows: list[ScenarioFactorRowV1] = Field(default_factory=list)
    source_artifacts: dict[str, str] = Field(default_factory=dict)
    normalization_warnings: list[dict[str, str]] = Field(default_factory=list)
    summary_source: str = ""
    sample_count: int | None = None


class ScenarioNarrativePacketV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_sidecar: ScenarioSidecarV1
    positive_narratives: list[dict[str, Any]] = Field(default_factory=list)
    hard_negative_narratives: list[dict[str, Any]] = Field(default_factory=list)
    paired_review: list[dict[str, Any]] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def direction_for_delta(factor: str, delta: float, *, flat_threshold: float = 1e-12) -> str:
    value = float(delta)
    if abs(value) <= float(flat_threshold):
        return "flat"
    name = str(factor).upper()
    if name in SPREAD_FACTORS or name.endswith("_OAS"):
        return "wider" if value > 0 else "tighter"
    return "up" if value > 0 else "down"


def magnitude_for_delta(delta: float, *, small: float = 0.05, medium: float = 0.20) -> str:
    value = abs(float(delta))
    if value <= 1e-12:
        return "flat"
    if value < small:
        return "small"
    if value < medium:
        return "medium"
    return "large"


def _format_number(value: float) -> str:
    return f"{float(value):.8g}"


def factor_row_from_start_end(
    *,
    factor: str,
    start: float,
    end: float,
    confidence: str = "medium",
) -> ScenarioFactorRowV1:
    delta = float(end) - float(start)
    return ScenarioFactorRowV1(
        factor=_compact(factor).upper(),
        start=float(start),
        end=float(end),
        delta=delta,
        direction=direction_for_delta(factor, delta),
        magnitude=magnitude_for_delta(delta),
        confidence=_compact(confidence) or "medium",
        evidence=(
            f"start={_format_number(float(start))}; "
            f"end={_format_number(float(end))}; "
            f"delta={_format_number(delta)}"
        ),
    )


def normalize_factor_table_csv_text(
    csv_text: str,
    *,
    scenario_id: str,
    horizon_days: int = 30,
) -> ScenarioSidecarV1:
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    required = {"factor", "start", "end"}
    columns = set(reader.fieldnames or [])
    missing_columns = sorted(required - columns)
    if missing_columns:
        raise ValueError(f"missing required column(s): {', '.join(missing_columns)}")

    rows: list[ScenarioFactorRowV1] = []
    for row_index, row in enumerate(reader, start=2):
        factor = _compact(row.get("factor"))
        if not factor:
            raise ValueError(f"row {row_index}: factor is required")
        try:
            start = float(str(row.get("start", "")).strip())
            end = float(str(row.get("end", "")).strip())
        except ValueError as exc:
            raise ValueError(f"row {row_index}: start and end must be numeric") from exc
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError(f"row {row_index}: start and end must be finite")
        rows.append(
            factor_row_from_start_end(
                factor=factor,
                start=start,
                end=end,
                confidence=_compact(row.get("confidence")) or "medium",
            )
        )
    if not rows:
        raise ValueError("factor table contains no factor rows")
    scenario_type: ScenarioType = (
        "factor_table_full"
        if len({row.factor for row in rows}) >= FULL_JOINT39_FACTOR_COUNT
        else "factor_table_partial"
    )
    warnings = []
    if scenario_type == "factor_table_partial":
        warnings.append(
            {
                "code": "partial_factor_coverage",
                "message": "Uploaded table does not contain the full Joint39 factor set.",
            }
        )
    return ScenarioSidecarV1(
        scenario_id=_compact(scenario_id),
        scenario_type=scenario_type,
        horizon_days=int(horizon_days),
        scenario_title=_compact(scenario_id),
        mechanical_summary=build_mechanical_summary(rows),
        factor_rows=rows,
        source_artifacts={"upload": "csv_text"},
        normalization_warnings=warnings,
        summary_source="uploaded_factor_table",
    )


def build_mechanical_summary(rows: list[ScenarioFactorRowV1]) -> str:
    pieces = [
        f"{row.factor} {row.direction} {row.magnitude}"
        for row in rows
        if row.direction != "flat"
    ]
    if not pieces:
        pieces = [f"{row.factor} flat" for row in rows[:8]]
    return "Mechanical baseline: " + "; ".join(pieces[:12]) + "."
```

- [ ] **Step 4: Run tests for Task 1**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_factor_table_requires_numeric_start_and_end test_code/test_nl_scenario_to_narrative_workbench.py::test_factor_table_normalizes_numeric_rows -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py test_code/test_nl_scenario_to_narrative_workbench.py
git commit -m "feat: add scenario sidecar schema and factor table parser"
```

---

### Task 2: Historical Joint39 Sidecar Normalizer

**Files:**
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench.py`

- [ ] **Step 1: Write failing historical normalizer test**

Append to `test_code/test_nl_scenario_to_narrative_workbench.py`:

```python
import json

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    normalize_historical_joint39_card,
)


def test_historical_joint39_card_normalizes_caption_fields(tmp_path) -> None:
    card = {
        "window_id": "joint39_train_0005",
        "scenario_title": "equity defensive pressure",
        "archetype": "liquidity_withdrawal",
        "caption_fields": {
            "evidence_used": ["SPX lower medium", "VIX higher small"],
        },
        "views": {},
    }
    # Sign-vector helpers read caption fields through the existing mechanical
    # summary path in production; this unit test pins the sidecar fields only.
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")

    sidecar = normalize_historical_joint39_card(card, source_path=cards_path)

    assert sidecar.scenario_id == "joint39_train_0005"
    assert sidecar.scenario_type == "historical_joint39"
    assert sidecar.scenario_title == "equity defensive pressure"
    assert sidecar.archetype == "liquidity_withdrawal"
    assert sidecar.source_artifacts["cards_jsonl"] == str(cards_path)
    assert sidecar.normalization_warnings == []
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_historical_joint39_card_normalizes_caption_fields -q
```

Expected: FAIL with `ImportError` for `normalize_historical_joint39_card`.

- [ ] **Step 3: Implement historical normalizer**

Add imports and functions to `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`:

```python
from experiments.backfill.block_ar.nl_hard_negative_bank_regenerate import (
    FACTORS,
    _mechanical_summary,
)
from experiments.backfill.block_ar.nl_sparse_variant_pilot import _evidence_used


def normalize_historical_joint39_card(
    card: dict[str, Any],
    *,
    source_path: str | Path,
) -> ScenarioSidecarV1:
    window_id = _compact(card.get("window_id"))
    if not window_id:
        raise ValueError("historical card is missing window_id")
    mechanical = _mechanical_summary(card)
    evidence_rows = _evidence_used(card)
    warnings = []
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
    *,
    cards_jsonl: str | Path,
    target_window_id: str,
) -> tuple[ScenarioSidecarV1, dict[str, Any]]:
    path = Path(cards_jsonl)
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_id[str(row.get("window_id", ""))] = row
    if target_window_id not in by_id:
        raise ValueError(f"target window not found: {target_window_id}")
    card = by_id[target_window_id]
    return normalize_historical_joint39_card(card, source_path=path), card
```

- [ ] **Step 4: Run historical tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_historical_joint39_card_normalizes_caption_fields -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit Task 2**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py test_code/test_nl_scenario_to_narrative_workbench.py
git commit -m "feat: normalize historical joint39 cases for workbench"
```

---

### Task 3: Generated Deck Sidecar Normalizer

**Files:**
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench.py`

- [ ] **Step 1: Write failing generated deck test**

Append to `test_code/test_nl_scenario_to_narrative_workbench.py`:

```python
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    normalize_generated_deck_summary,
)


def test_generated_deck_summary_converts_terminal_rows() -> None:
    deck_summary = {
        "summary_source": "report_terminal_delta_summary",
        "sample_count": 16,
        "future_len": 30,
        "factor_rows": [
            {
                "factor": "SPX",
                "direction": "up",
                "magnitude": "large",
                "terminal_mean_delta": 48.0,
                "terminal_p10_delta": -74.0,
                "terminal_p50_delta": 52.0,
                "terminal_p90_delta": 110.0,
            },
            {
                "factor": "BBB_OAS",
                "direction": "tighter",
                "magnitude": "small",
                "terminal_mean_delta": -0.08,
                "terminal_p10_delta": -0.20,
                "terminal_p50_delta": -0.06,
                "terminal_p90_delta": 0.09,
            },
        ],
    }

    sidecar = normalize_generated_deck_summary(
        deck_summary,
        scenario_id="fragile_risk_on_rebound_0",
        report_path="/tmp/report.json",
        arrays_path="/tmp/arrays.npz",
    )

    assert sidecar.scenario_type == "generated_deck"
    assert sidecar.sample_count == 16
    assert sidecar.factor_rows[0].factor == "SPX"
    assert sidecar.factor_rows[0].start == 0.0
    assert sidecar.factor_rows[0].end == 48.0
    assert sidecar.factor_rows[1].direction == "tighter"
    assert sidecar.source_artifacts["report"] == "/tmp/report.json"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_generated_deck_summary_converts_terminal_rows -q
```

Expected: FAIL with `ImportError` for `normalize_generated_deck_summary`.

- [ ] **Step 3: Implement generated deck normalizer**

Add to `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`:

```python
def normalize_generated_deck_summary(
    deck_summary: dict[str, Any],
    *,
    scenario_id: str,
    report_path: str | Path,
    arrays_path: str | Path = "",
) -> ScenarioSidecarV1:
    rows: list[ScenarioFactorRowV1] = []
    for item in deck_summary.get("factor_rows", []):
        if not isinstance(item, dict):
            continue
        factor = _compact(item.get("factor")).upper()
        if not factor:
            continue
        mean_delta = float(item.get("terminal_mean_delta", 0.0))
        direction = _compact(item.get("direction")) or direction_for_delta(factor, mean_delta)
        magnitude = _compact(item.get("magnitude")) or magnitude_for_delta(mean_delta)
        rows.append(
            ScenarioFactorRowV1(
                factor=factor,
                start=0.0,
                end=mean_delta,
                delta=mean_delta,
                direction=direction,
                magnitude=magnitude,
                confidence="distribution_mean",
                evidence=f"mean_terminal_delta={_format_number(mean_delta)}",
                p10_delta=(
                    None
                    if item.get("terminal_p10_delta") is None
                    else float(item["terminal_p10_delta"])
                ),
                p50_delta=(
                    None
                    if item.get("terminal_p50_delta") is None
                    else float(item["terminal_p50_delta"])
                ),
                p90_delta=(
                    None
                    if item.get("terminal_p90_delta") is None
                    else float(item["terminal_p90_delta"])
                ),
            )
        )
    if not rows:
        raise ValueError("generated deck summary contains no factor rows")
    artifacts = {"report": str(report_path)}
    if arrays_path:
        artifacts["arrays"] = str(arrays_path)
    return ScenarioSidecarV1(
        scenario_id=_compact(scenario_id),
        scenario_type="generated_deck",
        horizon_days=int(deck_summary.get("future_len", 30) or 30),
        scenario_title=_compact(scenario_id),
        archetype="mixed_ambiguous",
        mechanical_summary=build_mechanical_summary(rows),
        factor_rows=rows,
        source_artifacts=artifacts,
        normalization_warnings=[],
        summary_source=_compact(deck_summary.get("summary_source")) or "generated_deck",
        sample_count=int(deck_summary.get("sample_count", 0) or 0),
    )
```

- [ ] **Step 4: Run generated deck tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_generated_deck_summary_converts_terminal_rows -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit Task 3**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py test_code/test_nl_scenario_to_narrative_workbench.py
git commit -m "feat: normalize generated scenario decks for workbench"
```

---

### Task 4: Sidecar Negative Candidate Selection

**Files:**
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench.py`

- [ ] **Step 1: Write failing sidecar negative-selection test**

Append to `test_code/test_nl_scenario_to_narrative_workbench.py`:

```python
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    select_sidecar_negative_candidates,
)


def test_sidecar_negative_candidates_use_real_card_windows() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "near miss risk pressure",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower; DXY higher"},
        },
        {
            "window_id": "joint39_train_0200",
            "scenario_title": "same direction",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX higher; DXY lower"},
        },
    ]

    candidates = select_sidecar_negative_candidates(
        sidecar=sidecar,
        cards=cards,
        count=1,
    )

    assert candidates[0]["window_id"] == "joint39_train_0100"
    assert "SPX" in candidates[0]["contradiction_channels"]
    assert "DXY" in candidates[0]["contradiction_channels"]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_sidecar_negative_candidates_use_real_card_windows -q
```

Expected: FAIL with missing `select_sidecar_negative_candidates`.

- [ ] **Step 3: Implement sidecar candidate selection**

Add to `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`:

```python
def _direction_sign(direction: str) -> int:
    value = str(direction).lower()
    if value in {"up", "wider"}:
        return 1
    if value in {"down", "tighter"}:
        return -1
    return 0


def _card_direction_from_text(card: dict[str, Any], factor: str) -> int:
    text = " ".join(
        _compact(value)
        for value in [
            card.get("scenario_title"),
            card.get("archetype"),
            card.get("mechanical_summary"),
            (card.get("caption_fields") or {}).get("mechanical_summary")
            if isinstance(card.get("caption_fields"), dict)
            else "",
        ]
    ).lower()
    name = factor.lower()
    if name not in text:
        return 0
    factor_is_spread = factor.upper().endswith("_OAS")
    positive_words = ("wider", "higher", "up") if factor_is_spread else ("higher", "up", "bid", "firmer")
    negative_words = ("tighter", "lower", "down") if factor_is_spread else ("lower", "down", "offered", "softer")
    if any(word in text for word in positive_words):
        return 1
    if any(word in text for word in negative_words):
        return -1
    return 0


def select_sidecar_negative_candidates(
    *,
    sidecar: ScenarioSidecarV1,
    cards: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    target_signs = {row.factor: _direction_sign(row.direction) for row in sidecar.factor_rows}
    scored: list[tuple[float, dict[str, Any]]] = []
    for card in cards:
        channels: list[str] = []
        agreements = 0
        for factor, target_sign in target_signs.items():
            if target_sign == 0:
                continue
            candidate_sign = _card_direction_from_text(card, factor)
            if candidate_sign == 0:
                continue
            if candidate_sign * target_sign < 0:
                channels.append(factor)
            elif candidate_sign * target_sign > 0:
                agreements += 1
        if not channels:
            continue
        candidate = {
            "window_id": _compact(card.get("window_id")),
            "scenario_title": _compact(card.get("scenario_title")),
            "archetype": _compact(card.get("archetype")) or "mixed_ambiguous",
            "mechanical_summary": _compact(
                card.get("mechanical_summary")
                or (card.get("caption_fields") or {}).get("mechanical_summary")
                or card.get("scenario_title")
            ),
            "evidence_used": [],
            "contradiction_channels": channels,
            "contradiction_count": len(channels),
            "agreement_count": agreements,
        }
        score = 100.0 * len(channels) + agreements
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [candidate for _, candidate in scored[: int(count)]]
    if len(selected) < int(count):
        raise ValueError(
            f"only found {len(selected)} sidecar negative candidates, requested {count}"
        )
    return selected
```

- [ ] **Step 4: Run sidecar negative-selection test**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_sidecar_negative_candidates_use_real_card_windows -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit Task 4**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py test_code/test_nl_scenario_to_narrative_workbench.py
git commit -m "feat: select sidecar hard-negative candidates"
```

---

### Task 5: Extract Sidecar-Aware 14-View Runner

**Files:**
- Modify: `experiments/backfill/block_ar/nl_14_view_variant_pilot.py`
- Test: `test_code/test_nl_14_view_variant_pilot.py`

- [ ] **Step 1: Write failing dry-run extraction test**

Append to `test_code/test_nl_14_view_variant_pilot.py`:

```python
from types import SimpleNamespace

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot


def test_run_pilot_from_prepared_payload_dry_run_writes_prompt(tmp_path) -> None:
    target = {
        "window_id": "uploaded_factor_table",
        "scenario_title": "uploaded factor table",
        "archetype": "mixed_ambiguous",
        "mechanical_summary": "Mechanical baseline: SPX up small; DXY down medium.",
        "evidence_used": ["SPX start=100 end=110", "DXY start=90 end=85"],
    }
    negative_candidates = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "scenario_title": f"candidate {idx}",
            "archetype": "mixed_ambiguous",
            "mechanical_summary": "Mechanical baseline: SPX lower; DXY higher.",
            "evidence_used": ["SPX lower", "DXY higher"],
            "contradiction_channels": ["SPX", "DXY", "GOLD"],
            "contradiction_count": 3,
            "agreement_count": 1,
        }
        for idx in range(100, 140)
    ]
    args = SimpleNamespace(
        output_dir=tmp_path,
        support_cards_jsonl="support.jsonl",
        dry_run=True,
        model="gpt-test",
        reasoning_effort="low",
        timeout_seconds=10,
        validation_retries=0,
    )

    report = pilot.run_pilot_from_prepared_payload(
        args=args,
        target=target,
        negative_candidates=negative_candidates,
        source_paths={"cards_jsonl": "sidecar", "support_cards_jsonl": "support.jsonl"},
    )

    assert report["status"] == "fail"
    assert report["dry_run"] is True
    assert report["target"]["window_id"] == "uploaded_factor_table"
    assert (tmp_path / "fourteen_view_prompt.txt").exists()
    assert (tmp_path / "fourteen_view_schema.json").exists()
```

- [ ] **Step 2: Run the failing extraction test**

Run:

```bash
pytest test_code/test_nl_14_view_variant_pilot.py::test_run_pilot_from_prepared_payload_dry_run_writes_prompt -q
```

Expected: FAIL with missing `run_pilot_from_prepared_payload`.

- [ ] **Step 3: Extract implementation from `run_pilot`**

Modify `experiments/backfill/block_ar/nl_14_view_variant_pilot.py`:

1. Add this helper above `run_pilot`:

```python
def run_pilot_from_prepared_payload(
    *,
    args: argparse.Namespace,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    source_paths: dict[str, str],
    assigned_negative_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = _resolve(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if assigned_negative_candidates is None:
        assigned_negative_candidates = assign_negative_candidates_by_view(negative_candidates)
    prompt = build_prompt(
        target=target,
        negative_candidates=negative_candidates,
        assigned_negative_candidates=assigned_negative_candidates,
    )
    return _run_pilot_codex_loop(
        args=args,
        output_dir=output_dir,
        target=target,
        negative_candidates=negative_candidates,
        assigned_negative_candidates=assigned_negative_candidates,
        prompt=prompt,
        source_paths=source_paths,
    )
```

2. Move the body of the current `run_pilot` from `schema_path = ...` through final `return report` into a new private helper:

```python
def _run_pilot_codex_loop(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    assigned_negative_candidates: list[dict[str, Any]],
    prompt: str,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    schema_path = output_dir / "fourteen_view_schema.json"
    prompt_path = output_dir / "fourteen_view_prompt.txt"
    codex_output_path = output_dir / "fourteen_view_codex_output.json"
    events_path = output_dir / "fourteen_view_codex_events.jsonl"
    report_path = output_dir / "fourteen_view_report.json"
    review_path = output_dir / "fourteen_view_review.md"
```

After creating those path variables, move the current implementation block from the first existing `schema_path.write_text(...)` call through the final `return report` into `_run_pilot_codex_loop`. Do not change logic inside the moved block except for these mechanical replacements:

- Use the helper parameters `output_dir`, `target`, `negative_candidates`, `assigned_negative_candidates`, `prompt`, and `source_paths`.
- Remove any duplicate local creation of those values from the moved block.
- Keep artifact filenames, validation keys, retry behavior, event capture, Markdown output, and report JSON fields byte-for-byte equivalent unless the previous local variable name no longer exists.

3. Rewrite `run_pilot(args)` to prepare historical inputs and call the new entry point:

```python
def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    cards = _read_jsonl(Path(args.cards_jsonl))
    by_id = {str(card.get("window_id", "")): card for card in cards}
    if args.target_window_id not in by_id:
        raise ValueError(f"target window not found: {args.target_window_id}")
    target = _target_payload(by_id[args.target_window_id])
    target["evidence_used"] = _evidence_used(by_id[args.target_window_id])
    negative_candidates = select_negative_candidates_for_fourteen_view(
        cards=cards,
        target_window_id=args.target_window_id,
        count=max(int(args.negative_candidate_count), len(EXPECTED_VIEW_NAMES)),
        min_temporal_gap=int(args.min_temporal_gap),
    )
    return run_pilot_from_prepared_payload(
        args=args,
        target=target,
        negative_candidates=negative_candidates,
        source_paths={
            "cards_jsonl": str(args.cards_jsonl),
            "support_cards_jsonl": str(args.support_cards_jsonl),
        },
    )
```

Preserve all existing report keys, artifact names, validation behavior, and retry behavior.

- [ ] **Step 4: Run extraction and existing pilot tests**

Run:

```bash
pytest test_code/test_nl_14_view_variant_pilot.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add experiments/backfill/block_ar/nl_14_view_variant_pilot.py test_code/test_nl_14_view_variant_pilot.py
git commit -m "refactor: add sidecar-aware fourteen-view runner"
```

---

### Task 6: Workbench Packet Assembly

**Files:**
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench.py`

- [ ] **Step 1: Write failing dry-run packet test**

Append to `test_code/test_nl_scenario_to_narrative_workbench.py`:

```python
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    build_target_payload_from_sidecar,
    run_workbench_packet,
)


def test_build_target_payload_from_sidecar() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n",
        scenario_id="demo_upload",
    )

    target = build_target_payload_from_sidecar(sidecar)

    assert target["window_id"] == "demo_upload"
    assert target["scenario_title"] == "demo_upload"
    assert target["archetype"] == "mixed_ambiguous"
    assert "SPX up large" in target["mechanical_summary"]
    assert "SPX: start=100; end=110; delta=10" in target["evidence_used"]


def test_run_workbench_packet_dry_run(tmp_path) -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n",
        scenario_id="demo_upload",
    )
    candidates = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "scenario_title": f"candidate {idx}",
            "archetype": "mixed_ambiguous",
            "mechanical_summary": "Mechanical baseline: SPX lower; DXY higher.",
            "evidence_used": ["SPX lower", "DXY higher"],
            "contradiction_channels": ["SPX", "DXY", "GOLD"],
            "contradiction_count": 3,
            "agreement_count": 1,
        }
        for idx in range(100, 140)
    ]

    packet = run_workbench_packet(
        sidecar=sidecar,
        negative_candidates=candidates,
        output_dir=tmp_path,
        dry_run=True,
    )

    assert packet.scenario_sidecar.scenario_id == "demo_upload"
    assert packet.validation["status"] == "fail"
    assert "report" in packet.artifact_paths
```

- [ ] **Step 2: Run the failing packet tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_build_target_payload_from_sidecar test_code/test_nl_scenario_to_narrative_workbench.py::test_run_workbench_packet_dry_run -q
```

Expected: FAIL with missing symbols.

- [ ] **Step 3: Implement packet assembly**

Add to `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`:

```python
import argparse

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot
from experiments.backfill.block_ar.nl_codex_caption_batch import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
)


def build_target_payload_from_sidecar(sidecar: ScenarioSidecarV1) -> dict[str, Any]:
    return {
        "window_id": sidecar.scenario_id,
        "scenario_title": sidecar.scenario_title or sidecar.scenario_id,
        "archetype": sidecar.archetype or "mixed_ambiguous",
        "mechanical_summary": sidecar.mechanical_summary or build_mechanical_summary(sidecar.factor_rows),
        "evidence_used": [
            f"{row.factor}: {row.evidence}" for row in sidecar.factor_rows
        ][:12],
    }


def _packet_from_report(
    *,
    sidecar: ScenarioSidecarV1,
    report: dict[str, Any],
) -> ScenarioNarrativePacketV1:
    pairs = report.get("pairs", [])
    positives = [
        {"view_name": row.get("view_name"), "text": row.get("positive_text")}
        for row in pairs
    ]
    negatives = [
        {
            "view_name": row.get("view_name"),
            "negative_window_id": row.get("negative_window_id"),
            "text": row.get("negative_text"),
        }
        for row in pairs
    ]
    return ScenarioNarrativePacketV1(
        scenario_sidecar=sidecar,
        positive_narratives=positives,
        hard_negative_narratives=negatives,
        paired_review=pairs,
        validation=report.get("validation", {}),
        artifact_paths=report.get("artifact_paths", {}),
    )


def run_workbench_packet(
    *,
    sidecar: ScenarioSidecarV1,
    negative_candidates: list[dict[str, Any]],
    output_dir: str | Path,
    dry_run: bool,
    timeout_seconds: int = 1200,
    validation_retries: int = 2,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> ScenarioNarrativePacketV1:
    args = argparse.Namespace(
        output_dir=Path(output_dir),
        support_cards_jsonl="",
        dry_run=bool(dry_run),
        model=str(model),
        reasoning_effort=str(reasoning_effort),
        timeout_seconds=int(timeout_seconds),
        validation_retries=int(validation_retries),
    )
    report = pilot.run_pilot_from_prepared_payload(
        args=args,
        target=build_target_payload_from_sidecar(sidecar),
        negative_candidates=negative_candidates,
        source_paths={
            "cards_jsonl": sidecar.source_artifacts.get("cards_jsonl", sidecar.summary_source),
            "support_cards_jsonl": "",
            "sidecar": sidecar.model_dump_json(),
        },
    )
    packet = _packet_from_report(sidecar=sidecar, report=report)
    packet_path = Path(output_dir) / "scenario_narrative_packet.json"
    artifact_paths = dict(packet.artifact_paths)
    artifact_paths["packet"] = str(packet_path)
    packet = packet.model_copy(update={"artifact_paths": artifact_paths})
    packet_path.write_text(packet.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return packet
```

- [ ] **Step 4: Run packet tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_build_target_payload_from_sidecar test_code/test_nl_scenario_to_narrative_workbench.py::test_run_workbench_packet_dry_run -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 6**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py test_code/test_nl_scenario_to_narrative_workbench.py
git commit -m "feat: assemble scenario narrative packets"
```

---

### Task 7: Gradio Workbench Helper Functions

**Files:**
- Create: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench_app.py`

- [ ] **Step 1: Write failing UI helper tests**

Create `test_code/test_nl_scenario_to_narrative_workbench_app.py`:

```python
from __future__ import annotations

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    normalize_factor_table_csv_text,
)
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import (
    factor_rows_dataframe,
    packet_preview_rows,
    status_cards_markdown,
)


def test_factor_rows_dataframe_has_numeric_review_columns() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n",
        scenario_id="demo_upload",
    )

    frame = factor_rows_dataframe(sidecar)

    assert list(frame.columns) == [
        "Factor",
        "Start",
        "End",
        "Delta",
        "Direction",
        "Magnitude",
        "Confidence",
    ]
    assert frame.iloc[0]["Factor"] == "SPX"
    assert frame.iloc[0]["Delta"] == 10.0


def test_packet_preview_rows_pairs_positive_and_negative_text() -> None:
    packet = {
        "paired_review": [
            {
                "view_name": "sparse_user_query",
                "positive_text": "SPX up, DXY down.",
                "negative_window_id": "joint39_train_0100",
                "negative_text": "SPX lower, DXY firmer.",
            }
        ]
    }

    frame = packet_preview_rows(packet)

    assert frame.iloc[0]["View"] == "sparse_user_query"
    assert frame.iloc[0]["Positive"] == "SPX up, DXY down."
    assert frame.iloc[0]["Hard Negative"] == "SPX lower, DXY firmer."


def test_status_cards_markdown_summarizes_sidecar() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="demo_upload",
    )

    text = status_cards_markdown(sidecar, validation_status="waiting")

    assert "ScenarioSidecarV1" in text
    assert "`factor_table_partial`" in text
    assert "`waiting`" in text
```

- [ ] **Step 2: Run the failing UI helper tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench_app.py -q
```

Expected: FAIL with `ModuleNotFoundError` for the app module.

- [ ] **Step 3: Implement UI helper module**

Create `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py` with:

```python
#!/usr/bin/env python
"""Gradio analyst workbench for scenario-to-narrative generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioSidecarV1,
    normalize_factor_table_csv_text,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "scenario_to_narrative_workbench"
)
FACTOR_ROW_COLUMNS = [
    "Factor",
    "Start",
    "End",
    "Delta",
    "Direction",
    "Magnitude",
    "Confidence",
]
PACKET_COLUMNS = ["View", "Positive", "Negative Window", "Hard Negative"]


def factor_rows_dataframe(sidecar: ScenarioSidecarV1) -> pd.DataFrame:
    rows = [
        {
            "Factor": row.factor,
            "Start": row.start,
            "End": row.end,
            "Delta": row.delta,
            "Direction": row.direction,
            "Magnitude": row.magnitude,
            "Confidence": row.confidence,
        }
        for row in sidecar.factor_rows
    ]
    return pd.DataFrame(rows, columns=FACTOR_ROW_COLUMNS)


def packet_preview_rows(packet: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "View": row.get("view_name", ""),
            "Positive": row.get("positive_text", ""),
            "Negative Window": row.get("negative_window_id", ""),
            "Hard Negative": row.get("negative_text", ""),
        }
        for row in packet.get("paired_review", [])
        if isinstance(row, dict)
    ]
    return pd.DataFrame(rows, columns=PACKET_COLUMNS)


def status_cards_markdown(
    sidecar: ScenarioSidecarV1 | None,
    *,
    validation_status: str,
) -> str:
    if sidecar is None:
        return "## Status\n\n- Sidecar: `waiting`\n- Validation: `waiting`"
    return (
        "## Status\n\n"
        f"- Sidecar: `ScenarioSidecarV1`\n"
        f"- Type: `{sidecar.scenario_type}`\n"
        f"- Factors: `{len(sidecar.factor_rows)}`\n"
        f"- Warnings: `{len(sidecar.normalization_warnings)}`\n"
        f"- Validation: `{validation_status}`"
    )


def factor_move_plot(sidecar: ScenarioSidecarV1 | None) -> go.Figure:
    fig = go.Figure()
    if sidecar is None or not sidecar.factor_rows:
        fig.update_layout(title="No scenario loaded")
        return fig
    names = [row.factor for row in sidecar.factor_rows]
    deltas = [row.delta for row in sidecar.factor_rows]
    colors = ["#0f766e" if value >= 0 else "#b42318" for value in deltas]
    fig.add_bar(x=names, y=deltas, marker_color=colors)
    fig.update_layout(
        title="Factor terminal move",
        xaxis_title="Factor",
        yaxis_title="End minus start",
        margin={"l": 40, "r": 20, "t": 45, "b": 80},
        height=360,
    )
    return fig
```

- [ ] **Step 4: Run UI helper tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench_app.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit Task 7**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py test_code/test_nl_scenario_to_narrative_workbench_app.py
git commit -m "feat: add scenario-to-narrative workbench helpers"
```

---

### Task 8: Gradio Workbench Screen And Dry-Run Workflow

**Files:**
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench_app.py`

- [ ] **Step 1: Write failing dry-run callback test**

Append to `test_code/test_nl_scenario_to_narrative_workbench_app.py`:

```python
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import (
    normalize_factor_table_for_app,
)


def test_normalize_factor_table_for_app_returns_visual_outputs() -> None:
    csv_text = "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n"

    status, frame, warnings_json, sidecar_json = normalize_factor_table_for_app(csv_text)

    assert "ScenarioSidecarV1" in status
    assert frame.iloc[0]["Factor"] == "SPX"
    assert "partial_factor_coverage" in warnings_json
    assert '"scenario_id": "uploaded_factor_table"' in sidecar_json
```

- [ ] **Step 2: Run the failing callback test**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench_app.py::test_normalize_factor_table_for_app_returns_visual_outputs -q
```

Expected: FAIL with missing `normalize_factor_table_for_app`.

- [ ] **Step 3: Implement callback and Gradio screen**

Append to `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`:

```python
def normalize_factor_table_for_app(csv_text: str) -> tuple[str, pd.DataFrame, str, str]:
    sidecar = normalize_factor_table_csv_text(
        csv_text,
        scenario_id="uploaded_factor_table",
    )
    return (
        status_cards_markdown(sidecar, validation_status="normalized"),
        factor_rows_dataframe(sidecar),
        json.dumps(sidecar.normalization_warnings, indent=2, sort_keys=True),
        sidecar.model_dump_json(indent=2),
    )


def build_demo() -> Any:
    import gradio as gr

    with gr.Blocks(title="Scenario-to-Narrative Workbench") as demo:
        gr.Markdown(
            "# Scenario-to-Narrative Analyst Workbench\n"
            "Load a historical case, generated deck, or numeric factor table. "
            "Review the numerical evidence before generating narratives."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                source_mode = gr.Radio(
                    choices=["Factor Table"],
                    value="Factor Table",
                    label="Input mode",
                )
                factor_csv = gr.Textbox(
                    label="Factor table CSV",
                    lines=8,
                    value=(
                        "factor,start,end,confidence\n"
                        "SPX,1294.0,1311.0,medium\n"
                        "DXY,90.3,87.2,high\n"
                        "CRUDE_OIL,62.1,70.5,medium\n"
                        "GOLD,548.0,622.5,medium\n"
                    ),
                )
                normalize_button = gr.Button("Normalize and Visualize", variant="primary")
            with gr.Column(scale=2):
                status = gr.Markdown("## Status\n\n- Sidecar: `waiting`")
                factor_frame = gr.Dataframe(
                    headers=FACTOR_ROW_COLUMNS,
                    label="Factor moves",
                    interactive=False,
                )
                factor_plot = gr.Plot(label="Factor terminal move")
                warnings_json = gr.Code(language="json", label="Warnings")
                sidecar_json = gr.Code(language="json", label="ScenarioSidecarV1")
        normalize_button.click(
            fn=normalize_factor_table_for_app,
            inputs=[factor_csv],
            outputs=[status, factor_frame, warnings_json, sidecar_json],
            show_progress="full",
        ).then(
            fn=lambda text: factor_move_plot(
                normalize_factor_table_csv_text(text, scenario_id="uploaded_factor_table")
            ),
            inputs=[factor_csv],
            outputs=[factor_plot],
            show_progress="hidden",
        )
    return demo


def main() -> None:
    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=int(args.server_port),
        share=bool(args.share),
        theme=gr.themes.Origin(),
    )


if __name__ == "__main__":
    main()
```

This task intentionally implements factor-table normalization first. Historical and generated-deck selectors are added in later tasks after the core screen is stable.

- [ ] **Step 4: Run callback and app tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench_app.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py test_code/test_nl_scenario_to_narrative_workbench_app.py
git commit -m "feat: add scenario-to-narrative workbench screen"
```

---

### Task 9: Historical And Generated-Deck Input Modes

**Files:**
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench.py`
- Test: `test_code/test_nl_scenario_to_narrative_workbench_app.py`

- [ ] **Step 1: Write failing generated-deck artifact loader test**

Append to `test_code/test_nl_scenario_to_narrative_workbench.py`:

```python
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    load_generated_deck_sidecar_from_report,
)


def test_load_generated_deck_sidecar_from_report_uses_terminal_summary(tmp_path) -> None:
    report_path = tmp_path / "deck_report.json"
    report_path.write_text(
        json.dumps(
            {
                "generation": {
                    "forecast_steps": 30,
                    "sample_count": 16,
                    "generated_state_shape": [1, 16, 30, 39],
                    "terminal_delta_summary": [
                        {
                            "market": "SPX",
                            "mean_terminal_delta": 12.5,
                            "p10": -5.0,
                            "p50": 11.0,
                            "p90": 30.0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    sidecar = load_generated_deck_sidecar_from_report(report_path)

    assert sidecar.scenario_id == "deck_report"
    assert sidecar.scenario_type == "generated_deck"
    assert sidecar.factor_rows[0].factor == "SPX"
    assert sidecar.factor_rows[0].p90_delta == 30.0
```

- [ ] **Step 2: Run the failing generated-deck loader test**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py::test_load_generated_deck_sidecar_from_report_uses_terminal_summary -q
```

Expected: FAIL with missing loader.

- [ ] **Step 3: Implement generated-deck loader**

Add to `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py`:

```python
from experiments.backfill.block_ar.nl_reverse_caption_scenario_deck import (
    summarize_report_terminal_delta,
)


def load_generated_deck_sidecar_from_report(report_path: str | Path) -> ScenarioSidecarV1:
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: expected JSON object")
    summary = summarize_report_terminal_delta(report)
    if summary is None:
        raise ValueError("generated deck report has no terminal_delta_summary")
    return normalize_generated_deck_summary(
        summary,
        scenario_id=path.stem,
        report_path=path,
        arrays_path="",
    )
```

- [ ] **Step 4: Add app callbacks for historical/deck modes**

In `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`, add:

```python
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    load_generated_deck_sidecar_from_report,
    load_historical_joint39_sidecar,
)
from experiments.backfill.block_ar.nl_sparse_variant_pilot import DEFAULT_CARDS_JSONL


def normalize_historical_for_app(target_window_id: str) -> tuple[str, pd.DataFrame, str, str]:
    sidecar, _card = load_historical_joint39_sidecar(
        cards_jsonl=DEFAULT_CARDS_JSONL,
        target_window_id=target_window_id.strip(),
    )
    return (
        status_cards_markdown(sidecar, validation_status="normalized"),
        factor_rows_dataframe(sidecar),
        json.dumps(sidecar.normalization_warnings, indent=2, sort_keys=True),
        sidecar.model_dump_json(indent=2),
    )


def normalize_generated_deck_for_app(report_path: str) -> tuple[str, pd.DataFrame, str, str]:
    sidecar = load_generated_deck_sidecar_from_report(report_path.strip())
    return (
        status_cards_markdown(sidecar, validation_status="normalized"),
        factor_rows_dataframe(sidecar),
        json.dumps(sidecar.normalization_warnings, indent=2, sort_keys=True),
        sidecar.model_dump_json(indent=2),
    )
```

Then expand `build_demo()` source radio choices to `["Historical Joint39", "Generated Deck", "Factor Table"]` and add textboxes for historical window id and deck report path. Keep all three callbacks explicit; do not multiplex with stringly typed branching until tests cover all modes.

- [ ] **Step 5: Run core and app tests**

Run:

```bash
pytest test_code/test_nl_scenario_to_narrative_workbench.py test_code/test_nl_scenario_to_narrative_workbench_app.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 9**

```bash
git add experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py test_code/test_nl_scenario_to_narrative_workbench.py test_code/test_nl_scenario_to_narrative_workbench_app.py
git commit -m "feat: support historical and generated-deck workbench inputs"
```

---

### Task 10: TestFlight, Smoke Run, And Documentation

**Files:**
- Modify: `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`
- Create: `docs/research_protocols/scenario_to_narrative_workbench_testflight.md`
- Test: existing focused test suite

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest \
  test_code/test_nl_scenario_to_narrative_workbench.py \
  test_code/test_nl_scenario_to_narrative_workbench_app.py \
  test_code/test_nl_14_view_variant_pilot.py \
  test_code/test_nl_reverse_caption_scenario_deck.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run app import smoke**

Run:

```bash
python - <<'PY'
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import build_demo
demo = build_demo()
print(type(demo).__name__)
PY
```

Expected: prints `Blocks`.

- [ ] **Step 3: Run a dry-run packet smoke**

Run:

```bash
python - <<'PY'
from pathlib import Path
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    normalize_factor_table_csv_text,
    run_workbench_packet,
)

sidecar = normalize_factor_table_csv_text(
    "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\nGOLD,500,540,medium\n",
    scenario_id="dry_run_factor_table",
)
candidates = [
    {
        "window_id": f"joint39_train_{idx:04d}",
        "scenario_title": f"dry candidate {idx}",
        "archetype": "mixed_ambiguous",
        "mechanical_summary": "Mechanical baseline: SPX lower; DXY higher; gold lower.",
        "evidence_used": ["SPX lower", "DXY higher", "gold lower"],
        "contradiction_channels": ["SPX", "DXY", "GOLD"],
        "contradiction_count": 3,
        "agreement_count": 1,
    }
    for idx in range(100, 140)
]
packet = run_workbench_packet(
    sidecar=sidecar,
    negative_candidates=candidates,
    output_dir=Path("experiments/backfill/block_ar/nl_scenario_demo_outputs/scenario_to_narrative_workbench_dry_run"),
    dry_run=True,
)
print(packet.validation["status"])
print(packet.artifact_paths["packet"])
PY
```

Expected: prints `fail` for dry-run validation and a packet JSON path. Dry-run failure is expected because Codex is not invoked.

- [ ] **Step 4: Write TestFlight documentation**

Create `docs/research_protocols/scenario_to_narrative_workbench_testflight.md`:

```markdown
# Scenario-To-Narrative Workbench TestFlight

Date: 2026-06-05

## Scope

This TestFlight validates the reverse analyst workbench before external demo use.
The workbench normalizes numerical scenario evidence, visualizes factor moves,
and produces a strict 14-positive / 14-hard-negative narrative packet.

## Input Modes

- Historical Joint39 case: selected by `joint39_*` window id.
- Generated deck artifact: selected by saved report JSON path.
- Uploaded factor table: CSV with required `factor,start,end` columns and optional `confidence`.

## Guardrails

- Numeric start/end levels are required for uploaded factor tables.
- Direction-only uploaded tables are rejected for MVP generation.
- Local code computes facts and validation only; narrative prose must be Codex/GPT-authored.
- Partial factor tables are labeled `factor_table_partial`.
- Dry-run packets are diagnostic artifacts and are not valid narrative banks.

## Verification Commands

```bash
pytest \
  test_code/test_nl_scenario_to_narrative_workbench.py \
  test_code/test_nl_scenario_to_narrative_workbench_app.py \
  test_code/test_nl_14_view_variant_pilot.py \
  test_code/test_nl_reverse_caption_scenario_deck.py \
  -q
```

```bash
python - <<'PY'
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import build_demo
demo = build_demo()
print(type(demo).__name__)
PY
```
```

- [ ] **Step 5: Commit Task 10**

```bash
git add docs/research_protocols/scenario_to_narrative_workbench_testflight.md
git commit -m "docs: document scenario-to-narrative workbench testflight"
```

---

## Final Verification

- [ ] Run the focused workbench suite:

```bash
pytest \
  test_code/test_nl_scenario_to_narrative_workbench.py \
  test_code/test_nl_scenario_to_narrative_workbench_app.py \
  test_code/test_nl_14_view_variant_pilot.py \
  test_code/test_nl_reverse_caption_scenario_deck.py \
  -q
```

Expected: all tests pass.

- [ ] Run a repo import smoke:

```bash
python -m py_compile \
  experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py \
  experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py \
  experiments/backfill/block_ar/nl_14_view_variant_pilot.py
```

Expected: command exits `0`.

- [ ] Launch local app manually:

```bash
python experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py \
  --server-name 127.0.0.1 \
  --server-port 7861
```

Expected: Gradio serves the analyst workbench on `http://127.0.0.1:7861/`.

---

## Spec Coverage Self-Review

- Analyst workbench layout: Task 7 and Task 8.
- Historical Joint39 input: Task 2 and Task 9.
- Generated deck input: Task 3 and Task 9.
- Numeric factor table upload with required start/end: Task 1.
- `ScenarioSidecarV1` contract: Task 1 through Task 4.
- `ScenarioNarrativePacketV1` contract: Task 6.
- Real historical hard negatives when available: Task 4 and Task 6.
- Validation and leakage reuse through existing 14-view validator: Task 5 and Task 6.
- No rule-based narrative generation: Task 6 uses existing Codex/GPT runner and dry-run only for diagnostics.
- Human spot-check visualization before narratives: Task 7 and Task 8.
