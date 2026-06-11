#!/usr/bin/env python
"""995a/995c: condition-only narrative authoring for val-frame query windows.

995a PILOT (5 windows: 4010, 4120, 4230, 4340, 4450 -- spanning the 994a broad
val frame) validated the lane; ``prepare --all-stride5`` scales the same lane
to the FULL 89-window 994a query set (4010, 4015, ..., 4450; verified against
``val_frame_sanity_checks_994a.json``) for the 995c corpus run. This script
does NOT author any narrative prose locally. It has two subcommands:

  prepare -- build the Codex prompt bundles (structured facts only) for the
      pilot windows directly from the raw joint39 panel via the exact 994a
      0..4450 block frame (``build_val_block`` with eval_split=train,
      test_start=4511, val_size=0). This is required because the 939a support
      bank and the 972b source cards cover windows 0..4009 only; val-frame
      windows must be featurized from the panel itself. Per-factor z-scales are
      computed train-side (bank windows 0..4009 only) so no val data leaks into
      standardization. Output is a ``codex_testflight_pipeline_report.json`` +
      ``selected_support_cases.json`` consumed by the established Codex
      authoring lane:

          nl_episode_card_v3_codex_testflight.py run-multiformat

      which performs the actual gpt-5.5 (xhigh) authoring, validation, and
      982g-format card building (schema nl_episode_card_v3_codex_multiformat_card_v1).

  audit -- post-run leakage / schema audit of the generated
      ``multiformat_episode_cards.jsonl``: re-runs the LEAKAGE_PATTERNS scan,
      scans every card text for date/year references AFTER the window's
      history end date, checks forbidden numerics (VaR / P&L / terminal
      values), compares the card schema against a reference 982g card, and
      writes ``validation_report.json`` + ``provenance.json``.

Leakage rule: narratives describe ONLY the observed 30-day prefix ending at
the query window's history end date. Realized-future text is forbidden.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_994a_val_frame_start_only_eval import (  # noqa: E402
    BANK_TRAIN_WINDOW_COUNT,
    HISTORY_LEN,
    IV_DATE_PARQUET,
    SUPPORT_BANK_DIR,
    VAL_FIRST_WINDOW,
    VAL_LAST_WINDOW,
    _block_frame_namespace,
)
from experiments.backfill.block_ar.nl_episode_card_v3_codex_testflight import (  # noqa: E402
    _filter_negated_no_forecast_hits,
    render_selection_markdown,
    support_card_to_bundle,
)
from experiments.backfill.block_ar.nl_episode_narrative_cards import (  # noqa: E402
    LEAKAGE_PATTERNS,
    _snippets_for_patterns,
    build_episode_card,
)
from experiments.backfill.block_ar.nl_episode_narrative_support_cards import (  # noqa: E402
    _caption_from_support_row,
    _scales_from_history,
)
from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (  # noqa: E402
    PROMPT_VERSION,
)

PILOT_WINDOWS = (4010, 4120, 4230, 4340, 4450)
QUERY_STRIDE = 5
VAL_FRAME_SANITY_JSON = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only/val_frame_sanity_checks_994a.json"
)
# Token cost estimate assumptions (USD per 1M tokens, GPT-5-family API list
# rates used as a proxy for gpt-5.5; the Codex CLI lane runs under ChatGPT
# subscription auth, so marginal billed cost may be $0).
COST_RATES_USD_PER_MTOK = {
    "input_uncached": 1.25,
    "input_cached": 0.125,
    "output": 10.0,
}
DEFAULT_CHECKPOINT = (
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
    "best_model.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_narrative_pilot_995a"
)
REFERENCE_982G_CARDS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)

ISO_DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October"
    "|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
MONTH_YEAR_PATTERN = re.compile(
    rf"\b(?P<month>{MONTH_NAMES})\.?\s+(?:(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+)?"
    r"(?P<year>(?:19|20)\d{2})\b"
)
# Year-like 4-digit tokens, EXCLUDING decimal-number contexts: "2083.25" is a
# market-move magnitude (e.g. AAA_OAS wider by 2083.25), not a year, and
# "0.2083" is a fraction. Real year tokens ("2016", "in 2016.") still match.
BARE_YEAR_PATTERN = re.compile(r"(?<![\d.])\b((?:19|20)\d{2})\b(?!\.\d)")
FORBIDDEN_NUMERIC_PATTERNS = (
    re.compile(r"\bVaR\b"),
    re.compile(r"\bvalue[- ]at[- ]risk\b", re.I),
    re.compile(r"\bP&L\b|\bPnL\b", re.I),
    re.compile(r"\bexpected\s+shortfall\b", re.I),
    re.compile(r"\bterminal\s+(?:path|value|level|return|move)\b", re.I),
)
FUTURE_PHRASE_AUDIT_PATTERNS = (
    re.compile(r"\bnext\s+(?:week|month|quarter|year)\b", re.I),
    re.compile(r"\bcoming\s+(?:days|weeks|months|sessions)\b", re.I),
    re.compile(r"\bover\s+the\s+next\b", re.I),
    re.compile(r"\bin\s+the\s+weeks\s+ahead\b", re.I),
    re.compile(r"\bgoing\s+forward\b", re.I),
    re.compile(r"\bsubsequent(?:ly)?\b", re.I),
)
_MONTH_INDEX = {
    name.lower(): index
    for index, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _stride5_windows_verified(sanity_json: Path) -> list[int]:
    """Full 994a val-frame query set, verified against the 994a sanity report."""

    windows = list(range(VAL_FIRST_WINDOW, VAL_LAST_WINDOW + 1, QUERY_STRIDE))
    sanity = json.loads(_resolve(sanity_json).read_text(encoding="utf-8"))
    frame = sanity.get("frame", {})
    query_indices = [int(i) for i in frame.get("query_indices", [])]
    if windows != query_indices:
        raise ValueError(
            "--all-stride5 windows disagree with the 994a frame query set: "
            f"built {len(windows)} windows "
            f"({windows[0]}..{windows[-1]} stride {QUERY_STRIDE}), sanity file "
            f"has {len(query_indices)} query_indices"
        )
    if int(frame.get("query_stride", 0)) != QUERY_STRIDE:
        raise ValueError(
            f"994a sanity file reports query_stride={frame.get('query_stride')}"
            f", expected {QUERY_STRIDE}"
        )
    return windows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "all_stride5", False)):
        windows = _stride5_windows_verified(args.sanity_json)
    else:
        windows = [int(w) for w in args.windows]
    for w in windows:
        if not (VAL_FIRST_WINDOW <= w <= VAL_LAST_WINDOW):
            raise ValueError(
                f"window {w} outside the broad val frame "
                f"{VAL_FIRST_WINDOW}..{VAL_LAST_WINDOW}"
            )

    # Rebuild the exact 994a 0..4450 block frame from the raw panel.
    payload = torch.load(
        _resolve(args.checkpoint), map_location="cpu", weights_only=False
    )
    (
        _history_level,
        _history_norm,
        _center,
        _scale,
        _drift_feature,
        history_raw,
        _specs,
        _block,
    ) = build_val_block(_block_frame_namespace(), payload)
    history_raw = np.asarray(history_raw, dtype=np.float32)
    if history_raw.shape[0] != VAL_LAST_WINDOW + 1:
        raise ValueError(
            f"block frame has {history_raw.shape[0]} windows, "
            f"expected {VAL_LAST_WINDOW + 1}"
        )

    # Causality / consistency check vs the 939a bank (bank rows must equal the
    # rebuilt block rows 0..4009 and must NOT cover the val windows).
    bank_arrays_path = _resolve(SUPPORT_BANK_DIR / "support_bank_arrays.npz")
    with np.load(bank_arrays_path) as bank:
        bank_history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
        bank_support_indices = np.asarray(bank["support_indices"], dtype=np.int64)
    bank_max_index = int(bank_support_indices.max())
    bank_block_max_abs_diff = float(
        np.max(np.abs(bank_history_raw - history_raw[: bank_history_raw.shape[0]]))
    )
    if bank_max_index >= min(windows):
        raise ValueError(
            f"causality violation: bank max window {bank_max_index} >= "
            f"min pilot window {min(windows)}"
        )
    if bank_block_max_abs_diff > 1e-6:
        raise ValueError(
            "rebuilt block frame disagrees with the 939a bank on rows 0..4009 "
            f"(max abs diff {bank_block_max_abs_diff})"
        )

    # Train-side z-scales only (bank windows 0..4009).
    scales = _scales_from_history(history_raw[:BANK_TRAIN_WINDOW_COUNT])

    dates = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(_resolve(IV_DATE_PARQUET))["date"])
    )
    bundles: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    window_dates: dict[str, dict[str, str]] = {}
    for w in windows:
        history_start = str(dates[w].date())
        history_end = str(dates[w + HISTORY_LEN - 1].date())
        window_dates[str(w)] = {
            "history_start": history_start,
            "history_end": history_end,
        }
        metadata = {
            "window_id": f"joint39_val_{w:04d}",
            "manifest_split": "val_frame_994a",
            "window_index": w,
            "calendar_start_date": history_start,
            "calendar_end_date": history_end,
        }
        caption = _caption_from_support_row(
            history_raw=history_raw,
            scales=scales,
            row_index=w,
            metadata=metadata,
            source_path="raw_panel_994a_block_frame",
        )
        card = build_episode_card(caption, source_path="raw_panel_994a_block_frame")
        card["support_metadata"] = {
            "calendar_start_date": history_start,
            "calendar_end_date": history_end,
            "window_index": w,
            "raw_history_card": True,
            "support_move_rows": caption["support_move_rows"],
        }
        bundle = support_card_to_bundle(card)
        bundles.append(bundle)
        selected_rows.append(
            {
                "window_id": bundle["window_id"],
                "window_index": w,
                "history_start": history_start,
                "history_end": history_end,
                "target_angles": [card.get("scenario_title", "")],
                "source_title": card.get("scenario_title", ""),
                "source_archetype": card.get("archetype", ""),
                "market_implications": bundle["market_implications"],
            }
        )

    all_stride5 = bool(getattr(args, "all_stride5", False))
    if all_stride5:
        scope_note = (
            f"995c FULL RUN ({len(windows)} windows: the complete 994a "
            f"stride-{QUERY_STRIDE} val-frame query set "
            f"{windows[0]}..{windows[-1]}, verified against "
            f"{VAL_FRAME_SANITY_JSON.name}): condition-only val-frame "
            "narrative authoring bundles built from the raw joint39 panel "
            "(994a 0..4450 block frame), NOT from the 939a bank or 972b "
            "source cards (those stop at window 4009). Searchable prose must "
            "be Codex-authored via nl_episode_card_v3_codex_testflight.py "
            "run-multiformat."
        )
        selection_mode = "all_stride5_val_frame_windows_995c"
    else:
        scope_note = (
            "995a PILOT (5 windows only): condition-only val-frame narrative "
            "authoring bundles built from the raw joint39 panel (994a 0..4450 "
            "block frame), NOT from the 939a bank or 972b source cards (those "
            "stop at window 4009). Searchable prose must be Codex-authored via "
            "nl_episode_card_v3_codex_testflight.py run-multiformat."
        )
        selection_mode = "explicit_pilot_windows_995a"
    pipeline_report = {
        "schema_version": "episode_card_v3_codex_testflight_pipeline_v1",
        "scope_note": scope_note,
        "source_cards_jsonl": "raw_panel_994a_block_frame",
        "selection_mode": selection_mode,
        "rich_stride": 0,
        "prompt_version": PROMPT_VERSION,
        "narrative_bundles": bundles,
        "val_frame_995a": {
            "pilot_windows": windows,
            "window_dates": window_dates,
            "frame_definition": (
                "broad 441-window val frame: eval_split=train, test_start=4511, "
                "val_size=0 block (rows 0..4450); val queries are rows >= 4010"
            ),
            "checkpoint": str(args.checkpoint),
            "z_scale_provenance": (
                "per-factor std of 30d deltas over bank windows 0..4009 only "
                "(train-side; no val rows enter standardization)"
            ),
            "causality_checks": {
                "bank_max_window_index": bank_max_index,
                "min_pilot_window": int(min(windows)),
                "bank_max_lt_min_pilot": bool(bank_max_index < min(windows)),
                "bank_vs_block_history_raw_max_abs_diff": bank_block_max_abs_diff,
            },
            "leakage_rule": (
                "bundles expose only the observed 30-day prefix (history rows "
                "w..w+29); forecast dates are intentionally blank and no "
                "realized-future rows are included"
            ),
        },
    }
    pipeline_path = output_dir / "codex_testflight_pipeline_report.json"
    selection_path = output_dir / "selected_support_cases.json"
    review_path = output_dir / "selected_support_cases.md"
    _write_json(pipeline_path, pipeline_report)
    _write_json(selection_path, {"selected_cases": selected_rows})
    review_path.write_text(render_selection_markdown(selected_rows), encoding="utf-8")
    return {
        "status": "prepared",
        "pilot_windows": windows,
        "window_dates": window_dates,
        "pipeline_report": str(pipeline_path),
        "selected_support_cases": str(selection_path),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def _card_text_items(card: dict[str, Any]) -> list[tuple[str, str]]:
    """All (field, text) pairs of a multiformat card that could be searchable."""

    items: list[tuple[str, str]] = []
    views = card.get("views", {})
    if isinstance(views, dict):
        for name, value in views.items():
            if isinstance(value, str) and value.strip():
                items.append((f"views.{name}", value))
            elif isinstance(value, list):
                for i, text in enumerate(value):
                    if isinstance(text, str) and text.strip():
                        items.append((f"views.{name}[{i}]", text))
    fields = card.get("caption_fields", {})
    if isinstance(fields, dict):
        for name, value in fields.items():
            if isinstance(value, str) and value.strip():
                items.append((f"caption_fields.{name}", value))
            elif isinstance(value, list):
                for i, text in enumerate(value):
                    if isinstance(text, str) and text.strip():
                        items.append((f"caption_fields.{name}[{i}]", text))
    for name in ("scenario_title", "archetype"):
        value = card.get(name)
        if isinstance(value, str) and value.strip():
            items.append((name, value))
    extra = card.get("codex_multiformat_fields", {})
    if isinstance(extra, dict):
        for name in ("no_forecast_caveat", "quality_self_critique"):
            value = extra.get(name)
            if isinstance(value, str) and value.strip():
                items.append((f"codex_multiformat_fields.{name}", value))
            elif isinstance(value, list):
                for i, text in enumerate(value):
                    if isinstance(text, str) and text.strip():
                        items.append(
                            (f"codex_multiformat_fields.{name}[{i}]", text)
                        )
    return items


def _date_references_after(
    text: str, end_date: dt.date
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (violations, year_mentions) for date tokens after end_date."""

    violations: list[dict[str, str]] = []
    year_mentions: list[dict[str, str]] = []

    def snippet(start: int, end: int) -> str:
        return text[max(0, start - 50) : min(len(text), end + 50)].strip()

    for match in ISO_DATE_PATTERN.finditer(text):
        try:
            parsed = dt.date.fromisoformat(match.group(0))
        except ValueError:
            continue
        if parsed > end_date:
            violations.append(
                {
                    "kind": "iso_date_after_window_end",
                    "token": match.group(0),
                    "snippet": snippet(*match.span()),
                }
            )
    for match in MONTH_YEAR_PATTERN.finditer(text):
        month = _MONTH_INDEX.get(match.group("month").lower())
        year = int(match.group("year"))
        if month is None:
            continue
        day_text = match.group("day")
        day = int(day_text) if day_text else 1
        try:
            parsed = dt.date(year, month, min(day, 28) if day_text else 1)
        except ValueError:
            continue
        after = (
            parsed > end_date
            if day_text
            else (year, month) > (end_date.year, end_date.month)
        )
        if after:
            violations.append(
                {
                    "kind": "month_year_after_window_end",
                    "token": match.group(0),
                    "snippet": snippet(*match.span()),
                }
            )
    for match in BARE_YEAR_PATTERN.finditer(text):
        year = int(match.group(1))
        if year > end_date.year:
            violations.append(
                {
                    "kind": "year_after_window_end",
                    "token": match.group(1),
                    "snippet": snippet(*match.span()),
                }
            )
        else:
            year_mentions.append(
                {"token": match.group(1), "snippet": snippet(*match.span())}
            )
    return violations, year_mentions


def _aggregate_token_usage(event_files: list[Path]) -> dict[str, Any]:
    """Sum `turn.completed` usage records across all codex event files."""

    keys = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    totals = dict.fromkeys(keys, 0)
    per_batch: list[dict[str, Any]] = []
    for path in event_files:
        batch = {key: 0 for key in keys}
        batch["events_file"] = str(path.relative_to(ROOT))
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage", {})
            for key in keys:
                batch[key] += int(usage.get(key, 0) or 0)
        for key in keys:
            totals[key] += batch[key]
        per_batch.append(batch)
    uncached = totals["input_tokens"] - totals["cached_input_tokens"]
    estimated_cost_usd = (
        uncached * COST_RATES_USD_PER_MTOK["input_uncached"]
        + totals["cached_input_tokens"] * COST_RATES_USD_PER_MTOK["input_cached"]
        + totals["output_tokens"] * COST_RATES_USD_PER_MTOK["output"]
    ) / 1_000_000.0
    return {
        "totals": totals,
        "per_batch": per_batch,
        "estimated_cost_usd": round(estimated_cost_usd, 4),
        "cost_rate_assumptions_usd_per_mtok": dict(COST_RATES_USD_PER_MTOK),
        "cost_note": (
            "estimate uses GPT-5-family API list rates as a proxy for "
            "gpt-5.5; the Codex CLI lane runs under ChatGPT subscription "
            "auth, so the marginal billed cost may be $0"
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    pipeline = json.loads(
        (output_dir / "codex_testflight_pipeline_report.json").read_text(
            encoding="utf-8"
        )
    )
    window_dates = pipeline["val_frame_995a"]["window_dates"]
    index_by_id = {
        str(bundle["window_id"]): int(bundle["window_index"])
        for bundle in pipeline["narrative_bundles"]
    }
    cards_path = output_dir / "multiformat_episode_cards.jsonl"
    cards = [
        json.loads(line)
        for line in cards_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    codex_report = json.loads(
        (output_dir / "multiformat_codex_report.json").read_text(encoding="utf-8")
    )
    validation_by_id = {
        str(row.get("window_id", "")): row.get("issues", [])
        for row in codex_report.get("validation", [])
    }

    # Schema comparison vs a reference 982g card.
    schema_comparison: dict[str, Any] = {"reference": str(REFERENCE_982G_CARDS)}
    ref_path = _resolve(REFERENCE_982G_CARDS)
    if ref_path.exists():
        with ref_path.open("r", encoding="utf-8") as handle:
            reference_card = json.loads(handle.readline())
        ref_keys = set(reference_card)
        ref_views = set(reference_card.get("views", {}))
        card_keys = set(cards[0]) if cards else set()
        card_views = set(cards[0].get("views", {})) if cards else set()
        schema_comparison.update(
            {
                "reference_schema_version": reference_card.get("schema_version"),
                "pilot_schema_version": cards[0].get("schema_version")
                if cards
                else None,
                "schema_version_match": bool(
                    cards
                    and reference_card.get("schema_version")
                    == cards[0].get("schema_version")
                ),
                "top_level_keys_missing_vs_982g": sorted(ref_keys - card_keys),
                "top_level_keys_extra_vs_982g": sorted(card_keys - ref_keys),
                "view_keys_missing_vs_982g": sorted(ref_views - card_views),
                "view_keys_extra_vs_982g": sorted(card_views - ref_views),
            }
        )
    else:
        schema_comparison["status"] = "reference_cards_not_found"

    per_card: list[dict[str, Any]] = []
    for card in cards:
        window_id = str(card.get("window_id", ""))
        window_index = index_by_id.get(window_id)
        date_info = window_dates.get(str(window_index), {})
        end_date = dt.date.fromisoformat(date_info["history_end"])
        leakage_hits: list[dict[str, Any]] = []
        date_violations: list[dict[str, Any]] = []
        year_mentions: list[dict[str, Any]] = []
        numeric_hits: list[dict[str, Any]] = []
        future_phrase_warnings: list[dict[str, Any]] = []
        for field, text in _card_text_items(card):
            hits = _filter_negated_no_forecast_hits(
                _snippets_for_patterns(text, LEAKAGE_PATTERNS)
            )
            leakage_hits.extend({"field": field, **hit} for hit in hits)
            violations, mentions = _date_references_after(text, end_date)
            date_violations.extend({"field": field, **v} for v in violations)
            year_mentions.extend({"field": field, **m} for m in mentions)
            for pattern in FORBIDDEN_NUMERIC_PATTERNS:
                numeric_hits.extend(
                    {"field": field, **hit}
                    for hit in _snippets_for_patterns(text, (pattern,))
                )
            for pattern in FUTURE_PHRASE_AUDIT_PATTERNS:
                future_phrase_warnings.extend(
                    {"field": field, **hit}
                    for hit in _filter_negated_no_forecast_hits(
                        _snippets_for_patterns(text, (pattern,))
                    )
                )
        lane_issues = validation_by_id.get(window_id, [])
        lane_errors = [i for i in lane_issues if i.get("severity") == "error"]
        passed = (
            not leakage_hits
            and not date_violations
            and not numeric_hits
            and not lane_errors
            and bool(card.get("valid_for_training_retrieval"))
        )
        per_card.append(
            {
                "window_id": window_id,
                "window_index": window_index,
                "history_start": date_info.get("history_start"),
                "history_end": date_info.get("history_end"),
                "scenario_title": card.get("scenario_title"),
                "narrative_authoring": card.get("narrative_authoring"),
                "valid_for_training_retrieval": card.get(
                    "valid_for_training_retrieval"
                ),
                "card_leakage_flag": card.get("leakage", {}).get("has_leakage"),
                "leakage_scan_hits": leakage_hits,
                "post_window_date_violations": date_violations,
                "within_window_year_mentions": year_mentions,
                "forbidden_numeric_hits": numeric_hits,
                "future_phrase_warnings": future_phrase_warnings,
                "lane_validator_issues": lane_issues,
                "pass": passed,
            }
        )

    n_pass = sum(1 for row in per_card if row["pass"])
    report = {
        "schema_version": "nl_995a_val_frame_pilot_validation_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cards_jsonl": str(cards_path),
        "card_count": len(cards),
        "n_pass": n_pass,
        "status": "pass" if cards and n_pass == len(cards) else "needs_review",
        "lane_status": codex_report.get("status"),
        "lane_validation_error_count": codex_report.get("validation_error_count"),
        "lane_validation_warning_count": codex_report.get(
            "validation_warning_count"
        ),
        "schema_comparison_vs_982g": schema_comparison,
        "per_card": per_card,
    }
    _write_json(output_dir / "validation_report.json", report)

    # Provenance.
    run_dir = output_dir / "multiformat_codex_run"
    prompt_files = sorted((run_dir / "prompts").glob("*.txt"))
    batch_files = sorted((run_dir / "batches").glob("*.json"))
    event_files = sorted((run_dir / "codex_events").glob("*.jsonl"))
    try:
        codex_version = subprocess.run(
            ["codex", "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        codex_version = "unavailable"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        ).stdout.strip()
    except OSError:
        git_commit = "unavailable"

    def file_row(path: Path) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "mtime_utc": dt.datetime.fromtimestamp(
                path.stat().st_mtime, dt.timezone.utc
            ).isoformat(),
        }

    provenance = {
        "schema_version": "nl_995a_val_frame_pilot_provenance_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pilot_windows": pipeline["val_frame_995a"]["pilot_windows"],
        "window_dates": window_dates,
        "authoring_lane": (
            "experiments/backfill/block_ar/nl_episode_card_v3_codex_testflight.py "
            "run-multiformat (980a-approved standard; 982g card schema)"
        ),
        "codex_model": codex_report.get("codex_model"),
        "reasoning_effort": codex_report.get("reasoning_effort"),
        "codex_cli_version": codex_version,
        "prompt_version": pipeline.get("prompt_version"),
        "git_commit": git_commit,
        "commands": list(args.ran_command or []),
        "pipeline_report": file_row(
            output_dir / "codex_testflight_pipeline_report.json"
        ),
        "prompt_files": [file_row(p) for p in prompt_files],
        "batch_output_files": [file_row(p) for p in batch_files],
        "codex_event_files": [file_row(p) for p in event_files],
        "cards_jsonl": file_row(cards_path),
        "feature_provenance": pipeline["val_frame_995a"],
    }
    _write_json(output_dir / "provenance.json", provenance)

    # Final corpus report: counts, per-window status, token usage / cost,
    # schema check vs 982g. Failing cards are QUARANTINED (listed here and
    # excluded from the valid set) -- never deleted from the JSONL.
    requested_windows = [
        int(w) for w in pipeline["val_frame_995a"]["pilot_windows"]
    ]
    per_card_by_index = {
        int(row["window_index"]): row
        for row in per_card
        if row["window_index"] is not None
    }
    codex_errors_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in codex_report.get("errors", []):
        codex_errors_by_id.setdefault(str(row.get("window_id", "")), []).append(row)
    per_window_status: list[dict[str, Any]] = []
    missing_windows: list[int] = []
    quarantined_window_ids: list[str] = []
    warning_window_ids: list[str] = []
    for w in requested_windows:
        date_info = window_dates.get(str(w), {})
        row = per_card_by_index.get(w)
        window_id = f"joint39_val_{w:04d}"
        if row is None:
            missing_windows.append(w)
            per_window_status.append(
                {
                    "window_index": w,
                    "window_id": window_id,
                    "history_start": date_info.get("history_start"),
                    "history_end": date_info.get("history_end"),
                    "status": "missing",
                    "codex_errors": codex_errors_by_id.get(window_id, []),
                }
            )
            continue
        lane_warnings = [
            issue
            for issue in row["lane_validator_issues"]
            if issue.get("severity") != "error"
        ]
        n_warnings = len(lane_warnings) + len(row["future_phrase_warnings"])
        if not row["pass"]:
            quarantined_window_ids.append(row["window_id"])
        elif n_warnings:
            warning_window_ids.append(row["window_id"])
        per_window_status.append(
            {
                "window_index": w,
                "window_id": row["window_id"],
                "history_start": row["history_start"],
                "history_end": row["history_end"],
                "scenario_title": row["scenario_title"],
                "status": "valid" if row["pass"] else "quarantined",
                "leakage_scan_hit_count": len(row["leakage_scan_hits"]),
                "post_window_date_violation_count": len(
                    row["post_window_date_violations"]
                ),
                "forbidden_numeric_hit_count": len(row["forbidden_numeric_hits"]),
                "future_phrase_warning_count": len(row["future_phrase_warnings"]),
                "lane_validator_warning_count": len(lane_warnings),
                "lane_validator_error_count": len(row["lane_validator_issues"])
                - len(lane_warnings),
            }
        )
    token_usage = _aggregate_token_usage(event_files)
    corpus_report = {
        "schema_version": "nl_995c_val_frame_corpus_report_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authoring_lane": provenance["authoring_lane"],
        "codex_model": codex_report.get("codex_model"),
        "reasoning_effort": codex_report.get("reasoning_effort"),
        "counts": {
            "requested": len(requested_windows),
            "authored": len(cards),
            "valid": n_pass,
            "with_warnings": len(warning_window_ids),
            "quarantined": len(quarantined_window_ids),
            "missing": len(missing_windows),
        },
        "quarantine_policy": (
            "cards failing any leakage / post-window-date / forbidden-numeric "
            "/ lane-error check are quarantined (listed below, excluded from "
            "the valid set) but kept in multiformat_episode_cards.jsonl"
        ),
        "quarantined_window_ids": quarantined_window_ids,
        "warning_window_ids": warning_window_ids,
        "missing_window_indices": missing_windows,
        "lane_status": codex_report.get("status"),
        "lane_validation_error_count": codex_report.get("validation_error_count"),
        "lane_validation_warning_count": codex_report.get(
            "validation_warning_count"
        ),
        "codex_error_count": codex_report.get("codex_error_count"),
        "token_usage": token_usage,
        "schema_check_vs_982g": schema_comparison,
        "per_window_status": per_window_status,
        "artifact_paths": {
            "cards_jsonl": str(cards_path),
            "validation_report": str(output_dir / "validation_report.json"),
            "provenance": str(output_dir / "provenance.json"),
            "codex_report": str(output_dir / "multiformat_codex_report.json"),
        },
    }
    _write_json(output_dir / "final_corpus_report.json", corpus_report)
    return {
        "status": report["status"],
        "n_pass": n_pass,
        "card_count": len(cards),
        "counts": corpus_report["counts"],
        "estimated_cost_usd": token_usage["estimated_cost_usd"],
        "validation_report": str(output_dir / "validation_report.json"),
        "provenance": str(output_dir / "provenance.json"),
        "final_corpus_report": str(output_dir / "final_corpus_report.json"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prep.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    prep.add_argument(
        "--windows", type=int, nargs="+", default=list(PILOT_WINDOWS)
    )
    prep.add_argument(
        "--all-stride5",
        action="store_true",
        help=(
            "prepare the FULL 994a stride-5 val-frame query set "
            f"({VAL_FIRST_WINDOW}..{VAL_LAST_WINDOW}, 89 windows), verified "
            "against the 994a sanity-check JSON (overrides --windows)"
        ),
    )
    prep.add_argument(
        "--sanity-json", type=Path, default=VAL_FRAME_SANITY_JSON
    )

    aud = sub.add_parser("audit")
    aud.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    aud.add_argument(
        "--ran-command",
        action="append",
        default=[],
        help="exact command line(s) used for this pilot, recorded in provenance",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "audit":
        result = audit(args)
    else:  # pragma: no cover
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
