#!/usr/bin/env python
"""Reverse-caption generated 30-day scenario decks for narrative preservation.

This audit consumes saved generated-deck artifacts, summarizes the generated
factor movements locally, and asks Codex/GPT to caption the deck and compare it
with the original conditioning story. Local code does not author narrative
prose.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_codex_caption_batch import (  # noqa: E402
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
)
from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    FACTOR_INDEX,
)
from experiments.backfill.block_ar.nl_sparse_variant_pilot import (  # noqa: E402
    _extract_json_object,
    _require_all_properties,
)


DEFAULT_INPUT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_story_multistart_950b_5start_default_deck/"
    "start_0/fixed_start_live_story_deck_analysis.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "reverse_caption_scenario_deck_986f"
)
REVERSE_CAPTION_FACTORS = (
    "SPX",
    "VIX",
    "DXY",
    "Crude",
    "US10Y",
    "BBB_OAS",
    "Gold",
    "IV_ATM_1Y",
)
SPREAD_FACTORS = {"BBB_OAS"}
REPORT_MARKET_TO_FACTOR = {
    "SPX": "SPX",
    "VIX": "VIX",
    "DXY": "DXY",
    "CRUDE_OIL": "Crude",
    "Crude": "Crude",
    "US10Y": "US10Y",
    "BBB_OAS": "BBB_OAS",
    "GOLD": "Gold",
    "Gold": "Gold",
    "IV_ATM_1Y": "IV_ATM_1Y",
    "IV_SURFACE": "IV_SURFACE",
}


class ReverseCaptionCase(BaseModel):
    """Codex-authored reverse caption and alignment judgment for one deck."""

    model_config = ConfigDict(extra="forbid")

    case_name: str = Field(min_length=1)
    deck_caption: str = Field(min_length=10)
    source_alignment: str = Field(min_length=10)
    preserved_channels: list[str] = Field(default_factory=list)
    missing_or_weak_channels: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    alignment_score: float = Field(ge=0.0, le=1.0)
    reviewer_notes: list[str] = Field(default_factory=list)


class ReverseCaptionBatch(BaseModel):
    """Strict output schema for one Codex reverse-caption response."""

    model_config = ConfigDict(extra="forbid")

    cases: list[ReverseCaptionCase] = Field(default_factory=list)


def strict_schema() -> dict[str, Any]:
    return _require_all_properties(ReverseCaptionBatch.model_json_schema())


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


def _resolve_artifact_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    return base_dir / path


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _source_story(report: dict[str, Any]) -> str:
    for value in (
        _nested(report, "condition_only_case", "story"),
        _nested(report, "cached_query", "narrative_text"),
        _nested(report, "condition_only_case", "candidate_query_text"),
        _nested(report, "cached_query", "grounding", "cleaned_conditioning_text"),
    ):
        text = _compact(value)
        if text:
            return text
    return ""


def _cleaned_conditioning_text(report: dict[str, Any]) -> str:
    for value in (
        _nested(report, "condition_only_case", "condition_only_grounding", "cleaned_conditioning_text"),
        _nested(report, "cached_query", "grounding", "cleaned_conditioning_text"),
    ):
        text = _compact(value)
        if text:
            return text
    return ""


def _report_arrays_path(report: dict[str, Any], *, report_path: Path) -> str:
    for value in (
        _nested(report, "artifact_paths", "arrays"),
        _nested(report, "artifact_paths", "prefix_arrays_snapshot"),
    ):
        if value:
            return str(_resolve_artifact_path(str(value), base_dir=report_path.parent))
    fallback = report_path.parent / "prefix_arrays_snapshot.npz"
    return str(fallback)


def discover_review_cases(input_path: str | Path) -> list[dict[str, Any]]:
    """Discover single- or multi-case reverse-caption inputs."""

    path = Path(input_path)
    payload = _load_json(path)
    cases: list[dict[str, Any]] = []
    if isinstance(payload.get("cases"), list):
        for row in payload["cases"]:
            if not isinstance(row, dict):
                continue
            report_path = _resolve_artifact_path(
                str(
                    row.get("report_snapshot")
                    or row.get("prefix_report_snapshot_path")
                    or row.get("report_path")
                    or ""
                ),
                base_dir=path.parent,
            )
            report = _load_json(report_path) if report_path.exists() else {}
            arrays_value = row.get("arrays_snapshot") or row.get("prefix_arrays_snapshot_path")
            arrays_path = (
                _resolve_artifact_path(str(arrays_value), base_dir=path.parent)
                if arrays_value
                else Path(_report_arrays_path(report, report_path=report_path))
            )
            cases.append(
                {
                    "case_name": _compact(row.get("case_name")) or report_path.parent.name,
                    "report_path": str(report_path),
                    "arrays_path": str(arrays_path),
                    "source_story": _source_story(report),
                    "cleaned_conditioning_text": _cleaned_conditioning_text(report),
                    "market_implications": row.get("market_implications")
                    or _nested(report, "condition_only_case", "condition_only_grounding", "current_market_state_implications")
                    or [],
                    "operational_variant_index": _nested(
                        report,
                        "generation",
                        "narrative_ensemble_calibration",
                        "operational_variant_index",
                    ),
                }
            )
        return cases

    report = payload
    arrays_path = _report_arrays_path(report, report_path=path)
    cases.append(
        {
            "case_name": path.parent.name,
            "report_path": str(path),
            "arrays_path": arrays_path,
            "source_story": _source_story(report),
            "cleaned_conditioning_text": _cleaned_conditioning_text(report),
            "market_implications": _nested(
                report,
                "condition_only_case",
                "condition_only_grounding",
                "current_market_state_implications",
            )
            or [],
            "operational_variant_index": _nested(
                report,
                "generation",
                "narrative_ensemble_calibration",
                "operational_variant_index",
            ),
        }
    )
    return cases


def _select_state_array(
    arrays: dict[str, np.ndarray],
    *,
    operational_variant_index: int | None,
) -> np.ndarray:
    states = arrays.get("generated_states")
    if states is None:
        states = arrays.get("samples")
    if states is None:
        raise ValueError("expected generated_states or samples in arrays")
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[-1] == 39:
        if operational_variant_index is not None and 0 <= int(operational_variant_index) < arr.shape[0]:
            arr = arr[int(operational_variant_index)]
        else:
            arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
    if arr.ndim == 3 and arr.shape[-1] == 39:
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 39:
        return arr[None, :, :]
    raise ValueError(f"expected states with final factor dimension 39, got {arr.shape}")


def _select_start_vector(
    arrays: dict[str, np.ndarray],
    states: np.ndarray,
    *,
    operational_variant_index: int | None,
) -> np.ndarray:
    requested_raw = arrays.get("requested_raw")
    if requested_raw is not None:
        raw = np.asarray(requested_raw, dtype=np.float64)
        if raw.ndim == 2 and operational_variant_index is not None:
            idx = int(operational_variant_index)
            if 0 <= idx < raw.shape[0]:
                return raw[idx]
        if raw.ndim == 1 and raw.shape[0] == states.shape[-1]:
            return raw
    return np.asarray(states[:, 0, :], dtype=np.float64).mean(axis=0)


def _direction_label(factor: str, delta: float) -> str:
    if abs(float(delta)) < 1e-9:
        return "flat"
    if factor in SPREAD_FACTORS:
        return "wider" if float(delta) > 0 else "tighter"
    return "up" if float(delta) > 0 else "down"


def summarize_deck_states(
    *,
    arrays: dict[str, np.ndarray],
    operational_variant_index: int | None = None,
) -> dict[str, Any]:
    """Summarize generated factor movements for reverse captioning."""

    states = _select_state_array(arrays, operational_variant_index=operational_variant_index)
    start = _select_start_vector(
        arrays,
        states,
        operational_variant_index=operational_variant_index,
    )
    terminal_delta = states[:, -1, :] - start[None, :]
    factor_rows: list[dict[str, Any]] = []
    for factor in REVERSE_CAPTION_FACTORS:
        index = int(FACTOR_INDEX[factor])
        deltas = terminal_delta[:, index]
        mean_delta = float(np.mean(deltas))
        p10, p50, p90 = np.percentile(deltas, [10, 50, 90])
        factor_rows.append(
            {
                "factor": factor,
                "index": index,
                "direction": _direction_label(factor, mean_delta),
                "terminal_mean_delta": _round_float(mean_delta),
                "terminal_p10_delta": _round_float(float(p10)),
                "terminal_p50_delta": _round_float(float(p50)),
                "terminal_p90_delta": _round_float(float(p90)),
            }
        )
    abs_means = np.asarray([abs(float(row["terminal_mean_delta"])) for row in factor_rows])
    q33, q66 = np.percentile(abs_means, [33, 66])
    for row in factor_rows:
        value = abs(float(row["terminal_mean_delta"]))
        row["magnitude"] = "large" if value >= q66 else "medium" if value >= q33 else "small"
    return {
        "summary_source": "array_generated_states",
        "state_shape": [int(x) for x in states.shape],
        "sample_count": int(states.shape[0]),
        "future_len": int(states.shape[1]),
        "operational_variant_index": operational_variant_index,
        "factor_rows": factor_rows,
    }


def _magnitude_label(rows: list[dict[str, Any]], row: dict[str, Any]) -> str:
    abs_means = np.asarray(
        [abs(float(item["terminal_mean_delta"])) for item in rows],
        dtype=np.float64,
    )
    if abs_means.size == 0:
        return "small"
    q33, q66 = np.percentile(abs_means, [33, 66])
    value = abs(float(row["terminal_mean_delta"]))
    return "large" if value >= q66 else "medium" if value >= q33 else "small"


def summarize_report_terminal_delta(report: dict[str, Any]) -> dict[str, Any] | None:
    """Summarize calibrated report terminal deltas when available."""

    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    terminal_rows = generation.get("terminal_delta_summary", [])
    if not isinstance(terminal_rows, list) or not terminal_rows:
        return None
    factor_rows: list[dict[str, Any]] = []
    for row in terminal_rows:
        if not isinstance(row, dict):
            continue
        factor = REPORT_MARKET_TO_FACTOR.get(str(row.get("market", "")))
        if not factor:
            continue
        mean_delta = row.get("mean_terminal_delta")
        if mean_delta is None:
            continue
        factor_rows.append(
            {
                "factor": factor,
                "direction": _direction_label(factor, float(mean_delta)),
                "terminal_mean_delta": _round_float(float(mean_delta)),
                "terminal_p10_delta": (
                    None if row.get("p10") is None else _round_float(float(row["p10"]))
                ),
                "terminal_p50_delta": (
                    None if row.get("p50") is None else _round_float(float(row["p50"]))
                ),
                "terminal_p90_delta": (
                    None if row.get("p90") is None else _round_float(float(row["p90"]))
                ),
            }
        )
    if not factor_rows:
        return None
    for row in factor_rows:
        row["magnitude"] = _magnitude_label(factor_rows, row)
    generated_shape = generation.get("generated_state_shape", [])
    return {
        "summary_source": "report_terminal_delta_summary",
        "state_shape": generated_shape if isinstance(generated_shape, list) else [],
        "sample_count": int(generation.get("sample_count", 0) or 0),
        "future_len": int(generation.get("forecast_steps", 0) or 0),
        "operational_variant_index": _nested(
            report,
            "generation",
            "narrative_ensemble_calibration",
            "operational_variant_index",
        ),
        "factor_rows": factor_rows,
    }


def summarize_case_deck(case: dict[str, Any]) -> dict[str, Any]:
    """Return calibrated report summary when present, else array summary."""

    report_path = Path(str(case.get("report_path", "")))
    report = _load_json(report_path) if report_path.exists() else {}
    report_summary = summarize_report_terminal_delta(report)
    if report_summary is not None:
        return report_summary
    arrays = _load_npz_arrays(case["arrays_path"])
    return summarize_deck_states(
        arrays=arrays,
        operational_variant_index=(
            None
            if case.get("operational_variant_index") is None
            else int(case["operational_variant_index"])
        ),
    )


def _load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key].copy() for key in payload.files}


def build_reverse_caption_prompt(*, case: dict[str, Any], deck_summary: dict[str, Any]) -> str:
    payload = {
        "case_name": case.get("case_name"),
        "source_story": case.get("source_story"),
        "cleaned_conditioning_text": case.get("cleaned_conditioning_text"),
        "source_market_implications": case.get("market_implications", []),
        "generated_deck_summary": deck_summary,
    }
    return (
        "You are reverse-captioning a generated 30-day financial scenario deck. "
        "Return only JSON matching the supplied schema; do not wrap JSON in markdown.\n\n"
        "Task: infer the market narrative implied by the generated deck summary, "
        "then compare that reverse caption with the original source story. The "
        "deck summary is computed locally from generated paths; use it as the "
        "evidence for deck_caption. The source story is the conditioning story "
        "that generated the deck.\n\n"
        "Rules:\n"
        "- Do not invent named real-world news events.\n"
        "- Do not claim calibrated probabilities, VaR, ES, or P&L.\n"
        "- Keep deck_caption descriptive and current/deck-implied, not a news forecast.\n"
        "- source_alignment should explain preserved and missing conditionality in plain language.\n"
        "- preserved_channels should list source channels visibly present in the deck.\n"
        "- missing_or_weak_channels should list source channels weakly present or absent.\n"
        "- contradiction_flags should list deck channels that point against the source story.\n"
        "- alignment_score should be 0 to 1, where 1 means the deck strongly preserves the source story.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


def run_codex_caption(
    *,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    events_path: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> tuple[ReverseCaptionBatch | None, list[dict[str, Any]]]:
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
        str(model),
        "-c",
        f"model_reasoning_effort='{reasoning_effort}'",
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
        timeout=int(timeout_seconds),
        check=False,
    )
    _write_text(events_path, completed.stdout)
    if completed.returncode != 0:
        return None, [
            {
                "code": "codex_exec_failed",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-4000:],
            }
        ]
    try:
        return (
            ReverseCaptionBatch.model_validate_json(
                _extract_json_object(output_path.read_text(encoding="utf-8"))
            ),
            [],
        )
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        return None, [
            {
                "code": "codex_output_parse_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        ]


def build_review_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reverse-Caption Scenario Deck Audit",
        "",
        "Generated: 2026-06-03",
        "",
        "This packet reverse-captions generated 30-day scenario decks and compares "
        "the inferred deck narrative with the original conditioning story. "
        "Numerical deck summaries are computed locally; narrative captions and "
        "alignment judgments are authored by Codex/GPT.",
        "",
        "## Summary",
        "",
        f"- Status: `{report['status']}`",
        f"- Case count: `{len(report['cases'])}`",
        f"- Dry run: `{report['dry_run']}`",
        f"- Local prose generated: `{report['local_prose_generated']}`",
        "",
        "## Cases",
        "",
    ]
    for case in report["cases"]:
        lines.extend(
            [
                f"### {case['case_name']}",
                "",
                f"- Report: `{case['report_path']}`",
                f"- Arrays: `{case['arrays_path']}`",
                f"- Summary source: `{case['deck_summary'].get('summary_source', '')}`",
                f"- Sample count: `{case['deck_summary']['sample_count']}`",
                "",
                "Source story:",
                "",
                f"> {_compact(case.get('source_story'))}",
                "",
                "Generated deck factor read:",
                "",
            ]
        )
        for row in case["deck_summary"]["factor_rows"]:
            lines.append(
                "- "
                f"{row['factor']}: {row['direction']} {row['magnitude']} "
                f"(mean delta {row['terminal_mean_delta']}, "
                f"p10 {row.get('terminal_p10_delta')}, p90 {row.get('terminal_p90_delta')})"
            )
        caption = case.get("reverse_caption")
        if caption:
            lines.extend(
                [
                    "",
                    "Reverse caption:",
                    "",
                    f"> {_compact(caption['deck_caption'])}",
                    "",
                    "Alignment:",
                    "",
                    f"> {_compact(caption['source_alignment'])}",
                    "",
                    f"- Alignment score: `{caption['alignment_score']}`",
                    f"- Preserved channels: {', '.join(caption.get('preserved_channels', []))}",
                    f"- Missing or weak channels: {', '.join(caption.get('missing_or_weak_channels', []))}",
                    f"- Contradiction flags: {', '.join(caption.get('contradiction_flags', []))}",
                ]
            )
        if case.get("errors"):
            lines.extend(["", "Errors:", ""])
            for error in case["errors"]:
                lines.append(f"- `{json.dumps(error, sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def run_reverse_caption_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "reverse_caption_schema.json"
    _write_json(schema_path, strict_schema())
    discovered = discover_review_cases(args.input)
    if int(args.max_cases) > 0:
        discovered = discovered[: int(args.max_cases)]
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, case in enumerate(discovered, 1):
        deck_summary = summarize_case_deck(case)
        prompt = build_reverse_caption_prompt(case=case, deck_summary=deck_summary)
        case_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case["case_name"])).strip("_") or f"case_{index}"
        prompt_path = output_dir / f"{index:02d}_{case_slug}_prompt.txt"
        codex_output_path = output_dir / f"{index:02d}_{case_slug}_codex_output.json"
        events_path = output_dir / f"{index:02d}_{case_slug}_codex_events.jsonl"
        _write_text(prompt_path, prompt)
        reverse_caption: dict[str, Any] | None = None
        case_errors: list[dict[str, Any]] = []
        if bool(args.dry_run):
            case_errors.append({"code": "dry_run", "message": "Prompt/schema written; Codex not invoked."})
        else:
            batch, case_errors = run_codex_caption(
                prompt=prompt,
                schema_path=schema_path,
                output_path=codex_output_path,
                events_path=events_path,
                model=str(args.model),
                reasoning_effort=str(args.reasoning_effort),
                timeout_seconds=int(args.timeout_seconds),
            )
            if batch is not None:
                matching = [
                    row for row in batch.cases if row.case_name == str(case["case_name"])
                ]
                chosen = matching[0] if matching else (batch.cases[0] if batch.cases else None)
                if chosen is None:
                    case_errors.append({"code": "codex_returned_no_cases"})
                else:
                    reverse_caption = chosen.model_dump()
        errors.extend(case_errors)
        cases.append(
            {
                **case,
                "deck_summary": deck_summary,
                "reverse_caption": reverse_caption,
                "errors": case_errors,
                "artifact_paths": {
                    "prompt": str(prompt_path),
                    "codex_output": str(codex_output_path),
                    "codex_events": str(events_path),
                },
            }
        )
    report = {
        "schema_version": "reverse_caption_scenario_deck_report_v1",
        "status": "pass" if cases and (not errors or bool(args.dry_run)) else "fail",
        "dry_run": bool(args.dry_run),
        "input": str(args.input),
        "cases": cases,
        "errors": errors,
        "local_prose_generated": False,
        "codex_model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
        "elapsed_seconds": _round_float(time.time() - started),
        "artifact_paths": {
            "schema": str(schema_path),
            "report": str(output_dir / "reverse_caption_scenario_deck_report.json"),
            "markdown": str(output_dir / "reverse_caption_scenario_deck_report.md"),
        },
    }
    _write_json(report["artifact_paths"]["report"], report)
    _write_text(report["artifact_paths"]["markdown"], build_review_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_reverse_caption_audit(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": len(report["cases"]),
                "dry_run": report["dry_run"],
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
