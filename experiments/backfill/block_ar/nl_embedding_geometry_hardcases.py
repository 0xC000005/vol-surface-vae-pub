"""Compare raw text-embedding and bridge-condition hard-negative geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    anchor_similarity_metrics,
)


def _round(value: Any, digits: int = 12) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw != raw or raw in {float("inf"), float("-inf")}:
        return None
    return round(raw, digits)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _window_examples(
    examples: list[dict[str, Any]], window_id: str
) -> list[dict[str, Any]]:
    rows = [row for row in examples if str(row.get("window_id")) == str(window_id)]
    role_order = {"anchor": 0, "positive": 1, "negative": 2}
    rows.sort(
        key=lambda row: (role_order.get(str(row.get("role")), 99), row.get("kind", ""))
    )
    return rows


def _metrics_for_rows(
    rows: list[dict[str, Any]], vectors: np.ndarray
) -> dict[str, Any]:
    indices = [int(row["embedding_index"]) for row in rows]
    roles = [str(row["role"]) for row in rows]
    metrics = anchor_similarity_metrics(
        vectors[np.asarray(indices, dtype=np.int64)], roles
    )
    return {
        key: _round(value) if isinstance(value, (float, int)) else value
        for key, value in metrics.items()
    }


def _diagnosis(raw_margin: float | None, condition_margin: float | None) -> str:
    raw_low = raw_margin is not None and raw_margin < 0.50
    condition_low = condition_margin is not None and condition_margin < 0.50
    if raw_low and condition_low:
        return "raw_embedding_and_adapter_low_separation"
    if (not raw_low) and condition_low:
        return "adapter_collapses_raw_separation"
    if raw_low and (not condition_low):
        return "adapter_repairs_raw_separation"
    return "separation_ok"


def contrast_geometry_for_window(
    examples: list[dict[str, Any]],
    *,
    raw_embeddings: np.ndarray,
    condition_vectors: np.ndarray,
    window_id: str,
) -> dict[str, Any]:
    rows = _window_examples(examples, window_id)
    if not any(str(row.get("role")) == "anchor" for row in rows):
        raise ValueError(f"{window_id}: missing anchor example")
    if not any(str(row.get("role")) == "positive" for row in rows):
        raise ValueError(f"{window_id}: missing positive examples")
    if not any(str(row.get("role")) == "negative" for row in rows):
        raise ValueError(f"{window_id}: missing negative examples")
    raw_metrics = _metrics_for_rows(rows, raw_embeddings)
    condition_metrics = _metrics_for_rows(rows, condition_vectors)
    return {
        "window_id": str(window_id),
        "example_count": len(rows),
        "negative_count": sum(1 for row in rows if str(row.get("role")) == "negative"),
        "positive_count": sum(1 for row in rows if str(row.get("role")) == "positive"),
        "raw": raw_metrics,
        "condition": condition_metrics,
        "diagnosis": _diagnosis(
            raw_metrics.get("hard_margin"),
            condition_metrics.get("hard_margin"),
        ),
    }


def build_embedding_geometry_report(
    *,
    pipeline_report: dict[str, Any],
    pipeline_arrays: dict[str, np.ndarray],
    bridge_arrays: dict[str, np.ndarray],
    window_ids: list[str],
) -> dict[str, Any]:
    examples = build_bridge_examples(pipeline_report)
    raw_embeddings = np.asarray(pipeline_arrays["text_embeddings"], dtype=np.float32)
    condition_vectors = np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32)
    cases = [
        contrast_geometry_for_window(
            examples,
            raw_embeddings=raw_embeddings,
            condition_vectors=condition_vectors,
            window_id=window_id,
        )
        for window_id in window_ids
    ]
    diagnosis_counts: dict[str, int] = {}
    for case in cases:
        diagnosis = str(case["diagnosis"])
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
    raw_margins = [float(case["raw"]["hard_margin"]) for case in cases]
    condition_margins = [float(case["condition"]["hard_margin"]) for case in cases]
    return {
        "status": "ok",
        "scope_note": (
            "Offline geometry attribution. Raw text embeddings are compared to "
            "learned condition vectors for the same anchor/positive/negative groups."
        ),
        "window_ids": list(window_ids),
        "summary": {
            "case_count": len(cases),
            "diagnosis_counts": diagnosis_counts,
            "raw_hard_margin_mean": (
                _round(float(np.mean(raw_margins))) if raw_margins else None
            ),
            "condition_hard_margin_mean": (
                _round(float(np.mean(condition_margins))) if condition_margins else None
            ),
            "condition_minus_raw_margin_mean": (
                _round(
                    float(
                        np.mean(np.asarray(condition_margins) - np.asarray(raw_margins))
                    )
                )
                if raw_margins
                else None
            ),
        },
        "cases": cases,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Embedding Geometry Hard Cases",
        "",
        f"- Cases: `{report['summary']['case_count']}`",
        f"- Diagnosis counts: `{report['summary']['diagnosis_counts']}`",
        f"- Raw mean hard margin: `{report['summary']['raw_hard_margin_mean']}`",
        f"- Condition mean hard margin: `{report['summary']['condition_hard_margin_mean']}`",
        f"- Condition minus raw margin: `{report['summary']['condition_minus_raw_margin_mean']}`",
        "",
        "| Window | Diagnosis | Raw Margin | Condition Margin | Raw Gap | Condition Gap |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        lines.append(
            "| {window} | `{diagnosis}` | {raw_margin} | {cond_margin} | {raw_gap} | {cond_gap} |".format(
                window=case["window_id"],
                diagnosis=case["diagnosis"],
                raw_margin=case["raw"]["hard_margin"],
                cond_margin=case["condition"]["hard_margin"],
                raw_gap=case["raw"]["separation_mean"],
                cond_gap=case["condition"]["separation_mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", required=True)
    parser.add_argument("--pipeline-arrays", required=True)
    parser.add_argument("--bridge-arrays", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-ids", required=True)
    args = parser.parse_args()

    pipeline_report = json.loads(Path(args.pipeline_report).read_text(encoding="utf-8"))
    pipeline_arrays = _load_npz(args.pipeline_arrays)
    bridge_arrays = _load_npz(args.bridge_arrays)
    window_ids = [
        item.strip() for item in str(args.window_ids).split(",") if item.strip()
    ]
    report = build_embedding_geometry_report(
        pipeline_report=pipeline_report,
        pipeline_arrays=pipeline_arrays,
        bridge_arrays=bridge_arrays,
        window_ids=window_ids,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "embedding_geometry_hardcases.json"
    markdown_path = output_dir / "embedding_geometry_hardcases.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(markdown_path, report)
    print(
        json.dumps(
            {"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
