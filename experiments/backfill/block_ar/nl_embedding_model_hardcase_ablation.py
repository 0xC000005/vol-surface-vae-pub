"""Tiny embedding-model and representation ablation for hard cases."""

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
    embed_texts_with_openai,
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


def representation_text(text: str, *, representation: str) -> str:
    """Return the text view to embed for one example."""

    raw = str(text)
    if representation == "full":
        return raw
    if representation == "factor_tokens":
        marker = "MARKET_IMPLICATIONS:"
        if marker in raw:
            return raw.split(marker, 1)[1].split("\n", 1)[0].strip()
        return raw
    raise ValueError(f"unknown representation: {representation}")


def _window_rows(
    examples: list[dict[str, Any]],
    window_ids: list[str],
    *,
    representation: str,
) -> list[dict[str, Any]]:
    selected = {str(item) for item in window_ids}
    rows: list[dict[str, Any]] = []
    role_order = {"anchor": 0, "positive": 1, "negative": 2}
    for example in examples:
        window_id = str(example.get("window_id", ""))
        if window_id not in selected:
            continue
        rows.append(
            {
                "window_id": window_id,
                "role": str(example.get("role", "")),
                "kind": str(example.get("kind", "")),
                "text": representation_text(
                    str(example.get("text", "")), representation=representation
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            window_ids.index(row["window_id"]),
            role_order.get(row["role"], 99),
            row["kind"],
        )
    )
    return rows


def _case_metrics(
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    window_id: str,
) -> dict[str, Any]:
    local_indices = [
        idx for idx, row in enumerate(rows) if row["window_id"] == window_id
    ]
    roles = [rows[idx]["role"] for idx in local_indices]
    metrics = anchor_similarity_metrics(
        embeddings[np.asarray(local_indices, dtype=np.int64)],
        roles,
    )
    return {
        "window_id": window_id,
        "example_count": len(local_indices),
        "positive_count": sum(1 for role in roles if role == "positive"),
        "negative_count": sum(1 for role in roles if role == "negative"),
        "metrics": {
            key: _round(value) if isinstance(value, (float, int)) else value
            for key, value in metrics.items()
        },
    }


def build_embedding_model_ablation(
    *,
    pipeline_report: dict[str, Any],
    window_ids: list[str],
    embedding_model: str,
    representation: str,
    dotenv_path: str,
    batch_size: int,
) -> dict[str, Any]:
    examples = build_bridge_examples(pipeline_report)
    rows = _window_rows(examples, window_ids, representation=representation)
    if not rows:
        raise ValueError("no rows selected for ablation")
    embeddings = embed_texts_with_openai(
        [row["text"] for row in rows],
        model=embedding_model,
        dotenv_path=dotenv_path,
        batch_size=batch_size,
    )
    cases = [
        _case_metrics(rows, embeddings, window_id=window_id) for window_id in window_ids
    ]
    margins = [float(case["metrics"]["hard_margin"]) for case in cases]
    gaps = [float(case["metrics"]["separation_mean"]) for case in cases]
    return {
        "status": "ok",
        "scope_note": (
            "OpenAI embedding-only hard-case ablation. No labels, bridge "
            "training, memory targets, or scenario generation are changed."
        ),
        "embedding_model": embedding_model,
        "representation": representation,
        "window_ids": list(window_ids),
        "row_count": len(rows),
        "summary": {
            "case_count": len(cases),
            "mean_hard_margin": _round(float(np.mean(margins))),
            "mean_separation_gap": _round(float(np.mean(gaps))),
            "low_margin_count": int(np.sum(np.asarray(margins) < 0.50)),
        },
        "cases": cases,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Embedding Model Hard-Case Ablation",
        "",
        f"- Model: `{report['embedding_model']}`",
        f"- Representation: `{report['representation']}`",
        f"- Rows embedded: `{report['row_count']}`",
        f"- Mean hard margin: `{report['summary']['mean_hard_margin']}`",
        f"- Mean separation gap: `{report['summary']['mean_separation_gap']}`",
        f"- Low-margin cases: `{report['summary']['low_margin_count']}`",
        "",
        "| Window | Hard Margin | Gap | Positive Mean | Negative Mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        metrics = case["metrics"]
        lines.append(
            "| {window} | {margin} | {gap} | {pos} | {neg} |".format(
                window=case["window_id"],
                margin=metrics["hard_margin"],
                gap=metrics["separation_mean"],
                pos=metrics["positive_mean_cosine"],
                neg=metrics["negative_mean_cosine"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-ids", required=True)
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument(
        "--representation", choices=["full", "factor_tokens"], default="full"
    )
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--dotenv", default=".env")
    args = parser.parse_args()

    report = json.loads(Path(args.pipeline_report).read_text(encoding="utf-8"))
    window_ids = [
        item.strip() for item in str(args.window_ids).split(",") if item.strip()
    ]
    output = build_embedding_model_ablation(
        pipeline_report=report,
        window_ids=window_ids,
        embedding_model=str(args.embedding_model),
        representation=str(args.representation),
        dotenv_path=str(args.dotenv),
        batch_size=int(args.embedding_batch_size),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "embedding_model_hardcase_ablation.json"
    markdown_path = output_dir / "embedding_model_hardcase_ablation.md"
    json_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(markdown_path, output)
    print(
        json.dumps(
            {"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
