#!/usr/bin/env python
"""Convert condition-only grounding output into a prefix-latent condition report.

The output contract matches ``nl_prefix_latent_story_smoke.py --condition-report``:
the report contains a cached query payload and an arrays file with one
``text_memory`` row. The default embeds the clean condition query text, but
query-channel modes can preserve the raw narrative or compare implication-only
conditioning for grounding-bottleneck ablations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_start_sensitivity import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ADAPTER,
    _load_bridge_adapter,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    load_bridge_arrays,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    embed_texts_with_openai,
    normalize_rows,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_report_809a"
)

QUERY_CHANNELS = (
    "grounded_condition",
    "raw_narrative",
    "implications_only",
    "narrative_plus_implications",
    "narrative_plus_grounding",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_condition_case(
    *,
    case_json: str | Path | None,
    summary_json: str | Path | None,
    case_name: str | None,
    case_index: int,
) -> dict[str, Any]:
    """Load one condition-only grounding case payload."""

    if case_json:
        return _load_json(case_json)
    if not summary_json:
        raise ValueError("either --case-json or --summary-json is required")
    summary = _load_json(summary_json)
    cases = summary.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{summary_json}: expected non-empty cases list")
    if case_name:
        for row in cases:
            if isinstance(row, dict) and str(row.get("case_name", "")) == str(case_name):
                return row
        raise ValueError(f"{summary_json}: no case named {case_name!r}")
    idx = int(case_index)
    if idx < 0 or idx >= len(cases):
        raise IndexError(f"case_index {idx} outside {len(cases)} cases")
    row = cases[idx]
    if not isinstance(row, dict):
        raise ValueError(f"{summary_json}: case_index {idx} is not an object")
    return row


def _condition_implication_rows(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("current_market_state_implications", "recent_regime_implications"):
        items = grounding.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "market": str(item.get("market", "")),
                    "direction": str(item.get("direction", "")),
                    "magnitude": str(item.get("magnitude", "")),
                    "confidence": str(item.get("confidence", "")),
                    "evidence": item.get("evidence", []),
                    "inferred": bool(item.get("inferred", False)),
                    "horizon": str(item.get("horizon", "")),
                    "target_use": "support_prior",
                }
            )
    return rows


def _format_implication_for_query(item: dict[str, Any]) -> str:
    evidence = item.get("evidence", [])
    if isinstance(evidence, list):
        evidence_text = "; ".join(str(part) for part in evidence if str(part))
    else:
        evidence_text = str(evidence)
    return (
        f"{item.get('market', '')} {item.get('direction', '')} "
        f"{item.get('magnitude', '')} confidence={item.get('confidence', '')} "
        f"horizon={item.get('horizon', '')} target={item.get('target_use', '')} "
        f"inferred={bool(item.get('inferred', False))} evidence={evidence_text}"
    ).strip()


def _format_sidecar_items(label: str, items: Any) -> list[str]:
    if not isinstance(items, list) or not items:
        return [f"{label}: none"]
    lines = [f"{label}:"]
    for item in items:
        if isinstance(item, dict):
            compact = ", ".join(
                f"{key}={value}"
                for key, value in sorted(item.items())
                if isinstance(value, (str, int, float, bool))
            )
            lines.append(
                f"- {compact}" if compact else f"- {json.dumps(item, sort_keys=True)}"
            )
        else:
            lines.append(f"- {item}")
    return lines


def condition_query_text_for_channel(
    case: dict[str, Any],
    *,
    query_channel: str = "grounded_condition",
) -> str:
    """Build text for a condition-memory projection channel."""

    channel = str(query_channel)
    if channel not in QUERY_CHANNELS:
        raise ValueError(
            f"unknown query_channel {channel!r}; expected one of {QUERY_CHANNELS}"
        )
    story = str(case.get("story", "")).strip()
    grounding = case.get("condition_only_grounding", {})
    if not isinstance(grounding, dict):
        grounding = {}
    grounded_query = str(case.get("candidate_query_text", "")).strip()
    if channel == "grounded_condition":
        if not grounded_query:
            raise ValueError("condition case missing candidate_query_text")
        return grounded_query
    if channel == "raw_narrative":
        if not story:
            raise ValueError("condition case missing story")
        return "RAW_RISK_MANAGER_NARRATIVE:\n" + story

    implication_rows = _condition_implication_rows(grounding)
    implication_lines = [
        "EXPLICIT_MARKET_IMPLICATIONS:",
        *[f"- {_format_implication_for_query(row)}" for row in implication_rows],
    ]
    if len(implication_lines) == 1:
        implication_lines.append("- none")
    if channel == "implications_only":
        return "\n".join(implication_lines)

    if channel == "narrative_plus_implications":
        return "\n".join(
            [
                "RAW_RISK_MANAGER_NARRATIVE:",
                story or "missing",
                "",
                *implication_lines,
            ]
        )

    lines = [
        "RAW_RISK_MANAGER_NARRATIVE:",
        story or "missing",
        "",
        "GROUNDING_SIDECAR:",
        f"NARRATIVE_FRAME: {grounding.get('narrative_frame', '')}",
        f"CURRENT_MARKET_STATE_SUMMARY: {grounding.get('current_market_state_summary', '')}",
        f"RECENT_REGIME_SUMMARY: {grounding.get('recent_regime_summary', '')}",
        f"CLEANED_CONDITIONING_TEXT: {grounding.get('cleaned_conditioning_text', '')}",
        "",
        *implication_lines,
        "",
        *_format_sidecar_items(
            "GROUNDING_WARNINGS",
            grounding.get("grounding_warnings", []),
        ),
        "",
        *_format_sidecar_items(
            "UNSUPPORTED_CLAIMS",
            grounding.get("unsupported_claims", []),
        ),
        "",
        *_format_sidecar_items(
            "NON_CONDITIONING_FORWARD_LANGUAGE",
            grounding.get("non_conditioning_forward_language", []),
        ),
    ]
    return "\n".join(lines)


def compatible_grounding_from_condition_case(case: dict[str, Any]) -> dict[str, Any]:
    """Build the grounding shape consumed by analogue-mixture alignment."""

    grounding = case.get("condition_only_grounding", {})
    if not isinstance(grounding, dict):
        raise ValueError("condition case missing condition_only_grounding")
    return {
        "narrative_frame": str(grounding.get("narrative_frame", "")),
        "cleaned_conditioning_text": str(grounding.get("cleaned_conditioning_text", "")),
        "market_implications": _condition_implication_rows(grounding),
        "grounding_warnings": grounding.get("grounding_warnings", []),
        "unsupported_claims": grounding.get("unsupported_claims", []),
        "non_conditioning_forward_language": grounding.get(
            "non_conditioning_forward_language",
            [],
        ),
        "condition_only_grounding": grounding,
        "condition_only_validation": case.get("condition_only_validation", {}),
        "story_split": case.get("story_split", {}),
    }


def project_condition_query_text(
    *,
    query_text: str,
    embedding_model: str,
    bridge_adapter: str | Path,
    condition_dim: int,
    dotenv_path: str | Path,
    embedder: Any = embed_texts_with_openai,
    adapter_loader: Any = _load_bridge_adapter,
) -> dict[str, Any]:
    """Embed clean condition text and project it into condition-memory space."""

    embedding = np.asarray(
        embedder(
            [str(query_text)],
            model=str(embedding_model),
            dotenv_path=dotenv_path,
            batch_size=1,
        ),
        dtype=np.float32,
    )
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("embedder must return one 2-D embedding row")
    adapter = adapter_loader(
        bridge_adapter,
        embedding_dim=int(embedding.shape[1]),
        condition_dim=int(condition_dim),
    )
    adapter.eval() if hasattr(adapter, "eval") else None
    with torch.no_grad():
        condition = (
            adapter(torch.from_numpy(normalize_rows(embedding)).float())
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )
    return {
        "query_embedding": embedding[0].astype(np.float32),
        "query_condition": condition,
        "embedding_metadata": {
            "embedding_model": str(embedding_model),
            "embedding_dim": int(embedding.shape[1]),
            "condition_dim": int(condition_dim),
            "query_text_length": int(len(query_text)),
            "query_text_line_count": int(str(query_text).count("\n") + 1),
        },
    }


def build_condition_report(
    *,
    case: dict[str, Any],
    projection: dict[str, Any],
    output_dir: str | Path,
    bridge_arrays: str | Path,
    bridge_adapter: str | Path,
    query_text: str | None = None,
    query_channel: str = "grounded_condition",
) -> dict[str, Any]:
    """Write a condition report and arrays file for story smoke."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_query_text = (
        str(query_text)
        if query_text is not None
        else condition_query_text_for_channel(case, query_channel=query_channel)
    )
    if not selected_query_text:
        raise ValueError("condition case produced empty query text")
    query_condition = np.asarray(projection["query_condition"], dtype=np.float32)
    query_embedding = np.asarray(projection["query_embedding"], dtype=np.float32)
    arrays_path = output / "condition_only_report_arrays.npz"
    np.savez_compressed(
        arrays_path,
        text_memory=query_condition.reshape(1, -1).astype(np.float32),
        query_embedding=query_embedding.reshape(1, -1).astype(np.float32),
    )
    grounding = compatible_grounding_from_condition_case(case)
    validation = case.get("condition_only_validation", {})
    report_path = output / "condition_only_report.json"
    report = {
        "status": "ok",
        "scope_note": (
            "Condition-only narrative report for prefix-latent story smoke. "
            "Only clean current/recent condition text was embedded."
        ),
        "case_name": str(case.get("case_name", "")),
        "cached_query": {
            "condition_source": "condition_only_openai_story",
            "role": "condition_only_story",
            "kind": "condition_only_grounding",
            "window_id": str(case.get("case_name", "condition_only_story")),
            "embedding_index": -1,
            "narrative_text": str(case.get("story", "")),
            "query_text": selected_query_text,
            "grounding": grounding,
            "embedding_metadata": {
                **dict(projection.get("embedding_metadata", {})),
                "query_channel": str(query_channel),
                "grounding_model": str(
                    case.get("metadata", {}).get("model", "")
                    if isinstance(case.get("metadata"), dict)
                    else ""
                ),
                "condition_only_validation_status": str(
                    validation.get("status", "")
                    if isinstance(validation, dict)
                    else ""
                ),
                "bridge_arrays": str(bridge_arrays),
                "bridge_adapter": str(bridge_adapter),
            },
        },
        "artifact_paths": {
            "report": str(report_path),
            "arrays": str(arrays_path),
        },
    }
    _write_json(report_path, report)
    return report


def run_condition_only_report(args: argparse.Namespace) -> dict[str, Any]:
    case = select_condition_case(
        case_json=args.case_json,
        summary_json=args.summary_json,
        case_name=args.case_name,
        case_index=int(args.case_index),
    )
    query_channel = str(getattr(args, "query_channel", "grounded_condition"))
    query_text = condition_query_text_for_channel(case, query_channel=query_channel)
    arrays = load_bridge_arrays(args.bridge_arrays)
    condition_dim = int(np.asarray(arrays["memory_targets"]).shape[1])
    projection = project_condition_query_text(
        query_text=query_text,
        embedding_model=str(args.embedding_model),
        bridge_adapter=args.bridge_adapter,
        condition_dim=condition_dim,
        dotenv_path=args.dotenv,
    )
    return build_condition_report(
        case=case,
        projection=projection,
        output_dir=args.output_dir,
        bridge_arrays=args.bridge_arrays,
        bridge_adapter=args.bridge_adapter,
        query_text=query_text,
        query_channel=query_channel,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json")
    parser.add_argument("--summary-json")
    parser.add_argument("--case-name")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--bridge-adapter", default=DEFAULT_BRIDGE_ADAPTER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--query-channel",
        choices=QUERY_CHANNELS,
        default="grounded_condition",
        help=(
            "Text channel to embed. Use narrative_plus_grounding to preserve "
            "the full story with grounding as a sidecar."
        ),
    )
    parser.add_argument("--dotenv", default=".env")
    args = parser.parse_args()
    report = run_condition_only_report(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "arrays": report["artifact_paths"]["arrays"],
                "condition_source": report["cached_query"]["condition_source"],
                "text_memory_dim": report["cached_query"]["embedding_metadata"][
                    "condition_dim"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
